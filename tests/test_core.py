import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from vsid.config import Config
from vsid.models import kd_loss
from vsid.verifier import wilson_lower, class_entropy
from vsid.data import make_splits


def test_kd_loss_reduces_to_ce_and_kl():
    torch.manual_seed(0)
    zs, zt = torch.randn(8, 5), torch.randn(8, 5)
    y = torch.randint(0, 5, (8,))
    ce = F.cross_entropy(zs, y, reduction="none")
    kl = F.kl_div(F.log_softmax(zs, 1), F.softmax(zt, 1), reduction="none").sum(1)
    assert torch.allclose(kd_loss(zs, y, zt, alpha=1.0, T=1.0), ce, atol=1e-6)          # alpha=1 -> pure CE
    assert torch.allclose(kd_loss(zs, y, zt, alpha=0.0, T=1.0), kl, atol=1e-6)          # alpha=0, T=1 -> pure KL
    assert torch.allclose(kd_loss(zs, y, None, alpha=0.3, T=4.0), ce, atol=1e-6)        # no teacher -> CE


def test_kd_loss_temperature_scaling_and_unlabelled_rows():
    torch.manual_seed(1)
    zs, zt = torch.randn(6, 4), torch.randn(6, 4)
    y = torch.tensor([0, 1, -1, -1, 2, -1])                                           # -1 = no trusted label
    T = 4.0
    kl = F.kl_div(F.log_softmax(zs / T, 1), F.softmax(zt / T, 1), reduction="none").sum(1) * T * T
    out = kd_loss(zs, y, zt, alpha=0.5, T=T)
    assert torch.allclose(out[y < 0], kl[y < 0], atol=1e-6)                           # unlabelled rows: pure soft-target loss
    ce = F.cross_entropy(zs, y.clamp(min=0), reduction="none")
    assert torch.allclose(out[y >= 0], 0.5 * ce[y >= 0] + 0.5 * kl[y >= 0], atol=1e-6)


def test_wilson_and_entropy():
    assert wilson_lower(100, 100, 1.28) > 0.97
    assert wilson_lower(90, 100, 1.28) < 0.90 < wilson_lower(95, 100, 1.28)
    assert wilson_lower(0, 0, 1.28) == 0.0
    assert abs(class_entropy(np.arange(10), 10) - 1.0) < 1e-9
    assert class_entropy(np.zeros(50, dtype=int), 10) == 0.0


def _toy(n_per_class=60, C=6, seed=0):
    rng = np.random.RandomState(seed)
    vocab = [[f"w{c}_{j}" for j in range(8)] + ["common", "the", "my"] for c in range(C)]
    rows = []
    for c in range(C):
        for i in range(n_per_class):
            words = list(rng.choice(vocab[c], 5)) + list(rng.choice(["the", "my", "common", "please"], 2))
            rows.append((" ".join(words) + f" id{c}x{i}", f"intent{c}", c))
    return pd.DataFrame(rows, columns=["text", "intent", "label"]).sample(frac=1, random_state=seed).reset_index(drop=True)


def test_splits_are_leak_free_and_sized():
    cfg = Config(); cfg.n_dev, cfg.seed_per_class = 40, 4
    df = _toy(80); tr, te = df.iloc[:360].copy(), df.iloc[360:].copy()
    te = pd.concat([te, tr.iloc[:5]])                      # inject 5 train/test duplicates -> must be removed from train
    sp = make_splits(tr, te, [f"intent{c}" for c in range(6)], cfg, 0)
    assert sp.info["leakage_free"] and sp.info["n_train_rows_also_in_test_removed"] == 5
    assert len(sp.seed) == 24 and all(v == 0 for v in sp.info["pairwise_text_overlap"].values())


def test_loop_smoke_and_invariants():
    from vsid.features import TeacherFeaturizer
    from vsid.loop import run_loop, fit_teacher
    cfg = Config(); cfg.n_dev, cfg.seed_per_class = 60, 5
    cfg.rounds, cfg.epochs_teacher, cfg.n_boot, cfg.audit_per_round = 2, 8, 50, 20
    cfg.teacher_word_features, cfg.teacher_char_features, cfg.verifier_char_features = 300, 600, 600
    cfg.teacher_hidden, cfg.tau0, cfg.min_gain, cfg.p_boot = 32, 0.4, -1.0, 0.0
    df = _toy(100); tr, te = df.iloc[:480].copy(), df.iloc[480:].copy()
    sp = make_splits(tr, te, [f"intent{c}" for c in range(6)], cfg, 0)
    feat = TeacherFeaturizer(cfg.teacher_word_features, cfg.teacher_char_features).fit(sp.seed.text.tolist() + sp.pool.text.tolist())
    t0 = fit_teacher(cfg, feat, sp.seed.text.tolist(), sp.seed.label.tolist(), 0, 6)
    state, rounds = run_loop(cfg, "full", sp, feat, t0, 0, log=lambda *_: None)
    assert len(rounds) >= 1 and all("decision" in r and "gates" in r for r in rounds)
    # human-audited labels are true labels, and no dev/test text ever reaches the trusted set
    truth = dict(zip(sp.pool.text, sp.pool.label))
    n_seed = len(sp.seed)
    for t, l in list(zip(state["trusted_texts"], state["trusted_labels"]))[n_seed:]:
        assert truth[t] == l
    forbidden = set(sp.dev.text) | set(sp.test.text)
    assert not (set(state["trusted_texts"]) & forbidden)
    assert state["human_labels"] == sum(r.get("n_audited", 0) for r in rounds)
    # naive mode never spends human labels
    st2, _ = run_loop(cfg, "naive", sp, feat, t0, 0, log=lambda *_: None)
    assert st2["human_labels"] == 0 and len(st2["trusted_texts"]) == n_seed
