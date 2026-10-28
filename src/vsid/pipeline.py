"""End-to-end experiment for one seed -> results/run_seed{S}.json (+ model artifacts for seed 0)."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
import joblib
from collections import Counter

from .config import Config
from .data import load_banking77, make_splits
from .features import TeacherFeaturizer, StudentFeaturizer
from .models import predict_logits
from .loop import run_loop, fit_teacher
from .distill import run_student_suite, distill_one
from .evaluate import system_report, evaluate_model


def _json(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def run_seed(cfg: Config, seed: int, train, test, classes, log=print, save_artifacts=False) -> dict:
    t_start = time.time()
    torch.set_num_threads(cfg.torch_threads)
    C = len(classes)
    sp = make_splits(train, test, classes, cfg, seed)
    log(f"[seed {seed}] splits: {sp.info['sizes']}  leakage_free={sp.info['leakage_free']}")
    fit_texts = sp.seed["text"].tolist() + sp.pool["text"].tolist()          # unlabelled train-side text only
    tfeat = TeacherFeaturizer(cfg.teacher_word_features, cfg.teacher_char_features).fit(fit_texts)
    sfeat = StudentFeaturizer(cfg.student_word_features).fit(fit_texts)
    test_texts, ytest = sp.test["text"].tolist(), sp.test["label"].values
    Xt_teacher, Xt_student = tfeat.transform(test_texts), sfeat.transform(test_texts)

    # ---- initial teacher (supervised on the tiny seed) -------------------------------------------------
    t0 = fit_teacher(cfg, tfeat, sp.seed["text"].tolist(), sp.seed["label"].tolist(), seed * 100, C)
    log(f"[seed {seed}] teacher0 dev: {evaluate_model(t0, tfeat.transform(sp.dev['text'].tolist()), sp.dev['label'].values)['acc']:.3f}")

    # ---- recursive loops (full + 2 ablations) ----------------------------------------------------------
    states, loop_logs = {}, {}
    for mode in ("full", "verifier_only", "naive"):
        states[mode], loop_logs[mode] = run_loop(cfg, mode, sp, tfeat, t0, seed, log=log)

    # ---- baselines -------------------------------------------------------------------------------------
    k = states["full"]["human_labels"]
    rng = np.random.RandomState(500 + seed)
    ridx = rng.choice(len(sp.pool), size=k, replace=False) if k else np.array([], dtype=int)
    matched = fit_teacher(cfg, tfeat, sp.seed["text"].tolist() + sp.pool.iloc[ridx]["text"].tolist(),
                          sp.seed["label"].tolist() + sp.pool.iloc[ridx]["label"].tolist(), seed * 100 + 51, C)
    oracle = fit_teacher(cfg, tfeat, sp.seed["text"].tolist() + sp.pool["text"].tolist(),
                         sp.seed["label"].tolist() + sp.pool["label"].tolist(), seed * 100 + 52, C)

    # ---- students --------------------------------------------------------------------------------------
    students, sweep, best = run_student_suite(cfg, sp, states["full"], tfeat, sfeat, seed, log=log)
    state0 = {"teacher": t0, "trusted_texts": sp.seed["text"].tolist(), "trusted_labels": sp.seed["label"].tolist(),
              "trusted_pool_idx": []}
    s_from_t0 = distill_one(cfg, sp, state0, tfeat, sfeat, seed)

    # ---- FINAL TEST (touched once per system) ----------------------------------------------------------
    systems = []
    T_ = lambda name, m, extra=None: systems.append(system_report(name, m, tfeat, Xt_teacher, ytest, test_texts, cfg, {"kind": "teacher", **(extra or {})}))
    T_("T0 teacher, seed labels only (start of loop)", t0, {"human_labels": len(sp.seed)})
    T_("T1 teacher, label-budget-matched supervised", matched, {"human_labels": len(sp.seed) + k})
    T_("T2 teacher, naive self-training (no verifier, no gates)", states["naive"]["teacher"], {"human_labels": len(sp.seed)})
    T_("T3 teacher, verifier only (no audit, no gates)", states["verifier_only"]["teacher"], {"human_labels": len(sp.seed)})
    T_("T4 teacher, FULL verified loop", states["full"]["teacher"], {"human_labels": len(sp.seed) + k})
    T_("T5 teacher, oracle (all pool labels; upper bound)", oracle, {"human_labels": len(sp.seed) + len(sp.pool)})
    for name, m in students.items():
        systems.append(system_report(name, m, sfeat, Xt_student, ytest, test_texts, cfg, {"kind": "student"}))
    systems.append(system_report("S5 student, distilled from INITIAL teacher (no loop)", s_from_t0, sfeat, Xt_student, ytest,
                                 test_texts, cfg, {"kind": "student"}))

    # ---- failure analysis ------------------------------------------------------------------------------
    final_t, s4 = states["full"]["teacher"], students["S4 student, distilled (T, alpha tuned on dev)"]
    pt, ps = predict_logits(final_t, Xt_teacher).argmax(1), predict_logits(s4, Xt_student).argmax(1)
    p0 = predict_logits(t0, Xt_teacher).argmax(1)
    rec = lambda p: [float((p[ytest == c] == c).mean()) for c in range(C)]
    conf_pairs = Counter((classes[a], classes[b]) for a, b in zip(ytest, ps) if a != b).most_common(12)
    wrong = np.where(ps != ytest)[0][:25]
    failure = {
        "per_class_recall": {"teacher0": rec(p0), "teacher_full": rec(pt), "student_s4": rec(ps)},
        "top_confusions_student": [{"true": a, "pred": b, "count": n} for (a, b), n in conf_pairs],
        "examples": [{"text": test_texts[i], "true": classes[ytest[i]], "student": classes[ps[i]], "teacher": classes[pt[i]]} for i in wrong],
        "student_teacher_agreement": float((pt == ps).mean()),
    }
    out = {"seed": seed, "config": cfg.to_dict(), "split_info": sp.info, "classes": classes,
           "systems": systems, "loops": loop_logs, "sweep": sweep, "sweep_best": best,
           "human_labels_full": k, "failure": failure, "runtime_sec": time.time() - t_start}
    rd = Path(cfg.results_dir); rd.mkdir(parents=True, exist_ok=True)
    (rd / f"run_seed{seed}.json").write_text(json.dumps(out, default=_json, indent=1))
    if save_artifacts:
        ad = rd / "artifacts"; ad.mkdir(exist_ok=True)
        torch.save(final_t.state_dict(), ad / "teacher.pt"); torch.save(s4.state_dict(), ad / "student.pt")
        torch.save(students["S0 student, hard labels (baseline)"].state_dict(), ad / "student_hard.pt")
        joblib.dump({"tfeat": tfeat, "sfeat": sfeat, "classes": classes, "cfg": cfg.to_dict(),
                     "n_tf": tfeat.n_features, "n_sf": sfeat.n_features}, ad / "featurizers.joblib", compress=3)
    log(f"[seed {seed}] done in {out['runtime_sec']:.0f}s")
    return out


def run_all(cfg: Config, log=print, seeds=None):
    train, test, classes = load_banking77(cfg.data_dir)
    for i, s in enumerate(seeds if seeds is not None else cfg.seeds):
        run_seed(cfg, s, train, test, classes, log=log, save_artifacts=(i == 0))
