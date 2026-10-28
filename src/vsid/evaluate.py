"""Metrics, paired bootstrap, size and latency measurement."""
from __future__ import annotations
import time
import numpy as np
import torch
from sklearn.metrics import f1_score
from .models import csr_batch, n_params, model_size_mb, predict_logits


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {"acc": float((y_true == y_pred).mean()), "macro_f1": float(f1_score(y_true, y_pred, average="macro"))}


def bootstrap_ci(correct: np.ndarray, n_boot: int = 1000, seed: int = 0):
    rng = np.random.RandomState(seed)
    n = len(correct)
    means = np.array([correct[rng.randint(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def p_improve(correct_new: np.ndarray, correct_old: np.ndarray, n_boot: int, seed: int = 0) -> float:
    """Paired bootstrap: P(new accuracy > old accuracy) on the same examples."""
    rng = np.random.RandomState(seed)
    n = len(correct_new)
    d = correct_new.astype(float) - correct_old.astype(float)
    wins = sum(d[rng.randint(0, n, n)].mean() > 0 for _ in range(n_boot))
    return wins / n_boot


def evaluate_model(model, X, y) -> dict:
    logits = predict_logits(model, X)
    pred = logits.argmax(1)
    m = metrics(y, pred)
    m["correct"] = (pred == np.asarray(y))
    return m


def latency_profile(model, featurizer, texts, cfg) -> dict:
    """End-to-end CPU latency (featurise + forward), batch=1, 1 thread; plus batch-256 throughput."""
    torch.set_num_threads(cfg.torch_threads)
    model.eval()
    texts = list(texts)[: cfg.latency_queries + cfg.latency_warmup]
    lat = []
    lat_model = []
    with torch.no_grad():
        for i, t in enumerate(texts):
            best_total, best_model = 1e9, 1e9
            for _ in range(3):                                   # min-of-3 filters OS scheduling noise
                t0 = time.perf_counter()
                X = featurizer.transform([t])
                t1 = time.perf_counter()
                idx, off, w = csr_batch(X)
                model(idx, off, w)
                t2 = time.perf_counter()
                best_total, best_model = min(best_total, t2 - t0), min(best_model, t2 - t1)
            if i >= cfg.latency_warmup:
                lat.append(best_total * 1000)
                lat_model.append(best_model * 1000)
        big = texts[:256] if len(texts) >= 256 else texts
        t0 = time.perf_counter()
        Xb = featurizer.transform(big)
        idx, off, w = csr_batch(Xb)
        model(idx, off, w)
        thr = len(big) / (time.perf_counter() - t0)
    return {"p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
            "model_only_p50_ms": float(np.percentile(lat_model, 50)), "throughput_qps": float(thr)}


def system_report(name, model, featurizer, X_test, y_test, test_texts, cfg, extra=None) -> dict:
    ev = evaluate_model(model, X_test, y_test)
    lo, hi = bootstrap_ci(ev["correct"], 1000)
    out = {"name": name, "acc": ev["acc"], "macro_f1": ev["macro_f1"], "acc_ci95": [lo, hi],
           "params": n_params(model), "size_mb": model_size_mb(model),
           "latency": latency_profile(model, featurizer, test_texts, cfg)}
    if extra:
        out.update(extra)
    return out
