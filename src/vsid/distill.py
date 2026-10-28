"""Student suite: hard-label baseline vs. distilled students (+ T / alpha sweep)."""
from __future__ import annotations
import numpy as np
from .models import new_student, train_model, predict_logits
from .evaluate import evaluate_model


def build_transfer_set(splits, state):
    """Transfer set = trusted human-labelled texts  +  ALL unlabelled pool texts (the teacher can soft-label every one)."""
    pool = splits.pool
    trusted_texts, trusted_labels = list(state["trusted_texts"]), list(state["trusted_labels"])
    audited = set(state["trusted_pool_idx"])
    rest_idx = [i for i in range(len(pool)) if i not in audited]       # audited rows are already in trusted set
    texts = trusted_texts + pool.iloc[rest_idx]["text"].tolist()
    y_hard = np.array(trusted_labels + [-1] * len(rest_idx), dtype=np.int64)   # -1 = no trusted hard label
    return texts, y_hard


def run_student_suite(cfg, splits, state, tfeat, sfeat, seed, log=print):
    C = len(splits.classes)
    teacher = state["teacher"]
    texts, y_hard = build_transfer_set(splits, state)
    Xs = sfeat.transform(texts)
    t_logits = predict_logits(teacher, tfeat.transform(texts))
    has = y_hard >= 0
    Xs_dev, ydev = sfeat.transform(splits.dev["text"].tolist()), splits.dev["label"].values
    students, sweep = {}, []

    def fit(name, X, y, tl=None, alpha=0.5, T=4.0, s=seed):
        m = new_student(sfeat.n_features, C, cfg)
        m = train_model(m, X, y, cfg, s, cfg.epochs_student, teacher_logits=tl, alpha=alpha, T=T)
        students[name] = m
        return m

    # S0  hard-label student (non-distilled baseline): trusted labels only, plain CE
    fit("S0 student, hard labels (baseline)", Xs[has], y_hard[has])
    # S1  same data, soft targets only help?  (isolates the effect of soft targets from the effect of extra data)
    fit("S1 student, KD on labelled data only", Xs[has], y_hard[has], t_logits[has], cfg.kd_alpha, cfg.kd_temperature)
    # S2  hard pseudo-labels (argmax of teacher) on the full transfer set  (extra data, but NO soft targets)
    y_pseudo = np.where(has, y_hard, t_logits.argmax(1))
    fit("S2 student, hard pseudo-labels (no soft targets)", Xs, y_pseudo)
    # S3  full distillation on the whole transfer set (soft targets + extra unlabelled data)
    fit("S3 student, distilled (soft targets, full transfer set)", Xs, y_hard, t_logits, cfg.kd_alpha, cfg.kd_temperature)

    # temperature / alpha sweep (selection on DEV, never on test)
    for T in cfg.sweep_temperatures:
        for a in cfg.sweep_alphas:
            m = new_student(sfeat.n_features, C, cfg)
            m = train_model(m, Xs, y_hard, cfg, seed, cfg.epochs_student, teacher_logits=t_logits, alpha=a, T=T)
            ev = evaluate_model(m, Xs_dev, ydev)
            sweep.append({"T": T, "alpha": a, "dev_acc": ev["acc"], "dev_f1": ev["macro_f1"]})
            students[f"_sweep_T{T}_a{a}"] = m
    best = max(sweep, key=lambda d: d["dev_f1"])
    students["S4 student, distilled (T, alpha tuned on dev)"] = students[f"_sweep_T{best['T']}_a{best['alpha']}"]
    log(f"   sweep best on dev: T={best['T']} alpha={best['alpha']} (dev F1 {best['dev_f1']:.3f})")
    students = {k: v for k, v in students.items() if not k.startswith("_sweep_")}
    return students, sweep, best


def distill_one(cfg, splits, state, tfeat, sfeat, seed, alpha=None, T=None):
    """Single distilled student from an arbitrary teacher state (used for 'distil the INITIAL teacher' ablation)."""
    C = len(splits.classes)
    texts, y_hard = build_transfer_set(splits, state)
    Xs = sfeat.transform(texts)
    tl = predict_logits(state["teacher"], tfeat.transform(texts))
    m = new_student(sfeat.n_features, C, cfg)
    return train_model(m, Xs, y_hard, cfg, seed, cfg.epochs_student, teacher_logits=tl,
                       alpha=cfg.kd_alpha if alpha is None else alpha, T=cfg.kd_temperature if T is None else T)
