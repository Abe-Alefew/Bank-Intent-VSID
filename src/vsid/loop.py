"""Bounded recursive self-improvement loop with external verification, acceptance gates and rollback.

generate candidates -> verify (independent model + human audit = external ground truth) -> curate (class cap)
-> train a PROPOSAL -> evaluate on held-out dev -> accept / reject (rollback) -> adapt threshold -> repeat

Modes (ablations):
  full           : verifier + human-audit gate G1 + diversity gate G3 + dev gate G2 (with rollback) + adaptive threshold
  verifier_only  : independent-verifier agreement, but NO audit, NO gates, NO rollback (always accept)
  naive          : confidence >= tau only, NO verifier, NO audit, NO gates   (classic self-training / confirmation bias)

Honest scope note: this is *iterative, verified self-training* (bounded). The only "meta" element is the controller that
adapts the confidence threshold from audit evidence. It is NOT unrestricted recursive self-improvement.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .models import new_teacher, train_model, predict_logits, softmax_np
from .verifier import IndependentVerifier, wilson_lower, class_entropy
from .evaluate import evaluate_model, p_improve


def fit_teacher(cfg, feat, texts, labels, seed, n_classes, sample_weight=None):
    X = feat.transform(texts)
    m = new_teacher(feat.n_features, n_classes, cfg)
    return train_model(m, X, np.asarray(labels), cfg, seed, cfg.epochs_teacher, sample_weight=sample_weight)


def _cap_per_class(idx, pred, conf, cap):
    keep = []
    for c in np.unique(pred[idx]):
        ids = idx[pred[idx] == c]
        ids = ids[np.argsort(-conf[ids])][:cap]
        keep.extend(ids.tolist())
    return np.array(sorted(keep), dtype=int)


def run_loop(cfg, mode: str, splits, feat, teacher0, seed: int, log=print):
    assert mode in ("full", "verifier_only", "naive")
    C = len(splits.classes)
    pool, dev = splits.pool, splits.dev
    rng = np.random.RandomState(1000 + seed)
    order = rng.permutation(len(pool))
    slices = np.array_split(order, cfg.rounds)                   # each round sees a FRESH slice of the stream
    Xdev = feat.transform(dev["text"].tolist()); ydev = dev["label"].values

    trusted_texts = splits.seed["text"].tolist()
    trusted_labels = splits.seed["label"].tolist()
    trusted_pool_idx = []                                        # audited pool rows (true labels revealed)
    pseudo_idx, pseudo_lab = [], []                              # accepted pseudo-labelled pool rows

    verifier = None
    if mode != "naive":
        verifier = IndependentVerifier(cfg.verifier_char_features, C)
        verifier.fit_vectorizer(splits.seed["text"].tolist() + pool["text"].tolist())
        verifier.fit(trusted_texts, trusted_labels)

    incumbent = teacher0
    inc_eval = evaluate_model(incumbent, Xdev, ydev)
    tau = cfg.tau0
    rounds, rejections, human_labels = [], 0, 0

    for r, cand in enumerate(slices, start=1):
        texts = pool.iloc[cand]["text"].tolist()
        hidden = pool.iloc[cand]["label"].values                 # used ONLY for audit / diagnostics
        P = softmax_np(predict_logits(incumbent, feat.transform(texts)))
        conf, pred = P.max(1), P.argmax(1)
        m_conf = conf >= tau
        m = m_conf.copy()
        if verifier is not None:
            vp = verifier.predict_proba(texts)
            topk = np.argsort(-vp, axis=1)[:, : cfg.verifier_topk]
            m_ver = (topk == pred[:, None]).any(1)
            m &= m_ver
        else:
            m_ver = np.ones_like(m_conf)
        cap = int(np.ceil(cfg.class_cap_factor * len(cand) / C))
        sel = _cap_per_class(np.where(m)[0], pred, conf, cap)

        log_r = {"round": r, "tau": round(float(tau), 3), "n_candidates": int(len(cand)),
                 "n_conf_pass": int(m_conf.sum()), "n_verifier_pass": int(m.sum()), "n_selected": int(len(sel)),
                 "diag_true_precision_selected": float((pred[sel] == hidden[sel]).mean()) if len(sel) else None,
                 "diag_true_precision_conf_only": float((pred[m_conf] == hidden[m_conf]).mean()) if m_conf.any() else None}

        audited_local = np.array([], dtype=int)
        gates = {}
        if mode == "full" and len(sel):
            k = min(cfg.audit_per_round, max(1, len(sel) // 2))
            audited_local = rng.choice(sel, size=k, replace=False)      # simulated human spot-check
            n_ok = int((pred[audited_local] == hidden[audited_local]).sum())
            lb = wilson_lower(n_ok, k, cfg.wilson_z)
            human_labels += k
            # humans' labels are ground truth: they are ALWAYS banked into the trusted set (never rolled back)
            trusted_texts += [texts[i] for i in audited_local]
            trusted_labels += hidden[audited_local].tolist()
            trusted_pool_idx += cand[audited_local].tolist()
            sel = np.setdiff1d(sel, audited_local)
            ent = class_entropy(pred[sel], C)
            gates["G1_audit_precision"] = {"observed": n_ok / k, "wilson_lb": lb, "n_audited": k, "threshold": cfg.p_min,
                                           "pass": lb >= cfg.p_min}
            gates["G3_diversity"] = {"entropy": ent, "threshold": cfg.min_entropy, "pass": ent >= cfg.min_entropy}
            log_r.update({"n_audited": k, "audit_precision": n_ok / k, "audit_wilson_lb": lb, "class_entropy": ent})
            pre_ok = gates["G1_audit_precision"]["pass"] and gates["G3_diversity"]["pass"]
        else:
            pre_ok = True
            log_r.update({"class_entropy": class_entropy(pred[sel], C) if len(sel) else 0.0})

        decision, reason, proposal = "rejected", "", None
        if not pre_ok:
            reason = "pre-training gate failed (" + ", ".join(k for k, v in gates.items() if not v["pass"]) + ")"
        elif len(sel) == 0:
            reason = "no candidate passed selection -> nothing to learn from"
        else:
            add_idx = list(pseudo_idx) + cand[sel].tolist()
            add_lab = list(pseudo_lab) + pred[sel].tolist()
            tr_t = trusted_texts + pool.iloc[add_idx]["text"].tolist()
            tr_l = trusted_labels + add_lab
            proposal = fit_teacher(cfg, feat, tr_t, tr_l, seed * 100 + r, C)
            prop_eval = evaluate_model(proposal, Xdev, ydev)
            pimp = p_improve(prop_eval["correct"], inc_eval["correct"], cfg.n_boot, seed)
            gain = prop_eval["macro_f1"] - inc_eval["macro_f1"]
            log_r.update({"dev_acc_incumbent": inc_eval["acc"], "dev_f1_incumbent": inc_eval["macro_f1"],
                          "dev_acc_proposal": prop_eval["acc"], "dev_f1_proposal": prop_eval["macro_f1"],
                          "dev_gain_f1": gain, "p_improve": pimp})
            if mode == "full":
                gates["G2_dev_gain"] = {"gain_f1": gain, "p_improve": pimp, "min_gain": cfg.min_gain,
                                        "p_boot": cfg.p_boot, "pass": gain >= cfg.min_gain and pimp >= cfg.p_boot}
                ok = gates["G2_dev_gain"]["pass"]
                reason = "all gates passed" if ok else "G2 failed: no significant dev improvement -> rollback"
            else:
                ok, reason = True, "no gates in this ablation: always accept"
            if ok:
                decision, incumbent, inc_eval = "accepted", proposal, prop_eval
                pseudo_idx, pseudo_lab = add_idx, add_lab
        # rollback = discard proposal + pseudo-batch (incumbent unchanged); banked human labels are kept
        rejections = rejections + 1 if decision == "rejected" else 0

        # meta-adaptation of the threshold from audit evidence (bounded controller)
        tau_next = tau
        if mode == "full" and "audit_wilson_lb" in log_r:
            if log_r["audit_wilson_lb"] >= cfg.p_min + 0.05:
                tau_next = max(cfg.tau_min, tau - cfg.tau_step)
            elif log_r["audit_wilson_lb"] < cfg.p_min:
                tau_next = min(cfg.tau_max, tau + cfg.tau_step)
        # verifier refresh uses ONLY trusted (human) data
        if mode == "full" and len(audited_local):
            verifier.fit(trusted_texts, trusted_labels)

        pidx = np.array(pseudo_idx, dtype=int)
        log_r.update({"decision": decision, "reason": reason, "gates": gates, "tau_next": round(float(tau_next), 3),
                      "cum_human_labels": human_labels, "cum_pseudo_labels": len(pseudo_idx),
                      "cum_pseudo_true_precision": float((np.array(pseudo_lab) == pool.iloc[pidx]["label"].values).mean())
                      if len(pidx) else None,
                      "incumbent_dev_acc": inc_eval["acc"], "incumbent_dev_f1": inc_eval["macro_f1"]})
        rounds.append(log_r)
        log(f"   [{mode}] round {r}: tau={tau:.2f} sel={log_r['n_selected']} "
            f"prec(diag)={log_r['diag_true_precision_selected']} -> {decision} | dev F1 {inc_eval['macro_f1']:.3f}")
        tau = tau_next
        if mode == "full" and rejections >= cfg.patience:
            log(f"   [{mode}] stopping early: {rejections} consecutive rejections (patience={cfg.patience})")
            break

    state = {"teacher": incumbent, "trusted_texts": trusted_texts, "trusted_labels": trusted_labels,
             "pseudo_idx": pseudo_idx, "pseudo_lab": pseudo_lab, "human_labels": human_labels,
             "trusted_pool_idx": trusted_pool_idx}
    return state, rounds
