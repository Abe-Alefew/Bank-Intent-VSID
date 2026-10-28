"""PyTorch models + one generic trainer used for teachers AND students.

SparseMLP uses nn.EmbeddingBag(mode="sum") with per-sample weights = TF-IDF values, which is mathematically
identical to  Linear(sparse_tfidf_vector)  but touches only the non-zero features (fast on CPU).
"""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp


class SparseMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int, n_classes: int, dropout: float = 0.3, two_layer: bool = False):
        super().__init__()
        self.emb = nn.EmbeddingBag(n_features, hidden, mode="sum")
        nn.init.normal_(self.emb.weight, std=0.05)
        self.bias = nn.Parameter(torch.zeros(hidden))
        self.drop = nn.Dropout(dropout)
        if two_layer:
            h2 = max(hidden // 2, 32)
            self.head = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, h2), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(h2, n_classes))
        else:
            self.head = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, idx, offsets, w):
        h = self.emb(idx, offsets, per_sample_weights=w) + self.bias
        return self.head(h)


def csr_batch(X: sp.csr_matrix):
    return (torch.from_numpy(X.indices.astype(np.int64)),
            torch.from_numpy(X.indptr[:-1].astype(np.int64)),
            torch.from_numpy(X.data.astype(np.float32)))


def n_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def model_size_mb(model: nn.Module) -> float:
    return sum(p.numel() * 4 for p in model.parameters()) / 1e6   # float32 weights


@torch.no_grad()
def predict_logits(model: nn.Module, X: sp.csr_matrix, bs: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, X.shape[0], bs):
        idx, off, w = csr_batch(X[i:i + bs])
        out.append(model(idx, off, w).numpy())
    return np.vstack(out)


def softmax_np(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def kd_loss(student_logits, y_hard, teacher_logits, alpha: float, T: float, label_smoothing: float = 0.0):
    """Per-example distillation loss (Hinton et al., 2015).

        L_i = a_i * CE(y_i, z_s)  +  (1 - a_i) * T^2 * KL( softmax(z_t/T) || softmax(z_s/T) )

    a_i = alpha if example i has a (trusted) hard label (y_i >= 0), else 0  (pure soft-target learning).
    With teacher_logits=None it degrades to ordinary (label-smoothed) cross-entropy.
    Returns a vector of per-example losses.
    """
    has = y_hard >= 0
    ce = torch.zeros(student_logits.shape[0])
    if has.any():
        ce_has = F.cross_entropy(student_logits[has], y_hard[has], label_smoothing=label_smoothing, reduction="none")
        ce = ce.index_put((has.nonzero().squeeze(1),), ce_has)
    if teacher_logits is None:
        return ce
    log_ps = F.log_softmax(student_logits / T, dim=1)
    pt = F.softmax(teacher_logits / T, dim=1)
    kl = F.kl_div(log_ps, pt, reduction="none").sum(1)          # KL(teacher || student)
    a = torch.where(has, torch.full_like(ce, alpha), torch.zeros_like(ce))
    return a * ce + (1.0 - a) * (T ** 2) * kl


def train_model(model: nn.Module, X: sp.csr_matrix, y_hard: np.ndarray, cfg, seed: int, epochs: int,
                teacher_logits: np.ndarray | None = None, alpha: float = 0.5, T: float = 4.0,
                sample_weight: np.ndarray | None = None, lr: float | None = None) -> nn.Module:
    """Generic trainer. y_hard = -1 for examples without a trusted hard label."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    opt = torch.optim.AdamW(model.parameters(), lr=lr or cfg.lr, weight_decay=cfg.weight_decay)
    steps = max(1, epochs * int(np.ceil(n / cfg.batch_size)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr or cfg.lr, total_steps=steps, pct_start=0.15)
    yt = torch.from_numpy(np.asarray(y_hard, dtype=np.int64))
    tl = None if teacher_logits is None else torch.from_numpy(teacher_logits.astype(np.float32))
    sw = None if sample_weight is None else torch.from_numpy(sample_weight.astype(np.float32))
    for _ in range(epochs):
        model.train()
        perm = rng.permutation(n)
        for i in range(0, n, cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            idx, off, w = csr_batch(X[b])
            logits = model(idx, off, w)
            loss_i = kd_loss(logits, yt[b], None if tl is None else tl[b], alpha, T, cfg.label_smoothing)
            if sw is not None:
                loss_i = loss_i * sw[b]
            loss = loss_i.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    return model


def new_teacher(n_features: int, n_classes: int, cfg) -> SparseMLP:
    return SparseMLP(n_features, cfg.teacher_hidden, n_classes, cfg.dropout, two_layer=True)


def new_student(n_features: int, n_classes: int, cfg) -> SparseMLP:
    return SparseMLP(n_features, cfg.student_hidden, n_classes, cfg.dropout * 0.5, two_layer=False)


def clone(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)
