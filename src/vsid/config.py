"""Single source of truth for every hyper-parameter, gate and split size."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    # ---- data -------------------------------------------------------------
    dataset: str = "banking77"
    data_dir: str = str(ROOT / "data")
    results_dir: str = str(ROOT / "results")
    seeds: tuple = (0, 1, 2)            # >= 2 seeds required by Track B
    n_dev: int = 1000                   # gate set (re-used every round -> NOT final evidence)
    seed_per_class: int = 10            # tiny trusted labelled seed (77 classes -> 770 labels)

    # ---- featurisers ------------------------------------------------------
    teacher_word_features: int = 5000
    teacher_char_features: int = 15000
    student_word_features: int = 3000
    verifier_char_features: int = 30000

    # ---- models -----------------------------------------------------------
    teacher_hidden: int = 256
    student_hidden: int = 64
    dropout: float = 0.3
    label_smoothing: float = 0.05
    epochs_teacher: int = 20
    epochs_student: int = 25
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    torch_threads: int = 1              # 1 thread -> fair, reproducible latency comparison

    # ---- recursive loop ---------------------------------------------------
    rounds: int = 4
    audit_per_round: int = 100          # simulated human spot-check budget / round
    tau0: float = 0.60                  # initial teacher-confidence threshold
    tau_min: float = 0.30
    tau_max: float = 0.90
    tau_step: float = 0.05
    verifier_topk: int = 1              # verifier's top-k must contain the teacher label
    class_cap_factor: float = 1.5       # per-class cap = factor * (cand / n_classes)
    # acceptance gates
    p_min: float = 0.90                 # G1: Wilson lower bound of audited precision
    wilson_z: float = 1.2816            # one-sided 90 %
    min_entropy: float = 0.85           # G3: normalised class entropy of accepted batch (collapse guard)
    min_gain: float = 0.003             # G2: dev macro-F1 gain over incumbent
    p_boot: float = 0.70                # G2: bootstrap P(proposal > incumbent) on dev
    n_boot: int = 500
    patience: int = 2                   # stop after N consecutive rejections

    # ---- distillation -----------------------------------------------------
    kd_alpha: float = 0.5
    kd_temperature: float = 4.0
    sweep_temperatures: tuple = (1.0, 2.0, 4.0, 8.0)
    sweep_alphas: tuple = (0.0, 0.5)

    # ---- latency ----------------------------------------------------------
    latency_queries: int = 300
    latency_warmup: int = 30

    def to_dict(self) -> dict:
        return asdict(self)

    def quick(self) -> "Config":
        """Small/fast settings for smoke tests."""
        self.seeds = (0,)
        self.rounds = 2
        self.epochs_teacher = 6
        self.epochs_student = 6
        self.sweep_temperatures = (2.0, 4.0)
        self.sweep_alphas = (0.5,)
        self.n_boot = 100
        self.latency_queries = 60
        return self
