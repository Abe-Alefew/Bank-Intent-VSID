#!/usr/bin/env python
"""Run the full experiment.   python scripts/run_pipeline.py            (3 seeds, ~25-40 min on a laptop CPU)
                              python scripts/run_pipeline.py --quick    (1 seed, tiny settings: smoke test)"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vsid.config import Config
from vsid.pipeline import run_all

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true", help="tiny/fast settings for a smoke test")
ap.add_argument("--seeds", type=int, nargs="*", default=None)
ap.add_argument("--rounds", type=int, default=None)
ap.add_argument("--results-dir", default=None)
a = ap.parse_args()
cfg = Config()
if a.quick:
    cfg.quick()
if a.rounds:
    cfg.rounds = a.rounds
if a.results_dir:
    cfg.results_dir = a.results_dir
run_all(cfg, seeds=a.seeds)
