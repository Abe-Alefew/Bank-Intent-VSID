#!/usr/bin/env python
"""Quiet-machine latency benchmark of the saved models (teacher T4, hard student S0, distilled student S4).
Close other programs first.   python scripts/benchmark_latency.py   -> results/latency_benchmark.json"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import joblib, torch
from vsid.config import Config, ROOT
from vsid.data import load_banking77
from vsid.models import new_teacher, new_student, n_params, model_size_mb
from vsid.evaluate import latency_profile

cfg = Config(); cfg.latency_queries = 500; cfg.latency_warmup = 50
ad = ROOT / "results" / "artifacts"
meta = joblib.load(ad / "featurizers.joblib"); C = len(meta["classes"])
_, te, _ = load_banking77(cfg.data_dir)
texts = te["text"].tolist()
teacher = new_teacher(meta["n_tf"], C, cfg); teacher.load_state_dict(torch.load(ad / "teacher.pt"))
s_dist = new_student(meta["n_sf"], C, cfg); s_dist.load_state_dict(torch.load(ad / "student.pt"))
s_hard = new_student(meta["n_sf"], C, cfg); s_hard.load_state_dict(torch.load(ad / "student_hard.pt"))
out = {}
for name, m, f in (("T4 teacher (final)", teacher, meta["tfeat"]), ("S0 student (hard labels)", s_hard, meta["sfeat"]),
                   ("S4 student (distilled)", s_dist, meta["sfeat"])):
    r = latency_profile(m, f, texts, cfg); r.update({"params": n_params(m), "size_mb": model_size_mb(m)}); out[name] = r
    print(f"{name:28s} p50={r['p50_ms']:.3f} ms  p95={r['p95_ms']:.3f} ms  model-only p50={r['model_only_p50_ms']:.3f} ms  {r['throughput_qps']:.0f} q/s  params={r['params']:,}")
(ROOT / "results" / "latency_benchmark.json").write_text(json.dumps(out, indent=1))
