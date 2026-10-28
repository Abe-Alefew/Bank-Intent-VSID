#!/usr/bin/env python
"""Generate results/REPORT.md (tables only; write the interpretation yourself)."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
runs = {}
for p in sorted(RES.glob("run_seed*.json")):
    d = json.loads(p.read_text(encoding="utf-8")); runs[d["seed"]] = d
if not runs:
    sys.exit("No results/run_seed*.json found. Run scripts/run_pipeline.py first.")
seeds = sorted(runs)


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


names = [x["name"] for x in runs[seeds[0]]["systems"]]
rows = []
for n in names:
    xs = [next(x for x in runs[s]["systems"] if x["name"] == n) for s in seeds]
    acc = np.array([x["acc"] for x in xs]); f1 = np.array([x["macro_f1"] for x in xs])
    rows.append([n, f"{acc.mean():.3f} ± {acc.std(ddof=1) if len(acc) > 1 else 0:.3f}", f"{f1.mean():.3f} ± {f1.std(ddof=1) if len(f1) > 1 else 0:.3f}",
                 f"{xs[0]['params']:,}", f"{xs[0]['size_mb']:.2f}", f"{np.mean([x['latency']['p50_ms'] for x in xs]):.2f}",
                 f"{np.mean([x['latency']['p95_ms'] for x in xs]):.2f}", f"{np.mean([x['latency']['throughput_qps'] for x in xs]):.0f}",
                 xs[0].get("human_labels", "")])
md = [f"# Experiment report (auto-generated)\n\nSeeds: {seeds} · dataset: Banking77 · numbers are from YOUR run; interpret them yourself.\n",
      "## 1. Final test results (mean ± std over seeds)\n",
      table(["system", "test acc", "test macro-F1", "params", "MB (fp32)", "p50 ms", "p95 ms", "queries/s (b=256)", "human labels"], rows)]
lr = []
for s in seeds:
    for mode, lg in runs[s]["loops"].items():
        acc_n = sum(r["decision"] == "accepted" for r in lg)
        lr.append([s, mode, len(lg), acc_n, len(lg) - acc_n, lg[-1]["incumbent_dev_f1"], lg[-1]["cum_pseudo_labels"],
                   f"{lg[-1]['cum_pseudo_true_precision']:.3f}" if lg[-1]["cum_pseudo_true_precision"] is not None else "-", lg[-1]["cum_human_labels"]])
md += ["\n## 2. Loop summary per seed and variant\n",
       table(["seed", "variant", "rounds run", "accepted", "rejected", "final dev F1", "pseudo-labels used", "their true precision (diagnostic)", "human labels"],
             [[r[0], r[1], r[2], r[3], r[4], f"{r[5]:.3f}", r[6], r[7], r[8]] for r in lr])]
md += ["\n## 3. FULL loop round log (every seed)\n"]
rl = []
for s in seeds:
    for r in runs[s]["loops"]["full"]:
        rl.append([s, r["round"], r["tau"], r["n_selected"], r.get("audit_precision", "-") if "audit_precision" not in r else f"{r['audit_precision']:.2f}",
                   f"{r['audit_wilson_lb']:.2f}" if "audit_wilson_lb" in r else "-", r["decision"], r["reason"]])
md.append(table(["seed", "round", "τ", "selected", "audit precision", "Wilson LB", "decision", "reason"], rl))
gp = []
for s in seeds:
    t4 = next(x for x in runs[s]["systems"] if x["name"].startswith("T4"))
    gp.append([s, f"{runs[s]['loops']['full'][-1]['incumbent_dev_acc']:.3f}", f"{t4['acc']:.3f}",
               f"{runs[s]['loops']['full'][-1]['incumbent_dev_acc'] - t4['acc']:+.3f}"])
md += ["\n## 4. Dev-vs-test gap (repeated-validation check)\n", table(["seed", "final dev acc", "test acc (T4)", "gap"], gp)]
(RES / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
print("wrote", RES / "REPORT.md")
