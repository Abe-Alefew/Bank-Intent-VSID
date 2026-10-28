# Experiment report (auto-generated)

Seeds: [0, 1, 2] · dataset: Banking77 · numbers are from YOUR run; interpret them yourself.

## 1. Final test results (mean ± std over seeds)

| system | test acc | test macro-F1 | params | MB (fp32) | p50 ms | p95 ms | queries/s (b=256) | human labels |
|---|---|---|---|---|---|---|---|---|
| T0 teacher, seed labels only (start of loop) | 0.732 ± 0.007 | 0.732 ± 0.007 | 5,163,085 | 20.65 | 1.84 | 2.94 | 6311 | 770 |
| T1 teacher, label-budget-matched supervised | 0.786 ± 0.011 | 0.786 ± 0.010 | 5,163,085 | 20.65 | 1.12 | 2.67 | 5785 | 1170 |
| T2 teacher, naive self-training (no verifier, no gates) | 0.776 ± 0.011 | 0.774 ± 0.011 | 5,163,085 | 20.65 | 1.39 | 2.72 | 7011 | 770 |
| T3 teacher, verifier only (no audit, no gates) | 0.780 ± 0.002 | 0.776 ± 0.002 | 5,163,085 | 20.65 | 1.50 | 2.89 | 7074 | 770 |
| T4 teacher, FULL verified loop | 0.782 ± 0.009 | 0.780 ± 0.009 | 5,163,085 | 20.65 | 1.67 | 2.88 | 7357 | 1170 |
| T5 teacher, oracle (all pool labels; upper bound) | 0.912 ± 0.005 | 0.912 ± 0.005 | 5,163,085 | 20.65 | 1.50 | 2.83 | 7282 | 8992 |
| S0 student, hard labels (baseline) | 0.693 ± 0.006 | 0.690 ± 0.005 | 197,069 | 0.79 | 0.77 | 1.32 | 37643 |  |
| S1 student, KD on labelled data only | 0.712 ± 0.005 | 0.706 ± 0.005 | 197,069 | 0.79 | 0.72 | 1.37 | 37904 |  |
| S2 student, hard pseudo-labels (no soft targets) | 0.775 ± 0.004 | 0.773 ± 0.004 | 197,069 | 0.79 | 0.75 | 1.04 | 39067 |  |
| S3 student, distilled (soft targets, full transfer set) | 0.765 ± 0.009 | 0.763 ± 0.009 | 197,069 | 0.79 | 0.77 | 1.23 | 38114 |  |
| S4 student, distilled (T, alpha tuned on dev) | 0.770 ± 0.013 | 0.767 ± 0.013 | 197,069 | 0.79 | 0.69 | 1.36 | 30335 |  |
| S5 student, distilled from INITIAL teacher (no loop) | 0.721 ± 0.008 | 0.721 ± 0.007 | 197,069 | 0.79 | 0.57 | 1.30 | 41199 |  |

## 2. Loop summary per seed and variant

| seed | variant | rounds run | accepted | rejected | final dev F1 | pseudo-labels used | their true precision (diagnostic) | human labels |
|---|---|---|---|---|---|---|---|---|
| 0 | full | 4 | 2 | 2 | 0.753 | 1895 | 0.935 | 400 |
| 0 | verifier_only | 4 | 4 | 0 | 0.743 | 4417 | 0.936 | 0 |
| 0 | naive | 4 | 4 | 0 | 0.731 | 4605 | 0.921 | 0 |
| 1 | full | 4 | 3 | 1 | 0.734 | 2816 | 0.946 | 400 |
| 1 | verifier_only | 4 | 4 | 0 | 0.725 | 4400 | 0.935 | 0 |
| 1 | naive | 4 | 4 | 0 | 0.730 | 4562 | 0.923 | 0 |
| 2 | full | 4 | 2 | 2 | 0.775 | 1761 | 0.948 | 400 |
| 2 | verifier_only | 4 | 4 | 0 | 0.770 | 4360 | 0.936 | 0 |
| 2 | naive | 4 | 4 | 0 | 0.787 | 4522 | 0.924 | 0 |

## 3. FULL loop round log (every seed)

| seed | round | τ | selected | audit precision | Wilson LB | decision | reason |
|---|---|---|---|---|---|---|---|
| 0 | 1 | 0.6 | 836 | 0.98 | 0.95 | rejected | G2 failed: no significant dev improvement -> rollback |
| 0 | 2 | 0.55 | 915 | 0.96 | 0.93 | accepted | all gates passed |
| 0 | 3 | 0.55 | 1180 | 0.94 | 0.90 | accepted | all gates passed |
| 0 | 4 | 0.55 | 1286 | 0.95 | 0.91 | rejected | G2 failed: no significant dev improvement -> rollback |
| 1 | 1 | 0.6 | 852 | 0.95 | 0.91 | accepted | all gates passed |
| 1 | 2 | 0.6 | 1093 | 0.96 | 0.93 | accepted | all gates passed |
| 1 | 3 | 0.6 | 1196 | 0.92 | 0.88 | rejected | pre-training gate failed (G1_audit_precision) |
| 1 | 4 | 0.65 | 1171 | 0.97 | 0.94 | accepted | all gates passed |
| 2 | 1 | 0.6 | 829 | 0.96 | 0.93 | accepted | all gates passed |
| 2 | 2 | 0.6 | 1132 | 0.94 | 0.90 | accepted | all gates passed |
| 2 | 3 | 0.6 | 1206 | 0.92 | 0.88 | rejected | pre-training gate failed (G1_audit_precision) |
| 2 | 4 | 0.65 | 1129 | 0.90 | 0.85 | rejected | pre-training gate failed (G1_audit_precision) |

## 4. Dev-vs-test gap (repeated-validation check)

| seed | final dev acc | test acc (T4) | gap |
|---|---|---|---|
| 0 | 0.766 | 0.773 | -0.007 |
| 1 | 0.750 | 0.791 | -0.041 |
| 2 | 0.775 | 0.782 | -0.007 |