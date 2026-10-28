"""BankIntent-VSID dashboard.   streamlit run app/streamlit_app.py"""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "results"

st.set_page_config(page_title="BankIntent-VSID", page_icon="🏦", layout="wide")


@st.cache_data(show_spinner=False)
def load_runs():
    runs = {}
    for p in sorted(RES.glob("run_seed*.json")):
        d = json.loads(p.read_text())
        runs[d["seed"]] = d
    return runs


def systems_df(runs):
    rows = []
    for s, d in runs.items():
        for x in d["systems"]:
            rows.append({"seed": s, "name": x["name"], "kind": x["kind"], "acc": x["acc"], "macro_f1": x["macro_f1"],
                         "params": x["params"], "size_mb": x["size_mb"], "p50_ms": x["latency"]["p50_ms"],
                         "p95_ms": x["latency"]["p95_ms"], "qps": x["latency"]["throughput_qps"],
                         "human_labels": x.get("human_labels")})
    return pd.DataFrame(rows)


def agg(df):
    g = df.groupby(["kind", "name"], sort=False)
    a = g.agg(acc=("acc", "mean"), acc_std=("acc", "std"), f1=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
              params=("params", "first"), size_mb=("size_mb", "first"), p50_ms=("p50_ms", "mean"), p95_ms=("p95_ms", "mean"),
              qps=("qps", "mean"), human_labels=("human_labels", "first"), n_seeds=("seed", "nunique")).reset_index()
    a[["acc_std", "f1_std"]] = a[["acc_std", "f1_std"]].fillna(0)
    return a


runs = load_runs()
st.sidebar.title("🏦 BankIntent-VSID")
st.sidebar.caption("Verified self-improvement + knowledge distillation\n\niCog Labs · Emerging AI Paradigms (stretch goal)")
PAGES = ["1 · Concepts", "2 · Data & splits", "3 · Recursive loop", "4 · Distillation lab", "5 · Final evidence",
         "6 · Failure analysis", "7 · Live demo"]
page = st.sidebar.radio("Page", PAGES)
if not runs:
    st.error("No results found. Run `python scripts/run_pipeline.py` first (or `--quick`).")
    st.stop()
seeds = sorted(runs)
st.sidebar.write(f"Seeds loaded: {seeds}")
sdf = systems_df(runs)
A = agg(sdf)

# ----------------------------------------------------------------------------------------------------------------
if page == PAGES[0]:
    st.title("Verified self-improvement → distillation")
    st.markdown("A teacher improves itself from **unlabelled** bank-customer queries using *externally verified* feedback "
                "(independent verifier + human audit + gates + rollback). The improved teacher is then **distilled** into a "
                "small, fast student for cheap CPU deployment.")
    st.graphviz_chart("""digraph G { rankdir=LR; node [shape=box, style=rounded];
      L [label="770 labelled\\nqueries"]; U [label="8k unlabelled\\nstream"]; T0 [label="Teacher T0"];
      G [label="generate\\n(conf ≥ τ)"]; V [label="verify\\n(independent model)"]; A [label="human audit\\n(G1, G3)"];
      P [label="train proposal"]; D [label="dev gate G2\\naccept / ROLLBACK"]; T [label="Teacher T*"]; S [label="Student\\n(KD, 30x smaller)"];
      L -> T0; T0 -> G; U -> G; G -> V -> A -> P -> D; D -> T [label="accept"]; D -> G [label="next round", style=dashed];
      T -> S; U -> S [label="soft targets"]; }""")
    st.subheader("Three things called “recursive self-learning” (LO1)")
    st.table(pd.DataFrame([
        ["Test-time self-refinement", "output only (weights frozen)", "self-confirmation", "external checker vs first draft"],
        ["Iterative self-training  ← **this project**", "weights of a new checkpoint", "error reinforcement, collapse, contamination",
         "independent verifier + audit + untouched holdout + compare with previous checkpoint"],
        ["Recursive self-improvement (RSI)", "the improvement process itself", "reward hacking, runaway change",
         "independent measurement across rounds, bounds, human sign-off"]],
        columns=["Mechanism", "What changes", "Main risk", "Validation"]))
    st.info("Scope: this is *bounded, verified self-training*. Only the confidence threshold is adapted automatically. "
            "We do **not** claim unrestricted recursive self-improvement.")

# ----------------------------------------------------------------------------------------------------------------
elif page == PAGES[1]:
    st.title("Data & splits — Banking77")
    d = runs[seeds[0]]
    info = d["split_info"]
    c1, c2, c3, c4 = st.columns(4)
    for c, k, lab in zip((c1, c2, c3, c4), ("seed", "pool", "dev", "test"),
                         ("trusted labelled seed", "unlabelled pool (hidden labels)", "dev (gate set)", "test (final, once)")):
        c.metric(lab, f"{info['sizes'][k]:,}")
    st.dataframe(pd.DataFrame({"split": list(info["sizes"]), "size": list(info["sizes"].values())}), hide_index=True)
    st.write(f"Exact duplicates removed from train: **{info['n_exact_duplicates_removed']}** · "
             f"train rows also found in test (removed): **{info['n_train_rows_also_in_test_removed']}** · "
             f"77 intents · seed set = {info['seed_per_class']} per class")
    ov = pd.DataFrame({"pair": list(info["pairwise_text_overlap"]), "shared texts": list(info["pairwise_text_overlap"].values())})
    st.subheader("Leakage check (must be all zeros)")
    st.dataframe(ov, hide_index=True)
    st.success("Leakage-free ✅" if info["leakage_free"] else "LEAKAGE")
    f = ROOT / "data" / "banking77_train.csv"
    if f.exists():
        tr = pd.read_csv(f)
        st.subheader("Sample real customer queries")
        st.dataframe(tr.sample(8, random_state=1).rename(columns={"category": "intent"}), hide_index=True)

# ----------------------------------------------------------------------------------------------------------------
elif page == PAGES[2]:
    st.title("Recursive loop — generate → verify → curate → train → evaluate → accept / rollback")
    seed = st.selectbox("Seed", seeds)
    loops = runs[seed]["loops"]
    names = {"full": "FULL (verifier + audit + gates)", "verifier_only": "verifier only (no gates)", "naive": "naive self-training"}
    fig = go.Figure()
    for m, lg in loops.items():
        start = next((r["dev_f1_incumbent"] for r in lg if "dev_f1_incumbent" in r), lg[0]["incumbent_dev_f1"])
        ys = [start] + [r["incumbent_dev_f1"] for r in lg]
        fig.add_trace(go.Scatter(x=list(range(len(ys))), y=ys, mode="lines+markers", name=names[m]))
    fig.update_layout(title="Incumbent dev macro-F1 per round (round 0 = initial teacher)", xaxis_title="round",
                      yaxis_title="dev macro-F1", height=380)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Dev is re-used by the gates every round → it is a *gate*, not final evidence. See page 5 for test numbers.")
    full = loops["full"]
    rows = []
    for r in full:
        g = r["gates"]
        rows.append({"round": r["round"], "τ": r["tau"], "candidates": r["n_candidates"], "conf≥τ": r["n_conf_pass"],
                     "+verifier": r["n_verifier_pass"], "selected": r["n_selected"], "audited": r.get("n_audited"),
                     "audit prec": r.get("audit_precision"), "Wilson LB": r.get("audit_wilson_lb"),
                     "G1": g.get("G1_audit_precision", {}).get("pass"), "G3": g.get("G3_diversity", {}).get("pass"),
                     "G2": g.get("G2_dev_gain", {}).get("pass"), "dev F1 old": r.get("dev_f1_incumbent"),
                     "dev F1 proposal": r.get("dev_f1_proposal"), "P(better)": r.get("p_improve"),
                     "decision": r["decision"], "reason": r["reason"], "τ next": r["tau_next"]})
    st.subheader("FULL loop — round log (accepted vs rejected, with reasons)")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    c1, c2 = st.columns(2)
    pr = []
    for m, lg in loops.items():
        for r in lg:
            pr.append({"mode": names[m], "round": r["round"], "true precision of selected pseudo-labels": r["diag_true_precision_selected"],
                       "pseudo-labels used (cumulative)": r["cum_pseudo_labels"],
                       "cumulative pseudo precision": r["cum_pseudo_true_precision"]})
    pr = pd.DataFrame(pr)
    c1.plotly_chart(px.line(pr, x="round", y="true precision of selected pseudo-labels", color="mode", markers=True,
                            title="Diagnostic: true precision of the selected batch (hidden labels, analysis only)"), use_container_width=True)
    c2.plotly_chart(px.line(pr, x="round", y="cumulative pseudo precision", color="mode", markers=True,
                            title="Cumulative precision of accepted pseudo-labels"), use_container_width=True)
    st.write(f"Human labels spent by FULL: **{runs[seed]['human_labels_full']}** (audit budget). "
             "The label-budget-matched baseline (T1) receives the same number of random human labels.")

# ----------------------------------------------------------------------------------------------------------------
elif page == PAGES[3]:
    st.title("Distillation lab")
    st.subheader("Temperature explorer — what soft targets look like")
    logits, labels, note = None, None, ""
    try:
        import torch, joblib
        from vsid.config import Config
        from vsid.models import new_teacher, predict_logits
        from vsid.demo import load_artifacts
        art = load_artifacts()
        q = st.text_input("Query", "My card still has not arrived after two weeks")
        logits = predict_logits(art["teacher"], art["tfeat"].transform([q]))[0]
        labels = art["classes"]
        note = "real logits from the final teacher"
    except Exception as e:
        rng = np.random.RandomState(0)
        logits = np.array([6.0, 4.5, 3.8, 1.0, 0.5, 0.2, 0.0, -0.5]) + rng.randn(8) * 0.05
        labels = [f"intent_{i}" for i in range(8)]
        note = f"synthetic logits (artifacts unavailable: {e})"
    T = st.slider("Temperature T", 0.5, 16.0, 4.0, 0.5)
    alpha = st.slider("α (weight on hard-label CE for labelled rows)", 0.0, 1.0, 0.5, 0.05)
    top = np.argsort(-logits)[:8]
    def sm(z, T):
        z = z / T; z = z - z.max(); e = np.exp(z); return e / e.sum()
    df = pd.DataFrame({"intent": [labels[i] for i in top], "T=1": sm(logits, 1)[top], f"T={T}": sm(logits, T)[top]})
    st.plotly_chart(px.bar(df.melt("intent", var_name="softmax", value_name="probability"), x="intent", y="probability",
                           color="softmax", barmode="group", height=360), use_container_width=True)
    st.caption(f"{note}. At T=1 tiny probabilities are invisible; larger T reveals which wrong answers are *nearly right* (dark knowledge). "
               f"Loss per labelled example = {alpha:.2f}·CE + {1-alpha:.2f}·T²·KL(teacher‖student); unlabelled examples use the KL term only.")
    st.subheader("Student ladder (test, mean ± std over seeds)")
    S = A[A.kind == "student"][["name", "acc", "acc_std", "f1", "params", "size_mb", "p50_ms"]].copy()
    st.dataframe(S.style.format({"acc": "{:.3f}", "acc_std": "{:.3f}", "f1": "{:.3f}", "size_mb": "{:.2f}", "p50_ms": "{:.2f}", "params": "{:,}"}),
                 hide_index=True, use_container_width=True)
    st.markdown("**Reading the ladder:** S1−S0 = soft targets at equal data · S2−S0 = unlabelled data without soft targets · "
                "S3−S2 = soft vs hard on the same data · S3−S5 = value of the improved teacher.")
    sw = pd.concat([pd.DataFrame(d["sweep"]).assign(seed=s) for s, d in runs.items()])
    hm = sw.groupby(["T", "alpha"])["dev_f1"].mean().reset_index().pivot(index="alpha", columns="T", values="dev_f1")
    st.plotly_chart(px.imshow(hm, text_auto=".3f", aspect="auto", color_continuous_scale="Blues",
                              title="Dev macro-F1 of distilled student: temperature × α (mean over seeds)"), use_container_width=True)

# ----------------------------------------------------------------------------------------------------------------
elif page == PAGES[4]:
    st.title("Final evidence — untouched test set")
    st.caption(f"Mean ± std over {len(seeds)} seeds · latency: CPU, 1 thread, end-to-end (featurise + forward), batch = 1 · machine-dependent")
    T = A.copy()
    T["accuracy"] = T.apply(lambda r: f"{r.acc:.3f} ± {r.acc_std:.3f}", axis=1)
    T["macro-F1"] = T.apply(lambda r: f"{r.f1:.3f} ± {r.f1_std:.3f}", axis=1)
    st.dataframe(T[["name", "accuracy", "macro-F1", "params", "size_mb", "p50_ms", "p95_ms", "qps", "human_labels"]]
                 .rename(columns={"size_mb": "MB", "p50_ms": "p50 ms", "p95_ms": "p95 ms", "qps": "queries/s (b=256)", "human_labels": "human labels"})
                 .style.format({"MB": "{:.2f}", "p50 ms": "{:.2f}", "p95 ms": "{:.2f}", "queries/s (b=256)": "{:.0f}", "params": "{:,}"}),
                 hide_index=True, use_container_width=True)
    fig = px.bar(A, x="name", y="acc", error_y="acc_std", color="kind", title="Test accuracy", height=420)
    fig.update_xaxes(tickangle=-35); fig.update_yaxes(range=[max(0, A.acc.min() - 0.05), 1])
    st.plotly_chart(fig, use_container_width=True)
    fig = px.scatter(A, x="p50_ms", y="acc", size="params", color="kind", hover_name="name", size_max=45,
                     title="Accuracy vs end-to-end CPU latency (bubble = parameters)", log_x=True)
    st.plotly_chart(fig, use_container_width=True)
    t0 = A[A.name.str.startswith("T0")].iloc[0]; t4 = A[A.name.str.startswith("T4")].iloc[0]; t1 = A[A.name.str.startswith("T1")].iloc[0]
    s0 = A[A.name.str.startswith("S0")].iloc[0]; s3 = A[A.name.str.startswith("S4")].iloc[0]; tt = A[A.name.str.startswith("T4")].iloc[0]
    st.subheader("Headline comparisons (honest reading)")
    st.markdown(f"- **Loop vs start:** T4 − T0 = **{t4.acc - t0.acc:+.3f}** accuracy (seed std ≈ {max(t4.acc_std, t0.acc_std):.3f})\n"
                f"- **Loop vs label-budget-matched supervised:** T4 − T1 = **{t4.acc - t1.acc:+.3f}** "
                f"(T1 used the same {int(t1.human_labels)} human labels)\n"
                f"- **Distilled vs hard-label student:** S4 − S0 = **{s3.acc - s0.acc:+.3f}**\n"
                f"- **Student vs teacher:** accuracy gap {s3.acc - tt.acc:+.3f}, params ÷{tt.params / s3.params:.0f}, "
                f"latency ÷{tt.p50_ms / s3.p50_ms:.1f}")
    st.caption("If a difference is smaller than the seed std, treat it as *not established* — with 3 seeds this is a rough guide, not a significance test.")
    st.subheader("Dev-vs-test gap (repeated-validation check)")
    gap = []
    for s, d in runs.items():
        full = d["loops"]["full"][-1]
        t4s = next(x for x in d["systems"] if x["name"].startswith("T4"))
        gap.append({"seed": s, "final incumbent dev acc": full["incumbent_dev_acc"], "T4 test acc": t4s["acc"],
                    "gap (dev − test)": full["incumbent_dev_acc"] - t4s["acc"]})
    st.dataframe(pd.DataFrame(gap).style.format("{:.3f}", subset=["final incumbent dev acc", "T4 test acc", "gap (dev − test)"]), hide_index=True)
    st.subheader("Requirements checklist (from the training document)")
    ok = lambda b: "✅" if b else "❌"
    st.markdown(f"""
- {ok(len(seeds) >= 2)} ≥ 2 seeds for distillation comparison ({len(seeds)} seeds)
- {ok(all(d['split_info']['leakage_free'] for d in runs.values()))} train / dev / final-test separation, leakage assertions pass
- ✅ independent verifier + before/after + accepted/rejected updates logged (page 3)
- ✅ teacher, hard-label student (S0), distilled student (S3/S4) with accuracy, size, latency
- ✅ baselines: initial teacher, label-budget-matched, oracle; ablations: naive, verifier-only
- ✅ failure analysis (page 6) · ✅ risk analysis + limitations (docs/IMPLEMENTATION_PLAN.md §3.8, §13) · ✅ CPU only""")

# ----------------------------------------------------------------------------------------------------------------
elif page == PAGES[5]:
    st.title("Failure analysis")
    seed = st.selectbox("Seed", seeds)
    f = runs[seed]["failure"]; classes = runs[seed]["classes"]
    rec = pd.DataFrame({"intent": classes, "T0 (start)": f["per_class_recall"]["teacher0"], "T4 (loop)": f["per_class_recall"]["teacher_full"],
                        "S4 (student)": f["per_class_recall"]["student_s4"]})
    rec["T4−T0"] = rec["T4 (loop)"] - rec["T0 (start)"]
    c1, c2 = st.columns(2)
    c1.subheader("Biggest gains from the loop"); c1.dataframe(rec.sort_values("T4−T0", ascending=False).head(8), hide_index=True)
    c2.subheader("Classes that got WORSE"); c2.dataframe(rec.sort_values("T4−T0").head(8), hide_index=True)
    st.metric("Student–teacher prediction agreement", f"{f['student_teacher_agreement']:.1%}")
    st.subheader("Hardest classes for the student")
    st.dataframe(rec.sort_values("S4 (student)").head(10), hide_index=True)
    st.subheader("Top student confusions")
    st.dataframe(pd.DataFrame(f["top_confusions_student"]), hide_index=True)
    st.subheader("Misclassified test queries (student)")
    st.dataframe(pd.DataFrame(f["examples"]), hide_index=True, use_container_width=True)

# ----------------------------------------------------------------------------------------------------------------
else:
    st.title("Live demo — teacher vs student")
    try:
        import torch
        from vsid.demo import load_artifacts, classify
        art = load_artifacts()
    except Exception as e:
        st.error(f"Model artifacts not available ({e}). Run the pipeline first.")
        st.stop()
    examples = ["My card has not arrived yet", "Why was I charged twice for the same payment?", "How do I top up my account?",
                "I think someone stole my card", "What is the exchange rate for EUR?"]
    choice = st.selectbox("Try an example", examples)
    q = st.text_input("…or type a customer message", choice)
    if q.strip():
        c1, c2 = st.columns(2)
        for col, key, title in ((c1, "teacher", "Teacher (large, verified-loop)"), (c2, "student", "Student (distilled, small)")):
            top, ms = classify(art, key, q)
            col.subheader(title)
            col.metric("latency (end-to-end, this machine)", f"{ms:.2f} ms")
            col.dataframe(pd.DataFrame(top, columns=["intent", "probability"]).style.format({"probability": "{:.2%}"}), hide_index=True)
