"""Load saved artifacts (seed-0 models) for the Streamlit live demo."""
from __future__ import annotations
import time
from pathlib import Path
import joblib, numpy as np, torch
from .config import Config, ROOT
from .models import new_teacher, new_student, predict_logits, softmax_np

_CACHE = {}


def load_artifacts(results_dir=None):
    key = str(results_dir)
    if key in _CACHE:
        return _CACHE[key]
    ad = Path(results_dir or (ROOT / "results")) / "artifacts"
    meta = joblib.load(ad / "featurizers.joblib")
    cfg = Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in meta["cfg"].items()})
    C = len(meta["classes"])
    teacher = new_teacher(meta["n_tf"], C, cfg); teacher.load_state_dict(torch.load(ad / "teacher.pt")); teacher.eval()
    student = new_student(meta["n_sf"], C, cfg); student.load_state_dict(torch.load(ad / "student.pt")); student.eval()
    torch.set_num_threads(1)
    _CACHE[key] = {"teacher": teacher, "student": student, "tfeat": meta["tfeat"], "sfeat": meta["sfeat"], "classes": meta["classes"]}
    return _CACHE[key]


def classify(art, which, text, k=3):
    feat = art["tfeat"] if which == "teacher" else art["sfeat"]
    t0 = time.perf_counter()
    z = predict_logits(art[which], feat.transform([text]))
    ms = (time.perf_counter() - t0) * 1000
    p = softmax_np(z)[0]
    top = np.argsort(-p)[:k]
    return [(art["classes"][i], float(p[i])) for i in top], ms
