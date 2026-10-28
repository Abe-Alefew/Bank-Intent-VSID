"""Real dataset: Banking77 (PolyAI) - 13,083 genuine online-banking customer queries, 77 intents.

Downloaded at runtime from the public PolyAI GitHub repository (CSV).
Splits (per seed):
    seed  : tiny *trusted* labelled set (default 10 / class)
    pool  : large *unlabelled* stream (hidden labels are used ONLY for audit + diagnostics)
    dev   : gate set used by the acceptance gates every round (re-used -> weaker evidence)
    test  : official test split, touched ONCE at the end of each run (final evidence)
"""
from __future__ import annotations
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
FILES = {"train": "train.csv", "test": "test.csv"}


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", str(t).lower().strip())


def download(data_dir: str) -> None:
    d = Path(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    for split, fn in FILES.items():
        dst = d / f"banking77_{split}.csv"
        if dst.exists() and dst.stat().st_size > 1000:
            continue
        url = BASE + fn
        print(f"[data] downloading {url}")
        try:
            urllib.request.urlretrieve(url, dst)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                f"Could not download {url}: {e}\n"
                f"Download it manually and save it as {dst} (columns: text,category)."
            )


def load_banking77(data_dir: str):
    download(data_dir)
    tr = pd.read_csv(Path(data_dir) / "banking77_train.csv")
    te = pd.read_csv(Path(data_dir) / "banking77_test.csv")
    classes = sorted(tr["category"].unique())
    cid = {c: i for i, c in enumerate(classes)}
    tr = tr.rename(columns={"category": "intent"})
    te = te.rename(columns={"category": "intent"})
    tr["label"] = tr["intent"].map(cid)
    te["label"] = te["intent"].map(cid)
    return tr[["text", "intent", "label"]], te[["text", "intent", "label"]], classes


@dataclass
class Splits:
    seed: pd.DataFrame
    pool: pd.DataFrame      # 'label' column = HIDDEN ground truth (audit / diagnostics only)
    dev: pd.DataFrame
    test: pd.DataFrame
    classes: list
    info: dict              # leakage checks + sizes


def make_splits(train: pd.DataFrame, test: pd.DataFrame, classes: list, cfg, seed: int) -> Splits:
    rng = np.random.RandomState(seed)
    n0 = len(train)
    train = train.assign(_k=train["text"].map(_norm))
    test = test.assign(_k=test["text"].map(_norm))
    # leakage hygiene: drop exact duplicates inside train and train rows that also occur in test
    train = train.drop_duplicates("_k")
    n_dups = n0 - len(train)
    in_test = train["_k"].isin(set(test["_k"]))
    n_overlap = int(in_test.sum())
    train = train[~in_test].reset_index(drop=True)

    perm = rng.permutation(len(train))
    dev = train.iloc[perm[: cfg.n_dev]]
    rest = train.iloc[perm[cfg.n_dev:]]
    seed_idx = []
    for c in range(len(classes)):
        seed_idx += list(rest.index[rest["label"] == c][: cfg.seed_per_class])
    seed_df = rest.loc[seed_idx]
    pool = rest.drop(index=seed_idx)

    parts = {"seed": seed_df, "pool": pool, "dev": dev, "test": test}
    keys = {k: set(v["_k"]) for k, v in parts.items()}
    names = list(keys)
    overlaps = {f"{a}&{b}": len(keys[a] & keys[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    assert all(v == 0 for v in overlaps.values()), f"LEAKAGE between splits: {overlaps}"

    info = {
        "n_original_train": n0, "n_exact_duplicates_removed": int(n_dups),
        "n_train_rows_also_in_test_removed": n_overlap,
        "sizes": {k: int(len(v)) for k, v in parts.items()},
        "pairwise_text_overlap": overlaps, "leakage_free": True,
        "seed_per_class": cfg.seed_per_class,
    }
    cols = ["text", "intent", "label"]
    return Splits(seed_df[cols].reset_index(drop=True), pool[cols].reset_index(drop=True),
                  dev[cols].reset_index(drop=True), test[cols].reset_index(drop=True), classes, info)
