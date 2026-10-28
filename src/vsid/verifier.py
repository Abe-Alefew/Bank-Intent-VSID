"""Independent verifier + audit statistics.

The verifier is a *different model family* (logistic regression) on a *different feature view* (char n-grams 2-4)
and is trained ONLY on trusted human-labelled data (seed + audited examples). It never sees a pseudo-label, so it cannot
simply echo the teacher's mistakes (no feedback contamination).  It is still imperfect: errors can be correlated with the
teacher's, which is why the human-audit gate (external ground truth) is the final authority.
"""
from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class IndependentVerifier:
    def __init__(self, n_features: int, n_classes: int, C: float = 20.0):
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=n_features, sublinear_tf=True)
        self.n_classes = n_classes
        self.C = C
        self.clf = None

    def fit_vectorizer(self, texts):
        self.vec.fit(texts)
        return self

    def fit(self, texts, labels):
        X = self.vec.transform(texts)
        self.clf = LogisticRegression(C=self.C, max_iter=150)
        self.clf.fit(X, np.asarray(labels))
        return self

    def predict_proba(self, texts) -> np.ndarray:
        P = np.zeros((len(texts), self.n_classes), dtype=np.float32)
        P[:, self.clf.classes_] = self.clf.predict_proba(self.vec.transform(texts))
        return P


def wilson_lower(k: int, n: int, z: float) -> float:
    """Lower bound of the Wilson score interval for a binomial proportion (robust for small n)."""
    if n == 0:
        return 0.0
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((centre - margin) / den)


def class_entropy(labels: np.ndarray, n_classes: int) -> float:
    """Normalised Shannon entropy of a label histogram in [0,1]; 1 = perfectly balanced, 0 = single class (collapse)."""
    if len(labels) == 0:
        return 0.0
    h = np.bincount(labels, minlength=n_classes).astype(float)
    p = h[h > 0] / h.sum()
    return float(-(p * np.log(p)).sum() / np.log(n_classes))
