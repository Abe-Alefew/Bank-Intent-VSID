"""Text -> sparse TF-IDF featurisers. Fitted on unlabelled *training-side* text only (seed + pool); never on dev/test."""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer


class TeacherFeaturizer:
    """word 1-2 grams  +  char 3-5 grams (rich, slower)."""
    def __init__(self, n_word: int, n_char: int):
        self.w = TfidfVectorizer(ngram_range=(1, 2), max_features=n_word, sublinear_tf=True, lowercase=True)
        self.c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=n_char, sublinear_tf=True)

    def fit(self, texts):
        self.w.fit(texts); self.c.fit(texts); return self

    def transform(self, texts):
        return sp.hstack([self.w.transform(texts), self.c.transform(texts)], format="csr", dtype=np.float32)

    @property
    def n_features(self):
        return len(self.w.vocabulary_) + len(self.c.vocabulary_)


class StudentFeaturizer:
    """word 1-2 grams only, small vocabulary (cheap, fast)."""
    def __init__(self, n_word: int):
        self.w = TfidfVectorizer(ngram_range=(1, 2), max_features=n_word, sublinear_tf=True, lowercase=True)

    def fit(self, texts):
        self.w.fit(texts); return self

    def transform(self, texts):
        return self.w.transform(texts).astype(np.float32).tocsr()

    @property
    def n_features(self):
        return len(self.w.vocabulary_)
