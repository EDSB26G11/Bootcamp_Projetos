"""
Baselines da T2.1: a fasquia que os modelos candidatos têm de bater.

Os rankers são estimadores scikit-learn sem aprendizagem, para passarem pelo
mesmo cv_evaluate que os modelos a sério.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.data.split import SEED
from src.pipeline import build_preprocessor


class RandomRanker(ClassifierMixin, BaseEstimator):
    """Score ao acaso: o mínimo absoluto (PR-AUC ≈ taxa de positivos)."""

    def __init__(self, random_state=SEED):
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.random.RandomState(self.random_state).rand(len(X))
        return np.column_stack([1 - p, p])


class ColumnRanker(ClassifierMixin, BaseEstimator):
    """Ordena pelo valor de uma coluna; com threshold, vira regra binária."""

    def __init__(self, column="number_inpatient", threshold=None):
        self.column = column
        self.threshold = threshold

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = X[self.column].to_numpy(dtype=float)
        if self.threshold is not None:
            p = (p >= self.threshold).astype(float)
        else:
            p = p / max(p.max(), 1)
        return np.column_stack([1 - p, p])


def logistic_regression(**params):
    """Regressão logística com o pré-processamento comum (D16)."""
    return make_pipeline(build_preprocessor(), LogisticRegression(max_iter=2000, **params))


def baseline_models() -> dict:
    """Os quatro baselines da tabela de referência de docs/modelacao_base.md."""
    return {
        "random": RandomRanker(),
        "rule_inpatient_ge_1": ColumnRanker("number_inpatient", threshold=1),
        "rank_inpatient": ColumnRanker("number_inpatient"),
        "logreg": logistic_regression(),
    }
