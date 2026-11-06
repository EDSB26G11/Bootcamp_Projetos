"""
Baselines da T2.1: a fasquia que os modelos candidatos têm de bater.

Os rankers são estimadores scikit-learn sem aprendizagem, para passarem pelo
mesmo cv_evaluate que os modelos a sério.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.data.split import SEED
from src.pipeline import build_preprocessor


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


def simple_models() -> dict:
    """Variantes da regressão logística: pesos por classe e regularização L1."""
    return {
        "logreg_balanced": logistic_regression(class_weight="balanced"),
        "logreg_l1": logistic_regression(C=0.1, penalty="l1", solver="liblinear"),
    }


def hist_gradient_boosting(**preprocessor_kwargs):
    """Modelo v1 (F2·1). Os argumentos vão para build_preprocessor (ablações F2·2).

    Sem early stopping: a validação interna do sklearn é aleatória por
    encontro, não por doente (D13), e deixaria passar fuga por doente.
    """
    return make_pipeline(build_preprocessor(**preprocessor_kwargs), HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=100,
        l2_regularization=1.0, early_stopping=False, random_state=SEED))


def candidate_models() -> dict:
    """Modelos candidatos da T2.2, com o pré-processamento comum (D16) e uma
    configuração razoável; o tuning fica para a fase seguinte. O modelo v1 é
    o HistGradientBoosting (F2·1)."""
    return {
        "random_forest": make_pipeline(build_preprocessor(), RandomForestClassifier(
            n_estimators=300, min_samples_leaf=50, max_features="sqrt", n_jobs=-1, random_state=SEED)),
        "hist_gradient_boosting": hist_gradient_boosting(),
    }


def baseline_models() -> dict:
    """Os quatro baselines da tabela de referência de docs/modelacao_base.md."""
    return {
        # Ao acaso, como na W6: score constante = taxa de positivos do treino;
        # o top k% sai dos empates desfeitos ao acaso.
        "random": DummyClassifier(strategy="prior"),
        "rule_inpatient_ge_1": ColumnRanker("number_inpatient", threshold=1),
        "rank_inpatient": ColumnRanker("number_inpatient"),
        "logreg": logistic_regression(),
    }
