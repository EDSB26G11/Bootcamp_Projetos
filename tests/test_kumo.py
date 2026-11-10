"""Testes do desafiante Kumo Tabular (T2.3 / F2·4).

Não precisam de GPU, torch, sdm nem dos pesos: o Kumo é substituído por um mock
que regista o contexto e as linhas a prever de cada chamada.
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.data.population import NO_FOLLOW_UP_DISCHARGE
from src.features.build import CATEGORICAL_FEATURES, FEATURES
from src.models.kumo import KumoTabularClassifier, kumo_cv, stratified_context
from src.pipeline import PROCESSED, SPLITS, ModelData, load_model_data

HAS_DATA = os.path.exists(PROCESSED) and os.path.exists(SPLITS)


class RecordingBackend:
    """Mock do Kumo: guarda os índices do contexto e das linhas a prever."""

    def __init__(self):
        self.calls = []

    def __call__(self, X_context, y_context, X_query):
        self.calls.append({"context": X_context.index, "query": X_query.index,
                           "columns": list(X_context.columns)})
        return np.random.RandomState(0).rand(len(X_query))


def _synthetic_data(n_patients=400, seed=0) -> ModelData:
    """Doentes com 1-3 encontros; folds por doente (como D13)."""
    rng = np.random.RandomState(seed)
    patients = np.repeat(np.arange(n_patients), rng.randint(1, 4, n_patients))
    n = len(patients)
    X = pd.DataFrame({c: ("a" if c in CATEGORICAL_FEATURES else 0) for c in FEATURES}, index=range(n))
    X["discharge_disposition_id_desc"] = np.where(rng.rand(n) < 0.3, NO_FOLLOW_UP_DISCHARGE[3], "Home")
    y = pd.Series((rng.rand(n) < 0.2).astype(int))
    return ModelData(X_train=X, y_train=y, groups_train=pd.Series(patients),
                     folds_train=pd.Series(patients % 5), X_test=X.iloc[:0], y_test=y.iloc[:0])


def _assert_no_patient_in_both(backend, groups):
    assert len(backend.calls) == 5
    for call in backend.calls:
        ctx, query = set(groups.loc[call["context"]]), set(groups.loc[call["query"]])
        assert ctx and query and not ctx & query


def test_context_patients_never_in_query_rows():
    data = _synthetic_data()
    backend = RecordingBackend()
    kumo_cv("mock", KumoTabularClassifier(max_context=300, backend=backend), data, log=False)
    _assert_no_patient_in_both(backend, data.groups_train)


@pytest.mark.skipif(not HAS_DATA, reason="dados processados não disponíveis")
def test_context_patients_never_in_query_rows_real_folds():
    data = load_model_data()
    backend = RecordingBackend()
    per_fold, oof = kumo_cv("mock", KumoTabularClassifier(max_context=55_000, backend=backend), data,
                            log=False)
    _assert_no_patient_in_both(backend, data.groups_train)
    assert per_fold["n_context"].eq(55_000).all() and oof.notna().all()


def test_context_has_only_the_34_features():
    X = _synthetic_data().X_train.assign(patient_nbr=1, encounter_id=2)
    y = np.zeros(len(X), dtype=int)
    y[::5] = 1
    m = KumoTabularClassifier(max_context=100, backend=RecordingBackend()).fit(X, y)
    assert list(m.context_X_.columns) == FEATURES
    m.predict_proba(X)
    assert m.backend.calls[0]["columns"] == FEATURES


def test_stratified_context_keeps_positive_rate_and_is_reproducible():
    y = np.array([1] * 113 + [0] * 887)
    keep = stratified_context(y, 300)
    assert len(keep) == 300 and y[keep].mean() == pytest.approx(0.113, abs=0.004)
    assert np.array_equal(keep, stratified_context(y, 300))
    assert len(stratified_context(y, 5000)) == len(y)  # pede mais do que há: todas


def test_kumo_cv_logs_only_validation(tmp_path):
    data = _synthetic_data()
    path = tmp_path / "experiments.csv"
    per_fold, _ = kumo_cv("mock", KumoTabularClassifier(max_context=300, backend=RecordingBackend()),
                          data, path=str(path), history=None)
    log = pd.read_csv(path)
    assert log["dataset"].tolist() == ["validation"]
    assert "Só validação" in log.loc[0, "note"] and "backend" not in log.loc[0, "params"]
    assert {"recall_at_10_eleg", "precision_at_20_eleg", "seconds"} <= set(per_fold.columns)
    assert "recall_at_10_eleg" not in log.columns  # o registo mantém as colunas das outras linhas
