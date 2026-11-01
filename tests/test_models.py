import os

import numpy as np
import pandas as pd
import pytest

from src.models.baselines import ColumnRanker, RandomRanker, baseline_models
from src.models.evaluate import metrics_by_group, precision_at_k, recall_at_k, score_metrics, summarize
from src.models.threshold import capacity_table, threshold_for_capacity
from src.models.train import log_experiment
from src.pipeline import PROCESSED, SPLITS

HAS_DATA = os.path.exists(PROCESSED) and os.path.exists(SPLITS)


def test_recall_and_precision_at_k():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    s = np.arange(10, 0, -1)  # o primeiro tem o maior score
    assert recall_at_k(y, s, 0.2) == pytest.approx(1 / 3)
    assert precision_at_k(y, s, 0.2) == pytest.approx(0.5)
    assert recall_at_k(y, s, 1.0) == 1


def test_score_metrics_perfect_ranking():
    y = np.array([0] * 90 + [1] * 10)
    m = score_metrics(y, np.arange(100))
    assert m["pr_auc"] == pytest.approx(1) and m["roc_auc"] == pytest.approx(1)
    assert m["recall_at_10"] == 1 and m["precision_at_10"] == 1


def test_summarize_mean_and_std():
    s = summarize(pd.DataFrame({"pr_auc": [0.2, 0.4]}))
    assert s["pr_auc_mean"] == pytest.approx(0.3)
    assert s["pr_auc_std"] == pytest.approx(np.std([0.2, 0.4], ddof=1))


def test_rankers():
    X = pd.DataFrame({"number_inpatient": [0, 1, 3]})
    rule = ColumnRanker(threshold=1).fit(X, None).predict_proba(X)[:, 1]
    assert rule.tolist() == [0, 1, 1]
    rank = ColumnRanker().fit(X, None).predict_proba(X)[:, 1]
    assert np.all(np.diff(rank) > 0)
    p = RandomRanker().fit(X, None).predict_proba(X)
    assert np.allclose(p.sum(axis=1), 1)
    assert set(baseline_models()) == {"random", "rule_inpatient_ge_1", "rank_inpatient", "logreg"}


def test_capacity_threshold():
    s = np.linspace(0, 1, 1001)
    assert (s >= threshold_for_capacity(s, 0.1)).mean() == pytest.approx(0.1, abs=0.002)
    y = (s > 0.9).astype(int)
    tab = capacity_table(y, s, capacities=(0.1,))
    assert tab.loc[0.1, "recall"] == pytest.approx(1, abs=0.01)


def test_metrics_by_group_uses_global_top_k():
    y = np.array([1, 0] * 100)
    s = np.r_[np.ones(100), np.zeros(100)]  # só o grupo A tem score alto
    g = np.r_[["A"] * 100, ["B"] * 100]
    out = metrics_by_group(y, s, g, k=0.5)
    assert out.loc["A", "flagged_rate"] == 1 and out.loc["B", "flagged_rate"] == 0


def test_log_experiment_appends(tmp_path):
    path = tmp_path / "experiments.csv"
    log_experiment({"model": "a", "pr_auc_mean": 0.2}, path)
    log = log_experiment({"model": "b", "pr_auc_mean": 0.3}, path)
    assert log["model"].tolist() == ["a", "b"]
    assert len(pd.read_csv(path)) == 2


@pytest.mark.skipif(not HAS_DATA, reason="data/processed ou data/splits não gerados")
def test_baseline_reproduces_reference():
    from src.models.evaluate import cv_evaluate
    from src.pipeline import load_model_data

    # docs/modelacao_base.md. A PR-AUC não entra: com scores empatados depende
    # do método (o average_precision do sklearn dá 0,141, a tabela tem 0,150).
    m = cv_evaluate(ColumnRanker(threshold=1), load_model_data()).mean()
    assert m["roc_auc"] == pytest.approx(0.592, abs=0.002)
    assert m["recall_at_10"] == pytest.approx(0.148, abs=0.003)
    assert m["recall_at_20"] == pytest.approx(0.299, abs=0.003)
