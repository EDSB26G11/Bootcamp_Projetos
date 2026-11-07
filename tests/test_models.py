import os

import numpy as np
import pandas as pd
import pytest

from src.models.baselines import ColumnRanker, baseline_models, candidate_models, simple_models
from src.models.evaluate import cv_predict, metrics_by_group, precision_at_k, recall_at_k, score_metrics, summarize
from src.models.threshold import (capacity_table, confusion_at_k, gains_curve, rank_baseline_at_k,
                                  threshold_for_capacity)
from src.models.train import log_experiment, run_cv
from src.pipeline import PROCESSED, SPLITS, ModelData

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
    p = baseline_models()["random"].fit(X, [0, 1, 0]).predict_proba(X)[:, 1]
    assert np.allclose(p, 1 / 3)  # DummyClassifier(prior): a taxa de positivos
    assert set(baseline_models()) == {"random", "rule_inpatient_ge_1", "rank_inpatient", "logreg"}
    assert all(hasattr(m, "predict_proba") for m in {**simple_models(), **candidate_models()}.values())
    assert set(candidate_models()) == {"random_forest", "hist_gradient_boosting"}


def test_capacity_threshold():
    rng = np.random.RandomState(0)
    s = rng.rand(5000)
    y = (rng.rand(5000) < s * 0.25).astype(int)
    tab = capacity_table(y, s)
    for k in (0.10, 0.20):
        assert tab.loc[k, "pct_sinalizados"] == pytest.approx(k, abs=0.005)
        assert tab.loc[k, "lift"] == pytest.approx(tab.loc[k, "precision"] / y.mean())
        assert tab.loc[k, "readm_encontradas_por_1000_altas"] == pytest.approx(
            1000 * tab.loc[k, "pct_sinalizados"] * tab.loc[k, "precision"])
    assert "tier" not in " ".join(tab.columns)
    assert (s >= threshold_for_capacity(s, 0.1)).mean() == pytest.approx(0.1, abs=0.002)


def test_rank_baseline_ties():
    rng = np.random.RandomState(1)
    y = (rng.rand(2000) < 0.11).astype(int)
    out = rank_baseline_at_k(y, np.zeros(2000), 0.10)  # tudo empatado: ordem ao acaso
    assert out["recall_mean"] == pytest.approx(0.10, abs=0.02)
    assert out["recall_std"] > 0


def test_confusion_at_k_sums():
    rng = np.random.RandomState(2)
    s = rng.rand(1000)
    y = (rng.rand(1000) < s * 0.3).astype(int)
    c = confusion_at_k(y, s, 0.20)
    n_flag = capacity_table(y, s, capacities=(0.20,)).loc[0.20, "n_sinalizados"]
    assert c["tp"] + c["fp"] == n_flag
    assert c["tp"] + c["fn"] == y.sum()
    assert c["matrix"].sum() == 1000


def test_gains_curve_ends_at_100():
    rng = np.random.RandomState(3)
    s = rng.rand(500)
    y = (s > 0.8).astype(int)
    g = gains_curve(y, s)
    assert g["pct_readm_apanhadas"].iloc[-1] == pytest.approx(100)
    assert g["pct_readm_apanhadas"].is_monotonic_increasing


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


def test_run_cv_logs_current_and_history(tmp_path):
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"number_inpatient": rng.randint(0, 4, 200)})
    y = pd.Series((X["number_inpatient"] + rng.rand(200) > 2.5).astype(int))
    data = ModelData(X, y, pd.Series(range(200)), pd.Series(np.arange(200) % 5), X, y)
    cur, hist = tmp_path / "experiments.csv", tmp_path / "experiments_v1.csv"
    log_experiment({"model": "antiga"}, hist)
    folds, oof = run_cv("regra", ColumnRanker(threshold=1), data, path=cur, history=hist, return_oof=True)
    log = pd.read_csv(cur)
    assert log["dataset"].tolist() == ["train", "validation"]  # formato da W6
    assert {"date", "model", "dataset", "pr_auc", "log_loss", "brier", "note"} <= set(log.columns)
    assert pd.read_csv(hist)["model"].tolist() == ["antiga", "regra", "regra"]
    assert len(folds) == 5 and not oof.isna().any()


def test_oof_covers_every_training_row_once():
    rng = np.random.RandomState(4)
    X = pd.DataFrame({"number_inpatient": rng.randint(0, 4, 100)})
    y = pd.Series((rng.rand(100) < 0.3).astype(int))
    data = ModelData(X, y, pd.Series(range(100)), pd.Series(np.arange(100) % 5), X, y)
    train_folds, val_folds, oof = cv_predict(ColumnRanker(), data)
    assert len(oof) == 100 and not oof.isna().any()
    assert list(train_folds.index) == list(val_folds.index) == [0, 1, 2, 3, 4]


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


def test_calibration_table_perfect_and_shape():
    from src.models.evaluate import calibration_table

    rng = np.random.RandomState(0)
    p = rng.uniform(0, 0.5, 20000)
    y = (rng.uniform(size=p.size) < p).astype(int)  # y gerado com probabilidade p: bem calibrado
    t = calibration_table(y, p)
    assert len(t) == 10 and t["n"].sum() == p.size
    assert t["risco_previsto"].is_monotonic_increasing
    assert (t["diferenca"].abs() < 0.03).all()
