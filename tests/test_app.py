import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.app import INPUTS, make_record, predict_risk
from src.features.build import CATEGORICAL_FEATURES, FEATURES
from src.models.train import load_model, save_model
from src.pipeline import ModelData, build_preprocessor


def _bundle(tmp_path):
    rng = np.random.RandomState(0)
    n = 300
    X = pd.DataFrame({c: rng.randint(0, 5, n) for c in FEATURES if c not in CATEGORICAL_FEATURES})
    for c in CATEGORICAL_FEATURES:
        X[c] = rng.choice(["A", "B"], n)
    X = X[FEATURES]
    y = pd.Series((X["number_inpatient"] >= 3).astype(int))
    data = ModelData(X, y, None, None, X, y)
    model = make_pipeline(build_preprocessor(), LogisticRegression(max_iter=1000)).fit(X, y)
    path = save_model(model, data, threshold=0.5, path=tmp_path / "model.joblib")
    return load_model(path)


def test_bundle_has_everything_the_app_needs(tmp_path):
    b = _bundle(tmp_path)
    assert b["features"] == FEATURES
    assert set(INPUTS) <= set(b["defaults"])
    assert b["choices"]["insulin"] == ["A", "B"]


def test_make_record_fills_defaults_and_derived(tmp_path):
    b = _bundle(tmp_path)
    r = make_record(b, number_inpatient=2, number_emergency=1, number_outpatient=0)
    assert list(r.columns) == FEATURES
    assert r.loc[0, "n_visits_prior"] == 3


def test_predict_risk_increases_with_inpatient(tmp_path):
    b = _bundle(tmp_path)
    low = predict_risk(b, number_inpatient=0, number_emergency=0, number_outpatient=0)
    high = predict_risk(b, number_inpatient=4, number_emergency=0, number_outpatient=0)
    assert 0 <= low["risk"] < high["risk"] <= 1
    assert high["flag"] is True


def test_build_app(tmp_path):
    pytest.importorskip("gradio")
    from src.app import build_app

    assert build_app(_bundle(tmp_path)) is not None
