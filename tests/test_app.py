import json

import numpy as np
import pandas as pd
import pytest

from src.app import INPUTS, describe, make_record, predict_risk, risk_percentile
from src.features.build import CATEGORICAL_FEATURES, FEATURES
from src.models.baselines import candidate_models
from src.models.evaluate import predict_scores
from src.models.train import load_model, save_model
from src.pipeline import ModelData


def _data():
    rng = np.random.RandomState(0)
    n = 300
    X = pd.DataFrame({c: rng.randint(0, 5, n) for c in FEATURES if c not in CATEGORICAL_FEATURES})
    for c in CATEGORICAL_FEATURES:
        X[c] = rng.choice(["A", "B"], n)
    X = X[FEATURES]
    y = pd.Series((X["number_inpatient"] + rng.rand(n) * 2 >= 4).astype(int))
    return ModelData(X, y, None, None, X, y)


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    data = _data()
    model = candidate_models()["hist_gradient_boosting"].set_params(
        histgradientboostingclassifier__max_iter=20).fit(data.X_train, data.y_train)
    oof = predict_scores(model, data.X_train)  # no teste, chega um score por linha
    path = save_model(model, data, oof, path=tmp_path_factory.mktemp("v1"))
    return data, model, path


def test_skops_roundtrip(saved):
    data, model, path = saved
    assert {p.name for p in path.iterdir()} == {"model.skops", "config.json", "README.md"}
    assert not list(path.glob("*.joblib"))
    bundle = load_model(path)
    X = data.X_train.iloc[:100]
    assert np.allclose(predict_scores(bundle["model"], X), predict_scores(model, X))
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    assert config["threshold_k20"] <= config["threshold_k10"]
    assert {"python", "scikit-learn", "skops"} <= set(config["versions"])


def test_bundle_has_everything_the_app_needs(saved):
    bundle = load_model(saved[2])
    assert bundle["features"] == FEATURES
    assert set(INPUTS) <= set(bundle["defaults"])
    assert bundle["choices"]["insulin"] == ["A", "B"]


def test_make_record_fills_defaults_and_derived(saved):
    bundle = load_model(saved[2])
    r = make_record(bundle, number_inpatient=2, number_emergency=1, number_outpatient=0)
    assert list(r.columns) == FEATURES
    assert r.loc[0, "n_visits_prior"] == 3


def test_app_percentile(saved):
    bundle = load_model(saved[2])
    low = predict_risk(bundle, number_inpatient=0, number_emergency=0, number_outpatient=0)
    high = predict_risk(bundle, number_inpatient=4, number_emergency=0, number_outpatient=0)
    for out in (low, high):
        assert 0 <= out["percentile"] <= 100
    assert high["percentile"] > low["percentile"]
    # a ação muda ao passar o corte de 10%
    t = bundle["threshold_k10"]
    assert risk_percentile(bundle, t + 1e-9) >= 89
    assert high["follow_up_k10"] and not low["follow_up_k10"]
    assert "propor follow-up" in describe(high) and "sem follow-up" in describe(low)
    assert "%" in describe(high) and "probabilidade" not in describe(high).lower()


def test_build_app(saved):
    pytest.importorskip("gradio")
    from src.app import build_app

    assert build_app(load_model(saved[2])) is not None
