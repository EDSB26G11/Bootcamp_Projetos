import json

import numpy as np
import pandas as pd
import pytest

from src.app import (
    CAPACITY,
    INPUTS,
    capacity_threshold,
    chart_notes,
    describe,
    is_eligible,
    make_record,
    message,
    predict_risk,
    result_html,
    recommendation,
    risk_percentile,
)
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
    model = (
        candidate_models()["hist_gradient_boosting"]
        .set_params(histgradientboostingclassifier__max_iter=20)
        .fit(data.X_train, data.y_train)
    )
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
    # a recomendação muda ao passar o corte da capacidade (30%)
    t = capacity_threshold(bundle)
    assert risk_percentile(bundle, t + 1e-9) >= 100 - CAPACITY - 1
    assert high["follow_up"] and not low["follow_up"]
    assert "Recomendação: Follow-up" in describe(high) and "Recomendação: Sem follow-up" in describe(low)
    assert "%" in describe(high) and "probabilidade" not in describe(high).lower()


def test_build_app(saved):
    pytest.importorskip("gradio")
    from src.app import build_app

    assert build_app(load_model(saved[2])) is not None


def test_app_not_eligible_never_gets_follow_up(saved):
    bundle = load_model(saved[2])
    assert bundle["no_follow_up_discharge"] and bundle["n_eligible"] <= bundle["n_train"]
    dest = bundle["no_follow_up_discharge"][0]
    assert not is_eligible(bundle, discharge_disposition_id_desc=dest)
    out = predict_risk(
        bundle, discharge_disposition_id_desc=dest, number_inpatient=4, number_emergency=0, number_outpatient=0
    )
    assert not out["eligible"] and out["high_risk"] and not out["follow_up"]
    assert 0 <= out["percentile"] <= 100
    for text in (message(out), describe(out)):
        assert "Sem follow-up" in text and "Risco alto ou Destino com cuidados médicos" in text


def test_app_shows_estimated_risk_with_reference(saved):
    bundle = load_model(saved[2])
    out = predict_risk(bundle, number_inpatient=4, number_emergency=0, number_outpatient=0)
    assert 0 <= out["risk"] <= 1 and out["base_rate"] == pytest.approx(bundle["base_rate_eligible"])
    text = describe(out)
    assert "Risco de readmissão" in text and "média de" in text and "Em cada 100 doentes" in text
    html = result_html(out)
    assert "Risco de readmissão" in html and "em cada 100" in html and f"{CAPACITY}%" in html
    assert "elegíveis" not in text  # jargão do projeto (D21): na app diz-se "que podem receber follow-up"


def test_chart_notes_explain_reading_and_assumptions(saved):
    bundle = load_model(saved[2])
    notes = chart_notes(bundle)
    assert "Como ler" in notes and "Pressupostos principais" in notes
    assert f"{CAPACITY}%" in notes and "D21" in notes and "Destino sem / com cuidados médicos" in notes


def test_shown_percentile_matches_action_at_the_cut(saved):
    """Logo abaixo do corte de 30% a alta não pode aparecer como P70 (onde a lista começa)."""
    from src.app import _shown_percentile, _top

    bundle = load_model(saved[2])
    t = capacity_threshold(bundle)
    below = {"percentile": risk_percentile(bundle, t - 1e-9)}
    above = {"percentile": risk_percentile(bundle, t + 1e-9)}
    assert _shown_percentile(below) < 100 - CAPACITY <= _shown_percentile(above)
    assert _top(above) <= CAPACITY < _top(below)


def test_capacity_threshold_matches_saved_cuts(saved):
    """Sem threshold_k30 no config.json, o corte vem dos quantis OOF e bate com os cortes gravados."""
    bundle = load_model(saved[2])
    assert capacity_threshold(bundle, 10) == pytest.approx(bundle["threshold_k10"])
    q = np.asarray(bundle["oof_quantiles"])
    assert capacity_threshold(bundle, 20) == pytest.approx(np.quantile(q, 0.80))
    assert bundle["threshold_k20"] >= capacity_threshold(bundle, 30)


@pytest.mark.parametrize(
    "eligible, high, rec, why",
    [
        (True, True, "Follow-up", "Risco alto ou Destino sem cuidados médicos"),
        (True, False, "Sem follow-up", "Risco baixo ou Destino sem cuidados médicos"),
        (False, True, "Sem follow-up", "Risco alto ou Destino com cuidados médicos"),
        (False, False, "Sem follow-up", "Risco baixo ou Destino com cuidados médicos"),
    ],
)
def test_recommendation_and_reason(eligible, high, rec, why):
    out = {
        "eligible": eligible,
        "high_risk": high,
        "follow_up": eligible and high,
        "risk": 0.1,
        "base_rate": 0.096,
        "percentile": 80.0,
    }
    assert recommendation(out) == (rec, why)
    html = message(out)
    assert f'<span class="rec-value">{rec}</span>' in html and why in html


def test_result_html_matches_the_numbers():
    """O quadro mostra o mesmo número nos 100 pontos, no texto e na posição."""
    out = {"eligible": True, "high_risk": True, "follow_up": True, "risk": 0.19, "base_rate": 0.096, "percentile": 92.7}
    html = result_html(out)
    assert html.count('class="dot on"') == 19 and html.count('class="dot') == 100
    assert "19 em cada 100" in html and "19%" in html and "top 8%" in html and "Acima da média" in html
    assert "rec-box rec-yes" in html
    low = {**out, "high_risk": False, "follow_up": False, "risk": 0.004, "percentile": 5.0}
    html = result_html(low)
    assert html.count('class="dot on"') == 0 and "Menos de 1 em cada 100" in html and "rec-box rec-no" in html
