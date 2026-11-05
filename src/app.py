"""
Demo Gradio do modelo final (fase IV, deployment).

Lê o bundle gravado por src.models.train.save_model. O utilizador preenche as
features com mais peso; as restantes ficam com a mediana/moda do treino.
Correr a partir da raiz do repositório:
    python -m src.app
"""

import pandas as pd

from src.models.train import MODEL_PATH, load_model

# Campos mostrados na app; os outros vêm de bundle["defaults"].
NUMERIC_INPUTS = ["age_band", "time_in_hospital", "num_medications", "number_diagnoses",
                  "number_inpatient", "number_emergency", "number_outpatient"]
CATEGORICAL_INPUTS = ["discharge_disposition_id_desc", "admission_source_id_desc",
                      "diag_1_group", "A1Cresult", "insulin"]
INPUTS = NUMERIC_INPUTS + CATEGORICAL_INPUTS


def make_record(bundle: dict, **values) -> pd.DataFrame:
    """Uma linha com as features do modelo: valores dados + defaults do treino.

    As features derivadas que dependem das visitas são recalculadas aqui,
    para ficarem coerentes com o que o utilizador escreveu.
    """
    row = {**bundle["defaults"], **values}
    row["n_visits_prior"] = row["number_outpatient"] + row["number_emergency"] + row["number_inpatient"]
    return pd.DataFrame([row])[bundle["features"]]


def predict_risk(bundle: dict, **values) -> dict:
    """Probabilidade de readmissão < 30 dias e se fica acima do threshold (T3.3)."""
    p = float(bundle["model"].predict_proba(make_record(bundle, **values))[0, 1])
    t = bundle.get("threshold")
    return {"risk": p, "flag": None if t is None else p >= t}


def build_app(bundle: dict):
    import gradio as gr  # só é preciso para correr a app, não para os testes

    d = bundle["defaults"]
    inputs = [gr.Number(value=d[c], label=c, precision=0) for c in NUMERIC_INPUTS]
    inputs += [gr.Dropdown(bundle["choices"][c], value=d[c], label=c) for c in CATEGORICAL_INPUTS]

    def _predict(*args):
        out = predict_risk(bundle, **dict(zip(INPUTS, args)))
        label = "" if out["flag"] is None else (" — sinalizar para follow-up" if out["flag"] else " — sem follow-up")
        return f"Risco de readmissão < 30 dias: {out['risk']:.1%}{label}"

    return gr.Interface(fn=_predict, inputs=inputs, outputs=gr.Textbox(label="Resultado"),
                        title="Risco de readmissão a 30 dias (diabetes)",
                        description="Apoio à decisão no momento da alta. Não substitui o juízo clínico.")


if __name__ == "__main__":
    build_app(load_model(MODEL_PATH)).launch()
