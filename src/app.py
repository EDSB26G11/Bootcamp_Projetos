"""
Demo Gradio do modelo final (fase IV, deployment).

Lê a pasta gravada por src.models.train.save_model (model.skops + config.json).
O utilizador preenche as features com mais peso; as restantes ficam com a
mediana/moda do treino. A saída é texto: em que percentil de risco a alta
está face às previsões OOF do treino e a ação para cada capacidade. Não mostra
probabilidades, gauges nem semáforos.
Correr a partir da raiz do repositório:
    python -m src.app
"""

import numpy as np
import pandas as pd

from src.models.train import MODEL_DIR, load_model

# Campos mostrados na app; os outros vêm de bundle["defaults"].
NUMERIC_INPUTS = ["age_band", "time_in_hospital", "num_medications", "number_diagnoses",
                  "number_inpatient", "number_emergency", "number_outpatient"]
CATEGORICAL_INPUTS = ["discharge_disposition_id_desc", "admission_source_id_desc",
                      "diag_1_group", "A1Cresult", "insulin"]
INPUTS = NUMERIC_INPUTS + CATEGORICAL_INPUTS
CAPACITIES = {10: "threshold_k10", 20: "threshold_k20"}


def make_record(bundle: dict, **values) -> pd.DataFrame:
    """Uma linha com as features do modelo: valores dados + defaults do treino.

    As features derivadas que dependem das visitas são recalculadas aqui,
    para ficarem coerentes com o que o utilizador escreveu.
    """
    row = {**bundle["defaults"], **values}
    row["n_visits_prior"] = row["number_outpatient"] + row["number_emergency"] + row["number_inpatient"]
    return pd.DataFrame([row])[bundle["features"]]


def risk_percentile(bundle: dict, score: float) -> float:
    """Percentil (0–100) do score face às previsões OOF do treino."""
    q = np.asarray(bundle["oof_quantiles"])
    return 100 * np.searchsorted(q, score, side="right") / len(q)


def predict_risk(bundle: dict, **values) -> dict:
    """Percentil de risco, a fração de altas com risco igual ou maior e a ação
    (follow-up ou não) para cada capacidade."""
    score = float(bundle["model"].predict_proba(make_record(bundle, **values))[0, 1])
    pct = risk_percentile(bundle, score)
    return {"percentile": pct, "top_pct": 100 - pct,
            **{f"follow_up_k{k}": score >= bundle[key] for k, key in CAPACITIES.items()}}


def describe(out: dict) -> str:
    top = max(out["top_pct"], 1)
    lines = [f"Esta alta está nos {top:.0f}% com maior risco de readmissão em menos de 30 dias."]
    lines += [f"Com capacidade de {k}%: {'propor follow-up' if out[f'follow_up_k{k}'] else 'sem follow-up'}."
              for k in CAPACITIES]
    return "\n".join(lines)


def build_app(bundle: dict):
    import gradio as gr  # só é preciso para correr a app, não para os testes

    d = bundle["defaults"]
    inputs = [gr.Number(value=d[c], label=c, precision=0) for c in NUMERIC_INPUTS]
    inputs += [gr.Dropdown(bundle["choices"][c], value=d[c], label=c) for c in CATEGORICAL_INPUTS]

    def _predict(*args):
        return describe(predict_risk(bundle, **dict(zip(INPUTS, args))))

    return gr.Interface(fn=_predict, inputs=inputs, outputs=gr.Textbox(label="Resultado"),
                        title="Prioridade de follow-up após a alta (diabetes)",
                        description="Ordena as altas pelo risco de readmissão em menos de 30 dias. "
                                    "Uma pessoa revê cada caso. Não estima o efeito do follow-up.",
                        flagging_mode="never")


if __name__ == "__main__":
    build_app(load_model(MODEL_DIR)).launch()
