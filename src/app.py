"""
Demo Gradio do modelo final (fase IV, deployment).

Lê a pasta gravada por src.models.train.save_model (model.skops + config.json).
O utilizador preenche as features com mais peso; as restantes ficam com a
mediana/moda do treino. A saída diz em que posição de risco a alta está face
às previsões OOF do treino e a ação para cada capacidade. Não mostra
probabilidades, gauges nem semáforos.

Layout segundo a W8 (IBCS), onde se aplica a uma app:
- mensagem primeiro (uma frase que se pode verificar), título em três linhas
  (quem, o quê, quando) e rodapé com a fonte e os limites;
- cinzentos com significado: escuro = esta alta, claro = contexto;
- rótulos diretos em vez de legendas, eixos e grelhas;
- números em português (vírgula decimal).

Correr a partir da raiz do repositório:
    python -m src.app
"""

import numpy as np
import pandas as pd

from src.models.train import MODEL_DIR, load_model
from src.viz import pct

# Paleta da app: verde-azulado (próximo dos tons do Hospital da Luz), com
# significado fixo: escuro = esta alta, médio/claro = quem cabe no follow-up,
# cinzento = restantes altas. Contraste do texto sobre branco ≥ 4,5:1 (WCAG AA).
TEAL_DARK = "#00524D"    # títulos, esta alta (≈ 9:1)
TEAL = "#00857C"         # botão, follow-up com 10% (≈ 4,6:1)
TEAL_MID = "#5FB8AF"     # follow-up com 20%
TEAL_LIGHT = "#E6F4F2"   # fundos de cartão
CONTEXT = "#DDE3E5"      # restantes altas
TEXT = "#1F2A30"         # texto normal (≈ 15:1)
MUTED = "#4F5D66"        # rodapés e rótulos secundários (≈ 6,9:1)

# Campos mostrados na app; os outros vêm de bundle["defaults"].
NUMERIC_INPUTS = ["age_band", "time_in_hospital", "num_medications", "number_diagnoses",
                  "number_inpatient", "number_emergency", "number_outpatient"]
CATEGORICAL_INPUTS = ["discharge_disposition_id_desc", "admission_source_id_desc",
                      "diag_1_group", "A1Cresult", "insulin"]
INPUTS = NUMERIC_INPUTS + CATEGORICAL_INPUTS
CAPACITIES = {10: "threshold_k10", 20: "threshold_k20"}

# Rótulos em português (o valor passado ao modelo não muda).
LABELS = {
    "age_band": "Idade",
    "time_in_hospital": "Dias de internamento",
    "num_medications": "Medicamentos diferentes",
    "number_diagnoses": "Diagnósticos registados",
    "number_inpatient": "Internamentos",
    "number_emergency": "Idas à urgência",
    "number_outpatient": "Consultas",
    "discharge_disposition_id_desc": "Destino após a alta",
    "admission_source_id_desc": "Origem da admissão",
    "diag_1_group": "Diagnóstico principal",
    "A1Cresult": "HbA1c",
    "insulin": "Insulina",
}
CHOICE_LABELS = {
    "diag_1_group": {"Circulatory": "Circulatório", "Diabetes": "Diabetes", "Digestive": "Digestivo",
                     "Genitourinary": "Geniturinário", "Injury": "Lesão", "Missing": "Em falta",
                     "Musculoskeletal": "Musculoesquelético", "Neoplasms": "Neoplasia", "Other": "Outro",
                     "Respiratory": "Respiratório"},
    "A1Cresult": {">7": "> 7%", ">8": "> 8%", "None": "Não medida", "Norm": "Normal"},
    "insulin": {"No": "Não toma", "Steady": "Dose mantida", "Up": "Dose aumentada", "Down": "Dose reduzida"},
    "admission_source_id_desc": {"Emergency Room": "Urgência", "Physician Referral": "Referência do médico",
                                 "Clinic Referral": "Referência de clínica", "Transfer from a hospital":
                                 "Transferência de outro hospital"},
    "discharge_disposition_id_desc": {
        "Discharged to home": "Casa",
        "Discharged/transferred to home with home health service": "Casa, com apoio domiciliário",
        "Discharged/transferred to SNF": "Unidade de cuidados continuados (SNF)",
        "Discharged/transferred to another rehab fac including rehab units of a hospital .": "Reabilitação",
        "Discharged/transferred to another short term hospital": "Outro hospital (curta duração)",
        "Discharged/transferred to a long term care hospital.": "Hospital de longa duração",
        "Hospice / home": "Cuidados paliativos em casa",
        "Hospice / medical facility": "Cuidados paliativos em instituição",
        "Left AMA": "Saiu contra parecer médico",
    },
}
AGE_CHOICES = [(f"{10 * i}–{10 * i + 9} anos", i) for i in range(10)]


def make_record(bundle: dict, **values) -> pd.DataFrame:
    """Uma linha com as features do modelo: valores dados + defaults do treino.

    As features derivadas que dependem das visitas são recalculadas aqui,
    para ficarem coerentes com o que o utilizador escreveu.
    """
    row = {**bundle["defaults"], **values}
    row["n_visits_prior"] = row["number_outpatient"] + row["number_emergency"] + row["number_inpatient"]
    return pd.DataFrame([row])[bundle["features"]]


def risk_percentile(bundle: dict, score: float) -> float:
    """Percentil (0–100) do score face às previsões OOF das altas elegíveis do treino (D21)."""
    q = np.asarray(bundle["oof_quantiles"])
    return 100 * np.searchsorted(q, score, side="right") / len(q)


def is_eligible(bundle: dict, **values) -> bool:
    """A alta pode entrar na lista de follow-up? Não, se o destino da alta já
    tem acompanhamento institucional (D21)."""
    dest = values.get("discharge_disposition_id_desc", bundle["defaults"]["discharge_disposition_id_desc"])
    return dest not in set(bundle.get("no_follow_up_discharge", []))


def predict_risk(bundle: dict, **values) -> dict:
    """Percentil de risco, a fração de altas com risco igual ou maior e a ação
    (follow-up ou não) para cada capacidade. Uma alta fora da lista (D21) tem
    risco e posição, mas nunca follow-up."""
    score = float(bundle["model"].predict_proba(make_record(bundle, **values))[0, 1])
    pct_ = risk_percentile(bundle, score)
    eligible = is_eligible(bundle, **values)
    return {"percentile": pct_, "top_pct": 100 - pct_, "eligible": eligible,
            **{f"follow_up_k{k}": eligible and score >= bundle[key] for k, key in CAPACITIES.items()}}


def _top(out: dict) -> float:
    return max(out["top_pct"], 1)


NOT_ELIGIBLE = "fora da lista (já tem acompanhamento à saída)"


def _action(out: dict, k: int) -> str:
    if not out.get("eligible", True):
        return NOT_ELIGIBLE
    return "propor follow-up" if out[f"follow_up_k{k}"] else "sem follow-up"


def describe(out: dict) -> str:
    lines = [f"Esta alta está nos {pct(_top(out) / 100, 0)} com maior risco de readmissão em menos de 30 dias."]
    if not out.get("eligible", True):
        return lines[0] + "\nFora da lista de follow-up: o destino da alta já tem acompanhamento."
    lines += [f"Com capacidade de {k}%: {_action(out, k)}." for k in CAPACITIES]
    return "\n".join(lines)


def message(out: dict) -> str:
    """Mensagem IBCS: a conclusão primeiro, numa frase."""
    k_in = [k for k in CAPACITIES if out[f"follow_up_k{k}"]]
    top = pct(_top(out) / 100, 0)
    if not out.get("eligible", True):
        return ("### Fora da lista de follow-up: o destino da alta já tem acompanhamento\n"
                f"O doente continua internado, vai para outra instituição ou sai com apoio domiciliário ou "
                f"paliativo. O risco fica registado (nos {top} com maior risco), mas a capacidade da equipa "
                "fica para quem volta para casa sem acompanhamento.")
    if k_in:
        return (f"### Propor follow-up: esta alta está nos {top} com maior risco de readmissão em menos de 30 dias\n"
                f"Entra na lista com capacidade para {min(k_in)}% das altas ou mais.")
    return (f"### Sem follow-up: esta alta não está nas {max(CAPACITIES)}% com maior risco de readmissão "
            f"em menos de 30 dias\nFica nos {top} com maior risco; não entra na lista com capacidade até "
            f"{max(CAPACITIES)}% das altas.")


def action_table(out: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "Capacidade da equipa (contacta as altas de maior risco)": [f"{k}% das altas" for k in CAPACITIES],
        "Ação para esta alta": [_action(out, k).capitalize() for k in CAPACITIES],
    })


def risk_chart(out: dict):
    """Posição da alta na ordenação das altas de treino, com os cortes de capacidade.

    Eixo da esquerda (maior risco) para a direita; a faixa clara são todas as
    altas, a parte mais escura as que cabem na maior capacidade, o marcador
    escuro é esta alta. Rótulos diretos, sem legenda nem grelha.
    """
    from matplotlib.figure import Figure  # sem pyplot: não acumula figuras no servidor

    top = min(max(out["top_pct"], 0.5), 99.5)
    fig = Figure(figsize=(8, 2.3))
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0.015, 0.0, 0.97, 1.0])
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.7, 1.6)
    ax.axis("off")
    # Título do gráfico (o quê e unidade), como nas páginas do relatório.
    ax.text(0, 1.55, "Altas elegíveis para follow-up, ordenadas pelo risco de readmissão, em % das altas", fontsize=10,
            color=MUTED, va="top")
    # Faixa: cinzento = restantes altas; verde = quem entra na lista de follow-up (10% e 20%).
    ax.barh(0, 100, color=CONTEXT, height=0.62)
    ax.barh(0, 20, color=TEAL_MID, height=0.62)
    ax.barh(0, 10, color=TEAL, height=0.62)
    for k in CAPACITIES:
        ax.plot([k, k], [-0.31, 0.31], color="white", lw=2)
    for x, t, ha, c in [(0, "0%", "left", MUTED), (10, "10%", "center", TEAL_DARK),
                        (20, "20%", "center", TEAL_DARK), (100, "100%", "right", MUTED)]:
        ax.text(x, -0.45, t, fontsize=9.5, color=c, ha=ha, va="top")
    ax.text(0, -1.05, "← maior risco", fontsize=9.5, color=MUTED, ha="left", va="top")
    ax.text(100, -1.05, "menor risco →", fontsize=9.5, color=MUTED, ha="right", va="top")
    ax.text(22, -1.05, "a verde: entram na lista de follow-up (até 10% ou até 20% das altas elegíveis)", fontsize=9.5,
            color=TEAL_DARK, ha="left", va="top")
    # Esta alta: marcador escuro e rótulo direto.
    ax.plot(top, 0.55, marker="v", color=TEAL_DARK, markersize=12)
    ax.text(top + (1.2 if top < 60 else -1.2), 0.78, f"esta alta: {pct(_top(out) / 100, 0)} com maior risco",
            fontsize=11, fontweight="bold", color=TEAL_DARK, ha="left" if top < 60 else "right", va="center")
    return fig


CSS = f"""
.gradio-container {{max-width: 1200px !important; width: 100% !important; margin: 0 auto !important;
                    padding: 24px 32px !important; color: {TEXT}; font-size: 15px;}}
.gradio-container .prose, .gradio-container p, .gradio-container li {{color: {TEXT};}}
.app-header {{border-left: 6px solid {TEAL}; padding: 4px 0 4px 16px; margin-bottom: 12px;}}
.app-header h1 {{color: {TEAL_DARK}; font-size: 26px; font-weight: 700; margin: 0 0 6px 0;}}
.app-header p {{margin: 2px 0; color: {MUTED}; font-size: 15px;}}
.app-header strong {{color: {TEXT};}}
.card {{background: white !important; border: 1px solid {CONTEXT} !important; border-radius: 10px !important;
        padding: 16px 18px !important;}}
.card-result {{border-top: 4px solid {TEAL} !important;}}
.group-head h3 {{color: {TEAL_DARK} !important; font-size: 16px !important; margin: 10px 0 2px 0 !important;
                 text-transform: uppercase; letter-spacing: 0.04em;}}
.result-msg h3 {{color: {TEAL_DARK} !important; font-size: 21px !important; line-height: 1.3; margin: 0 0 4px 0;}}
.result-msg p {{font-size: 16px; color: {TEXT} !important;}}
.footnote p {{color: {MUTED} !important; font-size: 13px !important;}}
[data-testid='block-info'], label span {{color: {TEXT} !important; font-weight: 600 !important;}}
"""


# A paleta foi pensada para fundo claro: abre sempre em modo claro, mesmo com o sistema em modo escuro.
FORCE_LIGHT = """() => {
  const u = new URL(window.location.href);
  if (u.searchParams.get('__theme') !== 'light') { u.searchParams.set('__theme', 'light'); window.location.replace(u.href); }
}"""


def _theme(gr):
    teal = gr.themes.Color(c50="#E6F4F2", c100="#C7E7E3", c200="#9DD5CE", c300="#5FB8AF", c400="#2E9E94",
                           c500="#00857C", c600="#00756D", c700="#00625C", c800="#00524D", c900="#003F3B",
                           c950="#002B28")
    return gr.themes.Base(primary_hue=teal, neutral_hue="slate", radius_size="md").set(
        body_background_fill="#F4F7F7", body_text_color=TEXT, block_background_fill="white",
        button_primary_background_fill=TEAL, button_primary_background_fill_hover=TEAL_DARK,
        button_primary_text_color="white", block_label_text_color=TEXT, input_border_color_focus=TEAL)


def _dropdown(gr, bundle, col):
    names = CHOICE_LABELS.get(col, {})
    choices = sorted(((names.get(v, v), v) for v in bundle["choices"][col]), key=lambda t: t[1] not in names)
    return gr.Dropdown(choices, value=bundle["defaults"][col], label=LABELS[col])


def build_app(bundle: dict):
    import gradio as gr  # só é preciso para correr a app, não para os testes

    d = bundle["defaults"]
    n = bundle.get("n_eligible")
    base = (f"{n:,}".replace(",", " ") + " altas de treino elegíveis para follow-up" if n
            else "as altas de treino elegíveis para follow-up")

    with gr.Blocks(title="Prioridade de follow-up", fill_width=False) as demo:
        # Título em três linhas (quem, o quê, quando), como nas páginas do relatório.
        gr.HTML(
            '<div class="app-header"><h1>Prioridade de follow-up após a alta</h1>'
            '<p>Diabetes 130-US hospitals · modelo v1 (HistGradientBoosting)</p>'
            f'<p><strong>Posição no risco de readmissão &lt; 30 dias</strong>, face a {base}, '
            'e ação por capacidade da equipa</p>'
            '<p>Dados de 1999–2008 · risco comparado com as previsões out-of-fold do treino</p></div>')

        with gr.Row(equal_height=False):
            with gr.Column(scale=5, elem_classes="card"):
                gr.Markdown("### Doente e internamento", elem_classes="group-head")
                with gr.Row():
                    age = gr.Dropdown(AGE_CHOICES, value=int(d["age_band"]), label=LABELS["age_band"])
                    days = gr.Number(value=d["time_in_hospital"], label=LABELS["time_in_hospital"],
                                     precision=0, minimum=1, maximum=14)
                with gr.Row():
                    meds = gr.Number(value=d["num_medications"], label=LABELS["num_medications"], precision=0,
                                     minimum=0)
                    diags = gr.Number(value=d["number_diagnoses"], label=LABELS["number_diagnoses"],
                                      precision=0, minimum=1, maximum=16)
                diag1 = _dropdown(gr, bundle, "diag_1_group")
                source = _dropdown(gr, bundle, "admission_source_id_desc")
                dest = _dropdown(gr, bundle, "discharge_disposition_id_desc")

                gr.Markdown("### No ano anterior", elem_classes="group-head")
                with gr.Row():
                    inp = gr.Number(value=d["number_inpatient"], label=LABELS["number_inpatient"], precision=0, min_width=110,
                                    minimum=0)
                    em = gr.Number(value=d["number_emergency"], label=LABELS["number_emergency"], precision=0, min_width=110,
                                   minimum=0)
                    out_ = gr.Number(value=d["number_outpatient"], label=LABELS["number_outpatient"],
                                     precision=0, min_width=110, minimum=0)

                gr.Markdown("### Diabetes", elem_classes="group-head")
                with gr.Row():
                    a1c = _dropdown(gr, bundle, "A1Cresult")
                    ins = _dropdown(gr, bundle, "insulin")
                btn = gr.Button("Avaliar alta", variant="primary", size="lg")

            with gr.Column(scale=7, elem_classes=["card", "card-result"]):
                msg = gr.Markdown(elem_classes="result-msg")
                chart = gr.Plot(show_label=False, container=False)
                table = gr.Dataframe(interactive=False, show_label=False, wrap=True)
                gr.Markdown(
                    "Os campos que não aparecem ficam com o valor mais comum no treino. "
                    "Ficam fora da lista as altas para outro hospital ou instituição, internamento, "
                    "apoio domiciliário ou cuidados paliativos: já têm acompanhamento (D21). "
                    "É uma posição na ordenação, não uma probabilidade. Uma pessoa revê cada caso; "
                    "o modelo não estima o efeito do follow-up nem é causal.",
                    elem_classes="footnote")

        fields = {"age_band": age, "time_in_hospital": days, "num_medications": meds, "number_diagnoses": diags,
                  "number_inpatient": inp, "number_emergency": em, "number_outpatient": out_,
                  "discharge_disposition_id_desc": dest, "admission_source_id_desc": source,
                  "diag_1_group": diag1, "A1Cresult": a1c, "insulin": ins}
        components = [fields[c] for c in INPUTS]

        def _predict(*args):
            res = predict_risk(bundle, **dict(zip(INPUTS, args)))
            return message(res), risk_chart(res), action_table(res)

        btn.click(_predict, components, [msg, chart, table])
        demo.load(_predict, components, [msg, chart, table])
    return demo


if __name__ == "__main__":
    import gradio as gr

    build_app(load_model(MODEL_DIR)).launch(css=CSS, theme=_theme(gr), js=FORCE_LIGHT)
