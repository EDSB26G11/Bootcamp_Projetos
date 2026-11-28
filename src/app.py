"""
Demo Gradio do modelo final (fase IV, deployment).

Lê a pasta gravada por src.models.train.save_model (model.skops + config.json).
O utilizador preenche as features com mais peso; as restantes ficam com a
mediana/moda do treino. A saída responde às duas partes da pergunta: qual o
risco de readmissão em menos de 30 dias (o score do modelo, que bate certo com
as readmissões observadas nas previsões OOF, por decis) e que prioridade
(posição face às altas elegíveis e recomendação, com o motivo, para a
capacidade da equipa: CAPACITY). Sem gauges nem
semáforos.

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
TEAL_DARK = "#00524D"  # títulos, esta alta (≈ 9:1)
TEAL = "#02ACA0"  # botão, follow-up com 10% (≈ 4,6:1)
TEAL_MID = "#5FB8AF"  # follow-up com 20%
TEAL_LIGHT = "#E6F4F2"  # fundos de cartão
CONTEXT = "#DDE3E5"  # restantes altas
TEXT = "#1F2A30"  # texto normal (≈ 15:1)
MUTED = "#4F5D66"  # rodapés e rótulos secundários (≈ 6,9:1)

# Campos mostrados na app; os outros vêm de bundle["defaults"].
NUMERIC_INPUTS = [
    "age_band",
    "time_in_hospital",
    "num_medications",
    "number_diagnoses",
    "number_inpatient",
    "number_emergency",
    "number_outpatient",
]
CATEGORICAL_INPUTS = [
    "discharge_disposition_id_desc",
    "admission_source_id_desc",
    "diag_1_group",
    "A1Cresult",
    "insulin",
]
INPUTS = NUMERIC_INPUTS + CATEGORICAL_INPUTS
# Capacidade da equipa de follow-up: % das altas que podem receber follow-up,
# começando pelas de maior risco. "Risco alto" = estar dentro desta fração.
CAPACITY = 30

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
    "diag_1_group": {
        "Circulatory": "Circulatório",
        "Diabetes": "Diabetes",
        "Digestive": "Digestivo",
        "Genitourinary": "Geniturinário",
        "Injury": "Lesão",
        "Missing": "Em falta",
        "Musculoskeletal": "Musculoesquelético",
        "Neoplasms": "Neoplasia",
        "Other": "Outro",
        "Respiratory": "Respiratório",
    },
    "A1Cresult": {">7": "> 7%", ">8": "> 8%", "None": "Não medida", "Norm": "Normal"},
    "insulin": {"No": "Não toma", "Steady": "Dose mantida", "Up": "Dose aumentada", "Down": "Dose reduzida"},
    "admission_source_id_desc": {
        "Emergency Room": "Urgência",
        "Physician Referral": "Referência do médico",
        "Clinic Referral": "Referência de clínica",
        "Transfer from a hospital": "Transferência de outro hospital",
    },
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


def capacity_threshold(bundle: dict, k: int = CAPACITY) -> float:
    """Risco mínimo para uma alta ficar nos k% com maior risco.

    Usa threshold_k{k} do config.json, se existir. Se não, calcula-o a partir
    dos 1001 quantis das previsões OOF das altas elegíveis: para k inteiro dá o
    mesmo valor que threshold_for_capacity (ex.: o quantil 0,80 é o threshold_k20).
    """
    key = f"threshold_k{k}"
    if key in bundle:
        return float(bundle[key])
    return float(np.quantile(np.asarray(bundle["oof_quantiles"]), 1 - k / 100))


def predict_risk(bundle: dict, **values) -> dict:
    """Risco estimado (score do modelo), percentil de risco, se o risco é alto
    (dentro da capacidade) e a recomendação. Uma alta fora da lista (D21, destino
    com cuidados médicos) tem risco e posição, mas nunca follow-up."""
    score = float(bundle["model"].predict_proba(make_record(bundle, **values))[0, 1])
    pct_ = risk_percentile(bundle, score)
    eligible = is_eligible(bundle, **values)
    high = score >= capacity_threshold(bundle)
    return {
        "risk": score,
        "base_rate": bundle.get("base_rate_eligible"),
        "percentile": pct_,
        "top_pct": 100 - pct_,
        "eligible": eligible,
        "high_risk": high,
        "follow_up": eligible and high,
    }


def _shown_percentile(out: dict) -> int:
    """Percentil mostrado ao utilizador, arredondado para baixo: assim bate
    sempre certo com a ação. Ex.: 69,9 (logo abaixo do corte de 30%, sem
    follow-up) aparece como P69 e não como P70, que é onde a lista começa."""
    return int(np.floor(out["percentile"]))


def _top(out: dict) -> float:
    """'Nos X% com maior risco', coerente com o percentil mostrado (X = 100 − P)."""
    return max(100 - _shown_percentile(out), 1)


def _risk_lines(out: dict, bold: bool = False) -> list[str]:
    """O risco em linguagem simples, em frases curtas, uma por linha:

    'Risco de readmissão em menos de 30 dias: 9%, abaixo da média de 9,6%.'
    'Em cada 100 doentes como este, cerca de 9 voltam a ser internados.'
    'Está nos 41% com maior risco: fora dos 30% de risco alto.'

    A 2.ª frase é fiel ao modelo porque o risco está calibrado (D22): nas
    previsões out-of-fold, o previsto bate certo com o observado por decis.
    A média é a das altas que podem receber follow-up (elegíveis, D21).
    """
    if "risk" not in out:
        return []
    risk, base = out["risk"], out.get("base_rate")
    shown = "menos de 1%" if risk < 0.005 else pct(risk, 0)
    value = f"<strong>{shown}</strong>" if bold else shown
    first = f"Risco de readmissão em menos de 30 dias: {value}"
    if base:
        side = "próximo da" if abs(risk - base) < 0.005 else "acima da" if risk > base else "abaixo da"
        first += f", {side} média de {pct(base, 1)}"
    n = round(100 * risk)
    count = (
        "menos de 1 volta a ser internado"
        if n < 1
        else "cerca de 1 volta a ser internado" if n == 1 else f"cerca de {n} voltam a ser internados"
    )
    lines = [first + ".", f"Em cada 100 doentes como este, {count}."]
    if "percentile" in out:
        inside = "dentro" if out.get("high_risk") else "fora"
        lines.append(f"Está nos {pct(_top(out) / 100, 0)} com maior risco: {inside} dos {CAPACITY}% de risco alto.")
    return lines


def recommendation(out: dict) -> tuple[str, str]:
    """Recomendação e motivo, como no esboço da equipa:

    Recomendação: Follow-up        Motivo: Risco alto ou Destino sem cuidados médicos
    Recomendação: Sem follow-up    Motivo: Risco baixo ou Destino sem cuidados médicos
    Recomendação: Sem follow-up    Motivo: Risco alto ou Destino com cuidados médicos
    Recomendação: Sem follow-up    Motivo: Risco baixo ou Destino com cuidados médicos

    Risco alto = nos CAPACITY% com maior risco. Destino com cuidados médicos =
    destino da alta que já tem acompanhamento (fora da lista, D21).
    """
    rec = "Follow-up" if out["follow_up"] else "Sem follow-up"
    risk = "Risco alto" if out["high_risk"] else "Risco baixo"
    dest = "Destino sem cuidados médicos" if out.get("eligible", True) else "Destino com cuidados médicos"
    return rec, f"{risk} ou {dest}"


def describe(out: dict) -> str:
    """A mesma informação da app, em texto simples."""
    rec, why = recommendation(out)
    lines = [f"Recomendação: {rec}.", f"Motivo: {why}."]
    lines += _risk_lines(out)
    return "\n".join(lines)


def message(out: dict) -> str:
    """Topo do resultado (HTML): a recomendação numa caixa e o motivo logo abaixo.

    Follow-up = caixa preenchida a verde-azulado escuro; sem follow-up = caixa
    branca com contorno. Distingue-se de relance, sem semáforos."""
    rec, why = recommendation(out)
    kind = "rec-yes" if out["follow_up"] else "rec-no"
    return (
        f'<div class="rec-box {kind}"><span class="rec-label">Recomendação:</span> '
        f'<span class="rec-value">{rec}</span></div>'
        f'<p class="rec-why"><span class="rec-label">Motivo:</span> {why}</p>'
    )


def _risk_value(out: dict) -> str:
    """Risco desta alta em %, sem casas decimais (o mesmo número dos 100 pontos)."""
    return "<1%" if out["risk"] < 0.005 else pct(out["risk"], 0)


def _per_100(out: dict) -> int:
    return int(round(100 * out["risk"]))


def risk_html(out: dict) -> str:
    """Qual o risco, em dois blocos lado a lado:
    - o número grande (risco desta alta) com a média ao lado, por extenso;
    - 100 pontos, um por doente: os escuros são os que voltam a ser internados.
    As frequências naturais ("9 em cada 100") leem-se mais depressa do que %."""
    if "risk" not in out:
        return ""
    base, n = out.get("base_rate"), _per_100(out)
    side = ""
    if base:
        word = "próximo da" if abs(out["risk"] - base) < 0.005 else "acima da" if out["risk"] > base else "abaixo da"
        side = f'<div class="kpi-sub">{word.capitalize()} média das altas ({pct(base, 1)})</div>'
    dots = "".join(f'<span class="dot{" on" if i < n else ""}"></span>' for i in range(100))
    count = "Menos de 1" if n < 1 else str(n)
    verb = "volta" if n <= 1 else "voltam"
    return (
        '<div class="kpi-row">'
        '<div class="kpi">'
        '<div class="kpi-label">Risco de readmissão em menos de 30 dias</div>'
        f'<div class="kpi-value">{_risk_value(out)}</div>{side}</div>'
        '<div class="kpi kpi-icons">'
        f'<div class="icons" role="img" aria-label="{count} em cada 100 doentes">{dots}</div>'
        f'<div class="kpi-sub"><strong>{count} em cada 100</strong> doentes como este {verb} a ser '
        "internados em menos de 30 dias</div></div>"
        "</div>"
    )


def scale_html(out: dict) -> str:
    """Que prioridade: a barra das altas que podem receber follow-up, do maior
    para o menor risco, com a zona de risco alto (CAPACITY%) e o ponteiro desta
    alta. Escala de 20 em 20%, com o corte de CAPACITY% destacado."""
    top = _top(out)
    x = min(max(top, 0.5), 99.5)  # esquerda = maior risco
    align = "left" if x < 15 else "right" if x > 85 else "center"
    ticks = "".join(
        f'<span class="tick t-{t}" style="left:{t}%">{t}%</span>' for t in range(0, 101, 20) if abs(t - CAPACITY) >= 10
    )
    ticks += f'<span class="tick tick-cut" style="left:{CAPACITY}%">{CAPACITY}%</span>'
    return (
        '<div class="scale">'
        '<div class="scale-title">Posição entre as altas que podem receber follow-up</div>'
        '<div class="scale-flagrow">'
        f'<div class="scale-flag flag-{align}" style="left:{x}%">Esta alta: <strong>top {pct(top / 100, 0)}</strong></div>'
        "</div>"
        '<div class="scale-track">'
        f'<div class="scale-zone" style="width:{CAPACITY}%">risco alto</div>'
        f'<div class="scale-cut" style="left:{CAPACITY}%"></div>'
        f'<div class="scale-pin" style="left:{x}%"></div>'
        "</div>"
        f'<div class="scale-ticks">{ticks}</div>'
        '<div class="scale-ends"><span>← maior risco</span><span>menor risco →</span></div>'
        "</div>"
    )


def result_html(out: dict) -> str:
    """O quadro do resultado, de cima para baixo: recomendação, motivo, risco
    (número + 100 pontos) e posição na lista."""
    return message(out) + risk_html(out) + scale_html(out)


def chart_notes(bundle: dict) -> str:
    """Como ler e pressupostos principais, com os números do modelo gravado."""
    n, n_all = bundle.get("n_eligible"), bundle.get("n_train")
    num = lambda v: f"{v:,}".replace(",", " ")
    elig = f" ({num(n)} das {num(n_all)} altas de treino)" if n and n_all else ""
    base = bundle.get("base_rate_eligible")
    base_txt = f" ({pct(base, 1)})" if base else ""
    cut = capacity_threshold(bundle) if ("oof_quantiles" in bundle or f"threshold_k{CAPACITY}" in bundle) else None
    cuts_txt = f" Risco alto = risco de {pct(cut, 1)} ou mais." if cut is not None else ""
    return f"""**Como ler**

- **Risco.** O número grande é o risco estimado de esta alta voltar a ser internada em menos de 30 dias,
  comparado com a média das altas que podem receber follow-up{base_txt}. Os 100 pontos mostram o mesmo risco
  em doentes: cada ponto escuro é um doente, em 100 com este perfil, que volta a ser internado.
- **Posição.** A barra ordena as altas que podem receber follow-up, do maior risco (0%, à esquerda) para o
  menor (100%, à direita). "Top 15%" quer dizer que só 15% das altas têm risco igual ou maior. A zona
  colorida são os {CAPACITY}% com maior risco (risco alto).

**Pressupostos principais**

- **Capacidade.** A equipa faz follow-up a {CAPACITY}% das altas elegíveis, começando pelas de maior risco. O corte
  vem das previsões out-of-fold do treino.{cuts_txt}
- **Destino sem / com cuidados médicos.** Só as altas para casa sem acompanhamento podem receber
  follow-up{elig}. As restantes já têm acompanhamento à saída (outro hospital ou instituição, apoio domiciliário,
  cuidados paliativos): o modelo estima-lhes o risco, mas a recomendação é sempre sem follow-up (D21).
- **O que conta como readmissão.** Um novo internamento em menos de 30 dias nos hospitais do dataset. Uma
  readmissão noutro hospital não aparece nos dados.
- **O risco vale para estes dados.** Nas previsões out-of-fold, o risco previsto e o observado diferem menos
  de 1 pp em todos os decis. O dataset é dos EUA, de 1999 a 2008; noutro hospital ou noutro período, isto tem
  de ser verificado outra vez.
- **Campos não mostrados.** Ficam com o valor mais comum no treino.
- **Não é causal.** O modelo não estima o efeito do follow-up. Uma pessoa revê cada caso antes do follow-up."""


CSS = f"""
.gradio-container {{max-width: 1200px !important; width: 100% !important; margin: 0 auto !important;
                    padding: 20px 28px !important; color: {TEXT}; font-size: 15px;}}
.gradio-container .prose, .gradio-container p, .gradio-container li {{color: {TEXT};}}
.app-header {{border-left: 6px solid {TEAL}; padding: 2px 0 2px 16px; margin-bottom: 6px;}}
.app-header h1 {{color: {TEAL_DARK}; font-size: 26px; font-weight: 700; margin: 0 0 4px 0;}}
.app-header p {{margin: 2px 0; color: {MUTED}; font-size: 14px;}}
.card {{background: white !important; border: 1px solid {CONTEXT} !important; border-radius: 10px !important;
        padding: 16px 18px !important;}}
.card-result {{border-top: 4px solid {TEAL} !important;}}
.result-col {{position: sticky; top: 12px; align-self: flex-start;}}
.group-head h3 {{color: {TEAL_DARK} !important; font-size: 15px !important; margin: 8px 0 2px 0 !important;
                 text-transform: uppercase; letter-spacing: 0.04em;}}
.live-hint p {{color: {MUTED} !important; font-size: 13px !important; margin: 6px 0 0 0;}}

/* 1. Recomendação e motivo */
.rec-box {{border: 3px solid {TEXT}; border-radius: 8px; padding: 16px 22px; margin: 0 0 10px 0;
           font-size: clamp(20px, 2.4vw, 26px); line-height: 1.25; color: {TEXT}; background: white;}}
.rec-box .rec-label {{font-weight: 600;}}
.rec-box .rec-value {{font-weight: 800;}}
.rec-box.rec-yes {{background: {TEAL_DARK}; border-color: {TEAL_DARK}; color: white;}}
.rec-box.rec-yes .rec-label, .rec-box.rec-yes .rec-value {{color: white;}}
.rec-box.rec-no .rec-value {{color: {TEXT};}}
.rec-why {{font-size: 18px; margin: 0 0 18px 0 !important; color: {TEXT};}}
.rec-why .rec-label {{font-weight: 700;}}

/* 2. Risco: número grande + 100 pontos */
.kpi-row {{display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 18px;}}
.kpi {{flex: 1 1 170px; background: {TEAL_LIGHT}; border-radius: 8px; padding: 14px 16px;}}
.kpi.kpi-icons {{flex: 1.8 1 300px;}}
.kpi-label {{font-size: 14px; font-weight: 600; color: {MUTED};}}
.kpi-value {{font-size: 56px; font-weight: 800; line-height: 1.05; color: {TEAL_DARK}; margin: 6px 0 4px 0;}}
.kpi-sub {{font-size: 15px; color: {TEXT}; line-height: 1.35;}}
.kpi-icons {{display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center;}}
.kpi-icons .kpi-sub {{flex: 1 1 110px;}}
.icons {{display: grid; grid-template-columns: repeat(10, 11px); gap: 4px; flex: none;}}
.dot {{width: 11px; height: 11px; border-radius: 50%; background: white; border: 1px solid #B9C4C7;
       box-sizing: border-box;}}
.dot.on {{background: {TEAL_DARK}; border-color: {TEAL_DARK};}}

/* 3. Posição na lista */
.scale {{position: relative; padding-top: 4px;}}
.scale-title {{font-size: 14px; font-weight: 600; color: {MUTED}; margin-bottom: 6px;}}
.scale-flagrow {{position: relative; height: 22px;}}
.scale-flag {{position: absolute; bottom: 0; font-size: 15px; color: {TEXT}; white-space: nowrap;
              transform: translateX(-50%);}}
.scale-flag.flag-left {{transform: translateX(-8px);}}
.scale-flag.flag-right {{transform: translateX(calc(-100% + 8px));}}
.scale-track {{position: relative; height: 26px; background: {CONTEXT}; border-radius: 4px; margin-top: 16px;}}
.scale-zone {{position: absolute; left: 0; top: 0; bottom: 0; background: {TEAL_MID}; border-radius: 4px 0 0 4px;
              font-size: 13px; color: {TEXT}; display: flex; align-items: center; justify-content: center;}}
.scale-cut {{position: absolute; top: -4px; bottom: -4px; width: 3px; margin-left: -1.5px; background: {TEAL_DARK};}}
.scale-pin {{position: absolute; top: -15px; width: 0; height: 0; margin-left: -8px;
             border-left: 8px solid transparent; border-right: 8px solid transparent; border-top: 12px solid {TEXT};}}
.scale-ticks {{position: relative; height: 20px; margin-top: 4px; font-size: 12px; color: {MUTED};}}
.tick {{position: absolute; transform: translateX(-50%);}}
.tick:first-child {{transform: none;}}
.tick[style="left:100%"] {{transform: translateX(-100%);}}
.tick-cut {{font-weight: 700; font-size: 13px; color: {TEAL_DARK};}}
.scale-ends {{display: flex; justify-content: space-between; font-size: 12px; color: {MUTED}; margin-top: 2px;}}
@media (max-width: 560px) {{ .tick.t-20, .tick.t-40, .tick.t-80 {{display: none;}} .kpi-value {{font-size: 46px;}} }}

.chart-notes p, .chart-notes li {{font-size: 14px !important; line-height: 1.45; color: {TEXT} !important;}}
.chart-notes strong {{color: {TEAL_DARK};}}
.chart-notes ul {{margin: 2px 0 10px 0;}}
.footnote p {{color: {MUTED} !important; font-size: 13px !important;}}
[data-testid='block-info'], label span {{color: {TEXT} !important; font-weight: 600 !important;}}
"""


# A paleta foi pensada para fundo claro: abre sempre em modo claro, mesmo com o sistema em modo escuro.
FORCE_LIGHT = """() => {
  const u = new URL(window.location.href);
  if (u.searchParams.get('__theme') !== 'light') { u.searchParams.set('__theme', 'light'); window.location.replace(u.href); }
}"""


def _theme(gr):
    teal = gr.themes.Color(
        c50="#E6F4F2",
        c100="#C7E7E3",
        c200="#9DD5CE",
        c300="#5FB8AF",
        c400="#2E9E94",
        c500="#00857C",
        c600="#00756D",
        c700="#00625C",
        c800="#00524D",
        c900="#003F3B",
        c950="#002B28",
    )
    return gr.themes.Base(primary_hue=teal, neutral_hue="slate", radius_size="md").set(
        body_background_fill="#F4F7F7",
        body_text_color=TEXT,
        block_background_fill="white",
        button_primary_background_fill=TEAL,
        button_primary_background_fill_hover=TEAL_DARK,
        button_primary_text_color="white",
        block_label_text_color=TEXT,
        input_border_color_focus=TEAL,
    )


def _dropdown(gr, bundle, col):
    names = CHOICE_LABELS.get(col, {})
    choices = sorted(((names.get(v, v), v) for v in bundle["choices"][col]), key=lambda t: t[1] not in names)
    return gr.Dropdown(choices, value=bundle["defaults"][col], label=LABELS[col])


def build_app(bundle: dict):
    import gradio as gr  # só é preciso para correr a app, não para os testes

    d = bundle["defaults"]

    with gr.Blocks(title="Prioridade de follow-up", fill_width=False) as demo:
        gr.HTML(
            '<div class="app-header"><h1>Prioridade de follow-up após a alta</h1>'
            "<p>Doentes com diabetes · risco de readmissão em menos de 30 dias · "
            "modelo v1 (dados de 1999–2008)</p></div>"
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=5, elem_classes="card"):
                gr.Markdown("### Doente e internamento", elem_classes="group-head")
                with gr.Row():
                    age = gr.Dropdown(AGE_CHOICES, value=int(d["age_band"]), label=LABELS["age_band"])
                    days = gr.Number(
                        value=d["time_in_hospital"],
                        label=LABELS["time_in_hospital"],
                        precision=0,
                        minimum=1,
                        maximum=14,
                    )
                with gr.Row():
                    meds = gr.Number(
                        value=d["num_medications"], label=LABELS["num_medications"], precision=0, minimum=0
                    )
                    diags = gr.Number(
                        value=d["number_diagnoses"],
                        label=LABELS["number_diagnoses"],
                        precision=0,
                        minimum=1,
                        maximum=16,
                    )
                diag1 = _dropdown(gr, bundle, "diag_1_group")
                source = _dropdown(gr, bundle, "admission_source_id_desc")
                dest = _dropdown(gr, bundle, "discharge_disposition_id_desc")

                gr.Markdown("### No ano anterior", elem_classes="group-head")
                with gr.Row():
                    inp = gr.Number(
                        value=d["number_inpatient"],
                        label=LABELS["number_inpatient"],
                        precision=0,
                        min_width=110,
                        minimum=0,
                    )
                    em = gr.Number(
                        value=d["number_emergency"],
                        label=LABELS["number_emergency"],
                        precision=0,
                        min_width=110,
                        minimum=0,
                    )
                    out_ = gr.Number(
                        value=d["number_outpatient"],
                        label=LABELS["number_outpatient"],
                        precision=0,
                        min_width=110,
                        minimum=0,
                    )

                gr.Markdown("### Diabetes", elem_classes="group-head")
                with gr.Row():
                    a1c = _dropdown(gr, bundle, "A1Cresult")
                    ins = _dropdown(gr, bundle, "insulin")
                gr.Markdown(
                    "O resultado atualiza-se sozinho a cada dado alterado. Os campos não mostrados ficam "
                    "com o valor mais comum.",
                    elem_classes="live-hint",
                )

            # O resultado fica fixo no ecrã enquanto se preenchem os dados (sticky).
            with gr.Column(scale=7, elem_classes="result-col"):
                with gr.Column(elem_classes=["card", "card-result"]):
                    result = gr.HTML()
                with gr.Accordion("Como ler e pressupostos", open=False):
                    gr.Markdown(chart_notes(bundle), elem_classes="chart-notes")
                gr.Markdown(
                    "Apoio à decisão: uma pessoa revê cada caso. Não é causal nem substitui o juízo "
                    "clínico. Dados: UCI Diabetes 130-US hospitals (Strack et al., 2014).",
                    elem_classes="footnote",
                )

        fields = {
            "age_band": age,
            "time_in_hospital": days,
            "num_medications": meds,
            "number_diagnoses": diags,
            "number_inpatient": inp,
            "number_emergency": em,
            "number_outpatient": out_,
            "discharge_disposition_id_desc": dest,
            "admission_source_id_desc": source,
            "diag_1_group": diag1,
            "A1Cresult": a1c,
            "insulin": ins,
        }
        components = [fields[c] for c in INPUTS]

        def _predict(*args):
            # Um campo apagado a meio da escrita chega como None: usa-se o valor por defeito.
            values = {c: (d[c] if v is None else v) for c, v in zip(INPUTS, args)}
            return result_html(predict_risk(bundle, **values))

        # Atualiza em direto: ao abrir e sempre que um dado muda (sem botão).
        for comp in components:
            comp.change(_predict, components, result, show_progress="hidden")
        demo.load(_predict, components, result)
    return demo


if __name__ == "__main__":
    import gradio as gr

    build_app(load_model(MODEL_DIR)).launch(css=CSS, theme=_theme(gr), js=FORCE_LIGHT)
