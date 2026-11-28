"""
Gráficos da apresentação final (deck = relatório do projeto), no estilo do
modelo NOVA IMS usado no deck e com as regras IBCS da W8.

Cada gráfico sai do código e não de números escritos à mão: lê os dados de
treino, as previsões out-of-fold do v1 (reports/oof_v1.csv), os relatórios em
reports/ e o modelo gravado (models/v1). O teste (D13) nunca é lido.

Estilo (igual ao deck):
- preto = valor medido (AC); cinzento claro = contexto (ao acaso, restantes);
- verde = destaque (o modelo escolhido, o que fica na lista);
- verde/vermelho só nas variações (Δ) melhor/pior; azul = variação neutra;
- tracejado = previsto (FC); rótulos diretos, sem eixo de valores nem grelha;
- título, "quem / o quê / quando" e fonte ficam no diapositivo, não na imagem.

Correr a partir da raiz do repositório (escreve em reports/figures/deck/):
    python -m src.deck_figures
"""

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.population import follow_up_eligible  # noqa: E402
from src.features.build import model_frame  # noqa: E402
from src.models.evaluate import calibration_table, metrics_by_group  # noqa: E402
from src.models.threshold import rank_baseline_at_k  # noqa: E402
from src.models.train import MODEL_DIR, load_model  # noqa: E402
from src.pipeline import PROCESSED, SPLITS, read_processed  # noqa: E402
from src.viz import BAND, BLUE, CONTEXT, FONT, GREEN, GREEN_TEXT, INK, MID, RED, num, pct, report_page  # noqa: E402

OUT = Path("reports/figures/deck")
NUMBERS = Path("reports/deck_numbers.json")
OOF = Path("reports/oof_v1.csv")

# Paleta do modelo NOVA IMS do deck.
# Paleta e letra: as de src/viz.py (um só estilo para relatório, EDA e deck).
CAPACITY = 0.20      # D24
AGE_BANDS = ["Menos de 40 anos", "40–59 anos", "60–79 anos", "80 ou mais anos"]

# Destino da alta: nomes curtos em português (como no deck) e se fica na lista (D21).
DEST = {
    "Discharged to home": "Casa (sem apoio)",
    "Discharged/transferred to SNF": "Lar com enfermagem (SNF)",
    "Discharged/transferred to another rehab fac including rehab units of a hospital .": "Reabilitação",
    "Discharged/transferred to home with home health service": "Apoio domiciliário",
    "Discharged/transferred to another short term hospital": "Outro hospital",
    "Discharged/transferred to another type of inpatient care institution": "Outra instituição",
}
FEATURE_LABELS = {
    "number_inpatient": "Internamentos no ano anterior",
    "discharge_disposition_id_desc": "Destino da alta",
    "payer_code": "Pagador",
    "medical_specialty": "Especialidade",
    "n_visits_prior": "Visitas no ano anterior",
    "diag_1_group": "Diagnóstico principal",
    "number_emergency": "Idas à urgência no ano anterior",
    "number_diagnoses": "Diagnósticos registados",
    "age_band": "Idade",
}
MODEL_ROWS = [  # (modelo em experiments.csv, rótulo)
    ("random", "Ao acaso"),
    ("rank_inpatient", "Regra: internamentos anteriores"),
    ("logreg", "Regressão logística"),
    ("random_forest", "Random forest"),
    ("kumo_tabular_small ctx=55k", "Kumo Tabular (Hugging Face)"),
    ("hist_gradient_boosting", "HistGradientBoosting (v1)"),
]


def set_deck_style():
    """Arial (corpo do deck), sem grelha nem molduras, letra para ler num ecrã."""
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": FONT, "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.bottom": False,
        "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "savefig.dpi": 220, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
        "hatch.linewidth": 0.8,
    })


def load_train() -> pd.DataFrame:
    """Só as altas de treino (D13), pela mesma ordem de load_model_data(), com a
    previsão out-of-fold do v1 e a marca de elegível (D21). O teste não é lido."""
    df = read_processed(PROCESSED).merge(
        pd.read_csv(SPLITS)[["encounter_id", "split", "fold"]], on="encounter_id", validate="1:1")
    train = df[df["split"].eq("train")].reset_index(drop=True)
    oof = pd.read_csv(OOF)["oof_score"]
    assert len(oof) == len(train), "oof_v1.csv não tem uma linha por alta de treino"
    train["oof"] = oof.to_numpy()
    train["eligible"] = follow_up_eligible(train).to_numpy()
    return train


def _hbars(ax, labels, values, colors, texts, xmax=None, height=0.62, edge=None, hatch=None):
    """Barras horizontais com o valor escrito no fim (sem eixo de valores)."""
    y = np.arange(len(values))
    bars = ax.barh(y, values, color=colors, height=height, edgecolor=edge, hatch=hatch, linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_visible(False)
    xmax = xmax or max(values) * 1.2
    ax.set_xlim(0, xmax)
    for b, t in zip(bars, texts):
        ax.annotate(t, (b.get_width(), b.get_y() + b.get_height() / 2), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=FONT, fontweight="bold")
    return bars


# --- Problema: para onde iria a lista sem a regra D21 -------------------------------------------

def fig_destinos(tr: pd.DataFrame, k: float = CAPACITY):
    """Destino das altas no top k% de todas as altas de treino, antes de D21."""
    top = tr["oof"] >= np.quantile(tr["oof"], 1 - k)
    d = tr.loc[top]
    name = d["discharge_disposition_id_desc"].map(DEST)
    name = name.where(name.notna(), np.where(d["eligible"], "Outros, ficam na lista", "Outros, saem da lista"))
    share = name.value_counts(normalize=True)
    share = pd.concat([share.drop(["Outros, ficam na lista", "Outros, saem da lista"]).sort_values(ascending=False),
                       share[["Outros, ficam na lista", "Outros, saem da lista"]]])
    out_list = 1 - d["eligible"].mean()
    rest = float(share[["Outros, ficam na lista", "Outros, saem da lista"]].sum())
    stays = {"Casa (sem apoio)", "Outros, ficam na lista"}
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    _hbars(ax, list(share.index), list(share.values), [GREEN if s in stays else INK for s in share.index],
           [pct(v, 0) for v in share.values])
    return fig, {"k": k, "n_top": int(top.sum()), "pct_fora_da_lista": out_list,
                 "pct_casa": float(share["Casa (sem apoio)"]), "pct_outros": rest,
                 "partes": {k_: float(v) for k_, v in share.items()},
                 "n_train": len(tr), "n_eligible": int(tr["eligible"].sum()),
                 "pct_eligible": float(tr["eligible"].mean()),
                 "base_rate_eligible": float(tr.loc[tr["eligible"], "target"].mean())}


# --- Prova: dados (fuga por contagem de encontros) ------------------------------------------------

def fig_fuga(tr: pd.DataFrame):
    """Taxa de readmissão <30 dias pelo nº de encontros do doente no dataset:
    a contagem inclui o próprio regresso, por isso não se sabe na alta (D12)."""
    n_enc = tr.groupby("patient_nbr")["encounter_id"].transform("size")
    grp = pd.cut(n_enc, [0, 1, 2, np.inf], labels=["1 encontro", "2 encontros", "3 ou mais"])
    rate = tr.groupby(grp, observed=True)["target"].mean()
    share = grp.value_counts(normalize=True).reindex(rate.index)
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    _hbars(ax, list(rate.index), list(rate.values), [CONTEXT, CONTEXT, INK],
           [pct(v, 1) for v in rate.values], xmax=rate.max() * 1.3)
    return fig, {"taxa_1": float(rate.iloc[0]), "taxa_2": float(rate.iloc[1]), "taxa_3mais": float(rate.iloc[2]),
                 "pct_altas_1": float(share.iloc[0]), "pct_altas_3mais": float(share.iloc[2])}


# --- Prova: modelo contra a regra à mesma capacidade -----------------------------------------------

def capacity_numbers(tr: pd.DataFrame, k: float) -> dict:
    """Readmissões encontradas a k% das altas elegíveis: modelo, regra e acaso."""
    e = tr[tr["eligible"]]
    y, s = e["target"].to_numpy(), e["oof"].to_numpy()
    flag = s >= np.quantile(s, 1 - k)
    tp, pos = int(y[flag].sum()), int(y.sum())
    rule = rank_baseline_at_k(y, e["number_inpatient"].to_numpy(), k)
    rule_found = rule["recall_mean"] * pos
    return {"k": k, "n": len(e), "contactos": int(flag.sum()), "positivos": pos, "modelo": tp,
            "regra": float(rule_found), "acaso": k * pos, "recall_modelo": tp / pos,
            "recall_regra": float(rule["recall_mean"]), "precisao_modelo": tp / flag.sum(),
            "precisao_regra": float(rule["precision_mean"]), "delta": tp - rule_found,
            "por_1000_modelo": 1000 * tp / len(e), "por_1000_regra": 1000 * rule_found / len(e),
            "por_1000_acaso": 1000 * k * pos / len(e)}


def fig_modelo_vs_regra(c: dict):
    """AC por método e Δ face à regra, na mesma escala (CH 4.1)."""
    labels = ["Modelo v1", "Regra: internamentos anteriores", "Ao acaso"]
    vals = np.array([c["modelo"], c["regra"], c["acaso"]])
    delta = vals - c["regra"]
    lim = vals.max() * 1.22
    d_lo, d_hi = delta.min() * 1.35, max(delta.max(), 0) * 1.6 + 60
    fig, (ax, axd) = plt.subplots(1, 2, figsize=(9.2, 3.4),
                                  gridspec_kw={"width_ratios": [lim, d_hi - d_lo], "wspace": 0.05})
    _hbars(ax, labels, vals, [GREEN, INK, CONTEXT], [num(v) for v in vals], xmax=lim)
    ax.set_title("AC", loc="left", fontsize=FONT - 1, color=MID)
    y = np.arange(3)
    axd.barh(y, delta, color=[GREEN if d > 0 else RED for d in delta], height=0.62)
    axd.invert_yaxis()
    axd.axvline(0, color=INK, lw=1)
    axd.set_xlim(d_lo, d_hi)
    axd.set_yticks([])
    axd.xaxis.set_visible(False)
    axd.spines["left"].set_visible(False)
    for yi, d in zip(y, delta):
        if abs(d) < 0.5:
            continue
        axd.annotate(num(d, sign=True), (d, yi), xytext=(4 if d > 0 else -4, 0), textcoords="offset points",
                     va="center", ha="left" if d > 0 else "right", fontsize=FONT, fontweight="bold")
    axd.set_title("Δ face à regra", loc="left", fontsize=FONT - 1, color=MID)
    return fig


# --- Prova: matriz de confusão ------------------------------------------------------------------

def confusion(y, s, t) -> dict:
    """Matriz com o corte t (previsto positivo = risco ≥ t = follow-up) e as
    métricas da aula 08 (Evaluation Metrics): accuracy, error rate, precisão,
    recall, especificidade, F1 e balanced accuracy."""
    y, flag = np.asarray(y), np.asarray(s) >= t
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    fn, tn = int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum())
    n = tp + fp + fn + tn
    recall, spec = tp / (tp + fn), tn / (fp + tn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "recall": recall, "precisao": tp / (tp + fp),
            "especificidade": spec, "accuracy": (tp + tn) / n, "error_rate": (fp + fn) / n,
            "f1": 2 * tp / (2 * tp + fp + fn), "balanced_accuracy": (recall + spec) / 2,
            "sinalizadas": (tp + fp) / n, "accuracy_sempre_nao": (fp + tn) / n}


# Convenção da aula 08 (slide 53): classe real nas colunas, classe prevista nas
# linhas; TP em cima à esquerda, FP em cima à direita, FN em baixo à esquerda,
# TN em baixo à direita. Positivo = readmissão em menos de 30 dias.
CELLS = {(0, 0): ("tp", "TP", GREEN, INK, "readmissões encontradas"),
         (0, 1): ("fp", "FP", CONTEXT, INK, "follow-ups sem readmissão"),
         (1, 0): ("fn", "FN", INK, "white", "readmissões que escapam"),
         (1, 1): ("tn", "TN", "white", INK, "bem deixadas de fora")}


def _matrix(ax, m: dict, fs: float, show_labels: bool = True):
    """Desenha a matriz 2×2 em ax, com o cabeçalho da aula (Classe real / Prevista)."""
    for (r, c), (key, code, face, txt, lab) in CELLS.items():
        ax.add_patch(plt.Rectangle((c, 1 - r), 1, 1, facecolor=face, edgecolor=MID, lw=0.8))
        ax.text(c + 0.06, 2 - r - 0.1, code, ha="left", va="top", fontsize=fs - 1, fontweight="bold", color=txt)
        ax.text(c + 0.5, 1 - r + (0.55 if show_labels else 0.5), num(m[key]), ha="center", va="center",
                fontsize=fs + 6, fontweight="bold", color=txt)
        if show_labels:
            ax.text(c + 0.5, 1 - r + 0.24, lab, ha="center", va="center", fontsize=fs - 2, color=txt)
    ax.set_xlim(-0.62, 2)
    ax.set_ylim(0, 2.55)
    ax.axis("off")
    # Cabeçalhos: classe real (colunas) e classe prevista (linhas), como na aula.
    ax.text(1, 2.47, "Classe real", ha="center", va="center", fontsize=fs - 1, fontweight="bold")
    ax.text(0.5, 2.17, "Volta < 30 dias (+)", ha="center", va="center", fontsize=fs - 1)
    ax.text(1.5, 2.17, "Não volta (−)", ha="center", va="center", fontsize=fs - 1)
    ax.text(-0.56, 1.0, "Prevista", ha="center", va="center", fontsize=fs - 1, fontweight="bold", rotation=90)
    ax.text(-0.06, 1.5, "Follow-up\n(+)", ha="right", va="center", fontsize=fs - 1)
    ax.text(-0.06, 0.5, "Sem follow-up\n(−)", ha="right", va="center", fontsize=fs - 1)


def fig_confusao(m: dict):
    """Matriz do diapositivo principal (validação out-of-fold, capacidade de 20%)."""
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    _matrix(ax, m, FONT)
    return fig


def fig_confusao_anexo(m: dict, title: str):
    """Matriz pequena para o anexo, com o título e as métricas da aula por baixo."""
    fs = FONT - 3
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    _matrix(ax, m, fs, show_labels=False)
    ax.set_title(title, loc="left", fontsize=fs + 1, fontweight="bold", pad=2)
    fig.text(0.02, -0.02, f"Precisão {pct(m['precisao'])} · recall {pct(m['recall'])} · especificidade "
             f"{pct(m['especificidade'])} · F1 {num(m['f1'], 2)} · accuracy {pct(m['accuracy'])} · "
             f"balanced accuracy {pct(m['balanced_accuracy'])}", fontsize=fs - 1, color=INK, va="top")
    return fig


# --- Prova: risco calibrado ---------------------------------------------------------------------

def fig_calibracao(cal: pd.DataFrame):
    """Por decil de risco: observado (AC, sólido), previsto (FC, tracejado) e ΔFC em pp."""
    labels = [f"Decil {i}" for i in cal.index]
    obs, prev = 100 * cal["observado"].to_numpy(), 100 * cal["risco_previsto"].to_numpy()
    diff = obs - prev  # ΔFC = AC − FC (IBCS)
    fig, (ax, axd) = plt.subplots(1, 2, figsize=(9.2, 4.4), gridspec_kw={"width_ratios": [4, 1.4], "wspace": 0.08})
    y = np.arange(len(cal))
    ax.barh(y - 0.19, prev, height=0.36, color="white", edgecolor=INK, hatch="////", linewidth=0.8, label="FC")
    bars = ax.barh(y + 0.19, obs, height=0.36, color=INK, label="AC")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, max(obs.max(), prev.max()) * 1.2)
    for b, v, w in zip(bars, obs, np.maximum(obs, prev)):
        ax.annotate(num(v, 1), (w, b.get_y() + b.get_height() / 2), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=FONT - 1, fontweight="bold")
    ax.set_title("FC (tracejado) · AC (sólido), em %", loc="left", fontsize=FONT - 1, color=MID)
    lim = max(abs(diff).max() * 1.9, 1.0)
    axd.barh(y, diff, color=BLUE, height=0.55)
    axd.invert_yaxis()
    axd.axvline(0, color=INK, lw=1)
    axd.set_xlim(-lim, lim)
    axd.set_yticks([])
    axd.xaxis.set_visible(False)
    axd.spines["left"].set_visible(False)
    for yi, d in zip(y, diff):
        axd.annotate(num(d, 1, sign=True) if abs(d) >= 0.05 else "±0,0", (d, yi), xytext=(3 if d >= 0 else -3, 0),
                     textcoords="offset points", va="center", ha="left" if d >= 0 else "right", fontsize=FONT - 2)
    axd.set_title("ΔFC (pp)", loc="left", fontsize=FONT - 1, color=MID)
    return fig, {"max_diff_pp": float(abs(diff).max()), "decil_max": int(cal.index[np.argmax(abs(diff))]),
                 "decil10_obs": float(obs[-1]), "decil1_obs": float(obs[0])}


# --- Prova: equidade ----------------------------------------------------------------------------

def fairness_table(tr: pd.DataFrame, k: float = CAPACITY) -> pd.DataFrame:
    """Recall e altas sinalizadas por sexo, raça e idade, com o top k% das
    elegíveis calculado uma só vez para todas (a equipa é só uma)."""
    e = tr[tr["eligible"]].reset_index(drop=True)
    sex = e["gender"].map({"Female": "Mulheres", "Male": "Homens"})
    race = e["race"].fillna("Raça em falta").replace({"?": "Raça em falta"})
    lo = e["age"].str.extract(r"\[(\d+)", expand=False).astype(int)
    age = pd.cut(lo, [-1, 39, 59, 79, 200], labels=AGE_BANDS).astype(str)
    rows = []
    for dim, g in [("Sexo", sex), ("Raça", race), ("Idade", age)]:
        t = metrics_by_group(e["target"], e["oof"], g, k=k)
        t["dimensao"] = dim
        rows.append(t)
    out = pd.concat(rows)
    total = metrics_by_group(e["target"], e["oof"], pd.Series("Total", index=e.index), k=k)
    return out, float(total["recall"].iloc[0])


def fig_equidade(tab: pd.DataFrame, total: float):
    """Recall por grupo (AC) e Δ face ao recall no total, em pp (vermelho =
    o grupo fica para trás). Só grupos com ≥ 100 altas elegíveis."""
    keep = ["Mulheres", "Homens", "Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other",
            "Raça em falta", *AGE_BANDS]
    t = tab.loc[[g for g in keep if g in tab.index]]
    labels = [f"{g} ({num(n)})" for g, n in zip(t.index, t["n"])]
    rec = t["recall"].to_numpy()
    diff = 100 * (rec - total)
    fig, (ax, axd) = plt.subplots(1, 2, figsize=(9.2, 4.9), gridspec_kw={"width_ratios": [2.3, 1.5], "wspace": 0.06})
    _hbars(ax, labels, rec, [INK] * len(rec), [pct(v, 0) for v in rec], xmax=rec.max() * 1.25, height=0.58)
    ax.set_title(f"Recall, em % (total {pct(total, 0)})", loc="left", fontsize=FONT - 1, color=MID)
    y = np.arange(len(rec))
    axd.barh(y, diff, color=[GREEN if d > 0 else RED for d in diff], height=0.58)
    axd.invert_yaxis()
    axd.axvline(0, color=INK, lw=1)
    lim = abs(diff).max() * 1.45
    axd.set_xlim(-lim, lim)
    axd.set_yticks([])
    axd.xaxis.set_visible(False)
    axd.spines["left"].set_visible(False)
    for yi, d in zip(y, diff):
        axd.annotate(num(d, 0, sign=True) if round(d) != 0 else "±0", (d, yi), xytext=(3 if d >= 0 else -3, 0), textcoords="offset points",
                     va="center", ha="left" if d >= 0 else "right", fontsize=FONT - 1)
    axd.set_title("Δ face ao total (pp)", loc="left", fontsize=FONT - 1, color=MID)
    for sep in (1.5, 7.5):  # separa sexo, raça e idade
        for a in (ax, axd):
            a.axhline(sep, color=CONTEXT, lw=1)
    return fig


# --- Anexos: modelos e importância --------------------------------------------------------------

def fig_modelos(exp: pd.DataFrame):
    """PR-AUC na validação (média dos 5 folds) por modelo; verde = v1."""
    v = exp[exp["dataset"].eq("validation")].drop_duplicates("model", keep="last").set_index("model")
    vals = [float(v.loc[m, "pr_auc"]) for m, _ in MODEL_ROWS]
    labels = [lab for _, lab in MODEL_ROWS]
    colors = [CONTEXT] + [INK] * (len(vals) - 2) + [GREEN]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    _hbars(ax, labels, vals, colors, [num(x, 3) for x in vals], xmax=max(vals) * 1.2)
    return fig, dict(zip(labels, vals))


def fig_importancia(imp: pd.DataFrame, n: int = 6):
    """Perda de PR-AUC quando a variável é baralhada; verde = as duas que pesam mais."""
    t = imp.head(n)
    labels = [FEATURE_LABELS.get(i, i) for i in t.index]
    vals = t["importance"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    _hbars(ax, labels, vals, [GREEN, GREEN] + [INK] * (n - 2), [num(x, 3) for x in vals], xmax=vals.max() * 1.2)
    return fig


# --- Curva de ganhos (relatório e deck) ------------------------------------------------------------

def gains_table(y, scores, number_inpatient) -> pd.DataFrame:
    """% acumulada de readmissões encontradas por % de altas com follow-up (0 a 100),
    para o modelo, a regra (internamentos anteriores, 20 sorteios nos empates) e o acaso."""
    from src.models.threshold import gains_curve

    m = gains_curve(y, scores)
    # Regra com o mesmo método e sorteios de capacity_numbers(), para os números baterem com o deck.
    r = [100 * rank_baseline_at_k(y, number_inpatient, k)["recall_mean"] for k in m["k"]]
    out = pd.DataFrame({"k": np.r_[0, 100 * m["k"].to_numpy()], "modelo": np.r_[0, m["pct_readm_apanhadas"]],
                        "regra": np.r_[0, r]})
    out["acaso"] = out["k"]
    return out.round({"k": 0}).set_index("k")


def _gain_until(g: pd.DataFrame, step: int = 10) -> int:
    """Último k (múltiplo de step) até onde cada escalão de step% de follow-ups a mais
    encontra mais readmissões do que ao acaso (declive da curva do modelo > 1)."""
    last = 0
    for k in range(0, 100, step):
        if g.loc[k + step, "modelo"] - g.loc[k, "modelo"] > step:
            last = k + step
        else:
            break
    return last


def fig_ganhos(g: pd.DataFrame, k_cap: int = 20, page: bool = False, n: int | None = None,
               pos: int | None = None, figsize=None):
    """Curva de ganhos acumulados (IBCS): modelo a verde, regra a preto, acaso a
    tracejado; faixa cinzenta onde cada follow-up extra rende mais do que ao acaso;
    linha vertical na capacidade (D24) com os três valores escritos.
    page=True acrescenta a mensagem, o título em três linhas e o rodapé (relatório)."""
    cut = _gain_until(g)
    fig, ax = plt.subplots(figsize=figsize or ((10, 6.4) if page else (9.2, 4.6)))
    ax.axvspan(0, cut, color=BAND, zorder=0)
    ax.text(cut / 2, 97, "Cada follow-up extra rende\nmais do que ao acaso", ha="center", va="top",
            fontsize=FONT - 2, fontweight="bold", color=MID)
    ax.plot(g.index, g["acaso"], color="#9A9A9A", ls="--", lw=1.6, zorder=2)
    ax.plot(g.index, g["regra"], color=INK, lw=2, zorder=3)
    ax.plot(g.index, g["modelo"], color=GREEN, lw=3.2, zorder=4)
    # Rótulos diretos junto às linhas, onde estão afastadas (no fim as três juntam-se em 100%).
    for col, label, color, xk, dy, va in [("modelo", "Modelo v1", GREEN_TEXT, 55, 10, "bottom"),
                                          ("regra", "Regra simples", INK, 70, -10, "top"),
                                          ("acaso", "Ao acaso", "#7A7A7A", 85, -10, "top")]:
        ax.annotate(label, (xk, g.loc[xk, col]), xytext=(0, dy), textcoords="offset points", ha="center", va=va,
                    fontsize=FONT - 1, color=color, fontweight="bold" if col == "modelo" else None)
    # Capacidade escolhida (D24): os três valores escritos junto à linha.
    ax.axvline(k_cap, color=MID, lw=1, ls=":", zorder=1)
    for col, color, dx, ha, fw in [("modelo", GREEN_TEXT, -6, "right", "bold"), ("regra", INK, 6, "left", None),
                                    ("acaso", "#7A7A7A", 6, "left", None)]:
        yv = g.loc[k_cap, col]
        ax.plot(k_cap, yv, "o", ms=6, color=GREEN if col == "modelo" else color, zorder=5)
        ax.annotate(f"{num(yv, 0)}%", (k_cap, yv), xytext=(dx, 7 if col == "modelo" else -9),
                    textcoords="offset points", ha=ha, va="center", fontsize=FONT - 1, color=color, fontweight=fw)
    ax.annotate(f"Capacidade: {k_cap}% das altas (D24)", (k_cap, 2), xytext=(4, 0), textcoords="offset points",
                fontsize=FONT - 3, color=MID, va="bottom")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(0, 101, 10), [f"{x}%" for x in range(0, 101, 10)])
    ax.set_yticks([0, 25, 50, 75, 100], [f"{y}%" for y in (0, 25, 50, 75, 100)])
    ax.tick_params(length=0, labelsize=FONT - 2, colors=MID)
    for s in ("left",):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(INK)
    ax.set_xlabel("Altas com follow-up, em % das altas elegíveis (por ordem de risco)", fontsize=FONT - 1)
    if page:
        m, r = g.loc[k_cap, "modelo"], g.loc[k_cap, "regra"]
        report_page(
            fig, f"Com follow-up a {k_cap}% das altas, o modelo encontra {num(m, 0)}% das readmissões (regra "
            f"simples: {num(r, 0)}%; acaso: {k_cap}%); depois dos {cut}%, cada follow-up extra rende menos do que "
            "ao acaso",
            f"Rede hospitalar, altas de treino elegíveis para follow-up (n = {num(n)}; {num(pos)} readmissões; D21)",
            "Readmissões < 30 dias encontradas", "em % acumulada, por % de altas com follow-up",
            "Validação cruzada, 5 folds por doente (dados de 1999–2008)",
            "Fonte: Diabetes 130-US hospitals (UCI), 1999–2008. Faixa cinzenta: cada 10% de altas com follow-up a "
            "mais encontra mais de 10% das readmissões (mais do que ao acaso). Regra simples: ordenar pelos "
            "internamentos no ano anterior, empates ao acaso (média de 20 sorteios). Altas elegíveis: altas para a "
            "comunidade sem outro acompanhamento (D21). Previsões out-of-fold do treino; a confirmar no teste (T3.6).")
        return fig
    else:
        fig.subplots_adjust(right=0.97, top=0.97)
    return fig


def gains_page(y, scores, number_inpatient, k_cap: int = 20):
    """Página de relatório da curva de ganhos (usada no notebook 04 e no Projeto_Final)."""
    set_deck_style()
    g = gains_table(y, scores, number_inpatient)
    y = np.asarray(y)
    return fig_ganhos(g, k_cap, page=True, n=len(y), pos=int(y.sum())), g


# --- Números da demo -----------------------------------------------------------------------------

def demo_cases() -> dict:
    """As altas da demo, calculadas com o modelo gravado e as funções da app."""
    from src.app import capacity_threshold, predict_risk

    bundle = load_model(MODEL_DIR)
    a = dict(age_band=7, time_in_hospital=6, num_medications=20, number_diagnoses=9, number_inpatient=3,
             number_emergency=1, number_outpatient=0, discharge_disposition_id_desc="Discharged to home")
    cases = {"A_casa": a,
             "A_reabilitacao": {**a, "discharge_disposition_id_desc":
                                "Discharged/transferred to another rehab fac including rehab units of a hospital ."},
             "B_casa": dict(age_band=5, time_in_hospital=3, number_inpatient=0, number_emergency=0,
                            number_outpatient=0, discharge_disposition_id_desc="Discharged to home")}
    out = {}
    for name, v in cases.items():
        r = predict_risk(bundle, **v)
        out[name] = {"risco": r["risk"], "top_pct": r["top_pct"], "elegivel": r["eligible"],
                     "follow_up_app": r["follow_up"],
                     "follow_up_20": bool(r["eligible"] and r["risk"] >= capacity_threshold(bundle, 20))}
    return out


def train_confusions(tr: pd.DataFrame) -> dict:
    """Matrizes de treino (modelo final a prever as próprias altas) e de validação
    (OOF), a 10% e 20%, com os cortes gravados no config.json."""
    bundle = load_model(MODEL_DIR)
    e = tr[tr["eligible"]].reset_index(drop=True)
    s_train = bundle["model"].predict_proba(model_frame(e)[bundle["features"]])[:, 1]
    out = {}
    for k in (10, 20):
        t = bundle[f"threshold_k{k}"]
        out[f"validacao_k{k}"] = confusion(e["target"], e["oof"], t)
        out[f"treino_k{k}"] = confusion(e["target"], s_train, t)
    return out


def main():
    set_deck_style()
    OUT.mkdir(parents=True, exist_ok=True)
    tr = load_train()
    numbers = {"populacao": {"n_train": len(tr), "taxa_train": float(tr["target"].mean())}}

    def save(fig, name):
        fig.savefig(OUT / f"{name}.png", transparent=False, facecolor="white")
        plt.close(fig)

    fig, numbers["destinos"] = fig_destinos(tr)
    save(fig, "01_destinos_sem_D21")
    fig, numbers["fuga"] = fig_fuga(tr)
    save(fig, "02_fuga_contagem_encontros")
    for k in (0.10, 0.20, 0.30):
        numbers[f"capacidade_{int(k * 100)}"] = capacity_numbers(tr, k)
    save(fig_modelo_vs_regra(numbers["capacidade_20"]), "03_modelo_vs_regra_k20")
    numbers["confusao"] = train_confusions(tr)
    save(fig_confusao(numbers["confusao"]["validacao_k20"]), "04_matriz_confusao_k20")
    titles = {"validacao_k20": "20% · validação (out-of-fold)", "treino_k20": "20% · treino",
              "validacao_k10": "10% · validação (out-of-fold)", "treino_k10": "10% · treino"}
    for key, ttl in titles.items():
        save(fig_confusao_anexo(numbers["confusao"][key], ttl), f"A_matriz_{key}")
    e = tr[tr["eligible"]]
    cal = calibration_table(e["target"], e["oof"])
    fig, numbers["calibracao"] = fig_calibracao(cal)
    save(fig, "05_calibracao_decis")
    tab, total = fairness_table(tr)
    numbers["equidade"] = {"recall_total": total, **{g: {c: float(tab.loc[g, c]) for c in
                           ("n", "positive_rate", "flagged_rate", "recall")} for g in tab.index}}
    save(fig_equidade(tab, total), "06_equidade_recall_k20")
    fig, numbers["modelos_pr_auc"] = fig_modelos(pd.read_csv("reports/experiments.csv"))
    save(fig, "A_modelos_pr_auc")
    save(fig_importancia(pd.read_csv("reports/permutation_importance_v1.csv", index_col=0)), "A_importancia")
    fig, g = gains_page(e["target"], e["oof"], e["number_inpatient"])
    fig.savefig("reports/figures/gains_curve_v1.png", facecolor="white")
    plt.close(fig)
    save(fig_ganhos(g, figsize=(7.0, 3.9)), "07_curva_ganhos")  # tamanho do anexo A7: letra legível
    numbers["ganhos"] = {"ganho_ate": _gain_until(g), **{f"k{k}": g.loc[k].to_dict() for k in (10, 20, 30)}}
    numbers["demo"] = demo_cases()
    rate = tr.groupby(np.minimum(tr["number_inpatient"], 6))["target"].mean()
    numbers["internamentos"] = {"taxa_0": float(rate.loc[0]), "taxa_6_ou_mais": float(rate.loc[6])}
    from sklearn.metrics import average_precision_score
    numbers["pr_auc_elegiveis"] = {"modelo": float(average_precision_score(e["target"], e["oof"])),
                                   "regra": float(average_precision_score(e["target"], e["number_inpatient"]))}
    NUMBERS.write_text(json.dumps(numbers, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"Gráficos em {OUT}/ e números em {NUMBERS}")


if __name__ == "__main__":
    main()
