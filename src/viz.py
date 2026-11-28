"""
Gráficos no padrão IBCS (aula W8, T3.7), com o mesmo estilo em todo o projeto:
relatório (Projeto_Final), EDA (docs/figures) e deck (src/deck_figures.py).

Layout de uma página de relatório (report_page), igual à curva de ganhos:
- em cima, a mensagem: uma frase que se pode verificar, a negrito;
- o título em três linhas: quem (dados), o quê (a medida a negrito e a unidade)
  e quando (período e tipo de dados), a cinzento;
- em baixo, o rodapé: fonte, definições e o que ficou de fora.

Cores com significado fixo (paleta do modelo NOVA IMS usado no deck):
- preto = valor medido (AC); cinzento claro = contexto (ao acaso, restantes);
- verde = destaque (o modelo escolhido, o grupo que interessa);
- verde/vermelho só nas variações (Δ) melhor/pior; azul = variação neutra;
- tracejado = referência (acaso, taxa global) ou previsto (FC).
Rótulos diretos em vez de legendas e de eixos de valores; sem grelha.
Números em português (vírgula decimal): pct, num.
"""

import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Paleta (a mesma do deck).
INK = "#000000"         # valor medido (AC)
MID = "#4D4D4D"         # texto secundário, linhas de referência
CONTEXT = "#E7E7E7"     # contexto (ao acaso, restantes)
GREEN = "#0EE071"       # destaque / variação positiva
GREEN_TEXT = "#0A9F50"  # o mesmo verde, escurecido para texto
RED = "#FF4011"         # variação negativa
BLUE = "#3777FF"        # variação neutra
BAND = "#F2F2F2"        # faixa de fundo (zona a destacar)
FONT = 13

# Nomes antigos, para os notebooks que os importam (01_eda).
AC, GREY, LIGHT, GOOD, BAD, NOTE = INK, MID, CONTEXT, GREEN, RED, MID

# Nomes dos modelos no relatório (UN 1.1: o mesmo nome em todas as páginas).
MODEL_LABELS = {
    "hist_gradient_boosting": "Modelo v1 (HistGradientBoosting)",
    "random_forest": "Random forest",
    "logreg": "Regressão logística",
    "logreg_l1": "Regressão logística L1",
    "logreg_balanced": "Regressão logística com pesos",
    "rank_inpatient": "Regra: internamentos anteriores",
    "rule_inpatient_ge_1": "Regra: ≥ 1 internamento anterior",
    "random": "Ao acaso",
}


def set_style():
    """Estilo base: Arial (corpo do deck), sem grelha nem molduras, letra para ler num ecrã."""
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": FONT, "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": MID, "ytick.color": INK,
        "text.color": INK, "axes.prop_cycle": mpl.cycler(color=[INK, MID, CONTEXT]),
        "axes.titlesize": FONT - 1, "axes.titlelocation": "left", "legend.frameon": False,
        "figure.dpi": 100, "savefig.dpi": 220, "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
        "savefig.facecolor": "white", "hatch.linewidth": 0.8,
    })


def pct(x, decimals=1, sign=False):
    """0.1134 -> '11,3%' (com sign=True: '+2,1%')."""
    s = f"{x * 100:+.{decimals}f}%" if sign else f"{x * 100:.{decimals}f}%"
    return s.replace(".", ",")


def num(x, decimals=0, sign=False):
    """12345 -> '12 345'; 2.5 -> '2,5'."""
    s = f"{x:+,.{decimals}f}" if sign else f"{x:,.{decimals}f}"
    return s.replace(",", " ").replace(".", ",")


def _wrap(text, width_in, fontsize, bold=False):
    """Parte o texto em linhas que cabem em width_in polegadas (estimativa para Arial)."""
    per_char = fontsize / 72 * (0.56 if bold else 0.5)
    width = max(int(width_in / per_char), 20)
    return "\n".join("\n".join(textwrap.wrap(p, width)) for p in text.split("\n"))


def report_page(fig, message, who, what, what_unit, when, footnote, axes_left=0.08, axes_right=0.97):
    """Mensagem, título em três linhas e rodapé, com as mesmas posições (em polegadas)
    em todas as páginas. A área dos gráficos fica com o espaço que sobra."""
    w, h = fig.get_size_inches()
    lh = lambda fs: fs * 1.32 / 72  # noqa: E731  altura de uma linha, em polegadas
    y = h - 0.12
    msg = _wrap(message, w - 0.3, FONT + 1, bold=True)
    fig.text(0.03, y / h, msg, fontsize=FONT + 1, fontweight="bold", va="top")
    y -= lh(FONT + 1) * (msg.count("\n") + 1) + 0.22
    fig.text(0.03, y / h, who, fontsize=FONT - 1, va="top", color=MID)
    y -= lh(FONT - 1)
    b = fig.text(0.03, y / h, what, fontsize=FONT - 1, fontweight="bold", va="top")
    fig.canvas.draw()
    x_end = fig.transFigure.inverted().transform(b.get_window_extent().corners()[-1])[0]
    fig.text(x_end + 0.11 / w, y / h, what_unit, fontsize=FONT - 1, va="top", color=MID)
    y -= lh(FONT - 1)
    fig.text(0.03, y / h, when, fontsize=FONT - 1, va="top", color=MID)
    y -= lh(FONT - 1) + 0.45
    foot = _wrap(footnote, w - 0.3, FONT - 4)
    fig.text(0.03, 0.1 / h, foot, fontsize=FONT - 4, color=MID, va="bottom")
    bottom = 0.1 + lh(FONT - 4) * (foot.count("\n") + 1) + 0.75
    fig.subplots_adjust(top=y / h, bottom=bottom / h, left=axes_left, right=axes_right)
    return fig


def hbars(ax, labels, values, colors, texts, xmax=None, err=None, height=0.62, fontsize=FONT - 1):
    """Barras horizontais com o valor escrito no fim (sem eixo de valores).
    err: (baixo, alto) do IC; o valor fica escrito depois do IC."""
    values = np.asarray(values, dtype=float)
    y = np.arange(len(values))
    xerr = None if err is None else [values - np.asarray(err[0]), np.asarray(err[1]) - values]
    bars = ax.barh(y, values, color=colors, height=height, xerr=xerr,
                   error_kw=dict(ecolor=MID, lw=1, capsize=0))
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0, labelsize=fontsize)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ends = values if err is None else np.asarray(err[1])
    ax.set_xlim(0, xmax or ends.max() * 1.25)
    for b, end, t in zip(bars, ends, texts):
        ax.annotate(t, (end, b.get_y() + b.get_height() / 2), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=fontsize, fontweight="bold")
    return bars


def ref_line(ax, x, label):
    """Linha vertical tracejada de referência (ex.: taxa global), com rótulo direto em cima."""
    ax.axvline(x, color=MID, ls="--", lw=1)
    ax.annotate(label, (x, -0.55), ha="center", va="bottom", fontsize=FONT - 3, color=MID, annotation_clip=False)


def bars_page(values, message, who, what, what_unit, when, footnote, fmt=pct, highlight=(), ref=None,
              ref_label=None, figsize=(10, 4.8), err=None, n=None, panels=None):
    """Página com barras horizontais (estrutura na vertical, UN 3.4), valores escritos,
    destaque a verde e, se pedido, uma linha de referência (ex.: taxa global).

    values: Series (um painel), ou panels = [(Series, título, err, n), ...] para vários painéis.
    highlight: categorias a verde; sem destaque, todas a preto; com destaque, as outras a cinzento.
    err: (baixo, alto) do IC 95%; n: Series com o n de cada categoria (escrito ao lado do valor)."""
    panels = panels or [(values, None, err, n)]
    fig, axes = plt.subplots(1, len(panels), figsize=figsize, squeeze=False,
                             gridspec_kw={"wspace": 0.55} if len(panels) > 1 else None)
    for ax, (vals, title, e, nn) in zip(axes[0], panels):
        colors = [GREEN if i in highlight else CONTEXT for i in vals.index] if highlight else [INK] * len(vals)
        ns = nn.values if nn is not None else [None] * len(vals)
        texts = [fmt(v) + (f"  n={num(k)}" if k is not None else "") for v, k in zip(vals.values, ns)]
        top = max(vals.max(), np.max(e[1]) if e is not None else 0)
        hbars(ax, [str(i) for i in vals.index], vals.values, colors, texts, err=e,
              xmax=top * (1.6 if nn is not None else 1.25))
        if ref is not None:
            ref_line(ax, ref, ref_label or fmt(ref))
        if title:
            ax.set_title(title, loc="left", fontsize=FONT - 1, color=MID, pad=20 if ref is not None else 6)
    return report_page(fig, message, who, what, what_unit, when, footnote,
                       axes_left=0.2 if len(panels) == 1 else 0.12)


def model_vs_baseline(table, k, n_altas, footnote):
    """Página "modelo contra baseline à mesma capacidade" (exemplo da W8, slide 40).

    table: DataFrame com índice = método e coluna 'encontradas' (readmissões
    encontradas), com a linha de referência 'Ordenar por internamentos anteriores'.
    À esquerda, AC (modelo a verde, regra a preto, acaso a cinzento); à direita, a
    diferença face à regra, verde se melhor e vermelha se pior, na mesma escala (CH 4.1).
    """
    ref_name = "Ordenar por internamentos anteriores"
    vals = table["encontradas"]
    labels = [MODEL_LABELS["rank_inpatient"] if i == ref_name else "Ao acaso" if "acaso" in i else
              "Modelo v1" if i.startswith("Modelo") else i for i in vals.index]
    delta = vals - vals[ref_name]
    lim = max(vals.max(), 1) * 1.22
    d_lo, d_hi = min(delta.min(), 0) * 1.35, max(delta.max(), 0) * 1.6 + 60
    fig, (ax, axd) = plt.subplots(1, 2, figsize=(10, 4.8),
                                  gridspec_kw={"width_ratios": [lim, d_hi - d_lo], "wspace": 0.05})
    colors = [GREEN if lab == "Modelo v1" else CONTEXT if lab == "Ao acaso" else INK for lab in labels]
    hbars(ax, labels, vals.values, colors, [num(v) for v in vals.values], xmax=lim)
    ax.set_title("AC", loc="left", fontsize=FONT - 1, color=MID)
    y = np.arange(len(vals))
    axd.barh(y, delta.values, color=[GREEN if d > 0 else RED for d in delta.values], height=0.62)
    axd.invert_yaxis()
    axd.axvline(0, color=INK, lw=1)
    axd.set_xlim(d_lo, d_hi)
    axd.set_yticks([])
    axd.xaxis.set_visible(False)
    for s in ("bottom", "left"):
        axd.spines[s].set_visible(False)
    for yi, d in zip(y, delta.values):
        if abs(d) >= 0.5:
            axd.annotate(num(d, sign=True), (d, yi), xytext=(4 if d > 0 else -4, 0), textcoords="offset points",
                         va="center", ha="left" if d > 0 else "right", fontsize=FONT - 1, fontweight="bold")
    axd.set_title("Δ face à regra", loc="left", fontsize=FONT - 1, color=MID)
    model = vals.index[0]
    # Nº real de altas na lista (com os empates no corte pode passar de k × n): encontradas / precisão.
    n_list = round(vals[model] / table.loc[model, "precision"]) if "precision" in table else round(n_altas * k)
    msg = (f"Com follow-up às {num(n_list)} altas de maior risco ({pct(k, 0)} das {num(n_altas)} elegíveis), "
           f"o modelo encontra {num(vals[model])} readmissões em menos de 30 dias, "
           f"{num(delta[model])} mais do que a regra dos internamentos anteriores")
    return report_page(
        fig, msg,
        "Rede hospitalar, altas de treino elegíveis para follow-up (D21)",
        "Readmissões < 30 dias encontradas", f"em #, com follow-up a {pct(k, 0)} das altas elegíveis, por método",
        "Validação cruzada, 5 folds por doente (dados de 1999–2008) · AC e Δ face à regra", footnote,
        axes_left=0.24)


def lines_page(curves, message, who, what, what_unit, when, footnote, ref=None, ref_label=None,
               xlog=False, xticks=None, xlabel="", yfmt=None, figsize=(10, 5.6)):
    """Página com linhas (ex.: curva de aprendizagem), com o nome escrito no fim de cada linha.

    curves: lista de (Series indexada por x, rótulo, cor, destaque); destaque = linha grossa
    e valores escritos por cima; as outras têm os valores por baixo."""
    fig, ax = plt.subplots(figsize=figsize)
    fmt = yfmt or (lambda z: num(z, 3))
    for s, label, color, strong in curves:
        tc = GREEN_TEXT if color == GREEN else color
        ax.plot(s.index, s.values, color=color, lw=3 if strong else 2, marker="o", ms=6 if strong else 5)
        for x, v in zip(s.index, s.values):
            ax.annotate(fmt(v), (x, v), xytext=(0, 9 if strong else -15), textcoords="offset points",
                        ha="center", fontsize=FONT - 3, color=tc)
        ax.annotate(label, (s.index[-1], s.values[-1]), xytext=(14, 0), textcoords="offset points", va="center",
                    fontsize=FONT - 1, color=tc, fontweight="bold" if strong else None, annotation_clip=False)
    if xlog:
        ax.set_xscale("log")
    if xticks is not None:
        ax.set_xticks(xticks, [num(t) for t in xticks])
        ax.minorticks_off()
    if ref is not None:
        ax.axhline(ref, color=MID, ls="--", lw=1.2)
        ax.annotate(ref_label, (xticks[0] if xticks is not None else ax.get_xlim()[0], ref), xytext=(0, 5),
                    textcoords="offset points", fontsize=FONT - 3, color=MID)
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", length=0, labelsize=FONT - 2)
    ax.set_xlabel(xlabel, fontsize=FONT - 1)
    return report_page(fig, message, who, what, what_unit, when, footnote, axes_left=0.05, axes_right=0.8)
