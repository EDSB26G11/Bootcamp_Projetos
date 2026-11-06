"""
Gráficos no padrão IBCS (aula W8, T3.7).

Duas utilizações:
- gráficos de desenvolvimento (notebooks): só a notação. set_style() tira a
  grelha e as molduras; as cores têm significado (cinzento escuro = valor
  medido, AC) e os valores escrevem-se nas barras em vez de um eixo;
- páginas de relatório (Projeto_Final, deck): todas as regras. report_page()
  põe a mensagem em cima, o título em três linhas (quem, o quê, quando) e o
  rodapé (fonte, dados usados, o que ficou de fora).

Números em português: vírgula decimal (pct, num).
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

AC = "#404040"        # valor medido (actual)
GREY = "#8c8c8c"      # referência / baseline
LIGHT = "#c8c8c8"     # ordem ao acaso, contexto
GOOD = "#8CB400"      # variação desejável
BAD = "#E2001A"       # variação indesejável
NOTE = "#666666"      # rodapés e rótulos secundários

# Nomes dos modelos no relatório (UN 1.1: o mesmo nome em todas as páginas).
MODEL_LABELS = {
    "hist_gradient_boosting": "Modelo v1 (HistGradientBoosting)",
    "random_forest": "Random forest",
    "logreg": "Regressão logística",
    "logreg_l1": "Regressão logística L1",
    "logreg_balanced": "Regressão logística com pesos",
    "rank_inpatient": "Ordenar por internamentos anteriores",
    "rule_inpatient_ge_1": "Regra: ≥ 1 internamento anterior",
    "random": "Ordem ao acaso",
}


def set_style():
    """Estilo base: sem grelha, sem molduras, cinzentos, letra legível."""
    mpl.rcParams.update({
        "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": GREY, "axes.labelcolor": AC, "xtick.color": AC, "ytick.color": AC,
        "axes.prop_cycle": mpl.cycler(color=[AC, GREY, LIGHT]),
        "font.size": 10, "axes.titlesize": 11, "axes.titlelocation": "left",
        "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 150,
    })


def pct(x, decimals=1, sign=False):
    """0.1134 -> '11,3%' (com sign=True: '+2,1%')."""
    s = f"{x * 100:+.{decimals}f}%" if sign else f"{x * 100:.{decimals}f}%"
    return s.replace(".", ",")


def num(x, decimals=0, sign=False):
    """12345 -> '12 345'; 2.5 -> '2,5'."""
    s = f"{x:+,.{decimals}f}" if sign else f"{x:,.{decimals}f}"
    return s.replace(",", " ").replace(".", ",")


def label_bars(ax, bars, texts, color=AC, pad=3, fontsize=9):
    """Escreve o valor no fim de cada barra horizontal e retira o eixo de valores (SI 3.1)."""
    for b, t in zip(bars, texts):
        ax.annotate(t, (b.get_x() + b.get_width(), b.get_y() + b.get_height() / 2),
                    xytext=(pad, 0), textcoords="offset points", va="center", fontsize=fontsize, color=color)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)


def report_page(fig, message, who, what, what_unit, when, footnote, top=0.74, left=None):
    """Mensagem (uma frase que se pode verificar), título em três linhas e rodapé.

    what aparece a negrito (a medida); what_unit é o resto da linha (unidade, recorte).
    """
    fig.text(0.02, 0.985, message, fontsize=12, fontweight="bold", va="top", wrap=True)
    fig.text(0.02, top + 0.15, who, fontsize=9.5, va="top")
    bold = fig.text(0.02, top + 0.11, what, fontsize=9.5, fontweight="bold", va="top")
    fig.canvas.draw()  # mede o texto a negrito para pôr a unidade logo a seguir
    x_end = fig.transFigure.inverted().transform(bold.get_window_extent().corners()[-1])[0]
    fig.text(x_end + 0.008, top + 0.11, what_unit, fontsize=9.5, va="top")
    fig.text(0.02, top + 0.07, when, fontsize=9.5, va="top")
    fig.text(0.02, 0.01, footnote, fontsize=7.5, color=NOTE, va="bottom", wrap=True)
    fig.subplots_adjust(top=top, bottom=0.16, **({"left": left} if left else {}))
    return fig


def bars_page(values, message, who, what, what_unit, when, footnote, fmt=pct, highlight=(), ref=None,
              ref_label=None, figsize=(9, 4.6)):
    """Página com barras horizontais (estrutura na vertical, UN 3.4), valores escritos
    e, opcionalmente, uma linha de referência com rótulo direto (ex.: taxa global)."""
    fig, ax = plt.subplots(figsize=figsize)
    y = range(len(values))
    colors = [AC if (not highlight or name in highlight) else LIGHT for name in values.index]
    bars = ax.barh(list(y), values.values, color=colors, height=0.6)
    ax.invert_yaxis()
    ax.set_yticks(list(y), [str(i) for i in values.index])
    ax.tick_params(axis="y", length=0)
    label_bars(ax, bars, [fmt(v) for v in values.values])
    ax.set_xlim(0, values.max() * 1.18)
    if ref is not None:
        ax.axvline(ref, color=GREY, ls="--", lw=1)
        ax.annotate(ref_label or fmt(ref), (ref, len(values) - 0.4), ha="center", va="top", fontsize=8,
                    color=GREY, annotation_clip=False)
    return report_page(fig, message, who, what, what_unit, when, footnote, top=0.66)


def model_vs_baseline(table, k, n_altas, footnote):
    """Página "modelo contra baseline à mesma capacidade" (exemplo da W8, slide 40).

    table: DataFrame com índice = método e colunas 'encontradas' (readmissões
    encontradas) e a linha de referência 'Ordenar por internamentos anteriores'.
    Barras AC à esquerda; à direita, a diferença face à referência, verde se
    melhor e vermelha se pior, na mesma escala.
    """
    ref_name = "Ordenar por internamentos anteriores"
    vals = table["encontradas"]
    delta = vals - vals[ref_name]
    # Mesma escala nos dois níveis (CH 4.1): a largura de cada painel é proporcional ao intervalo que mostra.
    lim = max(vals.max(), 1)
    d_lo, d_hi = min(delta.min(), 0) * 1.4, max(delta.max(), 0) * 1.4
    fig, (ax, axd) = plt.subplots(1, 2, figsize=(10.5, 4.6),
                                  gridspec_kw={"width_ratios": [lim * 1.15, d_hi - d_lo], "wspace": 0.08})
    y = range(len(vals))
    bars = ax.barh(list(y), vals.values, color=AC, height=0.6)
    ax.invert_yaxis()
    ax.set_yticks(list(y), vals.index, fontsize=10)
    ax.tick_params(axis="y", length=0)
    label_bars(ax, bars, [num(v) for v in vals.values])
    ax.set_title("AC", color=NOTE, fontsize=9, loc="left")
    colors = [GOOD if d > 0 else BAD for d in delta.values]
    dbars = axd.barh(list(y), delta.values, color=colors, height=0.6)
    axd.invert_yaxis()
    axd.axvline(0, color=AC, lw=1)
    axd.set_yticks([])
    axd.set_xlim(d_lo, d_hi)
    for b, d, name in zip(dbars, delta.values, vals.index):
        if name == ref_name:
            continue
        x = b.get_width()
        axd.annotate(num(d, sign=True), (x, b.get_y() + b.get_height() / 2), xytext=(4 if d > 0 else -4, 0),
                     textcoords="offset points", va="center", ha="left" if d > 0 else "right", fontsize=9)
    axd.xaxis.set_visible(False)
    for s in ("bottom", "left"):
        axd.spines[s].set_visible(False)
    axd.set_title("Δ face a internamentos anteriores", color=NOTE, fontsize=9, loc="left")
    ax.set_xlim(0, lim * 1.15)
    model_name = vals.index[0]
    msg = (f"Contactando as {num(n_altas * k)} altas de maior risco ({pct(k, 0)} de {num(n_altas)}), "
           f"o modelo encontraria {num(vals[model_name])} readmissões em menos de 30 dias, "
           f"{num(delta[model_name])} mais do que ordenando por internamentos anteriores")
    return report_page(
        fig, msg,
        "Diabetes 130-US hospitals, encontros de treino (previsões out-of-fold, 5 folds por doente)",
        "Readmissões <30 dias encontradas", f"em #, com capacidade para {pct(k, 0)} das altas, por método",
        "1999–2008, AC e Δ face à ordenação por internamentos anteriores", footnote, top=0.66, left=0.27)
