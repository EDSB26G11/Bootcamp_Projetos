"""
Figuras de relatório, todas no mesmo layout (src/viz.py, igual à curva de ganhos):
- EDA (docs/figures/01–05), com os números do notebooks/01_eda;
- Projeto_Final (reports/figures/page_*.png e kumo_learning_curve.png).
A curva de ganhos e os gráficos do deck saem de src/deck_figures.py.

Os números são calculados aqui, não escritos à mão. O teste (D13) não é usado:
a EDA lê o ficheiro original (como o notebook 01), as páginas de modelação usam
as previsões out-of-fold do treino e os CSV de reports/.

Correr a partir da raiz do repositório:
    python -m src.report_figures
"""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.viz import INK, MID, bars_page, lines_page, model_vs_baseline, num, pct, set_style  # noqa: E402

DOCS_FIG = Path("docs/figures")
PAGES = Path("reports/figures")
REPORTS = Path("reports")
SOURCE = "Fonte: Diabetes 130-US hospitals (UCI), 1999–2008. "


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print("  ", path)


# --- EDA (ficheiro original, como no notebooks/01_eda) -----------------------------------------


def load_raw() -> pd.DataFrame:
    from src.data.load import load_diabetic_data, validate_diabetic_data
    from src.data.mappings import decode_ids, parse_ids_mapping

    df = load_diabetic_data(Path("data/raw/diabetic_data.csv"))
    validate_diabetic_data(df)
    df = decode_ids(df, parse_ids_mapping(Path("data/raw/IDS_mapping.csv")))
    df["target"] = (df["readmitted"] == "<30").astype(int)
    return df


def wilson(k, n, z=1.96):
    """Intervalo de confiança de Wilson (95%) para uma proporção."""
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return centre - half, centre + half


def rate_by(data, col, order=None):
    """Taxa de readmissão <30 por categoria, com n e IC 95% (como rate_by no notebook 01)."""
    g = data.groupby(col, dropna=False, observed=True)["target"].agg(n="size", pos="sum")
    g["rate"] = g["pos"] / g["n"]
    g["lo"], g["hi"] = wilson(g["pos"], g["n"])
    g.index = g.index.astype("object").where(g.index.notna(), "Em falta (?)")
    return g.reindex([o for o in order if o in g.index]) if order is not None else g


def eda_figures(df: pd.DataFrame):
    n, base = len(df), df["target"].mean()
    eda_who = f"Diabetes 130-US hospitals, todos os encontros do ficheiro original (n = {num(n)})"
    eda_when = "1999–2008 · AC · análise exploratória, antes das exclusões (D1, D2, D7)"

    # 01 · Target
    vc = df["readmitted"].value_counts(normalize=True).reindex(["NO", ">30", "<30"])
    vc.index = ["Não readmitido (NO)", "Readmitido depois de 30 dias (>30)", "Readmitido em menos de 30 dias (<30)"]
    save(
        bars_page(
            vc,
            f"{pct(vc.iloc[2])} dos encontros acabam numa readmissão em menos de 30 dias: o alvo é a "
            f"classe pequena; prever sempre «não» daria {pct(1 - vc.iloc[2])} de accuracy",
            eda_who,
            "Encontros",
            "em %, por resultado da readmissão",
            eda_when,
            SOURCE + "Target do projeto (D6): <30 contra o resto; os >30 contam como negativos.",
            highlight=[vc.index[2]],
            figsize=(10, 4.2),
        ),
        DOCS_FIG / "01_target.png",
    )

    # 02 · Missing contra não medido
    miss = df.drop(columns="target").isna().mean()
    miss = miss[miss > 0].sort_values(ascending=False)
    none = pd.Series({c: (df[c] == "None").mean() for c in ["max_glu_serum", "A1Cresult"]})
    v = pd.concat([miss, none])
    labels = [f"{c} (em falta)" for c in miss.index] + [f"{c} (não medido)" for c in none.index]
    v.index = labels
    fig = bars_page(
        v,
        f"{v.index[0].split(' ')[0]} falta em {pct(v.iloc[0])} dos encontros e saiu do modelo; "
        f"max_glu_serum e A1Cresult não foram medidos em {pct(none.iloc[0], 0)} e "
        f"{pct(none.iloc[1], 0)}, o que é informação e não falta (P2)",
        eda_who,
        "Valores em falta (?) e não medidos (None)",
        "em % dos encontros, por coluna",
        eda_when,
        SOURCE + "Verde: «None» nas análises quer dizer teste não feito e entra como categoria (P2). "
        "Só aparecem colunas com algum valor em falta.",
        highlight=[f"{c} (não medido)" for c in none.index],
        figsize=(10, 6.4),
    )
    # Sem destaque, as colunas com falta ficam a preto (AC), e não a cinzento.
    for bar, lab in zip(fig.axes[0].patches, labels):
        if "não medido" not in lab:
            bar.set_color(INK)
    save(fig, DOCS_FIG / "02_missing.png")

    # 03 · Encontros por doente
    enc = df.groupby("patient_nbr").size()
    counts = enc.clip(upper=10).value_counts().sort_index()
    counts.index = [f"{i} encontro" + ("s" if i > 1 else "") if i < 10 else "10 ou mais" for i in counts.index]
    rec_share = enc[enc > 1].sum() / len(df)
    save(
        bars_page(
            counts,
            f"{pct((enc == 1).mean(), 0)} dos {num(len(enc))} doentes têm um só encontro, mas os "
            f"doentes recorrentes somam {pct(rec_share, 0)} dos encontros: por isso o split é feito "
            "por doente (D13)",
            eda_who,
            "Doentes",
            "em #, por nº de encontros no ficheiro",
            eda_when,
            SOURCE + f"Máximo: {num(enc.max())} encontros num doente. Contar encontros por doente não entra "
            "no modelo: inclui o próprio regresso (D12).",
            fmt=num,
            figsize=(10, 6.6),
        ),
        DOCS_FIG / "03_encontros_por_doente.png",
    )

    # 04 · Taxa por idade e por raça, com IC 95%
    age_order = sorted(df["age"].dropna().unique(), key=lambda s: int(s.strip("[").split("-")[0]))
    ga = rate_by(df, "age", age_order)
    ga.index = [f"{a.strip('[)').split('-')[0]}–{int(a.strip('[)').split('-')[1]) - 1} anos" for a in ga.index]
    gr = rate_by(df, "race").sort_values("n", ascending=False)
    save(
        bars_page(
            None,
            f"A taxa de readmissão varia mais com a idade ({pct(ga['rate'].min())} a "
            f"{pct(ga['rate'].max())}) do que com a raça ({pct(gr['rate'].min())} a "
            f"{pct(gr['rate'].max())}); os grupos pequenos têm IC largos",
            eda_who,
            "Taxa de readmissão < 30 dias",
            "em %, com IC 95% (Wilson), por idade e por raça",
            eda_when,
            SOURCE + f"Linha tracejada: taxa global ({pct(base)}). Raça como vem na fonte. "
            "Os IC tratam os encontros como independentes, o que não é verdade para doentes "
            "recorrentes (P22): são algo estreitos demais.",
            ref=base,
            ref_label=f"global {pct(base)}",
            figsize=(12, 6.8),
            panels=[
                (ga["rate"], "Por idade", (ga["lo"], ga["hi"]), ga["n"]),
                (gr["rate"], "Por raça", (gr["lo"], gr["hi"]), gr["n"]),
            ],
        ),
        DOCS_FIG / "04_taxa_idade_raca.png",
    )

    # 05 · Taxa por internamentos no ano anterior, com IC 95%
    g = rate_by(
        df.assign(_i=df["number_inpatient"].clip(upper=6).astype(str).replace("6", "6 ou mais")),
        "_i",
        ["0", "1", "2", "3", "4", "5", "6 ou mais"],
    )
    save(
        bars_page(
            g["rate"],
            f"A taxa de readmissão sobe de {pct(g['rate'].iloc[0])} sem internamentos no ano "
            f"anterior para {pct(g['rate'].iloc[-1])} com 6 ou mais: o sinal mais forte",
            eda_who,
            "Taxa de readmissão < 30 dias",
            "em %, com IC 95% (Wilson), por internamentos no ano " "anterior",
            eda_when,
            SOURCE + f"Linha tracejada: taxa global ({pct(base)}). É a base da regra de comparação "
            "(ordenar por internamentos anteriores).",
            err=(g["lo"], g["hi"]),
            n=g["n"],
            ref=base,
            ref_label=f"global {pct(base)}",
            figsize=(10, 5.4),
        ),
        DOCS_FIG / "05_taxa_number_inpatient.png",
    )


# --- Páginas do Projeto_Final (cada uma é chamada no Projeto_Final.ipynb) ----------------------

WHO_POP = "Diabetes 130-US hospitals, população do projeto (sem óbitos nem registos impossíveis: D1, D2, D7)"


def page_target(df: pd.DataFrame):
    """Distribuição do resultado da readmissão na população do projeto."""
    base = df["target"].mean()
    dist = df["readmitted"].value_counts(normalize=True).reindex(["NO", ">30", "<30"])
    dist.index = ["Não readmitido", "Readmitido depois de 30 dias", "Readmitido em menos de 30 dias"]
    return bars_page(
        dist,
        f"{pct(base)} das {num(len(df))} altas foram readmitidas em menos de 30 dias: "
        "o grupo que o follow-up quer encontrar é pequeno",
        WHO_POP,
        "Altas",
        "em %, por resultado da readmissão",
        "1999–2008 · AC",
        SOURCE + f"n = {num(len(df))} altas de {num(df['patient_nbr'].nunique())} doentes. "
        "Readmissões noutros hospitais não aparecem nos dados.",
        highlight=["Readmitido em menos de 30 dias"],
        figsize=(10, 4.2),
    )


def page_inpatient(df: pd.DataFrame):
    """Taxa de readmissão por internamentos no ano anterior (a base da regra de comparação)."""
    base = df["target"].mean()
    rate = df.groupby(df["number_inpatient"].clip(upper=3))["target"].mean()
    rate.index = ["0", "1", "2", "3 ou mais"]
    ratio = rate.iloc[-1] / rate.iloc[0]
    return bars_page(
        rate,
        f"Doentes com 3 ou mais internamentos no ano anterior foram readmitidos {num(ratio, 1)} "
        f"vezes mais ({pct(rate.iloc[-1])}) do que doentes sem internamentos "
        f"({pct(rate.iloc[0])})",
        WHO_POP,
        "Taxa de readmissão < 30 dias",
        "em %, por internamentos no ano anterior",
        "1999–2008 · AC",
        SOURCE + f"Linha tracejada: taxa global ({pct(base)}). É a variável mais forte e a base da "
        "regra de comparação (ordenar por internamentos anteriores).",
        ref=base,
        ref_label=f"global {pct(base)}",
        figsize=(10, 4.4),
    )


def page_model_vs_baseline(k: float):
    """Modelo contra a regra à mesma capacidade k (reports/capacity_comparison_v1.csv, notebook 04)."""
    from src.deck_figures import load_train

    comp = pd.read_csv(REPORTS / "capacity_comparison_v1.csv")
    n_elig = int(load_train()["eligible"].sum())  # D21, só treino
    return model_vs_baseline(
        comp[comp["k"] == k].set_index("metodo"),
        k,
        n_altas=n_elig,
        footnote=SOURCE + f"Follow-up a {pct(k, 0)} das altas elegíveis (D21: sem as altas que já têm "
        "acompanhamento à saída). AC: readmissões < 30 dias entre as altas com follow-up, previsões out-of-fold "
        "do treino (o teste ainda não foi aberto). Regra: empates desfeitos ao acaso, média de 20 sorteios. "
        "Ao acaso: valor esperado.",
    )


def page_kumo_curve():
    """Curva de aprendizagem do Kumo contra a regressão logística (reports/kumo_learning_curve.csv,
    notebook 07; não volta a correr o Kumo)."""
    from src.deck_figures import load_train

    cur = pd.read_csv(REPORTS / "kumo_learning_curve.csv").groupby(["rotulados", "modelo"])["pr_auc"].mean()
    cur = cur.unstack()
    kumo, lr = cur["Kumo Tabular"], cur["logreg"]
    base_train = load_train()["target"].mean()
    return lines_page(
        [(kumo, "Kumo Tabular", INK, True), (lr, "Regressão logística", MID, False)],
        f"Com {num(kumo.index[0])} altas rotuladas, o Kumo Tabular tem PR-AUC {num(kumo.iloc[0], 3)} "
        f"contra {num(lr.iloc[0], 3)} da regressão logística; o projeto tem 80 041 rótulos, onde "
        "essa vantagem já não conta (D23)",
        "Diabetes 130-US hospitals, altas de treino (2 folds por doente)",
        "PR-AUC na validação",
        "por nº de altas rotuladas usadas para treinar (escala log)",
        "Validação cruzada, média de 2 folds (dados de 1999–2008)",
        SOURCE + "Linha tracejada: PR-AUC ao acaso (taxa de readmissão no treino). Mais alto é melhor. "
        "Kumo Tabular: modelo pré-treinado do Hugging Face que lê as altas rotuladas como contexto. "
        "Dados: reports/kumo_learning_curve.csv (notebooks/07).",
        ref=base_train,
        ref_label=f"ao acaso: {num(base_train, 3)}",
        xlog=True,
        xticks=list(kumo.index),
        xlabel="Altas rotuladas (escala log)",
    )


def projeto_final_pages():
    from src.pipeline import read_processed

    df = read_processed()
    save(page_target(df), PAGES / "page_target.png")
    save(page_inpatient(df), PAGES / "page_inpatient.png")
    for k in (0.10, 0.20):
        save(page_model_vs_baseline(k), PAGES / f"page_model_vs_baseline_k{int(k * 100)}.png")
    save(page_kumo_curve(), PAGES / "kumo_learning_curve.png")


def main():
    set_style()
    DOCS_FIG.mkdir(parents=True, exist_ok=True)
    print("EDA:")
    eda_figures(load_raw())
    print("Projeto_Final:")
    projeto_final_pages()


if __name__ == "__main__":
    main()
