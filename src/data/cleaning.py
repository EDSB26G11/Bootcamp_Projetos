"""
Limpeza de valores aprovada na EDA (D3, D4).
"""

import pandas as pd

DIAG_COLS = ["diag_1", "diag_2", "diag_3"]


def harmonize_race(df: pd.DataFrame) -> pd.DataFrame:
    """Dá a cada doente com mais de uma race a moda dos seus encontros (D3).

    Empate: fica a race do encontro mais recente (maior encounter_id, P17).
    Só mexe nos doentes com mais de uma race; os restantes ficam como estão.
    """
    df = df.copy()
    known = df.dropna(subset=["race"])
    multi = known.groupby("patient_nbr")["race"].nunique()
    multi = multi[multi > 1].index
    if multi.empty:
        return df
    sub = known[known["patient_nbr"].isin(multi)]
    counts = sub.groupby(["patient_nbr", "race"]).agg(
        n=("race", "size"), last=("encounter_id", "max")
    ).reset_index()
    best = counts.sort_values(["patient_nbr", "n", "last"]).groupby("patient_nbr").tail(1)
    lookup = best.set_index("patient_nbr")["race"]
    rows = df["patient_nbr"].isin(multi)
    df.loc[rows, "race"] = df.loc[rows, "patient_nbr"].map(lookup)
    return df


def pad_icd9(df: pd.DataFrame) -> pd.DataFrame:
    """Repõe os zeros à esquerda nos códigos ICD-9 numéricos ("8" -> "008") (D4)."""
    df = df.copy()
    for c in DIAG_COLS:
        df[c] = df[c].str.replace(r"^\d{1,2}(?=\.|$)", lambda m: m.group(0).zfill(3), regex=True)
    return df
