"""
Definição da população elegível (T1.4) e do target (T1.5).
"""

import pandas as pd

# discharge_disposition_id de óbito (11, 19, 20, 21). Um doente que morreu não
# pode ser readmitido. As altas para hospício (13, 14) ficam na população: há
# readmissões nesses encontros (43). Ver docs/decisions.md (D1).
DEATH_DISCHARGE_IDS = (11, 19, 20, 21)

# gender sem valor válido: 3 encontros, todos NO, sem outro encontro do mesmo
# doente para recuperar o valor. Ver docs/decisions.md (D7).
INVALID_GENDER = "Unknown/Invalid"

# Target binário: readmissão em menos de 30 dias contra o resto (D6).
POSITIVE_READMITTED = "<30"


def exclude_deaths(df: pd.DataFrame) -> pd.DataFrame:
    """Remove os encontros com alta por óbito."""
    keep = ~df["discharge_disposition_id"].isin(DEATH_DISCHARGE_IDS)
    return df[keep].reset_index(drop=True)


def exclude_invalid_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Remove os encontros com gender Unknown/Invalid."""
    keep = df["gender"] != INVALID_GENDER
    return df[keep].reset_index(drop=True)


def exclude_impossible(df: pd.DataFrame, mappings: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Remove os encontros impossíveis (regras "erro" de quality.py, D2).

    Corre sobre os dados completos, antes de define_population: a regra C5
    precisa dos encontros de óbito.
    """
    # Import local: quality.py importa DEATH_DISCHARGE_IDS deste módulo.
    from src.data.quality import impossible_encounters

    return df[~impossible_encounters(df, mappings)].reset_index(drop=True)


def define_population(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica todas as exclusões de linhas aprovadas (D1, D7)."""
    return exclude_invalid_gender(exclude_deaths(df))


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta a coluna target: 1 se readmitted == '<30', 0 se '>30' ou 'NO'."""
    df = df.copy()
    df["target"] = (df["readmitted"] == POSITIVE_READMITTED).astype(int)
    return df
