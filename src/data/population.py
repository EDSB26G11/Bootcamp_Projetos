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

# Destinos de alta fora da lista de follow-up (D21). O doente continua
# internado, vai para outra instituição ou já sai com acompanhamento
# (domiciliário, paliativo). Ficam no treino do modelo (D1 não muda) e o
# modelo estima-lhes o risco, mas não entram na lista de prioridades: o
# "acompanhamento adicional" é para quem volta para a comunidade sem
# acompanhamento. Descrições exatamente como no IDS_mapping.csv.
NO_FOLLOW_UP_DISCHARGE = {
    2: "Discharged/transferred to another short term hospital",
    3: "Discharged/transferred to SNF",
    4: "Discharged/transferred to ICF",
    5: "Discharged/transferred to another type of inpatient care institution",
    6: "Discharged/transferred to home with home health service",
    8: "Discharged/transferred to home under care of Home IV provider",
    9: "Admitted as an inpatient to this hospital",
    12: "Still patient or expected to return for outpatient services",
    13: "Hospice / home",
    14: "Hospice / medical facility",
    15: "Discharged/transferred within this institution to Medicare approved swing bed",
    22: "Discharged/transferred to another rehab fac including rehab units of a hospital .",
    23: "Discharged/transferred to a long term care hospital.",
    24: "Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare.",
    28: "Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital",
}


def follow_up_eligible(df: pd.DataFrame) -> pd.Series:
    """True se a alta pode entrar na lista de follow-up (D21): o destino da
    alta não é nenhum de NO_FOLLOW_UP_DISCHARGE. Os destinos desconhecidos
    ficam elegíveis."""
    return ~df["discharge_disposition_id_desc"].isin(NO_FOLLOW_UP_DISCHARGE.values())


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
