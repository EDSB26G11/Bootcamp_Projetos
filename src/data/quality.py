"""
Verificações de consistência do diabetic_data (T1.3, camada de qualidade).

Cada regra é uma função df -> Series booleana alinhada com o índice de df
(True = o encontro viola a regra). As regras só identificam; não removem nem
corrigem nada. As ações ficam em docs/decisions.md.

Camadas:
    A - coerência dentro do encontro (ex.: recém-nascido com 70 anos)
    B - formato e domínio dos valores
    C - coerência do mesmo doente entre encontros (ordem = encounter_id, P17)

Severidade:
    erro      - impossível
    suspeito  - improvável, mas com explicação possível
    limitacao - esperado pela forma como o dataset foi construído
"""

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from src.data.population import EXCLUDED_DISCHARGE_IDS

DIAG_COLS = ["diag_1", "diag_2", "diag_3"]

NEWBORN_ADMISSION_TYPE = 4
BIRTH_ADMISSION_SOURCES = (11, 12, 13, 14, 23, 24)  # partos e nascimentos
NEONATE_DISCHARGE = 10
DEATH_DISCHARGE_IDS = tuple(i for i in EXCLUDED_DISCHARGE_IDS if i not in (13, 14))
OBSTETRIC_SPECIALTY = r"Obstetric|Obsteric|Gynecolog|Perinatology"

# O dataset cobre 10 anos (1999-2008). Subir 2 bandas de idade exige pelo
# menos 11 anos, por isso é impossível.
MAX_AGE_JUMP = 10

# Domínio esperado de cada coluna categórica (sem contar NaN).
MED_VALUES = {"No", "Steady", "Up", "Down"}
CATEGORICAL_DOMAINS = {
    "gender": {"Female", "Male", "Unknown/Invalid"},
    "race": {"Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"},
    "age": {f"[{i}-{i + 10})" for i in range(0, 100, 10)},
    "weight": {f"[{i}-{i + 25})" for i in range(0, 200, 25)} | {">200"},
    "max_glu_serum": {"None", "Norm", ">200", ">300"},
    "A1Cresult": {"None", "Norm", ">7", ">8"},
    "change": {"No", "Ch"},
    "diabetesMed": {"No", "Yes"},
    "readmitted": {"NO", ">30", "<30"},
}
NUMERIC_RANGES = {
    "time_in_hospital": (1, 14),  # critério de inclusão (P8)
    "num_lab_procedures": (0, None),
    "num_procedures": (0, None),
    "num_medications": (0, None),
    "number_outpatient": (0, None),
    "number_emergency": (0, None),
    "number_inpatient": (0, None),
    "number_diagnoses": (1, None),
}

ICD9_PATTERN = r"^(\d{3}(\.\d{1,2})?|V\d{2}(\.\d{1,2})?|E\d{3}(\.\d)?)$"


@dataclass(frozen=True)
class Check:
    id: str
    camada: str
    descricao: str
    severidade: str
    func: Callable[[pd.DataFrame], pd.Series]


# --- helpers -----------------------------------------------------------------

def age_lower(df: pd.DataFrame) -> pd.Series:
    """Limite inferior da banda de idade ("[70-80)" -> 70)."""
    return df["age"].str.extract(r"\[(\d+)", expand=False).astype(int)


def icd9_numeric(codes: pd.Series) -> pd.Series:
    """Código ICD-9 como número; NaN para V/E e missing."""
    is_ve = codes.fillna("").str[0].isin(["V", "E"])
    return pd.to_numeric(codes.where(~is_ve), errors="coerce")


def any_diag_in(df: pd.DataFrame, ranges: list[tuple[float, float]]) -> pd.Series:
    """Algum de diag_1-3 em [lo, hi) para algum dos intervalos dados."""
    num = df[DIAG_COLS].apply(icd9_numeric)
    hit = pd.Series(False, index=df.index)
    for lo, hi in ranges:
        hit |= ((num >= lo) & (num < hi)).any(axis=1)
    return hit


def _by_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Encontros ordenados por doente e encounter_id (proxy da ordem temporal, P17)."""
    return df.sort_values(["patient_nbr", "encounter_id"])


# --- A. coerência dentro do encontro -----------------------------------------

def newborn_not_infant(df):
    return (df["admission_type_id"] == NEWBORN_ADMISSION_TYPE) & (age_lower(df) >= 10)


def birth_source_not_infant(df):
    return df["admission_source_id"].isin(BIRTH_ADMISSION_SOURCES) & (age_lower(df) >= 10)


def neonate_discharge_not_infant(df):
    return (df["discharge_disposition_id"] == NEONATE_DISCHARGE) & (age_lower(df) >= 10)


def pediatrics_adult(df):
    pediatric = df["medical_specialty"].fillna("").str.contains("Pediatric")
    return pediatric & (age_lower(df) >= 20)


def obstetric_male(df):
    obstetric = df["medical_specialty"].fillna("").str.contains(OBSTETRIC_SPECIALTY)
    pregnancy = any_diag_in(df, [(630, 680)])
    return (obstetric | pregnancy) & (df["gender"] == "Male")


def pregnancy_60_plus(df):
    return any_diag_in(df, [(630, 680)]) & (age_lower(df) >= 60)


def prostate_female(df):
    prostate = any_diag_in(df, [(185, 186), (600, 603)])
    return prostate & (df["gender"] == "Female")


def diag_order_gap(df):
    """diag_1 em falta mas diag_2/diag_3 preenchidos."""
    return df["diag_1"].isna() & df[["diag_2", "diag_3"]].notna().any(axis=1)


def diag_repeated(df):
    d1, d2, d3 = (df[c] for c in DIAG_COLS)
    return ((d1 == d2) | (d1 == d3) | (d2 == d3)).fillna(False).astype(bool)


def number_diagnoses_too_low(df):
    return df["number_diagnoses"] < df[DIAG_COLS].notna().sum(axis=1)


# --- B. formato e domínio ----------------------------------------------------

def icd9_malformed(df):
    bad = pd.Series(False, index=df.index)
    for c in DIAG_COLS:
        bad |= df[c].notna() & ~df[c].fillna("").str.match(ICD9_PATTERN)
    return bad


def out_of_domain(df, mappings: dict[str, pd.DataFrame] | None = None):
    """Valor fora do domínio esperado em alguma coluna (categóricas, IDs, ranges)."""
    bad = pd.Series(False, index=df.index)
    med_cols = df.columns[df.columns.get_loc("metformin"): df.columns.get_loc("metformin-pioglitazone") + 1]
    domains = {**CATEGORICAL_DOMAINS, **{c: MED_VALUES for c in med_cols}}
    if mappings is not None:
        domains |= {col: set(m[col]) for col, m in mappings.items()}
    for col, allowed in domains.items():
        bad |= df[col].notna() & ~df[col].isin(allowed)
    for col, (lo, hi) in NUMERIC_RANGES.items():
        bad |= df[col] < lo
        if hi is not None:
            bad |= df[col] > hi
    return bad


def duplicated_rows(df):
    """Linhas iguais em tudo exceto os IDs (todas as cópias ficam marcadas)."""
    return df.drop(columns=["encounter_id", "patient_nbr"]).duplicated(keep=False)


def no_diabetes_code(df):
    """Nenhum diag_1-3 é 250.xx. Limitação: o dataset só guarda 3 dos até 16 diagnósticos."""
    return ~any_diag_in(df, [(250, 251)])


# --- C. coerência do doente entre encontros ----------------------------------

def patient_gender_inconsistent(df):
    return df.groupby("patient_nbr")["gender"].transform("nunique") > 1


def patient_race_inconsistent(df):
    return df.groupby("patient_nbr")["race"].transform("nunique") > 1


def age_decreases(df):
    """Idade mais baixa do que no encontro anterior do mesmo doente."""
    d = _by_patient(df)
    same = d["patient_nbr"].eq(d["patient_nbr"].shift())
    return (same & (age_lower(d).diff() < 0)).reindex(df.index)


def age_jump(df):
    """Idade sobe 2 ou mais bandas face ao encontro anterior."""
    d = _by_patient(df)
    same = d["patient_nbr"].eq(d["patient_nbr"].shift())
    return (same & (age_lower(d).diff() > MAX_AGE_JUMP)).reindex(df.index)


def encounter_after_death(df):
    """Há um encontro anterior do mesmo doente com alta por óbito."""
    d = _by_patient(df)
    dead = d["discharge_disposition_id"].isin(DEATH_DISCHARGE_IDS).astype(int)
    died_before = dead.groupby(d["patient_nbr"]).transform(lambda s: s.cumsum().shift(fill_value=0)) > 0
    return died_before.reindex(df.index)


def no_readmission_but_next(df):
    """readmitted = NO, mas o doente tem um encontro seguinte no dataset."""
    d = _by_patient(df)
    has_next = d["patient_nbr"].eq(d["patient_nbr"].shift(-1))
    return ((d["readmitted"] == "NO") & has_next).reindex(df.index)


def readmitted_without_next(df):
    """readmitted = <30 no último encontro do doente (readmissão fora do dataset)."""
    d = _by_patient(df)
    has_next = d["patient_nbr"].eq(d["patient_nbr"].shift(-1))
    return ((d["readmitted"] == "<30") & ~has_next).reindex(df.index)


CHECKS = [
    Check("A1", "A", "Admissão Newborn (tipo 4) com idade >= 10", "erro", newborn_not_infant),
    Check("A2", "A", "Origem parto/nascimento (11-14, 23, 24) com idade >= 10", "erro", birth_source_not_infant),
    Check("A3", "A", "Alta de neonato (destino 10) com idade >= 10", "erro", neonate_discharge_not_infant),
    Check("A4", "A", "Especialidade pediátrica com idade >= 20", "suspeito", pediatrics_adult),
    Check("A5", "A", "Obstetrícia/ginecologia ou gravidez (630-679) em homens", "erro", obstetric_male),
    Check("A6", "A", "Gravidez (630-679) com idade >= 60", "erro", pregnancy_60_plus),
    Check("A7", "A", "Próstata (185, 600-602) em mulheres", "erro", prostate_female),
    Check("A8", "A", "diag_1 em falta com diag_2/diag_3 preenchidos", "suspeito", diag_order_gap),
    Check("A9", "A", "Mesmo código repetido em diag_1-3", "suspeito", diag_repeated),
    Check("A10", "A", "number_diagnoses menor que os diag_1-3 preenchidos", "erro", number_diagnoses_too_low),
    Check("B1", "B", "Código ICD-9 mal formado (ex.: zeros à esquerda perdidos)", "suspeito", icd9_malformed),
    Check("B2", "B", "Valor fora do domínio esperado (categorias, IDs, ranges)", "erro", out_of_domain),
    Check("B3", "B", "Linha duplicada (exceto IDs)", "erro", duplicated_rows),
    Check("B4", "B", "Nenhum diag_1-3 é diabetes (250.xx)", "limitacao", no_diabetes_code),
    Check("C1", "C", "Doente com mais de um gender", "erro", patient_gender_inconsistent),
    Check("C2", "C", "Doente com mais de uma race", "suspeito", patient_race_inconsistent),
    Check("C3", "C", "Idade desce face ao encontro anterior", "erro", age_decreases),
    Check("C4", "C", "Idade sobe 2+ bandas face ao encontro anterior", "erro", age_jump),
    Check("C5", "C", "Encontro depois de uma alta por óbito", "erro", encounter_after_death),
    Check("C6", "C", "readmitted = NO, mas há encontro seguinte", "suspeito", no_readmission_but_next),
    Check("C7", "C", "readmitted = <30 no último encontro do doente", "limitacao", readmitted_without_next),
]


def run_checks(df: pd.DataFrame, mappings: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Corre todas as regras e devolve uma linha por regra com n e % de encontros."""
    rows = []
    for check in CHECKS:
        mask = check.func(df, mappings) if check.func is out_of_domain else check.func(df)
        n = int(mask.sum())
        rows.append({
            "id": check.id,
            "camada": check.camada,
            "descricao": check.descricao,
            "severidade": check.severidade,
            "n": n,
            "%": round(100 * n / len(df), 3),
        })
    return pd.DataFrame(rows).set_index("id")


def check_masks(df: pd.DataFrame, mappings: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Uma coluna booleana por regra, para cruzar regras ou agregar violações."""
    return pd.DataFrame({
        c.id: c.func(df, mappings) if c.func is out_of_domain else c.func(df)
        for c in CHECKS
    })
