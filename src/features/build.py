"""
Feature engineering (T1.8, D14) e seleção de colunas para o modelo (T1.9, D15).

Todas as features são calculadas linha a linha, só com informação do próprio
encontro: nenhuma agregação por doente nem ordem do encounter_id (D12).
"""

import pandas as pd

from src.data.quality import icd9_numeric

DIAG_COLS = ["diag_1", "diag_2", "diag_3"]

MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]
# Medicamentos com uso suficiente para ficarem como coluna própria (≥1,5%).
# Os restantes só entram nas contagens agregadas (D15).
KEPT_MEDS = ["metformin", "repaglinide", "glimepiride", "glipizide", "glyburide",
             "pioglitazone", "rosiglitazone", "insulin"]

VISIT_COLS = ["number_outpatient", "number_emergency", "number_inpatient"]

# Grupos ICD-9 de Strack et al. (2014), tabela 2. V/E e o resto ficam em "Other".
ICD9_GROUPS = [
    ("Circulatory", [(390, 460), (785, 786)]),
    ("Respiratory", [(460, 520), (786, 787)]),
    ("Digestive", [(520, 580), (787, 788)]),
    ("Diabetes", [(250, 251)]),
    ("Injury", [(800, 1000)]),
    ("Musculoskeletal", [(710, 740)]),
    ("Genitourinary", [(580, 630), (788, 789)]),
    ("Neoplasms", [(140, 240)]),
]
MISSING = "Missing"


def icd9_group(codes: pd.Series) -> pd.Series:
    """Grupo de Strack et al. de cada código ICD-9; "Missing" se não houver código."""
    num = icd9_numeric(codes)
    out = pd.Series("Other", index=codes.index, dtype="object")
    for name, ranges in ICD9_GROUPS:
        hit = pd.Series(False, index=codes.index)
        for lo, hi in ranges:
            hit |= (num >= lo) & (num < hi)
        out[hit & (out == "Other")] = name
    out[codes.isna()] = MISSING
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta as features da T1.8 (D14). Não remove nenhuma coluna."""
    df = df.copy()
    # Idade: banda como ordinal 0-9 ("[70-80)" -> 7).
    df["age_band"] = df["age"].str.extract(r"\[(\d+)", expand=False).astype(int) // 10

    # Diagnósticos: grupo de Strack et al. e diabetes com complicações
    # (250.1-250.9; 250.0x é "sem complicações").
    for c in DIAG_COLS:
        df[f"{c}_group"] = icd9_group(df[c])
    num = df[DIAG_COLS].apply(icd9_numeric)
    df["diabetes_complicated"] = ((num >= 250.1) & (num < 251)).any(axis=1).astype(int)

    # Medicamentos: contagens agregadas sobre as 23 colunas.
    meds = df[MED_COLS]
    df["n_meds"] = meds.ne("No").sum(axis=1)
    df["n_meds_up"] = meds.eq("Up").sum(axis=1)
    df["n_meds_down"] = meds.eq("Down").sum(axis=1)

    # Histórico: total de visitas no ano anterior (P13).
    df["n_visits_prior"] = df[VISIT_COLS].sum(axis=1)
    return df


# --- T1.9: colunas que entram no modelo (D15) --------------------------------

NUMERIC_FEATURES = [
    "age_band", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_diagnoses", "n_meds", "n_meds_up", "n_meds_down",
]
# Contagens muito assimétricas (66-89% de zeros): log1p no pipeline (T1.10).
COUNT_FEATURES = VISIT_COLS + ["n_visits_prior"]
BINARY_FEATURES = ["diabetes_complicated"]
CATEGORICAL_FEATURES = [
    "race", "gender", "payer_code", "medical_specialty",
    "admission_type_id_desc", "discharge_disposition_id_desc", "admission_source_id_desc",
    "A1Cresult", "max_glu_serum",
    "diag_1_group", "diag_2_group", "diag_3_group",
] + KEPT_MEDS
FEATURES = NUMERIC_FEATURES + COUNT_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# Colunas que ficam de fora, com o motivo (D15; identificadores e target em D11).
DROPPED = {
    "weight": "96,9% missing, sem sinal",
    "examide": "constante",
    "citoglipton": "constante",
    "change": "redundante com as colunas de medicamentos (P20)",
    "diabetesMed": "igual a n_meds >= 1 (P19)",
    "diag_1": "substituído por diag_1_group",
    "diag_2": "substituído por diag_2_group",
    "diag_3": "substituído por diag_3_group",
    "age": "substituído por age_band",
    "admission_type_id": "substituído por admission_type_id_desc (D5)",
    "discharge_disposition_id": "substituído por discharge_disposition_id_desc (D5)",
    "admission_source_id": "substituído por admission_source_id_desc (D5)",
    **{m: "quase constante (>= 99,3% No); só nas contagens" for m in MED_COLS
       if m not in KEPT_MEDS + ["examide", "citoglipton"]},
}


def model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz de features para o modelo: build_features, colunas de FEATURES e
    "Missing" como categoria nas categóricas com valores em falta (D15)."""
    x = build_features(df)[FEATURES].copy()
    x[CATEGORICAL_FEATURES] = x[CATEGORICAL_FEATURES].astype("object").fillna(MISSING)
    return x
