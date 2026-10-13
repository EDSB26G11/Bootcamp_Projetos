"""
Leitura e validação do diabetic_data.csv (T1.1).
"""

import pandas as pd

EXPECTED_ROWS = 101_766
EXPECTED_PATIENTS = 71_518
EXPECTED_MISSING = {"race": 2_273, "weight": 98_569}
READMITTED_VALUES = {"<30", ">30", "NO"}
REQUIRED_COLUMNS = [
    "encounter_id",
    "patient_nbr",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "readmitted",
]


def load_diabetic_data(path: str) -> pd.DataFrame:
    """Lê o diabetic_data.csv preservando os campos que são bandas/códigos.

    "?" é o único marcador de dados em falta usado no ficheiro, por isso é
    convertido para NaN. age, weight e diag_1/2/3 ficam como texto porque
    representam intervalos ou códigos ICD-9, não valores numéricos.

    keep_default_na=False: em A1Cresult e max_glu_serum a string "None"
    significa "teste não realizado" (uma categoria válida), e o pandas
    converte "None" para NaN por omissão.
    """
    dtype_as_string = {
        "age": "string",
        "weight": "string",
        "diag_1": "string",
        "diag_2": "string",
        "diag_3": "string",
        "payer_code": "string",
    }
    return pd.read_csv(path, na_values="?", keep_default_na=False, dtype=dtype_as_string)


def validate_diabetic_data(df: pd.DataFrame) -> None:
    """Confirma que o dataset lido bate com o descrito no guia do projeto.

    Levanta ValueError com todos os problemas encontrados (não só o primeiro).
    """
    problems = []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Colunas em falta: {missing_cols}")

    if len(df) != EXPECTED_ROWS:
        problems.append(f"linhas: {len(df)} (esperado {EXPECTED_ROWS})")

    n_patients = df["patient_nbr"].nunique()
    if n_patients != EXPECTED_PATIENTS:
        problems.append(f"doentes: {n_patients} (esperado {EXPECTED_PATIENTS})")

    unexpected = set(df["readmitted"].dropna().unique()) - READMITTED_VALUES
    if unexpected or df["readmitted"].isna().any():
        problems.append(f"valores inesperados em readmitted: {sorted(unexpected)} / NaN")

    for col, expected in EXPECTED_MISSING.items():
        found = int(df[col].isna().sum())
        if found != expected:
            problems.append(f"missing em {col}: {found} (esperado {expected})")

    if problems:
        raise ValueError("Validação do diabetic_data falhou: " + "; ".join(problems))


def describe_dtypes(df: pd.DataFrame) -> pd.Series:
    """Tipo de cada coluna, como texto, para registo em documentação/notebook."""
    return df.dtypes.astype(str).rename("dtype")
