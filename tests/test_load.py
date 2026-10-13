import os

import pandas as pd
import pytest

from src.data.load import (
    EXPECTED_PATIENTS,
    EXPECTED_ROWS,
    describe_dtypes,
    load_diabetic_data,
    validate_diabetic_data,
)

RAW_DATA = "data/raw/diabetic_data.csv"


def test_load_converts_question_mark_and_keeps_text(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text(
        "encounter_id,patient_nbr,race,age,weight,diag_1,payer_code\n"
        "1,10,?,[0-10),?,250.83,MC\n"
        "2,11,Caucasian,[10-20),[75-100),V57,?\n"
    )
    df = load_diabetic_data(str(csv))
    assert df["race"].isna().tolist() == [True, False]
    assert df["weight"].isna().tolist() == [True, False]
    assert df["age"].iloc[0] == "[0-10)"
    assert df["diag_1"].iloc[0] == "250.83"
    assert pd.api.types.is_string_dtype(df["diag_1"])


def test_none_is_kept_as_not_measured(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text("A1Cresult,max_glu_serum,race\nNone,None,?\n>8,Norm,Caucasian\n")
    df = load_diabetic_data(str(csv))
    assert df["A1Cresult"].tolist() == ["None", ">8"]
    assert df["max_glu_serum"].iloc[0] == "None"
    assert df["race"].isna().tolist() == [True, False]


def test_validate_reports_missing_columns():
    with pytest.raises(ValueError, match="Colunas em falta"):
        validate_diabetic_data(pd.DataFrame({"a": [1]}))


def test_validate_reports_wrong_counts():
    df = pd.DataFrame(
        {
            "encounter_id": [1],
            "patient_nbr": [1],
            "admission_type_id": [1],
            "discharge_disposition_id": [1],
            "admission_source_id": [1],
            "readmitted": ["NO"],
            "race": ["x"],
            "weight": ["y"],
        }
    )
    with pytest.raises(ValueError, match="linhas") as exc:
        validate_diabetic_data(df)
    assert "doentes" in str(exc.value)


@pytest.mark.skipif(not os.path.exists(RAW_DATA), reason="data/raw não disponível")
def test_real_file_passes_validation():
    df = load_diabetic_data(RAW_DATA)
    validate_diabetic_data(df)
    assert len(df) == EXPECTED_ROWS
    assert df["patient_nbr"].nunique() == EXPECTED_PATIENTS
    assert len(describe_dtypes(df)) == df.shape[1]
