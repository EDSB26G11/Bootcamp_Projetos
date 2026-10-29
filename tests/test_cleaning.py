import pandas as pd

from src.data.cleaning import harmonize_race, pad_icd9


def test_harmonize_race_uses_mode():
    df = pd.DataFrame({
        "encounter_id": [1, 2, 3, 4, 5],
        "patient_nbr": [10, 10, 10, 20, 20],
        "race": ["Caucasian", "Caucasian", "Other", "Asian", None],
    })
    out = harmonize_race(df)
    assert out["race"].tolist()[:3] == ["Caucasian"] * 3
    # Doente 20 só tem uma race: não mexe, nem no missing.
    assert out["race"].iloc[3] == "Asian"
    assert pd.isna(out["race"].iloc[4])


def test_harmonize_race_tie_uses_latest_encounter():
    df = pd.DataFrame({
        "encounter_id": [5, 9, 7],
        "patient_nbr": [10, 10, 10],
        "race": ["Caucasian", "AfricanAmerican", None],
    })
    assert harmonize_race(df)["race"].tolist() == ["AfricanAmerican"] * 3


def test_pad_icd9_only_short_numeric_codes():
    df = pd.DataFrame({
        "diag_1": ["8", "38", "250.83", "V57", "E812"],
        "diag_2": ["41.1", "414", None, "9", "E9"],
        "diag_3": ["3", "8.5", "428", "V4", "250"],
    })
    out = pad_icd9(df)
    assert out["diag_1"].tolist() == ["008", "038", "250.83", "V57", "E812"]
    assert out["diag_2"].tolist()[:2] == ["041.1", "414"]
    assert pd.isna(out["diag_2"].iloc[2])
    assert out["diag_2"].tolist()[3:] == ["009", "E9"]
    assert out["diag_3"].tolist() == ["003", "008.5", "428", "V4", "250"]
