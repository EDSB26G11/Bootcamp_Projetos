import os

import pandas as pd
import pytest

from src.data import quality as q
from src.data.load import load_diabetic_data
from src.data.mappings import parse_ids_mapping

RAW_DATA = "data/raw/diabetic_data.csv"
IDS_MAPPING = "data/raw/IDS_mapping.csv"

BASE_ROW = {
    "encounter_id": 1,
    "patient_nbr": 1,
    "race": "Caucasian",
    "gender": "Female",
    "age": "[70-80)",
    "weight": pd.NA,
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 3,
    "medical_specialty": "InternalMedicine",
    "num_lab_procedures": 40,
    "num_procedures": 1,
    "num_medications": 10,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "diag_1": "250.83",
    "diag_2": "428",
    "diag_3": "401",
    "number_diagnoses": 5,
    "max_glu_serum": "None",
    "A1Cresult": "None",
    "metformin": "No",
    "metformin-pioglitazone": "No",
    "change": "No",
    "diabetesMed": "Yes",
    "readmitted": "NO",
}


def make(*overrides):
    """Um encontro por dicionário, com os valores de BASE_ROW por omissão."""
    rows = [{**BASE_ROW, "encounter_id": i + 1, **o} for i, o in enumerate(overrides)]
    df = pd.DataFrame(rows)
    for c in q.DIAG_COLS:
        df[c] = df[c].astype("string")
    return df


def flags(func, *overrides):
    return func(make(*overrides)).tolist()


def test_newborn_birth_and_neonate_need_infant():
    assert flags(q.newborn_not_infant,
                 {"admission_type_id": 4}, {"admission_type_id": 4, "age": "[0-10)"}) == [True, False]
    assert flags(q.birth_source_not_infant,
                 {"admission_source_id": 11}, {"admission_source_id": 11, "age": "[0-10)"}) == [True, False]
    assert flags(q.neonate_discharge_not_infant,
                 {"discharge_disposition_id": 10}, {"discharge_disposition_id": 10, "age": "[0-10)"}) == [True, False]


def test_pediatrics_adult():
    assert flags(q.pediatrics_adult,
                 {"medical_specialty": "Pediatrics-Endocrinology"},
                 {"medical_specialty": "Surgery-Pediatric", "age": "[10-20)"},
                 {"medical_specialty": pd.NA}) == [True, False, False]


def test_sex_and_age_specific_diagnoses():
    assert flags(q.obstetric_male,
                 {"gender": "Male", "medical_specialty": "ObstetricsandGynecology"},
                 {"gender": "Male", "diag_2": "648"},
                 {"gender": "Female", "diag_2": "648"}) == [True, True, False]
    assert flags(q.pregnancy_60_plus,
                 {"diag_3": "648.83"}, {"diag_3": "648.83", "age": "[30-40)"}) == [True, False]
    assert flags(q.prostate_female,
                 {"diag_1": "600"}, {"diag_1": "600", "gender": "Male"}, {"diag_1": "V57"}) == [True, False, False]


def test_diagnosis_structure():
    assert flags(q.diag_order_gap, {"diag_1": pd.NA}, {"diag_1": pd.NA, "diag_2": pd.NA, "diag_3": pd.NA}) == [True, False]
    assert flags(q.diag_repeated, {"diag_3": "428"}, {}, {"diag_2": pd.NA, "diag_3": pd.NA}) == [True, False, False]
    assert flags(q.number_diagnoses_too_low, {"number_diagnoses": 2}, {"number_diagnoses": 3}) == [True, False]


def test_icd9_format():
    assert flags(q.icd9_malformed,
                 {"diag_1": "8"}, {"diag_1": "V57", "diag_2": "E888", "diag_3": "250.01"}) == [True, False]


def test_out_of_domain():
    assert flags(q.out_of_domain,
                 {"metformin": "Maybe"}, {"time_in_hospital": 15}, {"age": "70"}, {}) == [True, True, True, False]
    mappings = {"admission_type_id": pd.DataFrame({"admission_type_id": [1, 2]})}
    assert q.out_of_domain(make({"admission_type_id": 9}, {}), mappings).tolist() == [True, False]


def test_duplicated_rows_ignore_ids():
    assert flags(q.duplicated_rows, {"patient_nbr": 1}, {"patient_nbr": 2}, {"age": "[60-70)"}) == [True, True, False]


def test_no_diabetes_code():
    assert flags(q.no_diabetes_code, {}, {"diag_1": "428"}) == [False, True]


def test_patient_demographics_consistency():
    assert flags(q.patient_gender_inconsistent,
                 {"gender": "Male"}, {}, {"patient_nbr": 2}) == [True, True, False]
    assert flags(q.patient_race_inconsistent,
                 {"race": "Other"}, {}, {"patient_nbr": 2, "race": pd.NA}) == [True, True, False]


def test_age_sequence_uses_encounter_order():
    # encounter_id 1 e 2 do doente 1, por ordem inversa nas linhas
    df = make({"encounter_id": 2, "age": "[60-70)"}, {"encounter_id": 1, "age": "[70-80)"})
    assert q.age_decreases(df).tolist() == [True, False]
    assert flags(q.age_jump, {"age": "[50-60)"}, {"age": "[70-80)"}, {"age": "[80-90)"}) == [False, True, False]


def test_encounter_after_death():
    assert flags(q.encounter_after_death,
                 {}, {"discharge_disposition_id": 11}, {}, {}) == [False, False, True, True]


def test_readmitted_vs_next_encounter():
    rows = ({"readmitted": "NO"}, {"readmitted": "<30"}, {"patient_nbr": 2, "readmitted": "<30"})
    assert flags(q.no_readmission_but_next, *rows) == [True, False, False]
    assert flags(q.readmitted_without_next, *rows) == [False, True, True]


def test_run_checks_summary():
    out = q.run_checks(make({"admission_type_id": 4}, {}))
    assert list(out.index) == [c.id for c in q.CHECKS]
    assert out.loc["A1", "n"] == 1
    assert out.loc["A1", "%"] == 50.0


@pytest.mark.skipif(not os.path.exists(RAW_DATA), reason="data/raw não disponível")
def test_real_file_counts():
    df = load_diabetic_data(RAW_DATA)
    out = q.run_checks(df, parse_ids_mapping(IDS_MAPPING))
    expected = {
        "A1": 9, "A2": 5, "A3": 6, "A4": 151, "A5": 0, "A6": 0, "A7": 0, "A8": 20,
        "A9": 5_492, "A10": 0, "B1": 5_567, "B2": 0, "B3": 0, "B4": 63_742,
        "C1": 7, "C2": 404, "C3": 18, "C4": 1, "C5": 14, "C6": 490, "C7": 3_224,
    }
    assert out["n"].to_dict() == expected
