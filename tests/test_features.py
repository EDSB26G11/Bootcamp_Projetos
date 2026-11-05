import pandas as pd

from src.features.build import (
    CATEGORICAL_FEATURES,
    DROPPED,
    FEATURES,
    MED_COLS,
    build_features,
    icd9_group,
    model_frame,
)


def _row(**kw):
    base = {"age": "[70-80)", "diag_1": "428", "diag_2": "250.43", "diag_3": None,
            "number_outpatient": 1, "number_emergency": 0, "number_inpatient": 2}
    base.update({m: "No" for m in MED_COLS})
    base.update(kw)
    return base


def test_icd9_group_follows_strack():
    codes = pd.Series(["428", "786.5", "250.01", "820", "V57", "E812", "008", None], dtype="string")
    assert icd9_group(codes).tolist() == [
        "Circulatory", "Respiratory", "Diabetes", "Injury", "Other", "Other", "Other", "Missing"]


def test_build_features_row_level():
    df = pd.DataFrame([_row(insulin="Up", metformin="Steady", glipizide="Down"),
                       _row(age="[0-10)", diag_2="250.01")])
    out = build_features(df)
    assert out["age_band"].tolist() == [7, 0]
    assert out["diag_2_group"].tolist() == ["Diabetes", "Diabetes"]
    assert out["diabetes_complicated"].tolist() == [1, 0]
    assert out["n_meds"].tolist() == [3, 0]
    assert out["n_meds_up"].tolist() == [1, 0]
    assert out["n_meds_down"].tolist() == [1, 0]
    assert out["n_visits_prior"].tolist() == [3, 3]


def test_features_and_dropped_do_not_overlap():
    assert not set(FEATURES) & set(DROPPED)
    assert len(FEATURES) == len(set(FEATURES))


def test_model_frame_fills_missing_categories():
    df = pd.DataFrame([_row()])
    for c in ["race", "gender", "payer_code", "medical_specialty", "A1Cresult", "max_glu_serum",
              "admission_type_id_desc", "discharge_disposition_id_desc", "admission_source_id_desc"]:
        df[c] = None
    for c in ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
              "number_diagnoses"]:
        df[c] = 1
    x = model_frame(df)
    assert list(x.columns) == FEATURES
    assert (x[CATEGORICAL_FEATURES[:4]] == "Missing").all().all()
