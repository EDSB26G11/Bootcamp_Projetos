import os

import pandas as pd
import pytest

from src.data.split import assign_splits

PROCESSED = "data/processed/diabetic_data_clean.csv"


def _toy(n_patients=200):
    rows = []
    for p in range(n_patients):
        for e in range(1 + p % 3):
            rows.append({"encounter_id": len(rows), "patient_nbr": p, "target": int((p + e) % 7 == 0)})
    return pd.DataFrame(rows)


def test_no_patient_in_two_parts():
    s = assign_splits(_toy())
    parts = s.assign(part=s["split"] + s["fold"].astype(str))
    assert (parts.groupby("patient_nbr")["part"].nunique() == 1).all()
    assert set(s.loc[s["split"] == "test", "fold"]) == {-1}
    assert set(s.loc[s["split"] == "train", "fold"]) == {0, 1, 2, 3, 4}


def test_is_deterministic():
    assert assign_splits(_toy()).equals(assign_splits(_toy()))


@pytest.mark.skipif(not os.path.exists(PROCESSED), reason="data/processed não gerado")
def test_real_split_is_stratified():
    df = pd.read_csv(PROCESSED, usecols=["encounter_id", "patient_nbr", "target"])
    s = assign_splits(df).merge(df[["encounter_id", "target"]], on="encounter_id")
    assert abs((s["split"] == "test").mean() - 0.2) < 0.01
    rates = s.groupby(["split", "fold"])["target"].mean()
    assert (rates - df["target"].mean()).abs().max() < 0.005
