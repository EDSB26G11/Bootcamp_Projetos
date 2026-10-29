import os

import pandas as pd
import pytest

from src.data.load import load_diabetic_data
from src.data.population import (
    DEATH_DISCHARGE_IDS,
    add_target,
    define_population,
    exclude_deaths,
    exclude_invalid_gender,
)

RAW_DATA = "data/raw/diabetic_data.csv"


def test_excludes_deaths_only():
    df = pd.DataFrame({"discharge_disposition_id": [1, 11, 13, 14, 19, 20, 21, 3]})
    out = exclude_deaths(df)
    assert out["discharge_disposition_id"].tolist() == [1, 13, 14, 3]
    assert out.index.tolist() == [0, 1, 2, 3]


def test_excludes_invalid_gender():
    df = pd.DataFrame({"gender": ["Female", "Unknown/Invalid", "Male"]})
    assert exclude_invalid_gender(df)["gender"].tolist() == ["Female", "Male"]


def test_add_target_is_under_30_vs_rest():
    df = pd.DataFrame({"readmitted": ["<30", ">30", "NO"]})
    out = add_target(df)
    assert out["target"].tolist() == [1, 0, 0]
    assert "readmitted" in out.columns


@pytest.mark.skipif(not os.path.exists(RAW_DATA), reason="data/raw não disponível")
def test_real_file_population():
    df = add_target(define_population(load_diabetic_data(RAW_DATA)))
    assert len(df) == 100_111
    assert df["patient_nbr"].nunique() == 70_436
    assert not df["discharge_disposition_id"].isin(DEATH_DISCHARGE_IDS).any()
    assert df["discharge_disposition_id"].isin([13, 14]).sum() == 771
    assert (df["gender"] != "Unknown/Invalid").all()
    assert round(100 * df["target"].mean(), 2) == 11.34


@pytest.mark.skipif(not os.path.exists(RAW_DATA), reason="data/raw não disponível")
def test_real_file_without_impossible_encounters():
    from src.data.mappings import parse_ids_mapping
    from src.data.population import exclude_impossible

    raw = load_diabetic_data(RAW_DATA)
    df = define_population(exclude_impossible(raw, parse_ids_mapping("data/raw/IDS_mapping.csv")))
    assert len(raw) - len(exclude_impossible(raw)) == 59
    assert len(df) == 100_052
    assert df["patient_nbr"].nunique() == 70_412
