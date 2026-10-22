import os

import pandas as pd
import pytest

from src.data.load import load_diabetic_data
from src.data.population import EXCLUDED_DISCHARGE_IDS, exclude_deaths

RAW_DATA = "data/raw/diabetic_data.csv"


def test_excludes_deaths_only():
    df = pd.DataFrame({"discharge_disposition_id": [1, 11, 13, 14, 19, 20, 21, 3]})
    out = exclude_deaths(df)
    assert out["discharge_disposition_id"].tolist() == [1, 13, 14, 3]
    assert out.index.tolist() == [0, 1, 2, 3]


@pytest.mark.skipif(not os.path.exists(RAW_DATA), reason="data/raw não disponível")
def test_real_file_population():
    df = exclude_deaths(load_diabetic_data(RAW_DATA))
    assert len(df) == 100_114
    assert df["patient_nbr"].nunique() == 70_439
    assert not df["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS).any()
