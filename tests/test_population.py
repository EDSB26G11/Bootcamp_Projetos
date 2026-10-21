import os

import pandas as pd
import pytest

from src.data.load import load_diabetic_data
from src.data.population import EXCLUDED_DISCHARGE_IDS, exclude_dead_hospice

RAW_DATA = "data/raw/diabetic_data.csv"


def test_excludes_dead_and_hospice_only():
    df = pd.DataFrame({"discharge_disposition_id": [1, 11, 13, 14, 19, 20, 21, 3]})
    out = exclude_dead_hospice(df)
    assert out["discharge_disposition_id"].tolist() == [1, 3]
    assert out.index.tolist() == [0, 1]


@pytest.mark.skipif(not os.path.exists(RAW_DATA), reason="data/raw não disponível")
def test_real_file_population():
    df = exclude_dead_hospice(load_diabetic_data(RAW_DATA))
    assert len(df) == 99_343
    assert df["patient_nbr"].nunique() == 69_990
    assert not df["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS).any()
