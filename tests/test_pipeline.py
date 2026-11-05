import os

import numpy as np
import pandas as pd
import pytest

from src.features.build import CATEGORICAL_FEATURES, DROPPED, FEATURES
from src.pipeline import PROCESSED, SPLITS, build_preprocessor, cv_splits, load_model_data, read_processed

HAS_DATA = os.path.exists(PROCESSED) and os.path.exists(SPLITS)


def _frame(n, specialty):
    rng = np.random.RandomState(0)
    x = pd.DataFrame({c: rng.randint(0, 5, n) for c in FEATURES if c not in CATEGORICAL_FEATURES})
    for c in CATEGORICAL_FEATURES:
        x[c] = "A"
    x["medical_specialty"] = specialty
    return x


def test_preprocessor_handles_unseen_and_rare_categories():
    train = _frame(300, ["Common"] * 250 + ["Rare"] * 50)
    pre = build_preprocessor().fit(train)
    names = list(pre.get_feature_names_out())
    assert "medical_specialty_Common" in names
    assert "medical_specialty_Rare" not in names  # < 100 no treino: junta em infrequent
    out = pre.transform(_frame(2, ["Never seen", "Common"]))
    assert not np.isnan(out).any()


@pytest.mark.skipif(not HAS_DATA, reason="data/processed ou data/splits não gerados")
def test_all_processed_columns_are_accounted_for():
    cols = set(read_processed().columns)
    known = set(FEATURES) | set(DROPPED) | {"encounter_id", "patient_nbr", "readmitted", "target"}
    assert cols - known == set()


@pytest.mark.skipif(not HAS_DATA, reason="data/processed ou data/splits não gerados")
def test_model_data_matches_split():
    data = load_model_data()
    assert len(data.X_train) == 80_041 and len(data.X_test) == 20_011
    assert list(data.X_train.columns) == FEATURES
    assert not data.X_train.isna().any().any()
    assert abs(data.y_train.mean() - data.y_test.mean()) < 0.001
    for tr, va in cv_splits(data):
        assert not set(data.groups_train.iloc[tr]) & set(data.groups_train.iloc[va])
