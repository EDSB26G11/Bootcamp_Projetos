"""
Pipeline reprodutível de pré-processamento (T1.10, D16).

Junta o ficheiro processado, o split por doente (D13) e as features (T1.8/T1.9)
e devolve um ColumnTransformer que só é ajustado no treino. Uso típico:

    data = load_model_data()
    pipe = make_pipeline(build_preprocessor(), LogisticRegression(max_iter=1000))
    pipe.fit(data.X_train, data.y_train)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features.build import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    COUNT_FEATURES,
    NUMERIC_FEATURES,
    model_frame,
)

PROCESSED = "data/processed/diabetic_data_clean.csv"
SPLITS = "data/splits/splits.csv"
TEXT_COLUMNS = ["age", "weight", "payer_code", "diag_1", "diag_2", "diag_3"]

# Categorias com menos encontros do que isto no treino juntam-se numa só
# ("infrequent"). Resolve os níveis raros de medical_specialty sem usar o teste.
MIN_CATEGORY_COUNT = 100


@dataclass
class ModelData:
    X_train: pd.DataFrame
    y_train: pd.Series
    groups_train: pd.Series  # patient_nbr, para CV agrupada
    folds_train: pd.Series   # fold 0-4 de D13
    X_test: pd.DataFrame
    y_test: pd.Series


def read_processed(path: str = PROCESSED) -> pd.DataFrame:
    """Relê o CSV processado sem perder "None", "NULL" nem zeros à esquerda."""
    return pd.read_csv(path, keep_default_na=False, na_values="",
                       dtype={c: "string" for c in TEXT_COLUMNS})


def load_model_data(processed: str = PROCESSED, splits: str = SPLITS) -> ModelData:
    """Features e target separados pelo split por doente de data/splits/."""
    df = read_processed(processed).merge(
        pd.read_csv(splits)[["encounter_id", "split", "fold"]], on="encounter_id", validate="1:1"
    )
    X = model_frame(df)
    train = df["split"].eq("train").to_numpy()
    return ModelData(
        X_train=X[train].reset_index(drop=True),
        y_train=df.loc[train, "target"].reset_index(drop=True),
        groups_train=df.loc[train, "patient_nbr"].reset_index(drop=True),
        folds_train=df.loc[train, "fold"].reset_index(drop=True),
        X_test=X[~train].reset_index(drop=True),
        y_test=df.loc[~train, "target"].reset_index(drop=True),
    )


def build_preprocessor() -> ColumnTransformer:
    """Numéricas escaladas, contagens com log1p, categóricas em one-hot.

    Serve para modelos lineares e de árvores (a escala não muda as árvores).
    """
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("counts", make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                                     StandardScaler()), COUNT_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=MIN_CATEGORY_COUNT,
                                  sparse_output=False), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def cv_splits(data: ModelData):
    """Pares (treino, validação) dos 5 folds de D13, para cross_validate(cv=...)."""
    folds = data.folds_train.to_numpy()
    return [(np.where(folds != k)[0], np.where(folds == k)[0]) for k in sorted(set(folds))]
