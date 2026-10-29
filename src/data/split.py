"""
Split agrupado por doente (T1.7, D13).
"""

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
TEST_FOLDS = 5  # 1 de 5 partes = 20% dos doentes para teste
CV_FOLDS = 5


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por encontro: split ("train"/"test") e fold de CV (0-4; -1 no teste).

    Teste e folds usam StratifiedGroupKFold por patient_nbr, estratificado pelo
    target: nenhum doente aparece em duas partes e todas ficam com a mesma taxa
    de positivos. O teste só se usa uma vez, na avaliação final (T3.6).
    """
    out = df[["encounter_id", "patient_nbr", "target"]].reset_index(drop=True)
    outer = StratifiedGroupKFold(n_splits=TEST_FOLDS, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(outer.split(out, out["target"], out["patient_nbr"]))
    out["split"] = "train"
    out.loc[test_idx, "split"] = "test"
    out["fold"] = -1

    train = out.loc[train_idx]
    inner = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, val) in enumerate(inner.split(train, train["target"], train["patient_nbr"])):
        out.loc[train.index[val], "fold"] = k
    return out[["encounter_id", "patient_nbr", "split", "fold"]]
