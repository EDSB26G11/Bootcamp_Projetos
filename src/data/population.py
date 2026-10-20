"""
Definição da população elegível (T1.4).
"""

import pandas as pd

# discharge_disposition_id de óbito (11, 19, 20, 21) e hospício (13, 14).
# Um doente que morreu não pode ser readmitido e no hospício o objetivo dos
# cuidados não é evitar o regresso ao hospital. Ver docs/decisions.md (D1).
EXCLUDED_DISCHARGE_IDS = (11, 13, 14, 19, 20, 21)


def exclude_dead_hospice(df: pd.DataFrame) -> pd.DataFrame:
    """Remove os encontros com alta por óbito ou para hospício."""
    keep = ~df["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS)
    return df[keep].reset_index(drop=True)
