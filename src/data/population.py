"""
Definição da população elegível (T1.4).
"""

import pandas as pd

# discharge_disposition_id de óbito (11, 19, 20, 21). Um doente que morreu não
# pode ser readmitido. Os hospícios (13, 14) ficam na população: o doente
# pode voltar ao hospital. Ver docs/decisions.md (D1).
EXCLUDED_DISCHARGE_IDS = (11, 19, 20, 21)


def exclude_deaths(df: pd.DataFrame) -> pd.DataFrame:
    """Remove os encontros com alta por óbito."""
    keep = ~df["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS)
    return df[keep].reset_index(drop=True)
