"""
Ponto de corte pela capacidade da equipa de follow-up (T3.3).

Nesta fase compara-se a ordenação; o threshold só se fixa depois de escolhido
o modelo, a partir da fração de altas que a equipa consegue contactar.
"""

import numpy as np
import pandas as pd

CAPACITIES = (0.05, 0.10, 0.15, 0.20, 0.30)


def threshold_for_capacity(scores, capacity: float) -> float:
    """Score mínimo para sinalizar a fração `capacity` de altas com maior risco."""
    return float(np.quantile(np.asarray(scores), 1 - capacity))


def capacity_table(y, scores, capacities=CAPACITIES) -> pd.DataFrame:
    """Para cada capacidade: threshold, altas sinalizadas, recall, precisão e
    readmissões apanhadas por cada 100 contactos."""
    y, scores = np.asarray(y), np.asarray(scores)
    rows = []
    for c in capacities:
        t = threshold_for_capacity(scores, c)
        flag = scores >= t
        tp = (flag & (y == 1)).sum()
        rows.append({"capacity": c, "threshold": t, "flagged": flag.mean(),
                     "recall": tp / y.sum(), "precision": tp / max(flag.sum(), 1),
                     "caught_per_100_calls": 100 * tp / max(flag.sum(), 1)})
    return pd.DataFrame(rows).set_index("capacity")
