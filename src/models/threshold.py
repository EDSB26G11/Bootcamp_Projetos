"""
Ponto de corte pela capacidade da equipa de follow-up (T3.3).

Um só corte por capacidade k (sem tiers): sinaliza-se a fração k de altas com
maior risco. Calcula-se sobre as previsões OOF do treino; o teste só serve
para confirmar, na avaliação final, que o corte sinaliza ≈ k das altas.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.data.split import SEED

CAPACITIES = (0.10, 0.20)


def threshold_for_capacity(scores, capacity: float) -> float:
    """Score mínimo para sinalizar a fração `capacity` de altas com maior risco."""
    return float(np.quantile(np.asarray(scores), 1 - capacity))


def capacity_table(y, scores, capacities=CAPACITIES) -> pd.DataFrame:
    """Para cada capacidade k: threshold, altas sinalizadas, recall, precisão,
    lift face à taxa base e readmissões encontradas por 100 follow-ups e por
    1 000 altas."""
    y, scores = np.asarray(y), np.asarray(scores)
    base_rate = y.mean()
    rows = []
    for k in capacities:
        t = threshold_for_capacity(scores, k)
        flag = scores >= t
        tp = (flag & (y == 1)).sum()
        precision = tp / max(flag.sum(), 1)
        rows.append({"k": k, "threshold": t, "n_sinalizados": int(flag.sum()),
                     "pct_sinalizados": flag.mean(), "recall": tp / y.sum(),
                     "precision": precision, "lift": precision / base_rate,
                     "readm_por_100_followups": 100 * precision,
                     "readm_encontradas_por_1000_altas": 1000 * flag.mean() * precision})
    return pd.DataFrame(rows).set_index("k")


def _top_k_draw(values, k: float, rng) -> np.ndarray:
    """Top k% por valor, com os empates desfeitos por um sorteio de rng."""
    values = np.asarray(values, dtype=float)
    n = int(round(k * len(values)))
    perm = rng.permutation(len(values))
    top = np.zeros(len(values), dtype=bool)
    top[perm[np.argsort(-values[perm], kind="stable")[:n]]] = True
    return top


def rank_baseline_at_k(y, number_inpatient, k: float, n_draws: int = 20, seed: int = SEED) -> dict:
    """Baseline à mesma capacidade: ordenar por number_inpatient e sinalizar o
    top k%, com os (muitos) empates desfeitos ao acaso. Média e desvio de
    n_draws sorteios."""
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    rec, prec = [], []
    for _ in range(n_draws):
        top = _top_k_draw(number_inpatient, k, rng)
        rec.append(y[top].sum() / y.sum())
        prec.append(y[top].mean())
    return {"k": k, "recall_mean": np.mean(rec), "recall_std": np.std(rec),
            "precision_mean": np.mean(prec), "precision_std": np.std(prec)}


def confusion_at_k(y, scores, k: float) -> dict:
    """Matriz de confusão com o corte da capacidade k (sinalizado = previsto 1)."""
    y, scores = np.asarray(y), np.asarray(scores)
    flag = (scores >= threshold_for_capacity(scores, k)).astype(int)
    m = confusion_matrix(y, flag, labels=[0, 1])
    tn, fp, fn, tp = m.ravel()
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn), "matrix": m}


def gains_curve(y, scores, ks=None, n_draws: int = 1, seed: int = SEED) -> pd.DataFrame:
    """Percentagem de readmissões apanhadas ao sinalizar o top k%, para k de 1%
    a 100%. Com scores empatados (ex.: number_inpatient), faz a média de
    n_draws sorteios."""
    y = np.asarray(y)
    ks = np.arange(1, 101) / 100 if ks is None else np.asarray(ks)
    rng = np.random.RandomState(seed)
    caught = np.zeros(len(ks))
    for _ in range(n_draws):
        caught += [y[_top_k_draw(scores, k, rng)].sum() / y.sum() for k in ks]
    return pd.DataFrame({"k": ks, "pct_readm_apanhadas": 100 * caught / n_draws})
