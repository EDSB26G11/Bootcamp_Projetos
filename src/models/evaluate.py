"""
Métricas e avaliação em CV (T2.1–T2.4, regras 4 e 5 de docs/modelacao_base.md).

A métrica principal é a PR-AUC. O recall@k responde à pergunta do negócio:
que fração das readmissões se apanha se a equipa de follow-up só chegar aos
k% de altas com maior risco.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data.split import SEED
from src.pipeline import ModelData, cv_splits

TOP_K = (0.10, 0.20)


def _top_k(scores, k: float) -> np.ndarray:
    """Máscara dos k% de encontros com maior score.

    Empates desfeitos ao acaso (seed fixa), não pela ordem das linhas: a ordem
    do ficheiro segue o encounter_id e não pode decidir quem é sinalizado (D12).
    """
    scores = np.asarray(scores)
    n = int(round(k * len(scores)))
    perm = np.random.RandomState(SEED).permutation(len(scores))
    top = np.zeros(len(scores), dtype=bool)
    top[perm[np.argsort(-scores[perm], kind="stable")[:n]]] = True
    return top


def recall_at_k(y, scores, k: float) -> float:
    y = np.asarray(y)
    return y[_top_k(scores, k)].sum() / y.sum()


def precision_at_k(y, scores, k: float) -> float:
    return np.asarray(y)[_top_k(scores, k)].mean()


def score_metrics(y, scores) -> dict:
    """PR-AUC, ROC-AUC e recall/precisão nos top 10% e 20%."""
    out = {"pr_auc": average_precision_score(y, scores), "roc_auc": roc_auc_score(y, scores)}
    for k in TOP_K:
        pct = int(k * 100)
        out[f"recall_at_{pct}"] = recall_at_k(y, scores, k)
        out[f"precision_at_{pct}"] = precision_at_k(y, scores, k)
    return out


def predict_scores(model, X) -> np.ndarray:
    """Score de risco (probabilidade da classe 1)."""
    return model.predict_proba(X)[:, 1]


def cv_evaluate(model, data: ModelData) -> pd.DataFrame:
    """Uma linha por fold de D13 com as métricas de score_metrics.

    O modelo é clonado e ajustado em cada fold; o teste não é usado.
    """
    rows = []
    for k, (tr, va) in enumerate(cv_splits(data)):
        m = clone(model).fit(data.X_train.iloc[tr], data.y_train.iloc[tr])
        scores = predict_scores(m, data.X_train.iloc[va])
        rows.append({"fold": k, **score_metrics(data.y_train.iloc[va], scores)})
    return pd.DataFrame(rows).set_index("fold")


def summarize(folds: pd.DataFrame) -> pd.Series:
    """Média e desvio dos folds: pr_auc_mean, pr_auc_std, ..."""
    return pd.Series({f"{c}_{s}": getattr(folds[c], s)() for c in folds.columns for s in ("mean", "std")})


def metrics_by_group(y, scores, groups, k: float = 0.10, min_n: int = 100) -> pd.DataFrame:
    """Métricas por grupo (idade, raça, sexo) para a fairness (T3.5).

    O top k% é calculado na população toda (a equipa de follow-up é só uma),
    e depois vê-se, em cada grupo, a taxa de sinalizados e o recall.
    """
    df = pd.DataFrame({"y": np.asarray(y), "s": np.asarray(scores), "g": np.asarray(groups),
                       "flag": _top_k(scores, k)})
    rows = []
    for g, d in df.groupby("g", observed=True):
        if len(d) < min_n:
            continue
        rows.append({"group": g, "n": len(d), "positive_rate": d["y"].mean(),
                     "flagged_rate": d["flag"].mean(),
                     "recall": d.loc[d["y"] == 1, "flag"].mean(),
                     "precision": d.loc[d["flag"], "y"].mean() if d["flag"].any() else np.nan,
                     "pr_auc": average_precision_score(d["y"], d["s"]) if d["y"].nunique() > 1 else np.nan})
    return pd.DataFrame(rows).set_index("group")
