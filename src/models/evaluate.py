"""
Métricas e avaliação em CV (T2.1–T2.4, regras 4 e 5 de docs/modelacao_base.md).

A métrica principal é a PR-AUC. O recall@k responde à pergunta do negócio:
que fração das readmissões se apanha se a equipa de follow-up só chegar aos
k% de altas com maior risco. Tudo o que vem depois da escolha do modelo
(threshold, importância, fairness) usa as previsões out-of-fold (OOF) do
treino; o teste só se abre na avaliação final.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

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
    """PR-AUC, ROC-AUC, recall/precisão nos top 10% e 20%, log loss e Brier."""
    out = {"pr_auc": average_precision_score(y, scores), "roc_auc": roc_auc_score(y, scores)}
    for k in TOP_K:
        pct = int(k * 100)
        out[f"recall_at_{pct}"] = recall_at_k(y, scores, k)
        out[f"precision_at_{pct}"] = precision_at_k(y, scores, k)
    # Log loss e Brier só fazem sentido para probabilidades, não para rankings.
    is_prob = np.min(scores) >= 0 and np.max(scores) <= 1
    out["log_loss"] = log_loss(y, np.clip(scores, 1e-15, 1 - 1e-15), labels=[0, 1]) if is_prob else np.nan
    out["brier"] = brier_score_loss(y, scores) if is_prob else np.nan
    return out


def predict_scores(model, X) -> np.ndarray:
    """Score de risco (probabilidade da classe 1)."""
    return model.predict_proba(X)[:, 1]


def cv_predict(model, data: ModelData):
    """Ajusta o modelo em cada fold de D13.

    Devolve as métricas por fold no treino do fold e na validação, e as
    previsões OOF: o score de cada encontro de treino quando estava na validação.
    """
    train_rows, val_rows = [], []
    oof = pd.Series(np.nan, index=data.y_train.index, name="oof_score")
    for k, (tr, va) in enumerate(cv_splits(data)):
        m = clone(model).fit(data.X_train.iloc[tr], data.y_train.iloc[tr])
        train_rows.append({"fold": k, **score_metrics(data.y_train.iloc[tr],
                                                      predict_scores(m, data.X_train.iloc[tr]))})
        scores = predict_scores(m, data.X_train.iloc[va])
        oof.iloc[va] = scores
        val_rows.append({"fold": k, **score_metrics(data.y_train.iloc[va], scores)})
    return pd.DataFrame(train_rows).set_index("fold"), pd.DataFrame(val_rows).set_index("fold"), oof


def cv_evaluate(model, data: ModelData) -> pd.DataFrame:
    """Uma linha por fold de D13 com as métricas de validação; o teste não é usado."""
    return cv_predict(model, data)[1]


def summarize(folds: pd.DataFrame) -> pd.Series:
    """Média e desvio dos folds: pr_auc_mean, pr_auc_std, ..."""
    return pd.Series({f"{c}_{s}": getattr(folds[c], s)() for c in folds.columns for s in ("mean", "std")})


def oof_permutation_importance(model, data: ModelData, n_repeats: int = 5) -> pd.DataFrame:
    """Importância por permutação de cada uma das 34 features, medida na
    validação de cada fold (as mesmas previsões das OOF) com a PR-AUC.

    importance = descida média da PR-AUC quando a coluna é baralhada;
    std = desvio dessa média entre os 5 folds.
    """
    per_fold = []
    for tr, va in cv_splits(data):
        m = clone(model).fit(data.X_train.iloc[tr], data.y_train.iloc[tr])
        r = permutation_importance(m, data.X_train.iloc[va], data.y_train.iloc[va],
                                   scoring="average_precision", n_repeats=n_repeats,
                                   random_state=SEED)
        per_fold.append(pd.Series(r.importances_mean, index=data.X_train.columns))
    imp = pd.concat(per_fold, axis=1)
    return (pd.DataFrame({"importance": imp.mean(axis=1), "std": imp.std(axis=1)})
            .sort_values("importance", ascending=False))


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


def calibration_table(y, scores, n_bins: int = 10) -> pd.DataFrame:
    """Risco previsto contra readmissões observadas, por grupos de igual
    tamanho ordenados pelo score (decis com n_bins=10).

    Responde à 1.ª parte da pergunta ("qual o risco"): se o modelo diz 20%,
    cerca de 20 em cada 100 altas desse grupo voltam em menos de 30 dias.
    Sem calibração isotónica/Platt (não dada nas aulas): só se verifica.
    """
    d = pd.DataFrame({"y": np.asarray(y), "s": np.asarray(scores)})
    d["grupo"] = pd.qcut(d["s"].rank(method="first"), n_bins, labels=range(1, n_bins + 1))
    out = d.groupby("grupo", observed=True).agg(n=("y", "size"), risco_previsto=("s", "mean"),
                                                observado=("y", "mean"))
    out["diferenca"] = out["risco_previsto"] - out["observado"]
    return out
