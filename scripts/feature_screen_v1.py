"""Triagem de duas features novas (D28). Correr a partir da raiz: python scripts/feature_screen_v1.py

Triagem de duas features novas, só no treino (5 folds por doente, D13). O teste não é usado.

1. Comorbilidades de Charlson (adaptação de Deyo, só com diag_1–3): índice ponderado e indicadores.
2. HbA1c medida × medicação alterada (Strack et al., 2014).

Filtro rápido: nos mesmos folds, uma regressão logística de y sobre o logit do score OOF do v1
(só recalibra, não muda a ordem) contra o mesmo + a feature. Se a feature não melhora fora do fold,
o v1 já tem essa informação.
Confirmação: o v1 com a feature (extra_binary, como na ablação D18), fold a fold contra o v1.
"""

import dataclasses
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss

sys.path.insert(0, ".")
from src.deck_figures import load_train  # noqa: E402
from src.models.baselines import hist_gradient_boosting  # noqa: E402
from src.models.evaluate import cv_predict, recall_at_k  # noqa: E402
from src.pipeline import load_model_data  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "reports/feature_screen_v1.csv"

# Charlson (Deyo, 1992), categorias por código ICD-9 de 3 dígitos; peso; a diabetes fica de fora
# porque todos os doentes a têm.
CHARLSON = {
    "enfarte": ([(410, 410), (412, 412)], 1),
    "insuf_cardiaca": ([(428, 428)], 1),
    "vascular_periferica": ([(440, 441), (443, 443)], 1),
    "cerebrovascular": ([(430, 438)], 1),
    "demencia": ([(290, 290)], 1),
    "pulmonar_cronica": ([(490, 496), (500, 505)], 1),
    "reumatologica": ([(710, 710), (714, 714), (725, 725)], 1),
    "ulcera_peptica": ([(531, 534)], 1),
    "figado_ligeira": ([(571, 571)], 1),
    "hemiplegia": ([(342, 342), (344, 344)], 2),
    "renal": ([(582, 583), (585, 586), (588, 588)], 2),
    "cancro": ([(140, 172), (174, 195), (200, 208)], 2),
    "figado_grave": ([(572, 572), (456, 456)], 3),
    "metastases": ([(196, 199)], 6),
    "sida": ([(42, 44)], 6),
}


def icd3(s: pd.Series) -> pd.Series:
    """Parte inteira de um código ICD-9 numérico (V/E e em falta ficam NaN)."""
    x = s.astype("string").str.extract(r"^(\d{3})", expand=False)
    return pd.to_numeric(x, errors="coerce")


def features(df: pd.DataFrame) -> pd.DataFrame:
    codes = pd.concat([icd3(df[c]) for c in ("diag_1", "diag_2", "diag_3")], axis=1)
    out = pd.DataFrame(index=df.index)
    score = np.zeros(len(df))
    for name, (ranges, w) in CHARLSON.items():
        hit = np.zeros(len(df), dtype=bool)
        for lo, hi in ranges:
            hit |= ((codes >= lo) & (codes <= hi)).any(axis=1).to_numpy()
        out[f"ch_{name}"] = hit.astype(int)
        score += w * hit
    out["charlson"] = score
    measured = df["A1Cresult"].astype(str).ne("None")
    out["a1c_medida_x_mudanca"] = (measured & df["change"].eq("Ch")).astype(int)
    return out


tr = load_train()  # treino, pela ordem de load_model_data, com o OOF do v1
F = features(tr)
y = tr["target"].to_numpy()
oof = tr["oof"].to_numpy()
elig = tr["eligible"].to_numpy()
fold = tr["fold"].to_numpy()
logit = np.log(oof / (1 - oof))
print(
    "prevalência:",
    {
        c: round(F[c].mean(), 4)
        for c in ["ch_insuf_cardiaca", "ch_renal", "ch_pulmonar_cronica", "ch_cancro", "a1c_medida_x_mudanca"]
    },
    "charlson>0:",
    round((F["charlson"] > 0).mean(), 3),
)

SETS = {
    "Charlson (índice)": ["charlson"],
    "Charlson (índice + 15 indicadores)": ["charlson"] + [c for c in F if c.startswith("ch_")],
    "HbA1c medida × medicação alterada": ["a1c_medida_x_mudanca"],
}
rows = []
for name, cols in SETS.items():
    for k in np.unique(fold):
        trn, val = fold != k, fold == k
        res = {}
        for tag, X in [("base", logit[:, None]), ("com", np.column_stack([logit, F[cols].to_numpy()]))]:
            m = LogisticRegression(max_iter=1000).fit(X[trn], y[trn])
            p = m.predict_proba(X[val])[:, 1]
            res[tag] = (
                log_loss(y[val], p),
                average_precision_score(y[val], p),
                recall_at_k(y[val][elig[val]], p[elig[val]], 0.30),
            )
        rows.append(
            {
                "etapa": "filtro",
                "feature": name,
                "fold": k,
                "d_logloss": res["com"][0] - res["base"][0],
                "d_pr_auc": res["com"][1] - res["base"][1],
                "d_recall30": 100 * (res["com"][2] - res["base"][2]),
            }
        )

# Taxa observada por decil de risco do v1, com e sem insuficiência cardíaca (leitura do filtro).
tr["decil"] = pd.qcut(tr["oof"].rank(method="first"), 10, labels=range(1, 11))
tab = tr.assign(chf=F["ch_insuf_cardiaca"]).groupby(["decil", "chf"], observed=True)["target"].agg(["mean", "size"])
print(tab.unstack().round(3).to_string())

# Confirmação: o v1 com as features (todas as candidatas, para não depender do filtro).
data = load_model_data()
assert len(data.X_train) == len(tr)
for name, cols in [("v1", [])] + list(SETS.items()):
    X = data.X_train.copy()
    for c in cols:
        X[c] = F[c].to_numpy()
    d = dataclasses.replace(data, X_train=X)
    _, val, o = cv_predict(hist_gradient_boosting(extra_binary=cols), d)
    o = o.to_numpy()
    for k in np.unique(fold):
        v = fold == k
        rows.append(
            {
                "etapa": "ablacao",
                "feature": name,
                "fold": k,
                "pr_auc": average_precision_score(y[v], o[v]),
                "recall30": recall_at_k(y[v][elig[v]], o[v][elig[v]], 0.30),
            }
        )
    print("ablação feita:", name, flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
print("gravado", OUT)
