"""Medição do tuning do v1 (D27). Correr a partir da raiz: python scripts/tuning_v1.py

Medição do tuning do v1, só no treino (5 folds por doente, D13). O teste não é usado.

Para cada configuração treina até MAX_ITER iterações e mede, de STEP em STEP iterações,
a PR-AUC de validação (todas as altas) e o recall@30% nas altas elegíveis (D21, D26).
Isto é o early stopping feito com os folds por doente.
"""

import sys
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline

sys.path.insert(0, ".")
from src.data.population import follow_up_eligible  # noqa: E402
from src.data.split import SEED  # noqa: E402
from src.models.evaluate import recall_at_k  # noqa: E402
from src.pipeline import build_preprocessor, cv_splits, load_model_data  # noqa: E402

MAX_ITER, STEP = 600, 25
CONFIGS = {  # v1 = os parâmetros atuais (src/models/baselines.py)
    "v1 (31 folhas, msl 100, l2 1)": dict(max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=1.0),
    "15 folhas, msl 200, l2 1": dict(max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0),
    "7 folhas, msl 200, l2 1": dict(max_leaf_nodes=7, min_samples_leaf=200, l2_regularization=1.0),
    "31 folhas, msl 300, l2 5": dict(max_leaf_nodes=31, min_samples_leaf=300, l2_regularization=5.0),
}
OUT = sys.argv[1] if len(sys.argv) > 1 else "reports/tuning_v1.csv"

data = load_model_data()
X, y = data.X_train, data.y_train.to_numpy()
eligible = follow_up_eligible(X).to_numpy()
rows = []
for name, params in CONFIGS.items():
    t0 = time.time()
    for k, (tr, va) in enumerate(cv_splits(data)):
        pre = build_preprocessor().fit(X.iloc[tr])
        Xtr, Xva = pre.transform(X.iloc[tr]), pre.transform(X.iloc[va])
        clf = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=MAX_ITER, early_stopping=False, random_state=SEED, **params
        ).fit(Xtr, y[tr])
        yv, ev = y[va], eligible[va]
        for it, p in enumerate(clf.staged_predict_proba(Xva), start=1):
            if it % STEP:
                continue
            s = p[:, 1]
            rows.append(
                {
                    "config": name,
                    "fold": k,
                    "iter": it,
                    "pr_auc": average_precision_score(yv, s),
                    "recall30_eleg": recall_at_k(yv[ev], s[ev], 0.30),
                }
            )
    print(f"{name}: {time.time() - t0:.0f} s", flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
print("gravado", OUT)
