"""Últimas duas variantes medidas antes do teste (D29), só no treino (5 folds por doente, D13).

1. Média das previsões do v1 e da regressão logística (as duas com o pré-processamento de D16).
2. O HistGradientBoosting com as categóricas tratadas por ele próprio (OrdinalEncoder + categorical_features)
   em vez de one-hot, com os mesmos parâmetros do v1.

Critério da D18: só conta se ganhar ao v1 por mais do que o desvio entre folds (0,008 de PR-AUC).
Correr a partir da raiz: python scripts/last_checks_v1.py
"""

import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

sys.path.insert(0, ".")
from src.data.population import follow_up_eligible  # noqa: E402
from src.data.split import SEED  # noqa: E402
from src.features.build import CATEGORICAL_FEATURES  # noqa: E402
from src.models.baselines import hist_gradient_boosting, logistic_regression  # noqa: E402
from src.models.evaluate import cv_predict, recall_at_k  # noqa: E402
from src.pipeline import load_model_data  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "reports/last_checks_v1.csv"

data = load_model_data()
y = data.y_train.to_numpy()
elig = follow_up_eligible(data.X_train).to_numpy()
fold = data.folds_train.to_numpy()
num_cols = [c for c in data.X_train.columns if c not in CATEGORICAL_FEATURES]
native = make_pipeline(
    ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1, min_frequency=100
                ),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", num_cols),
        ],
        verbose_feature_names_out=False,
    ),
    HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=SEED,
        categorical_features=list(range(len(CATEGORICAL_FEATURES))),
    ),
)
oof = {}
for name, model in [
    ("v1", hist_gradient_boosting()),
    ("logreg", logistic_regression()),
    ("hgb categóricas nativas", native),
]:
    oof[name] = cv_predict(model, data)[2].to_numpy()
    print("feito:", name, flush=True)
oof["média v1 + logreg"] = (oof["v1"] + oof["logreg"]) / 2
rows = []
for name, s in oof.items():
    for k in np.unique(fold):
        v = fold == k
        rows.append(
            {
                "variante": name,
                "fold": k,
                "pr_auc": average_precision_score(y[v], s[v]),
                "recall30_eleg": recall_at_k(y[v][elig[v]], s[v][elig[v]], 0.30),
            }
        )
pd.DataFrame(rows).to_csv(OUT, index=False)
print("gravado", OUT)
