"""
Treino, registo de experiências (T2.4) e gravação do modelo final.

Cada corrida que entra na comparação fica numa linha de reports/experiments.csv,
incluindo as que correm mal (docs/modelacao_base.md, "Armadilhas").
"""

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone

from src.data.split import SEED
from src.features.build import FEATURES
from src.models.evaluate import cv_evaluate, summarize
from src.pipeline import ModelData

EXPERIMENTS = "reports/experiments.csv"
MODEL_PATH = "models/model.joblib"


def _params(model) -> str:
    """Parâmetros do último passo do pipeline (o classificador), em JSON."""
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    return json.dumps({k: v for k, v in est.get_params(deep=False).items()}, default=str, sort_keys=True)


def log_experiment(row: dict, path: str = EXPERIMENTS) -> pd.DataFrame:
    """Acrescenta uma linha ao registo (cria o ficheiro se não existir)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    log = pd.read_csv(path) if path.exists() else pd.DataFrame()
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    log.to_csv(path, index=False)
    return log


def run_cv(name: str, model, data: ModelData, features=FEATURES, notes: str = "",
           log: bool = True, path: str = EXPERIMENTS) -> pd.DataFrame:
    """Avalia o modelo nos 5 folds de D13 e regista a corrida. Devolve as métricas por fold."""
    folds = cv_evaluate(model, data)
    if log:
        log_experiment({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": name,
            "n_features": len(features),
            "features": ";".join(features),
            "params": _params(model),
            "seed": SEED,
            **summarize(folds).round(4).to_dict(),
            "notes": notes,
        }, path)
    return folds


def fit_final(model, data: ModelData):
    """Reajusta o modelo escolhido no treino completo (todos os folds)."""
    return clone(model).fit(data.X_train, data.y_train)


def save_model(model, data: ModelData, threshold: float | None = None,
               path: str = MODEL_PATH, **meta) -> Path:
    """Grava um bundle com o modelo, o threshold (T3.3) e os valores por defeito
    de cada feature (mediana/moda do treino), usados pela app para os campos
    que o utilizador não preenche."""
    X = data.X_train
    defaults = {c: (X[c].median() if pd.api.types.is_numeric_dtype(X[c]) else X[c].mode().iat[0])
                for c in X.columns}
    choices = {c: sorted(X[c].unique().tolist()) for c in X.columns
               if not pd.api.types.is_numeric_dtype(X[c])}
    bundle = {"model": model, "threshold": threshold, "features": list(X.columns),
              "defaults": defaults, "choices": choices, "meta": {"saved_at": datetime.now().isoformat(timespec="seconds"), **meta}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_model(path: str = MODEL_PATH) -> dict:
    return joblib.load(path)
