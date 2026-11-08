"""
Treino, registo de experiências (T2.4) e gravação do modelo final.

Cada corrida que entra na comparação fica em reports/experiments.csv, incluindo
as que correm mal (docs/modelacao_base.md, "Armadilhas"), no formato da W6:
uma linha para o treino e outra para a validação (OOF), com a média e o desvio
dos 5 folds. Esse ficheiro é a comparação atual e pode ser limpo antes de voltar
a correr um notebook; as mesmas linhas vão sempre também para
reports/experiments_v1.csv, o histórico, que nunca se limpa.

O modelo final grava-se com skops (não com pickle/joblib): ao carregar, só se
aceitam os tipos da lista TRUSTED_TYPES, revista à mão.
"""

import json
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import skops
import skops.io as sio
import sklearn
from sklearn.base import clone

from src.data.population import NO_FOLLOW_UP_DISCHARGE, follow_up_eligible
from src.data.split import SEED
from src.models.evaluate import cv_predict
from src.models.threshold import threshold_for_capacity
from src.pipeline import ModelData

EXPERIMENTS = "reports/experiments.csv"
HISTORY = "reports/experiments_v1.csv"
MODEL_DIR = "models/v1"

# Tipos que o skops 0.16 não confia por defeito num pipeline
# build_preprocessor() + HistGradientBoostingClassifier (saída de
# skops.io.get_untrusted_types). Revisto: é a classe do scikit-learn que guarda
# as árvores. Um tipo novo obriga a revê-lo e a acrescentá-lo aqui; até lá,
# load_model recusa o ficheiro.
TRUSTED_TYPES = [
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]


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


def _log_row(name: str, dataset: str, folds: pd.DataFrame, note: str, params: str) -> dict:
    """Linha no formato da W6: date, model, dataset, métricas, note."""
    metrics = {c: round(folds[c].mean(), 4) for c in folds.columns}
    stds = {f"{c}_std": round(folds[c].std(), 4) for c in folds.columns}
    return {"date": datetime.now().isoformat(timespec="seconds"), "model": name, "dataset": dataset,
            **metrics, **stds, "note": note, "params": params}


def run_cv(name: str, model, data: ModelData, notes: str = "", log: bool = True,
           path: str = EXPERIMENTS, history: str | None = HISTORY, return_oof: bool = False):
    """Avalia o modelo nos 5 folds de D13 e regista a corrida (treino e
    validação) no registo atual e no histórico. Devolve as métricas de
    validação por fold e, com return_oof, também as previsões OOF."""
    train_folds, val_folds, oof = cv_predict(model, data)
    if log:
        params = _params(model)
        for dataset, folds in (("train", train_folds), ("validation", val_folds)):
            row = _log_row(name, dataset, folds, notes, params)
            log_experiment(row, path)
            if history:
                log_experiment(row, history)
    return (val_folds, oof) if return_oof else val_folds


def fit_final(model, data: ModelData):
    """Reajusta o modelo escolhido no treino completo (todos os folds)."""
    return clone(model).fit(data.X_train, data.y_train)


def _jsonable(v):
    return v.item() if isinstance(v, np.generic) else v


def save_model(model, data: ModelData, oof_scores, path: str = MODEL_DIR, **meta) -> Path:
    """Grava uma pasta com:

    - model.skops: o pipeline;
    - config.json: features, valores por defeito e opções de cada campo (para
      a app), destinos de alta fora da lista de follow-up (D21), thresholds a
      10% e 20% de capacidade e quantis das previsões OOF (para o percentil de
      risco), ambos calculados só nas altas elegíveis; versões e seed;
    - README.md: o model card (só se ainda não existir, para não o apagar).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    X = data.X_train
    oof = np.asarray(oof_scores)
    eligible = follow_up_eligible(X).to_numpy()
    oof_e = oof[eligible]  # a lista de prioridades só tem altas elegíveis (D21)
    config = {
        "features": list(X.columns),
        "defaults": {c: _jsonable(X[c].median() if pd.api.types.is_numeric_dtype(X[c]) else X[c].mode().iat[0])
                     for c in X.columns},
        "choices": {c: sorted(X[c].unique().tolist()) for c in X.columns
                    if not pd.api.types.is_numeric_dtype(X[c])},
        "no_follow_up_discharge": sorted(NO_FOLLOW_UP_DISCHARGE.values()),
        "threshold_k10": threshold_for_capacity(oof_e, 0.10),
        "threshold_k20": threshold_for_capacity(oof_e, 0.20),
        "oof_quantiles": np.quantile(oof_e, np.linspace(0, 1, 1001)).tolist(),
        "n_train": int(len(oof)),
        "n_eligible": int(eligible.sum()),
        # referência para o risco estimado na app: taxa observada de readmissão <30 dias
        "base_rate_eligible": float(np.asarray(data.y_train)[eligible].mean()),
        "base_rate_all": float(np.asarray(data.y_train).mean()),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "skops": skops.__version__},
        "seed": SEED,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        **meta,
    }
    sio.dump(model, path / "model.skops")
    (path / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    card = path / "README.md"
    if not card.exists():
        card.write_text("# Model card: readmissão a 30 dias (diabetes)\n\n"
                        "_A preencher: dados, uso previsto, métricas (OOF e teste), thresholds por "
                        "capacidade, limites (não causal; sem explicação por doente; labels incompletos)._\n",
                        encoding="utf-8")
    return path


def load_model(path: str = MODEL_DIR) -> dict:
    """Carrega o modelo e o config.json. Recusa o ficheiro se tiver algum tipo
    fora de TRUSTED_TYPES."""
    path = Path(path)
    file = path / "model.skops"
    untrusted = sio.get_untrusted_types(file=file)
    unknown = sorted(set(untrusted) - set(TRUSTED_TYPES))
    if unknown:
        raise ValueError(f"Tipos não revistos em {file}: {unknown}")
    model = sio.load(file, trusted=untrusted)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    return {"model": model, **config}
