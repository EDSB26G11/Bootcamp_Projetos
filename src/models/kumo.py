"""
Kumo Tabular (NVIDIA, Hugging Face Hub) como modelo desafiante do v1 (T2.3 / F2·4).

Não substitui o HistGradientBoosting v1 (D17), não entra na app nem em models/.
Precisa do ambiente à parte de requirements-kumo.txt (torch + sdm); o torch e o
sdm só se importam quando o modelo corre, para os testes não precisarem deles.

O Kumo Tabular não é treinado nos nossos dados: recebe as altas rotuladas como
*contexto* no momento da previsão (Lab W5_1). Por isso:

- `fit` só guarda o contexto: uma amostra estratificada pelo target, tirada só
  das linhas que recebe (o treino do fold), com as 34 features em bruto (D15),
  sem build_preprocessor nem one-hot e sem identificadores (D11);
- `predict_proba` corre o Kumo com esse contexto.

`kumo_cv` é o ciclo de CV do desafiante. Não usa cv_predict/run_cv porque
esses calculam também as métricas no treino do fold: no Kumo seria lento e sem
significado (essas linhas estão no próprio contexto). Usa os mesmos folds de
D13 (cv_splits), as mesmas métricas (score_metrics) e o mesmo registo
(log_experiment), só com dataset="validation". O teste não se abre.
"""

import json
import time
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

from src.data.population import follow_up_eligible
from src.data.split import SEED
from src.features.build import CATEGORICAL_FEATURES, FEATURES
from src.models.evaluate import TOP_K, score_metrics
from src.models.train import EXPERIMENTS, HISTORY, _log_row, log_experiment
from src.pipeline import ModelData, cv_splits

TARGET = "target"
# Linhas das tabelas do pré-treino: até ~60k. O treino de cada fold tem ~64k.
DEFAULT_CONTEXT = 30_000
# Sem GPU, contextos acima disto demoram demasiado: kumo_cv recusa sem allow_cpu.
CPU_CONTEXT_LIMIT = 5_000

# Backend: função (X_context, y_context, X_query) -> probabilidade da classe 1.
Backend = Callable[[pd.DataFrame, np.ndarray, pd.DataFrame], np.ndarray]

_MODELS: dict = {}  # um KumoTabular por (size, device): os pesos só se carregam uma vez


def stratified_context(y, n: int, random_state: int = SEED) -> np.ndarray:
    """Posições (0..len(y)-1, ordenadas) de uma amostra de n linhas com a mesma
    taxa de positivos de y. Com n >= len(y), devolve todas."""
    pos = np.arange(len(y))
    if n >= len(y):
        return pos
    keep, _ = train_test_split(pos, train_size=n, stratify=np.asarray(y), random_state=random_state)
    return np.sort(keep)


def column_types(columns) -> dict:
    """Tipos de coluna declarados, como no Lab W5_1: as categóricas de D15 e o
    target como categóricos, o resto numérico."""
    import sdm

    cat = set(CATEGORICAL_FEATURES) | {TARGET}
    return {c: sdm.Stype.categorical if c in cat else sdm.Stype.numerical for c in columns}


def default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def kumo_backend(size: str = "small", num_estimators: int = 4, device: str | None = None,
                 random_state: int = SEED, query_batch_size: int = 20_000) -> Backend:
    """Backend real: o KumoTabular do sdm (pesos do Hugging Face Hub, ~110 MB
    para o "small"). Com a mesma seed no gerador, a mesma previsão em cada
    corrida; partir as linhas a prever em lotes não muda o resultado (as linhas
    a prever só olham para o contexto)."""
    import sdm
    import torch

    device = device or default_device()

    def predict(X_context, y_context, X_query):
        key = (size, device)
        if key not in _MODELS:
            _MODELS[key] = sdm.models.KumoTabular(task="classification", size=size, device=device)
        model = _MODELS[key]
        stypes = column_types(list(X_context.columns) + [TARGET])
        table = lambda df: sdm.TableTensor.from_pandas(
            df=df, stypes={c: stypes[c] for c in df.columns}, device=device)
        context = table(X_context.assign(**{TARGET: np.asarray(y_context).astype(str)}))
        x_context, y_ctx = context.drop_columns(TARGET), context[:, TARGET]
        out = []
        for start in range(0, len(X_query), query_batch_size):
            query = table(X_query.iloc[start:start + query_batch_size])
            with torch.inference_mode():
                pred = model(x_context=x_context, y_context=y_ctx, x_query=query,
                             num_estimators=num_estimators,
                             generator=torch.Generator(device).manual_seed(random_state))
            out.append(pred.to_pandas()["1"].to_numpy())
        return np.concatenate(out)

    predict.device = device
    return predict


class KumoTabularClassifier(ClassifierMixin, BaseEstimator):
    """Interface scikit-learn para o Kumo Tabular (como o ColumnRanker de baselines.py).

    fit(X, y): guarda o contexto (amostra estratificada de max_context linhas
    de X, só com as colunas de FEATURES). predict_proba(X): corre o backend
    com esse contexto. `backend=None` usa o Kumo real; os testes passam um mock.
    """

    def __init__(self, max_context=DEFAULT_CONTEXT, size="small", num_estimators=4,
                 device=None, random_state=SEED, query_batch_size=20_000, backend=None):
        self.max_context = max_context
        self.size = size
        self.num_estimators = num_estimators
        self.device = device
        self.random_state = random_state
        self.query_batch_size = query_batch_size
        self.backend = backend

    def fit(self, X, y):
        X = X[FEATURES]  # sem encounter_id nem patient_nbr (D11), mesmo se vierem em X
        keep = stratified_context(y, self.max_context, self.random_state)
        self.context_X_ = X.iloc[keep]
        self.context_y_ = np.asarray(y)[keep].astype(int)
        self.context_index_ = X.index[keep]  # para verificar a ausência de fuga por doente
        self.classes_ = np.array([0, 1])
        return self

    def _backend(self) -> Backend:
        if self.backend is not None:
            return self.backend
        return kumo_backend(self.size, self.num_estimators, self.device, self.random_state,
                            self.query_batch_size)

    def predict_proba(self, X):
        backend = self._backend()
        cuda = getattr(backend, "device", None) == "cuda"
        if cuda:
            import torch
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        p = np.asarray(backend(self.context_X_, self.context_y_, X[FEATURES]), dtype=float)
        if cuda:
            torch.cuda.synchronize()
            self.last_peak_gb_ = torch.cuda.max_memory_allocated() / 2**30
        else:
            self.last_peak_gb_ = np.nan
        self.last_seconds_ = time.perf_counter() - t0
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def eligible_at_k(y, scores, eligible) -> dict:
    """Recall e precisão @10% e @20% só nas altas elegíveis (D21), como no notebook 04."""
    m = score_metrics(np.asarray(y)[eligible], np.asarray(scores)[eligible])
    return {f"{c}_eleg": m[c] for k in TOP_K
            for c in (f"recall_at_{int(k * 100)}", f"precision_at_{int(k * 100)}")}


VALIDATION_ONLY_NOTE = (
    "Só validação: no Kumo as linhas de treino do fold estão no próprio contexto, "
    "por isso a métrica de treino não tem significado (e seria lenta). "
    "@k em todas as altas, como nas outras linhas; @k nas elegíveis (D21) em reports/kumo_cv_folds.csv."
)


def _params(model: KumoTabularClassifier) -> str:
    p = {k: v for k, v in model.get_params().items() if k != "backend"}
    return json.dumps(p, default=str, sort_keys=True)


def kumo_cv(name: str, model: KumoTabularClassifier, data: ModelData, notes: str = "",
            folds=None, log: bool = True, path: str = EXPERIMENTS, history: str | None = HISTORY,
            allow_cpu: bool = False):
    """CV do desafiante nos folds de D13 (por defeito os 5; `folds` escolhe alguns).

    Devolve (por_fold, oof): por_fold tem as métricas de score_metrics na
    validação (todas as altas), as @k nas elegíveis (sufixo _eleg), o tamanho
    do contexto, os segundos e a memória de pico da GPU da previsão; oof tem o
    score de cada encontro de validação (NaN nos folds não corridos).
    Com log, regista uma linha dataset="validation" (só com os 5 folds)."""
    splits = cv_splits(data)
    folds = range(len(splits)) if folds is None else folds
    eligible = follow_up_eligible(data.X_train).to_numpy()
    oof = pd.Series(np.nan, index=data.y_train.index, name="oof_score")
    rows = []
    for k in folds:
        tr, va = splits[k]
        m = model.__class__(**model.get_params()).fit(data.X_train.iloc[tr], data.y_train.iloc[tr])
        if model.backend is None and not allow_cpu and default_device() == "cpu" \
                and len(m.context_X_) > CPU_CONTEXT_LIMIT:
            raise RuntimeError(f"Sem GPU: contexto de {len(m.context_X_):,} linhas demoraria horas. "
                               "Usar uma GPU (Colab T4) ou allow_cpu=True.")
        scores = m.predict_proba(data.X_train.iloc[va])[:, 1]
        oof.iloc[va] = scores
        y_va = data.y_train.iloc[va]
        rows.append({"fold": k, **score_metrics(y_va, scores), **eligible_at_k(y_va, scores, eligible[va]),
                     "n_context": len(m.context_X_), "n_query": len(va),
                     "seconds": m.last_seconds_, "gpu_peak_gb": m.last_peak_gb_})
    per_fold = pd.DataFrame(rows).set_index("fold")
    if log and len(per_fold) == len(splits):
        extra = ("n_context", "n_query", "seconds", "gpu_peak_gb")
        metrics = per_fold[[c for c in per_fold.columns  # mesmas colunas das outras linhas
                            if not c.endswith("_eleg") and c not in extra]]
        note = " ".join(x for x in (notes, VALIDATION_ONLY_NOTE) if x)
        row = _log_row(name, "validation", metrics, note, _params(model))
        log_experiment(row, path)
        if history:
            log_experiment(row, history)
    return per_fold, oof
