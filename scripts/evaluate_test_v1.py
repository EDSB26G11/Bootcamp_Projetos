"""Avaliação única no teste (T3.6) do modelo v1 congelado (D27–D29), com as metas aceites na D20.

Corre uma só vez: recusa voltar a correr se reports/test_results_v1.json já existir. Usa o modelo e o corte
gravados (models/v1, threshold_k30 = capacidade de 30%, D26), sem nada aprendido no teste. Os intervalos de
confiança (95%) vêm de um bootstrap agrupado por doente (1000 reamostragens, seed 42).

Correr a partir da raiz: python scripts/evaluate_test_v1.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

sys.path.insert(0, ".")
from src.data.population import follow_up_eligible  # noqa: E402
from src.data.split import SEED  # noqa: E402
from src.models.evaluate import calibration_table  # noqa: E402
from src.models.threshold import rank_baseline_at_k  # noqa: E402
from src.models.train import MODEL_DIR, load_model  # noqa: E402
from src.pipeline import PROCESSED, SPLITS, load_model_data, read_processed  # noqa: E402

OUT = Path("reports/test_results_v1.json")
N_BOOT = 1000
CAPACITIES = (0.30, 0.20, 0.10)  # 30%: D26; 20% e 10%: sensibilidade

# Ensaio (--ensaio): o mesmo código sobre as previsões OOF do treino, para verificar o script sem abrir o teste.
ENSAIO = "--ensaio" in sys.argv
if ENSAIO:
    OUT = Path(sys.argv[sys.argv.index("--ensaio") + 1])
    N_BOOT = 100  # o ensaio só verifica o código
elif OUT.exists():
    raise SystemExit(f"{OUT} já existe: o teste só se avalia uma vez (D13).")

bundle = load_model(MODEL_DIR)
data = load_model_data()
df = read_processed(PROCESSED).merge(pd.read_csv(SPLITS)[["encounter_id", "split"]], on="encounter_id", validate="1:1")
part = "train" if ENSAIO else "test"
rows = df[df["split"].eq(part)].reset_index(drop=True)
X = data.X_train if ENSAIO else data.X_test
y = (data.y_train if ENSAIO else data.y_test).to_numpy()
assert len(rows) == len(X) and (rows["target"].to_numpy() == y).all()
if ENSAIO:
    s = pd.read_csv("reports/oof_v1.csv")["oof_score"].to_numpy()
else:
    s = bundle["model"].predict_proba(X[bundle["features"]])[:, 1]
elig = follow_up_eligible(X).to_numpy()
inp = X["number_inpatient"].to_numpy()
patient = rows["patient_nbr"].to_numpy()
base_elig = bundle["base_rate_eligible"]  # taxa base das elegíveis, conhecida antes do teste (treino)


def recall_at(yv, sv, k):
    """Recall e precisão nos k% de score mais alto (empates ao acaso, seed fixa), e altas na lista."""
    rng = np.random.RandomState(SEED)
    n = int(round(k * len(sv)))
    perm = rng.permutation(len(sv))
    top = np.zeros(len(sv), dtype=bool)
    top[perm[np.argsort(-sv[perm], kind="stable")[:n]]] = True
    return yv[top].sum() / max(yv.sum(), 1), yv[top].mean()


def metrics(idx):
    """Métricas de uma (re)amostra de linhas do teste."""
    yv, sv, ev, iv = y[idx], s[idx], elig[idx], inp[idx]
    out = {"pr_auc_modelo": average_precision_score(yv, sv), "pr_auc_regra": average_precision_score(yv, iv)}
    ye, se, ie = yv[ev], sv[ev], iv[ev]
    for k in CAPACITIES:
        r, p = recall_at(ye, se, k)
        rr = rank_baseline_at_k(ye, ie, k)["recall_mean"]
        tag = int(round(100 * k))
        out[f"recall{tag}_modelo"], out[f"precisao{tag}_modelo"], out[f"recall{tag}_regra"] = r, p, rr
        out[f"delta_recall{tag}_pp"] = 100 * (r - rr)
        out[f"por1000_{tag}_modelo"] = 1000 * ye.sum() / len(ye) * r
        out[f"por1000_{tag}_regra"] = 1000 * ye.sum() / len(ye) * rr
    return out


point = metrics(np.arange(len(y)))

# Bootstrap agrupado por doente: reamostram-se doentes, com todas as suas altas.
groups = pd.Series(np.arange(len(y))).groupby(patient).apply(np.array).to_list()
rng = np.random.RandomState(SEED)
boot = []
for b in range(N_BOOT):
    pick = rng.randint(0, len(groups), len(groups))
    boot.append(metrics(np.concatenate([groups[i] for i in pick])))
boot = pd.DataFrame(boot)
ci = {c: [float(boot[c].quantile(0.025)), float(boot[c].quantile(0.975))] for c in boot}

# Calibração (D22) e Brier, nas altas elegíveis e em todas.
cal_e = calibration_table(y[elig], s[elig])
cal_a = calibration_table(y, s)
calib = {
    "max_diff_pp_elegiveis": float(100 * cal_e["diferenca"].abs().max()),
    "max_diff_pp_todas": float(100 * cal_a["diferenca"].abs().max()),
    "decis_elegiveis": (100 * cal_e[["risco_previsto", "observado"]]).round(2).reset_index().to_dict("records"),
    "brier_modelo_elegiveis": float(brier_score_loss(y[elig], s[elig])),
    "brier_taxa_base_elegiveis": float(brier_score_loss(y[elig], np.full(elig.sum(), base_elig))),
}

# Critério operacional: o corte gravado (threshold_k30) aplicado às altas elegíveis do teste.
flag = s[elig] >= bundle["threshold_k30"]
ye = y[elig]
tp, fp = int((flag & (ye == 1)).sum()), int((flag & (ye == 0)).sum())
fn, tn = int((~flag & (ye == 1)).sum()), int((~flag & (ye == 0)).sum())
oper = {
    "fracao_com_follow_up": float(flag.mean()),
    "threshold_k30": bundle["threshold_k30"],
    "matriz": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    "recall": tp / (tp + fn),
    "precisao": tp / (tp + fp),
    "accuracy": (tp + tn) / len(ye),
    "f1": 2 * tp / (2 * tp + fp + fn),
}

metas = {  # D20 (aceite a 28/11)
    "prioridade_pr_auc_acima_da_regra": point["pr_auc_modelo"] > point["pr_auc_regra"],
    "prioridade_recall30_acima_da_regra": point["recall30_modelo"] > point["recall30_regra"],
    "risco_decis_menos_de_2pp": calib["max_diff_pp_elegiveis"] < 2,
    "risco_brier_abaixo_da_taxa_base": calib["brier_modelo_elegiveis"] < calib["brier_taxa_base_elegiveis"],
    "operacional_30pct_mais_menos_1pp": abs(oper["fracao_com_follow_up"] - 0.30) <= 0.01,
    "negocio_por1000_acima_da_regra": point["por1000_30_modelo"] > point["por1000_30_regra"],
}
res = {
    "ensaio_com_oof_do_treino": ENSAIO,
    "avaliado_em": datetime.now().isoformat(timespec="seconds"),
    "n_teste": int(len(y)),
    "n_elegiveis": int(elig.sum()),
    "taxa_teste": float(y.mean()),
    "taxa_elegiveis": float(y[elig].mean()),
    "pontual": point,
    "ic95_bootstrap_por_doente": ci,
    "calibracao": calib,
    "operacional": oper,
    "metas_D20": {k: bool(v) for k, v in metas.items()},
}
OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
print(
    json.dumps(
        {"metas_D20": res["metas_D20"], "pontual": {k: round(v, 4) for k, v in point.items()}},
        indent=1,
        ensure_ascii=False,
    )
)
