# Features aprovadas (T1.6)

Colunas de `data/processed/diabetic_data_clean.csv` que podem entrar no modelo
por serem conhecidas **no momento da alta**. Resulta das decisões D8–D12 em
[decisions.md](decisions.md), fechadas no roadmap a 29/10.

Aprovar aqui quer dizer só "não há fuga de informação". Se cada coluna fica no
modelo final (constantes, redundantes, quase sem variação, código vs.
descrição) decide-se em T1.8/T1.9 e está no
[dicionário de variáveis](dicionario_variaveis.md).

## Resumo

| | Colunas |
|---|---|
| Originais aprovadas | 47 (as 50 originais menos `readmitted`, `encounter_id`, `patient_nbr`) |
| Descrições criadas no pipeline | 3 (`*_desc`; a mesma informação que o código, escolher um dos dois em T1.9) |
| Excluídas por definição (D11) | `readmitted`, `encounter_id`, `patient_nbr` |
| Target | `target` (D6) |

## Aprovadas com pressuposto escrito (D9)

Estes pressupostos têm de aparecer no relatório final.

| Colunas | Pressuposto |
|---|---|
| `diag_1`, `diag_2`, `diag_3`, `number_diagnoses` | Os diagnósticos estão codificados na alta (P14). A codificação ICD-9 pode ser finalizada depois, para faturação |
| `A1Cresult`, `max_glu_serum` | O resultado é conhecido na alta (P15). Pode chegar perto ou depois dela |
| 23 colunas de medicamentos (`metformin` … `metformin-pioglitazone`), `change`, `diabetesMed` | Refletem o internamento e o estado à alta (P16) |

## Aprovadas sem reservas (D10)

| Grupo | Colunas |
|---|---|
| Destino da alta | `discharge_disposition_id` (também define a população, D1) |
| Administrativas | `admission_type_id`, `admission_source_id`, `payer_code`, `medical_specialty` |
| Internamento | `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications` |
| Histórico (ano anterior, P13) | `number_outpatient`, `number_emergency`, `number_inpatient` |
| Demográficas | `age`, `gender`, `race` (uso por fairness decide-se em T3.5) |

`weight` não entrou em nenhuma das decisões da T1.6. É registado no
internamento, por isso não há risco de fuga, mas tem 96,9% de missing e a
proposta é descartá-lo em T1.9. Fica contado nas 47 colunas originais até essa
decisão.

## Regra para variáveis derivadas (D12)

Qualquer feature construída em T1.8 tem de cumprir:

1. **Nada de agregações por doente sobre o dataset completo** (ex.: nº de
   encontros do doente, média de internamentos do doente). Incluem
   internamentos posteriores à alta.
2. **Nada de ordem temporal a partir do `encounter_id`** (ex.: "encontro
   anterior", "nº de encontros até agora"). A ordem não é garantidamente
   temporal (P17).

## Features do modelo (T1.8–T1.10)

Das colunas aprovadas acima, o modelo usa **34 features**, definidas em
`src/features/build.py` (D14, D15). O pré-processamento está em
`src/pipeline.py` (D16) e dá 160 colunas depois do one-hot.

| Tipo no pipeline | Features | Tratamento |
|---|---|---|
| Numéricas (9) | `age_band`, `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`, `number_diagnoses`, `n_meds`, `n_meds_up`, `n_meds_down` | `StandardScaler` |
| Contagens (4) | `number_outpatient`, `number_emergency`, `number_inpatient`, `n_visits_prior` | `log1p` + `StandardScaler` |
| Binária (1) | `diabetes_complicated` | sem transformação |
| Categóricas (20) | `race`, `gender`, `payer_code`, `medical_specialty`, 3 `*_id_desc`, `A1Cresult`, `max_glu_serum`, `diag_1–3_group`, 8 medicamentos (`metformin`, `repaglinide`, `glimepiride`, `glipizide`, `glyburide`, `pioglitazone`, `rosiglitazone`, `insulin`) | one-hot; em falta = `Missing`; < 100 encontros no treino = `infrequent` |

Saem do modelo (D15): `weight`, `examide`, `citoglipton`, os 13 medicamentos
com ≥ 99,3% `No` (só contam em `n_meds*`), `change`, `diabetesMed`, `diag_1–3`
e `age` em bruto e os 3 `*_id` (ficam as `_desc`). O motivo de cada uma está
em `DROPPED`.

Como usar na modelação (T2):

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from src.pipeline import build_preprocessor, cv_splits, load_model_data

data = load_model_data()  # X_train, y_train, groups_train, folds_train, X_test, y_test
pipe = make_pipeline(build_preprocessor(), LogisticRegression(max_iter=2000))
cross_val_score(pipe, data.X_train, data.y_train, cv=cv_splits(data), scoring="average_precision")
```

O teste (`data.X_test`) só se usa na avaliação final (T3.6).

## Ainda por decidir noutras tarefas

- T2: confirmar se `payer_code`, `max_glu_serum` e o indicador de registo
  incompleto (D5) ajudam o modelo.
- T3.5: usar `race`, `gender` e `age_band` no modelo ou só para avaliar fairness.
