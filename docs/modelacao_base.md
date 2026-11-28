# Modelação: regras, números de referência e decisões

Regras seguidas na modelação (T2.1–T2.4), os números de referência dos modelos comparados e as decisões
tomadas. As decisões estão em [decisions.md](decisions.md); os critérios de sucesso em
[problem_framing.md](problem_framing.md).

## Base

| Peça | Onde | O que dá |
|---|---|---|
| População e target | `data/processed/diabetic_data_clean.csv` | 100 052 encontros, 70 412 doentes, 11,34% de positivos (`<30`) |
| Split por doente (D13) | `data/splits/splits.csv` | Treino 80 041 encontros com 5 folds; teste 20 011. Todas as partes com 11,34% de positivos |
| Features (D14, D15) | `src/features/build.py` | 34 features, sem fuga de informação (D8–D12) |
| Pré-processamento (D16) | `src/pipeline.py` | `ColumnTransformer` ajustado só no treino; 160 colunas |
| Registo de experiências | `reports/experiments.csv` (comparação) e `reports/experiments_v1.csv` (histórico) | uma linha de treino e uma de validação por modelo, via `run_cv` em `src/models/train.py` |

## Regras

1. **O teste não se abre.** `data.X_test` só se usa uma vez, na avaliação final (T3.6). Toda a
   comparação é feita nos 5 folds de `cv_splits(data)`.
2. **Sempre os mesmos folds.** Usar `cv_splits(data)`, não `KFold` nem `StratifiedKFold`: os folds de D13
   garantem que nenhum doente está no treino e na validação ao mesmo tempo.
3. **O pré-processamento vai dentro do pipeline.** Nada de `fit` no conjunto todo antes da CV, nem
   reamostragem fora do fold.
4. **Métrica principal: PR-AUC** (`average_precision`). Reportar também ROC-AUC e o recall nos k% de maior
   risco (a fração de readmissões encontrada com follow-up a k% das altas; k = 30%, D26). A accuracy não
   serve: prever sempre "não" dá 88,7%.
5. **Média e desvio dos 5 folds.** Uma diferença menor do que o desvio entre folds não é uma diferença.
6. **Features novas respeitam D12.** Nenhuma agregação por doente sobre o dataset e nenhuma ordem a partir
   do `encounter_id`. Uma feature nova é uma decisão da equipa (decisions.md).
7. **Seed 42** em tudo o que é aleatório.

## Números de referência

Médias nos 5 folds de CV do treino, em todas as altas (a escolha do modelo usa todas; as métricas da lista
de follow-up usam só as elegíveis, D21):

| Modelo | PR-AUC | ROC-AUC | Recall@10% | Precisão@10% | Recall@20% |
|---|---|---|---|---|---|
| Ao acaso | 0,113 | 0,500 | 0,100 | 0,113 | 0,200 |
| Regra: `number_inpatient` ≥ 1 | 0,141 | 0,592 | 0,146 | 0,166 | 0,297 |
| Ordenar por `number_inpatient` | 0,176 | 0,607 | 0,209 | 0,237 | 0,341 |
| Regressão logística (sem tuning) | 0,210 | 0,664 | 0,229 | 0,260 | 0,385 |
| Random forest (sem tuning) | 0,221 | 0,672 | 0,234 | 0,265 | 0,386 |
| **HistGradientBoosting, modelo v1 (sem tuning)** | **0,234** | **0,677** | **0,247** | **0,280** | **0,400** |

As regras com `number_inpatient` usam o `average_precision` do scikit-learn e empates desfeitos ao acaso.

Como ler: com a regressão logística, dar follow-up aos 10% de altas com maior risco encontra 23% das
readmissões em menos de 30 dias. Um em cada quatro doentes com follow-up seria readmitido, contra um em
cada nove ao acaso.

**O que esperar.** Na literatura sobre este dataset, os melhores modelos ficam em ROC-AUC ≈ 0,65–0,70. O
ganho do gradient boosting sobre a regressão logística é pequeno: o dataset não tem sinais clínicos fortes
(análises detalhadas, sinais vitais, notas).

## Armadilhas

- **Fuga por doente.** É o erro que mais infla as métricas neste dataset: num split aleatório por encontro,
  41% dos encontros de validação seriam de doentes que o modelo já viu. Os folds de D13 evitam isto.
- **Escolher o corte cedo demais.** O corte decide-se pela capacidade da equipa de follow-up (D26), não
  durante a comparação de modelos, que compara a ordenação (PR-AUC, recall@k).
- **Tuning em excesso.** Com cerca de 9 000 positivos no treino, um grid grande ajusta-se ao ruído dos
  folds. Chegam poucas combinações por modelo.
- **Esquecer o registo.** Cada corrida que entra na comparação vai para `reports/experiments.csv`, incluindo
  as que correram mal.

## Decisões da modelação

1. Gradient boosting: **HistGradientBoosting**, que é o modelo v1 (D17, 10/11).
2. Ablação: `payer_code` fica, o indicador de registo incompleto não entra e `max_glu_serum` fica (D18).
3. Pré-processamento: o comum de D16, para todos os modelos.
4. Hugging Face Hub: só para publicar (D23, 8/11). O Kumo Tabular fica como desafiante e empata com o v1
   ([anexo_kumo_tabular.md](anexo_kumo_tabular.md)). O Space `murtaza65/diabetes-readmission-prediction-live`
   foi analisado e não serve (sem licença, treinado com os nossos doentes de teste, fuga por doente).
5. Tuning: medido nos folds por doente (early stopping e três níveis de regularização); nenhuma variante
   bate o v1, que se mantém (D27, `scripts/tuning_v1.py`).
6. Features novas: comorbilidades de Charlson e HbA1c × medicação alterada; nenhuma melhora o v1 (D28,
   `scripts/feature_screen_v1.py`).
