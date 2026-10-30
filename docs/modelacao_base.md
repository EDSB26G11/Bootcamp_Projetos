# Fase 2 — Modelação base: contexto

Ponto de partida para a fase 2 do roadmap (T2.1–T2.4): o que já está pronto,
o que esta fase tem de entregar, as regras a respeitar e os números de
referência. As decisões que levaram até aqui estão em
[decisions.md](decisions.md) (D1–D16).

## Onde estamos

- **Calendário.** A fase 2 estava prevista para as semanas 4–7. Estamos no
  fim da semana 8. O **status report 3 é na semana 9 (2–6 de novembro)** e
  pede o **modelo v1 congelado**. A fase 3 (desequilíbrio, métricas e
  threshold, W6–W8) também ainda não começou. Por isso, esta fase tem de ser
  curta: comparar poucos modelos bem, não muitos à pressa.
- **Fase 1 completa.** População, target, split, features e pipeline estão
  implementados e testados (44 testes).

## O que já está pronto

| Peça | Onde | O que dá |
|---|---|---|
| População e target | `data/processed/diabetic_data_clean.csv` | 100 052 encontros, 70 412 doentes, 11,34% de positivos (`<30`) |
| Split por doente (D13) | `data/splits/splits.csv` | Treino 80 041 encontros com 5 folds; teste 20 011. Todas as partes com 11,34% de positivos |
| Features (D14, D15) | `src/features/build.py` | 34 features, sem fuga de informação (D8–D12) |
| Pré-processamento (D16) | `src/pipeline.py` | `ColumnTransformer` ajustado só no treino; 160 colunas |

Para começar:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from src.pipeline import build_preprocessor, cv_splits, load_model_data

data = load_model_data()
pipe = make_pipeline(build_preprocessor(), LogisticRegression(max_iter=2000))
cross_validate(pipe, data.X_train, data.y_train, cv=cv_splits(data),
               scoring=["average_precision", "roc_auc"])
```

## O que a fase tem de entregar

Um conjunto pequeno de modelos **comparáveis entre si**, todos avaliados da
mesma forma, e a escolha justificada do candidato a modelo v1.

| Tarefa | O quê | Output | Papel |
|---|---|---|---|
| T2.1 Baselines | Regra simples e regressão logística. São a fasquia: tudo o que vier depois tem de as bater | Tabela baseline | R2 |
| T2.2 Modelos candidatos | Um gradient boosting (HistGradientBoosting, LightGBM ou XGBoost) e uma random forest, com tuning modesto | `notebooks/03_models.ipynb` | R2 |
| T2.3 Hugging Face Hub | Ver se há modelos ou datasets tabulares úteis; ler model cards e licenças; decidir e justificar | Nota em `decisions.md` | R3 |
| T2.4 Registo de experiências | Uma linha por corrida: data, modelo, features, parâmetros, seed, métricas | `docs/experiments.md` | R2 |

## Regras

1. **O teste não se abre.** `data.X_test` só se usa uma vez, na avaliação
   final (T3.6). Toda a comparação nesta fase é feita nos 5 folds de
   `cv_splits(data)`.
2. **Sempre os mesmos folds.** Usar `cv_splits(data)`, não `KFold` nem
   `StratifiedKFold`: os folds de D13 garantem que nenhum doente está no
   treino e na validação ao mesmo tempo.
3. **O pré-processamento vai dentro do pipeline.** Nada de `fit` no
   conjunto todo antes da CV, nem SMOTE fora do fold (isso é na T3.1).
4. **Métrica principal: PR-AUC** (`average_precision`). Reportar também
   ROC-AUC e recall@10% / recall@20% (a fração de readmissões apanhada se a
   equipa de follow-up só chegar a 10% ou 20% das altas). A accuracy não serve:
   prever sempre "não" dá 88,7%.
5. **Média e desvio dos 5 folds.** Uma diferença menor do que o desvio entre
   folds não é uma diferença.
6. **Features novas respeitam D12.** Nenhuma agregação por doente sobre o
   dataset e nenhuma ordem a partir do `encounter_id`. Se surgir uma ideia de
   feature, vai ao roadmap como decisão.
7. **Seed 42** em tudo o que é aleatório.

## Números de referência

Médias nos 5 folds de CV do treino:

| Modelo | PR-AUC | ROC-AUC | Recall@10% | Precisão@10% | Recall@20% |
|---|---|---|---|---|---|
| Ao acaso | 0,113 | 0,500 | 0,100 | 0,113 | 0,200 |
| Regra: `number_inpatient` ≥ 1 | 0,150 | 0,592 | 0,148 | 0,168 | 0,299 |
| Ordenar por `number_inpatient` | 0,183 | 0,611 | 0,210 | 0,238 | 0,347 |
| Regressão logística (sem tuning) | 0,210 | 0,664 | 0,229 | 0,260 | 0,385 |

Como ler: com a regressão logística, chamar os 10% de altas com maior risco
apanha 23% das readmissões em menos de 30 dias. Um em cada quatro doentes
contactados seria de facto readmitido, contra um em cada nove ao acaso.

**Expectativa realista.** Na literatura sobre este dataset, os melhores
modelos ficam tipicamente em ROC-AUC ≈ 0,65–0,70. O ganho do gradient
boosting sobre a regressão logística deve ser pequeno. Não é falha da equipa:
o dataset não tem sinais clínicos fortes (análises detalhadas, sinais vitais,
notas). Isto tem de aparecer no relatório e no deck.

## Armadilhas desta fase

- **Fuga por doente.** É o erro que mais infla as métricas neste dataset:
  num split aleatório por encontro, 41% dos encontros de validação seriam de
  doentes que o modelo já viu. Os folds de D13 evitam isto, desde que se usem.
- **Otimizar o threshold agora.** O ponto de corte decide-se na T3.3, pela
  capacidade da equipa de follow-up. Nesta fase compara-se a ordenação
  (PR-AUC, recall@k), não um threshold.
- **Tuning em excesso.** Com cerca de 9 000 positivos no treino, um grid
  grande ajusta-se ao ruído dos folds. Chegam poucas combinações por modelo.
- **Esquecer o registo.** Cada corrida que entra na comparação vai para
  `docs/experiments.md`, incluindo as que correram mal.

## Decisões que vão surgir

Levar ao roadmap quando aparecerem:

1. Qual o gradient boosting (HistGradientBoosting já vem com o scikit-learn;
   LightGBM e XGBoost são dependências novas).
2. Se `payer_code`, `max_glu_serum` e o indicador de registo incompleto (D5)
   ficam. Testar retirando-os (ablação) na CV.
3. Se o gradient boosting usa o pré-processamento comum (D16) ou categorias
   nativas.
4. Se se usa algo do Hugging Face Hub (T2.3).
5. Qual o candidato a modelo v1 para o status report 3.
