# Anexo: Kumo Tabular como modelo desafiante (T2.3 / F2·4)

Nota da T2.3 (avaliar o Hugging Face Hub). Bancada: [notebooks/07_kumo_challenger.ipynb](../notebooks/07_kumo_challenger.ipynb).
Decisão: D23 em [decisions.md](decisions.md).

## O que se avaliou no Hugging Face Hub

1. **Space `murtaza65/diabetes-readmission-prediction-live`** (6/11): não serve. Não tem licença, foi
   treinado com o dataset inteiro, incluindo os nossos doentes de teste, e o split não é por doente
   (fuga). Ver ponto 4 de [modelacao_base.md](modelacao_base.md).
2. **Kumo Tabular** (NVIDIA, Apache-2.0, pesos de ~110 MB no Hub): modelo tabular pré-treinado (Lab W5_1).
   Não se treina nos nossos dados: recebe altas rotuladas como *contexto* no momento da previsão.
   Testado como **desafiante** do v1 (HistGradientBoosting, D17).

## Como se comparou

- Os mesmos 5 folds por doente (D13). O teste não foi aberto: só previsões out-of-fold.
- As mesmas 34 features (D15), em bruto (o Kumo trata as categóricas e os missing). Sem `encounter_id`
  nem `patient_nbr` (D11).
- Contexto: amostra estratificada tirada só do treino do fold, com 10k, 30k e 55k encontros. Modelo
  `small` (28 M parâmetros), `num_estimators=4`, seed 42.
- @k só nas altas elegíveis (D21).
- **Verificação de fuga:** em nenhum fold há doentes do contexto na validação. A fração de linhas de
  validação com uma linha igual no contexto é ≤ 0,01%, e vem de doentes diferentes.

## Resultados (OOF, média ± desvio nos 5 folds)

| Modelo | PR-AUC | ROC-AUC | Brier | Recall@10% (elegíveis) | Recall@20% (elegíveis) |
|---|---|---|---|---|---|
| HGB v1 | 0,234 ± 0,008 | 0,677 | 0,0953 | 25,2% ± 0,8 | 40,1% ± 1,4 |
| Kumo, contexto 55k | 0,233 ± 0,005 | 0,676 | 0,0954 | 25,1% ± 0,9 | 40,4% ± 1,9 |
| Kumo, contexto 30k | 0,226 ± 0,006 | 0,671 | 0,0958 | 24,8% ± 1,1 | 39,9% ± 1,7 |
| Kumo, contexto 10k | 0,216 ± 0,004 | 0,663 | 0,0963 | 24,8% ± 1,0 | 39,2% ± 1,6 |
| Regressão logística | 0,210 ± 0,008 | 0,664 | 0,0965 | 24,5% ± 1,3 | 39,4% ± 1,4 |

Fonte: `reports/kumo_vs_v1_summary.csv`, `reports/kumo_cv_folds.csv`.

1. **Ordena melhor do que o v1?** Não. Com o melhor contexto (55k), a diferença fold a fold é de
   −0,001 ± 0,004 em PR-AUC e o Kumo só ganha em 1 dos 5 folds. No recall@10% ganha em 2 folds, no
   recall@20% em 3. Todas as diferenças são menores do que o desvio entre folds, ou seja, é um empate
   (regra 5 de [modelacao_base.md](modelacao_base.md)). O contexto foi escolhido nestes mesmos folds, o
   que favorece ligeiramente o Kumo.
2. **Com poucos rótulos?** Aí ajuda. Com os mesmos encontros rotulados, como contexto do Kumo ou como
   treino da regressão logística (média de 2 folds): com 500 rótulos, PR-AUC 0,165 contra 0,148; com
   2 000, 0,197 contra 0,168; com 55 000, 0,233 contra 0,206. Neste projeto há 80 041 altas rotuladas no
   treino, por isso esta vantagem não se aplica. Fonte: `reports/kumo_learning_curve.csv`,
   `reports/figures/kumo_learning_curve.png`.
3. **Custos** (`reports/kumo_costs.csv`):
   - O Kumo com contexto de 55k demora 129 s por fold só a prever, numa GPU (RTX 4050). O v1 treina e
     prevê em 9 s, em CPU.
   - O Kumo precisa de uma GPU para ser utilizável. Pesa ~110 MB, contra 2,9 MB do `model.skops` do v1.
   - **O contexto com dados de doentes tem de viajar com o modelo**: ~10 MB de altas rotuladas (55k), que
     a app teria de guardar e proteger. O v1 só guarda parâmetros.
4. **Explicabilidade** (importância por permutação, D19): as duas ordens das features concordam pouco
   (Spearman 0,43; 7 das 10 primeiras em comum). `number_inpatient` domina nos dois. No Kumo, o destino da
   alta pesa muito menos (−0,015 contra −0,050 de PR-AUC). Além disso, a importância do Kumo depende do
   contexto: com outro contexto, a mesma alta pode ter outro risco.

## Conclusão

O Kumo Tabular **empata** com o v1 na ordenação e na calibração, mas custa mais: precisa de GPU, é cerca
de 14 vezes mais lento por fold, obriga a enviar dados de doentes com o modelo e é menos estável de
explicar. Só ajudaria com poucos rótulos, o que não é o nosso caso. **Recomendação: manter o
HistGradientBoosting como modelo v1.** O Kumo fica como desafiante documentado e não entra na app nem
em `models/`. O Hugging Face Hub usa-se para publicar o modelo v1 e o Space da app (T4.1, T4.3).

Estado: a F2·4 foi fechada no roadmap a 8/11 (opção A: Hub só para publicar). Este teste confirma-a.
