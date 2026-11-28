# Model card: readmissão a 30 dias em doentes com diabetes (modelo v1)

Grupo EDSB26_G11, Enterprise Data Science Bootcamp 2026-27, NOVA IMS. Ana Carapinha (20250306), Filipa
Filipe (20250313), Sandra Figueiredo (20251705), Tiago Marques (20251844).

Ficheiros em `models/v1/`: `model.skops` (pipeline completo), `config.json` (features, cortes, quantis,
versões) e este `README.md`. A fonte deste texto é `docs/model_card.md`, versionado; o `save_model()` copia-o
para `models/v1/README.md`.

> **Estado:** validação cruzada (previsões out-of-fold do treino) e avaliação única no teste a 28/11
> (secção “Avaliação no teste”).

## Uso previsto

- **Para quê:** no momento da alta hospitalar, estimar o risco de o doente ser readmitido em menos de 30 dias
  e ordenar as altas elegíveis por esse risco, para a equipa dar follow-up primeiro às de maior risco,
  até ao limite da sua capacidade: 30% das altas elegíveis (D26).
- **Para quem:** a Direção de Care Quality e a equipa de follow-up de uma rede hospitalar.
- **Como:** o modelo propõe; **uma pessoa revê cada caso** e decide o follow-up. A app (`src/app.py`) mostra o
  risco estimado, a posição da alta face às outras e a recomendação, com o motivo (capacidade de 30%, D26).
- **Altas elegíveis (D21):** altas para a comunidade sem acompanhamento. As altas para outro hospital ou
  instituição, internamento, apoio domiciliário ou cuidados paliativos aparecem como “fora da lista”.

## Usos excluídos

- Decisões clínicas automáticas, ou recusar cuidados a quem tem risco baixo.
- Afirmar que o follow-up **causa** menos readmissões: o modelo não é causal (ver Limites).
- Explicar o risco de um doente concreto: a importância das variáveis é global, não por doente.
- Usar noutras populações, outros países ou anos recentes sem nova validação (dados dos EUA, 1999–2008).
- Usar o risco como avaliação do desempenho de médicos, serviços ou hospitais.

## Dados

- **Fonte:** UCI *Diabetes 130-US hospitals for years 1999–2008* (Strack et al., 2014,
  DOI 10.1155/2014/781670). Um encontro (internamento) por linha.
- **População (D1–D7):** 101 766 encontros brutos; saem os óbitos (não podem ser readmitidos), 59 encontros
  impossíveis e 3 com `gender` Unknown/Invalid. Ficam 100 052 encontros de 70 412 doentes.
- **Target (D6):** 1 se `readmitted == "<30"`; 0 se `>30` ou `NO`. Positivos: 11,34%.
- **Split (D13):** por doente (`StratifiedGroupKFold`, seed 42): teste com 20% dos doentes (20 011 encontros),
  fechado até à T3.6; treino com 80 041 encontros e 5 folds de validação cruzada. O mesmo doente nunca está
  no treino e na validação.

## Modelo

- `HistGradientBoostingClassifier` do scikit-learn (D17): `learning_rate=0.05`, `max_iter=300`,
  `max_leaf_nodes=31`, `min_samples_leaf=100`, `l2_regularization=1.0`, sem early stopping (a validação interna
  do scikit-learn não separa por doente). O tuning foi medido nos folds por doente e não melhorou o modelo
  (D27); duas features novas (comorbilidades de Charlson, HbA1c × medicação) também não (D28).
- **34 features** conhecidas no momento da alta (D8–D15): demografia, tipo, origem e destino da alta,
  especialidade, pagador, duração e atos do internamento, diagnósticos agrupados (Strack et al.), análises,
  medicação e visitas no ano anterior. Sem identificadores e sem agregações por doente (D11–D12).
- **Pré-processamento (D16):** ajustado só no treino; `log1p` nas contagens, escala nas numéricas, one-hot nas
  categóricas com as raras (< 100) agrupadas; valores em falta como categoria própria.
- Retreinado com as 80 041 altas de treino. Gravado com **skops**: o `load_model()` só aceita os tipos revistos.

## Avaliação (validação cruzada, 5 folds por doente)

| Métrica | Modelo v1 | Ordenar por internamentos anteriores | Ao acaso |
|---|---|---|---|
| PR-AUC (todas as altas) | 0,234 ± 0,008 | 0,176 | 0,113 |
| ROC-AUC | 0,677 | 0,607 | 0,50 |
| **Recall@30% (altas elegíveis; capacidade, D26)** | **52,2%** | **49,6%** | **30,0%** |
| Precisão@30% (altas elegíveis) | 16,6% | 15,8% | 9,6% |
| Recall@20% / @10% (sensibilidade) | 40,1% / 25,0% | 37,6% / 24,0% | 20,0% / 10,0% |
| Readmissões encontradas por 1 000 altas elegíveis, a 30% | 50 | 47 | 29 |

- **Calibração:** nas altas elegíveis, o risco previsto e o observado diferem no máximo 0,8 pp por decil; Brier
  0,083 contra 0,086 de prever sempre a taxa base. Não foi preciso recalibrar.
- **Cortes por capacidade** (sobre as previsões out-of-fold das altas elegíveis, no `config.json`): risco ≥
  0,103 para os 30% de maior risco (D26); 0,125 para os 20% e 0,169 para os 10% (sensibilidade).
- **Até 30%, cada follow-up a mais encontra mais readmissões do que ao acaso;** a partir daí rende menos
  (curva de ganhos, `reports/figures/gains_curve_v1.png`).
- **Variáveis que mais pesam** (importância por permutação, perda de PR-AUC): internamentos no ano anterior
  (−0,052) e destino da alta (−0,050); muito atrás, pagador, especialidade e visitas anteriores.
- Para comparação, a literatura neste dataset fica em ROC-AUC 0,65–0,70.

## Avaliação no teste

Avaliação única, a 28/11, com o modelo e o corte congelados (D27–D29): 20 011 altas de 14 103 doentes que nunca
foram usadas antes (D13), das quais 13 101 elegíveis (D21). IC 95% por bootstrap agrupado por doente.
`scripts/evaluate_test_v1.py`, `reports/test_results_v1.json`.

| Meta da D20 | Teste | Cumpre? |
|---|---|---|
| PR-AUC acima da regra (todas as altas) | 0,227 [0,210; 0,245] contra 0,169 [0,156; 0,184] | sim |
| Recall@30% acima da regra (elegíveis) | 52,4% contra 51,1%: +1,3 pp [−0,8; +3,3] | sim, mas o IC inclui o zero |
| Previsto e observado a menos de 2 pp em cada decil | 2,2 pp nas elegíveis (decil 10: 24,6% previsto, 22,4% observado); 0,8 pp em todas as altas | não, por 0,25 pp |
| Brier abaixo de prever sempre a taxa base (elegíveis) | 0,082 contra 0,085 | sim |
| 30% ± 1 pp das altas elegíveis com follow-up pelo corte gravado | 31,0% | sim |
| Readmissões encontradas por 1 000 altas acima da regra | 49 contra 48 | sim |

Leitura: o modelo generaliza (PR-AUC 0,227 no teste contra 0,234 na validação) e ordena claramente melhor do
que a regra simples. Na decisão a 30% de capacidade, a vantagem sobre a regra é pequena e incerta (+1,3 pp; na
validação, +2,6 pp). O risco é fiável em todas as altas; nas elegíveis, o decil de maior risco sobrestima cerca
de 2 pp. Com o corte gravado, 656 das 1 230 readmissões das altas elegíveis entram na lista (53,3%), com
precisão de 16,2%. Nada foi afinado no teste.

## Equidade

Com 30% de capacidade (D26), nas previsões out-of-fold das altas elegíveis (recall no total: 52,2%):

- **Sexo:** equilibrado (com follow-up 29,6% das mulheres e 30,5% dos homens; recall 52,4% e 52,0%).
- **Raça em falta:** só 12,0% destas altas entram na lista (recall 24,4%, contra 46–55% nos outros grupos).
- **90–99 anos:** recall 45,6%, 7 pp abaixo do total, apesar da taxa de readmissão mais alta (12,5%).
- Grupos com menos de 100 altas não se reportam.

Quem revê a lista deve conhecer estas diferenças; a monitorização segue a % de altas com follow-up e o recall
por grupo ([monitoring.md](monitoring.md)).

## Limites

- **Não causal:** diz quem tem mais risco, não a quem o follow-up evita uma readmissão. Medi-lo exige um
  piloto com grupo de controlo.
- **Labels incompletos:** os `>30` contam como negativos, e as readmissões noutros hospitais não ficam
  registadas.
- **Pressupostos de disponibilidade na alta** (P14–P16): diagnósticos, análises e medicação são tratados como
  conhecidos no momento da alta.
- **Dados antigos e sem datas:** não há validação no tempo; a capacidade é uma % das altas, não “por dia”.
- **Sinal fraco:** a vantagem sobre a regra simples é de cerca de 2,6 pp de recall a 20% e a 30% e pequena a
  10%. A 30%, cerca de 1 readmissão em cada 6 follow-ups.
- **Sobreajuste:** PR-AUC 0,379 no treino contra 0,234 na validação. Modelos mais regularizados validaram
  pior (D27), por isso esta diferença não esconde desempenho perdido.

## Reprodutibilidade

Python 3.13.13, scikit-learn 1.9.1, skops 0.16.0 (as mesmas versões em `requirements.txt`); seed 42.
Treino e gravação: `notebooks/06_deployment.ipynb`. Decisões D1–D23 em `docs/decisions.md`.
