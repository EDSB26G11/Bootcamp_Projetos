# Framing da decisão e critérios de sucesso

Este documento responde às tarefas T0.5 (framing da decisão) e T0.6
(critérios de sucesso a 3 níveis). Começa na decisão que o modelo apoia
(aula W1, "Start from the decision"; aula W8, "Start from the decision the
page supports, not from a model metric") e só depois chega às métricas.

## 0. A pergunta

> No momento da alta hospitalar, qual o risco de um doente voltar a ser
> internado em menos de 30 dias, e usar esse risco para estabelecer
> prioridades no acompanhamento adicional.

Tem duas partes, e cada uma tem a sua avaliação (D22, 7/11):

1. **Qual o risco** de cada doente. O modelo dá um risco estimado (score
   entre 0 e 1). Para o poder dizer como risco, tem de bater certo com o que
   acontece: entre altas com risco estimado de 20%, cerca de 20 em cada 100
   voltam. Avalia-se com a **calibração** (risco previsto contra observado por
   decis, Brier, log loss).
2. **Que prioridade.** As altas elegíveis (D21) ordenam-se pelo risco e a
   equipa dá follow-up às primeiras até esgotar a capacidade. Avalia-se com a
   **ordenação**: PR-AUC e recall e precisão no top k%.

A versão anterior (6/11) tratava só a segunda parte: a app mostrava a
posição, não o risco.

## 1. A decisão

| Pergunta | Resposta |
|---|---|
| **Quem decide** | A Direção de Care Quality da rede hospitalar ("Director of Care Quality" no guia do projeto). Quem executa é uma equipa de follow-up com capacidade limitada. |
| **Que decisão** | No momento da alta, que doentes recebem follow-up adicional e por que ordem. |
| **Quem pode entrar na lista** | Só quem volta para a comunidade sem acompanhamento (D21). Saem as altas em que o doente continua internado, vai para outro hospital ou instituição (cuidados continuados, reabilitação, longa duração, psiquiatria) ou já sai com apoio domiciliário ou cuidados paliativos. Sem esta regra, 59% da lista a 10% iria para estes doentes. O modelo estima o risco de todos; a lista só ordena os elegíveis (52 436 das 80 041 altas de treino, 65,5%). |
| **Ação** | Dar follow-up (telefone, consulta precoce, revisão da medicação) às altas com maior risco de readmissão em menos de 30 dias, até esgotar a capacidade. |
| **Baseline atual** | Não há um modelo em uso. A comparação justa é com uma regra simples que um hospital já podia aplicar: ordenar as altas pelo número de internamentos no ano anterior (`number_inpatient`). A ordem ao acaso é o mínimo. |
| **Restrição** | Capacidade: a equipa só consegue dar follow-up a uma fração das altas elegíveis. A equipa fixou **30%** (D26, 28/11), com 10% e 20% como análise de sensibilidade. É o limite até onde cada follow-up a mais encontra mais readmissões do que ao acaso. |
| **Custo de errar** | Falso negativo: uma readmissão que não recebeu follow-up. Falso positivo: um follow-up gasto num doente que não seria readmitido. Não temos custos reais, por isso não se inventa uma análise custo-benefício: a capacidade fixa quantos follow-ups há, e o modelo decide a quem. |

## 2. Limites (a repetir no relatório, no deck e na app)

- **Não é causal.** O modelo estima o risco observado de readmissão. Não prova
  que o follow-up evita readmissões (guia do projeto). Para isso seria preciso
  um piloto ou experiência.
- **Labels incompletos.** `readmitted` só vê os registos destes hospitais: um
  `NO` pode ser uma readmissão noutro hospital (guia do projeto). Os `>30`
  contam como negativos (D6).
- **Dados de 1999–2008, sem datas.** Não há períodos para ver se o modelo
  continua certo ao longo do tempo; a validação é feita por doente (D13).
- **Revisão humana.** O modelo ordena; uma pessoa decide cada follow-up.
- **O risco vale para estes dados.** O risco estimado está calibrado para os
  hospitais e o período do dataset (EUA, 1999–2008). Noutro hospital ou noutro
  período teria de ser verificado outra vez.

## 3. Critérios de sucesso a 3 níveis

Cada nível tem uma medida, uma meta (o PL, no vocabulário do IBCS da W8),
o valor atual e como se confirma. Os valores atuais são das previsões
out-of-fold do treino (5 folds por doente) com o modelo v1
(HistGradientBoosting), **só nas altas elegíveis** (D21). A meta só se
confirma uma vez, no teste, na avaliação final (T3.6).

**Estado: aceite (D20, 28/11).** As metas são para a capacidade de 30% (D26) e foram fixadas **antes** de
abrir o teste, para não se escolher a meta depois de ver o resultado.

| Nível | Medida | Meta (PL) | Atual (OOF, capacidade 30%) | Como se confirma |
|---|---|---|---|---|
| **Modelo: prioridade** (2.ª parte da pergunta) | PR-AUC; recall e precisão nos 30% de maior risco | Bater a ordenação por `number_inpatient` à mesma capacidade, em PR-AUC e em recall@30% | PR-AUC 0,202 (ordenação por `number_inpatient`: 0,169; ao acaso: 0,096). Recall@30% 52,2% (49,6%; ao acaso 30%): +2,6 pp ± 1,8 entre folds, melhor em 4 de 5. Sensibilidade: recall@20% 40,1% (37,6%), recall@10% 25,0% (24,0%) | Teste, uma vez (T3.6) |
| **Modelo: risco** (1.ª parte da pergunta) | Risco previsto contra readmissões observadas, por decis; Brier | Em cada decil, previsto e observado a menos de 2 pp; Brier abaixo do de prever sempre a taxa base | Previsto e observado a menos de 1 pp em todos os decis (maior diferença: 0,6 pp em todas as altas, 0,8 pp nas elegíveis). Brier nas elegíveis 0,083 (prever sempre a taxa base: 0,086). `reports/calibration_v1.csv` | Teste, uma vez (T3.6) |
| **Operacional** | Fração de altas elegíveis com follow-up pelo corte, face à capacidade | O corte gravado (`threshold_k30` em `config.json`) sinaliza 30% ± 1 pp das altas elegíveis do teste | Nas OOF, 30,0% por construção | Teste, uma vez (B6 no `04_tuning_selection`) |
| **Negócio** | Readmissões em menos de 30 dias encontradas por cada 1 000 altas elegíveis | Mais do que a ordenação por `number_inpatient` à mesma capacidade | 50 por 1 000 altas elegíveis (47 com `number_inpatient`, 29 ao acaso): 2 615 contra 2 487 readmissões encontradas em 15 731 follow-ups | Teste, uma vez; no deck, o diapositivo "modelo contra a regra" |

**Resultado no teste (28/11, avaliação única):** cinco das seis metas cumprem-se. PR-AUC 0,227 contra 0,169
da regra; recall@30% 52,4% contra 51,1% (+1,3 pp, IC 95% de −0,8 a +3,3); 49 readmissões por 1 000 altas
contra 48; Brier 0,082 contra 0,085; 31,0% das altas elegíveis com follow-up pelo corte gravado. Falha a
calibração nas elegíveis por 0,25 pp: no decil de maior risco, 24,6% previsto contra 22,4% observado (em
todas as altas, a maior diferença é 0,8 pp). Detalhe no [model card](model_card.md).

O nível de negócio que interessa à Direção (menos readmissões) não se mede com dados históricos: o modelo
prevê, não prova que o follow-up evita o regresso. Fica proposto um piloto com sorteio dentro da lista.

### Porque mudou o critério operacional

A primeira versão dos critérios falava em "doentes sinalizados por dia". O dataset não tem
datas, por isso não é possível medir por dia. A capacidade passa a ser uma
fração das altas; para traduzir em follow-ups por dia, basta multiplicar pelo
número de altas diárias de cada hospital, que é um dado da operação e não
do modelo.

### O que mudou com D21 (7/11)

Antes de D21, a lista incluía as altas para instituições, que têm taxas de
readmissão altas (reabilitação 27,9%) e que o modelo põe no topo. A vantagem
sobre a ordenação por `number_inpatient` parecia maior (a 10%: +349
readmissões encontradas; agora +54). Só nas altas elegíveis, a vantagem é
clara a 20% e a 30% e pequena a 10%. Por isso a capacidade ficou em 30% (D26).

### O que não é critério de sucesso

- **Accuracy.** Prever sempre "não readmitido" dá 88,7%.
- **Um AUC alto por si só.** Na literatura sobre este dataset, os melhores
  modelos ficam em ROC-AUC ≈ 0,65–0,70. O que interessa é quantas readmissões
  a equipa encontra com os follow-ups que tem.

## 4. Onde está a evidência

- Métricas e registo: `reports/experiments.csv` e `reports/experiments_v1.csv`.
- Capacidade, comparação à mesma capacidade e matrizes de confusão:
  `notebooks/04_tuning_selection.ipynb`, `reports/capacity_comparison_v1.csv`,
  `reports/confusion_k10.csv`, `reports/confusion_k20.csv`; a 30% (D26), matrizes e números em
  `reports/deck_numbers.json` (`src/deck_figures.py`).
- Curva de ganhos: `reports/figures/gains_curve_v1.png`.
