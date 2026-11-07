# Framing da decisão e critérios de sucesso

Este documento responde às tarefas T0.5 (framing da decisão) e T0.6
(critérios de sucesso a 3 níveis). Começa na decisão que o modelo apoia
(aula W1, "Start from the decision"; aula W8, "Start from the decision the
page supports, not from a model metric") e só depois chega às métricas.

## 1. A decisão

| Pergunta | Resposta |
|---|---|
| **Quem decide** | O Director of Care Quality da rede hospitalar (guia do projeto). Quem executa é uma equipa de follow-up com capacidade limitada. |
| **Que decisão** | No momento da alta, que doentes recebem follow-up adicional e por que ordem. |
| **Quem pode entrar na lista** | Só quem volta para a comunidade sem acompanhamento (D21). Saem as altas em que o doente continua internado, vai para outro hospital ou instituição (cuidados continuados, reabilitação, longa duração, psiquiatria) ou já sai com apoio domiciliário ou cuidados paliativos. Sem esta regra, 59% da lista a 10% iria para estes doentes. O modelo estima o risco de todos; a lista só ordena os elegíveis (52 436 das 80 041 altas de treino, 65,5%). |
| **Ação** | Contactar (telefone, consulta precoce, revisão da medicação) as altas com maior risco de readmissão em menos de 30 dias, até esgotar a capacidade. |
| **Baseline atual** | Não há um modelo em uso. A comparação justa é com uma regra simples que um hospital já podia aplicar: ordenar as altas pelo número de internamentos no ano anterior (`number_inpatient`). A ordem ao acaso é o mínimo. |
| **Restrição** | Capacidade: a equipa só consegue contactar uma fração das altas elegíveis. Trabalhamos com **10% e 20%**; a escolha final é da equipa de negócio (decisão em aberto no roadmap). |
| **Custo de errar** | Falso negativo: uma readmissão que não recebeu follow-up. Falso positivo: um contacto gasto num doente que não seria readmitido. Não temos custos reais, por isso não se inventa uma análise custo-benefício: a capacidade fixa quantos contactos há, e o modelo decide a quem. |

## 2. Limites (a repetir no relatório, no deck e na app)

- **Não é causal.** O modelo estima o risco observado de readmissão. Não prova
  que o follow-up evita readmissões (guia do projeto). Para isso seria preciso
  um piloto ou experiência.
- **Labels incompletos.** `readmitted` só vê os registos destes hospitais: um
  `NO` pode ser uma readmissão noutro hospital (guia do projeto). Os `>30`
  contam como negativos (D6).
- **Dados de 1999–2008, sem datas.** Não há períodos para ver se o modelo
  continua certo ao longo do tempo; a validação é feita por doente (D13).
- **Revisão humana.** O modelo ordena; uma pessoa decide cada contacto.

## 3. Critérios de sucesso a 3 níveis

Cada nível tem uma medida, uma meta (o PL, no vocabulário do IBCS da W8),
o valor atual e como se confirma. Os valores atuais são das previsões
out-of-fold do treino (5 folds por doente) com o modelo v1
(HistGradientBoosting), **só nas altas elegíveis** (D21). A meta só se
confirma uma vez, no teste, na avaliação final (T3.6).

| Nível | Medida | Meta (PL) | Atual (OOF, k = 10%) | Como se confirma |
|---|---|---|---|---|
| **Modelo** | PR-AUC; recall e precisão no top k% | Bater a ordenação por `number_inpatient` à mesma capacidade, em PR-AUC e em recall@10% e @20% | PR-AUC 0,202 (ordenação por `number_inpatient`: 0,169; ao acaso: 0,096). Recall@10% 25,0% (24,0%; 10%): +1,3 pp ± 1,2 entre folds, melhor em 4 de 5. Recall@20% 40,1% (37,6%; 20%): +2,6 pp ± 1,3, melhor nos 5 | Teste, uma vez (T3.6) |
| **Operacional** | Fração de altas elegíveis sinalizadas pelo corte, face à capacidade | O corte gravado (`threshold_k10`, `threshold_k20` em `config.json`) sinaliza k ± 1 pp das altas elegíveis do teste | Nas OOF, 10,0% e 20,0% por construção | Teste, uma vez (B6 no `04_tuning_selection`) |
| **Negócio** | Readmissões em menos de 30 dias encontradas por cada 1 000 altas elegíveis | Mais do que a ordenação por `number_inpatient` à mesma capacidade | k = 10%: 24 por 1 000 altas elegíveis (23 com `number_inpatient`, 10 ao acaso), 1 255 contra 1 201 readmissões encontradas. k = 20%: 38 (36; 19), 2 011 contra 1 882 | Teste, uma vez; no deck, a página "modelo contra baseline" |

### Porque mudou o critério operacional

A versão do roadmap falava em "doentes sinalizados por dia". O dataset não tem
datas, por isso não é possível medir por dia. A capacidade passa a ser uma
fração das altas; para traduzir em contactos por dia, basta multiplicar pelo
número de altas diárias de cada hospital, que é um dado da operação e não
do modelo.

### O que mudou com D21 (7/11)

Antes de D21, a lista incluía as altas para instituições, que têm taxas de
readmissão altas (reabilitação 27,9%) e que o modelo põe no topo. A vantagem
sobre a ordenação por `number_inpatient` parecia maior (a 10%: +349
readmissões encontradas; agora +54). Só nas altas elegíveis, a vantagem é
clara a 20% e pequena a 10%. Isto pesa na escolha da capacidade e nas metas
(T0.6, por validar em equipa).

### O que não é critério de sucesso

- **Accuracy.** Prever sempre "não readmitido" dá 88,7%.
- **Um AUC alto por si só.** Na literatura sobre este dataset, os melhores
  modelos ficam em ROC-AUC ≈ 0,65–0,70. O que interessa é quantas readmissões
  a equipa encontra com os contactos que tem.

## 4. Onde está a evidência

- Métricas e registo: `reports/experiments.csv` e `reports/experiments_v1.csv`.
- Capacidade, comparação à mesma capacidade e matrizes de confusão:
  `notebooks/04_tuning_selection.ipynb`, `reports/capacity_comparison_v1.csv`,
  `reports/confusion_k10.csv`, `reports/confusion_k20.csv`.
- Curva de ganhos: `reports/figures/gains_curve_v1.png`.
