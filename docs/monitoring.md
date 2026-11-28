# Plano de monitorização do modelo v1 (T4.5)

**Estado: proposta.** O modelo ainda não está em uso num hospital. Este plano diz o que medir quando
estiver, com que frequência, contra que valores de referência e o que fazer quando um valor sai do
intervalo. Os valores de referência vêm das previsões out-of-fold do treino, só nas altas elegíveis (D21),
com a capacidade de 30% (D26). O teste (28/11) confirma-os de perto: recall@30% de 52,4%, 31,0% das
altas com follow-up e risco a menos de 1 pp do observado em todas as altas (detalhe no model card).

## Porque é preciso

- **Os dados são de 1999–2008, dos EUA.** Noutro hospital ou noutro período, a mistura de doentes, os
  códigos e as práticas mudam. O risco tem de ser verificado com dados do próprio hospital antes do uso e
  depois, periodicamente.
- **O modelo é calibrado** (risco previsto e observado a menos de 1 pp por decil, D22). A app diz "em cada
  100 doentes como este, cerca de N voltam": se a calibração se perder, essa frase deixa de ser verdade.
- **O corte de 30% foi fixado nas previsões do treino** (`threshold_k30` = 0,1026 no `config.json`). Se a
  distribuição do risco mudar, o mesmo corte passa a pôr mais ou menos de 30% das altas na lista, e a equipa
  de follow-up deixa de ter a carga que planeou.
- **Há grupos que ficam para trás** (altas sem raça registada: recall de 24% contra 52% no total). Isto tem
  de ser seguido, não só reportado uma vez.

## O que medir

### 1. Entradas (todas as semanas; não precisa de esperar pelo resultado)

| Indicador | Referência (treino, altas elegíveis) | Alarme |
|---|---|---|
| % de altas elegíveis (D21) entre todas as altas | 65,5% | mudança de mais de 5 pp: verificar o registo do destino da alta |
| % de altas com ≥ 1 internamento no ano anterior (a variável que mais pesa) | 30,1% | mudança de mais de 5 pp |
| % de altas com ≥ 1 ida à urgência no ano anterior | 10,2% | mudança de mais de 5 pp |
| % de altas com HbA1c medida | 17,9% | mudança de mais de 5 pp |
| % de altas com raça em falta | 2,3% | subida: o grupo com pior recall cresce |
| % de altas com 70 ou mais anos | 34,9% | mudança de mais de 5 pp |
| Valores novos nas categóricas (destino, origem, especialidade, pagador) | nenhum | qualquer valor novo: o modelo trata-o como "infrequente" (D16) e convém rever |

Os limites de 5 pp são um ponto de partida para a equipa afinar com os primeiros meses de dados, não um
valor estatístico.

### 2. Saídas (todas as semanas)

| Indicador | Referência | Alarme |
|---|---|---|
| % de altas elegíveis com follow-up pelo corte gravado | 30,0% | fora de 30% ± 1 pp (o critério operacional da D20) |
| Risco médio previsto nas altas elegíveis | 9,6% | mudança de mais de 1 pp |
| % de altas com follow-up por grupo (sexo, raça, idade) | por grupo em `reports/deck_numbers.json` | um grupo a afastar-se do seu valor de referência |

### 3. Resultado (todos os meses; só 30 dias depois da alta se sabe se houve readmissão)

| Indicador | Referência | Alarme |
|---|---|---|
| Taxa de readmissão < 30 dias nas altas elegíveis | 9,6% | mudança de mais de 2 pp |
| Calibração: risco previsto contra observado, por decil | diferença < 1 pp em todos os decis | algum decil com diferença ≥ 2 pp (meta da D20) |
| Recall nos 30% de maior risco | 52,2% | abaixo do recall da regra simples (internamentos anteriores) no mesmo mês |
| Precisão nos 30% de maior risco | 16,6% (cerca de 1 em cada 6 follow-ups) | descida continuada durante 3 meses |
| Recall por grupo (sexo, raça, idade) | por grupo em `reports/deck_numbers.json` | um grupo com recall mais de 10 pp abaixo do total |

Atenção: as readmissões noutros hospitais não aparecem nos dados (D6). O recall medido é um mínimo.

### 4. Operação (todos os meses, com a equipa de follow-up)

- % das altas na lista que receberam de facto o follow-up;
- tempo entre a alta e o follow-up;
- casos em que quem revê a lista discordou do modelo, e porquê.

Estes números dizem se a lista está a ser usada como previsto. Não dizem se o follow-up evita
readmissões: isso só um piloto com sorteio dentro da lista mostra (o modelo prevê, não prova causa).

## O que fazer quando há alarme

| Situação | Ação |
|---|---|
| Lista fora de 30% ± 1 pp, com o resto estável | recalcular o corte para 30% com as altas recentes (só o corte, sem retreinar) |
| Calibração perdida (algum decil ≥ 2 pp), ordenação estável | recalibrar ou retreinar com dados recentes e voltar a validar por doente (D13) |
| Recall abaixo da regra simples | parar de usar a lista e voltar à regra até haver novo modelo validado |
| Grupo a ficar para trás | avisar quem revê a lista; investigar o registo desse grupo; decidir em equipa (decisions.md) |
| Valor novo numa categórica ou mudança de registo | rever com quem faz o registo antes de mexer no modelo |

Qualquer modelo novo passa pelo mesmo processo do v1: validação cruzada por doente, comparação com a regra
simples à mesma capacidade, um teste usado uma só vez e uma decisão registada em `docs/decisions.md`.

## Quem e como

- **Responsável pelo modelo:** a definir pela Direção de Care Quality. Revê os indicadores, decide as
  ações e regista-as.
- **Dados:** as previsões e as entradas de cada alta ficam guardadas com a data, para calcular os
  indicadores quando o resultado chegar (30 dias depois).
- **Código:** os indicadores 2 e 3 usam as mesmas funções do projeto (`src/models/threshold.py`,
  `src/models/evaluate.py`: `capacity_table`, `calibration_table`, `metrics_by_group`), para os números
  serem comparáveis com os do relatório.
