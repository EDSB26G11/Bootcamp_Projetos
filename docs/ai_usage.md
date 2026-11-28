# Uso de IA no projeto

A W1 e a W3 pedem que se diga como foi usada a IA e como se garantiu um uso responsável. Este documento
resume o que fizemos. O registo detalhado, sessão a sessão, é interno.

## Que ferramenta e para quê

Usámos um assistente de programação com IA (Claude Code, da Anthropic, no VS Code) como apoio ao trabalho da
equipa, sobretudo em tarefas de execução:

- **Código:** primeiras versões de funções em `src/`, testes em `tests/`, refatorações e correções, a partir
  do que a equipa definia (o que calcular, com que regras e com que dados).
- **Verificação:** procurar inconsistências entre o código, os números e os documentos (por exemplo, números
  que já não batiam depois de uma decisão nova) e propor como as corrigir.
- **Documentação e gráficos:** rascunhos de texto (README, docs, notas do deck) e o código dos gráficos no
  padrão IBCS, que a equipa reviu e ajustou.
- **Pesquisa de apoio:** por exemplo, a caracterização dos 23 medicamentos do dataset
  ([anexo_medicamentos.md](anexo_medicamentos.md)), com as afirmações não confirmadas assinaladas.

## O que ficou com a equipa

- **O enquadramento e as decisões.** A pergunta, quem pode receber follow-up, a capacidade, as métricas e o
  que fica fora do projeto foram decididos pela equipa. Cada decisão está em [decisions.md](decisions.md), com
  a evidência e a data. A IA preparava opções com números; várias propostas foram recusadas ou reduzidas
  (por exemplo, métodos fora das aulas, D19, e a componente LLM, D25).
- **A leitura dos resultados.** O que os números querem dizer para o hospital, os limites e o que se diz à
  Direção de Care Quality.
- **A revisão.** O código e os textos produzidos com apoio da IA foram lidos e alterados pela equipa antes de
  entrarem no repositório.

## Como garantimos um uso responsável

- **Os números vêm do código, não do texto.** Os valores do relatório e do deck são calculados por funções
  versionadas (por exemplo, `src/deck_figures.py` gera `reports/deck_numbers.json`) e podem ser
  reproduzidos. Uma frase escrita pela IA sem número verificável não entra no relatório.
- **Testes automáticos.** O GitHub Actions corre os testes (`pytest`) em cada push; a app só é publicada se
  passarem.
- **O teste fica fechado.** O conjunto de teste (D13) não foi usado para escolher nada; a IA trabalhou só com o
  treino e as previsões out-of-fold.
- **Verificar o que é crítico** (aula W4): afirmações sobre os dados, o enunciado ou as aulas foram
  confirmadas nas fontes (enunciado, slides, o próprio dataset). Quando não foi possível, ficaram assinaladas
  como pressupostos ([decisions.md](decisions.md), P1–P28).
- **Dados.** O dataset é público (UCI, anonimizado). Não foram partilhados dados pessoais nem credenciais com a
  ferramenta.

## Limites

- A IA pode errar e escrever com mais confiança do que a evidência permite. Por isso os números passam pelo
  código e pelos testes, e as decisões pela equipa.
- O modelo do projeto não usa IA generativa: é um HistGradientBoosting treinado nos dados (D17). O Kumo
  Tabular, um modelo pré-treinado do Hugging Face, foi avaliado como desafiante e não entrou na app (D23).
