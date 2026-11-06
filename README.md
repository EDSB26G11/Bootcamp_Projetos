# Projeto: Readmissão hospitalar a 30 dias em doentes com diabetes


Projeto do grupo para o **Enterprise Data Science Bootcamp (EDSB 2026-27)** — NOVA IMS.

## Visão geral do projeto

**Pergunta.** No momento da alta, que doentes com diabetes têm maior risco de
voltar a ser internados em **menos de 30 dias**?

**Para quem.** Para uma equipa de follow-up que só consegue contactar uma parte
das altas (por exemplo, 10% ou 20%). O modelo ordena as altas por risco, para
os contactos chegarem primeiro a quem mais precisa.

**Dados.** Diabetes 130-US hospitals (1999–2008), em `data/raw/diabetic_data.csv`:
um encontro (internamento) por linha, com dados demográficos, diagnósticos,
análises, medicação e histórico de utilização. Depois das exclusões
([docs/decisions.md](docs/decisions.md), D1–D7) ficam 100 052 encontros de
70 412 doentes.

**Target.** `target` = 1 se `readmitted == "<30"`, 0 se `>30` ou `NO` (D6).
Positivos: 11,34%.

**Como se avalia.**

- O split é feito por doente: o mesmo doente nunca aparece no treino e na validação (D13).
- A métrica principal é a PR-AUC.
- A métrica de negócio é o recall@10% / @20%: a fração das readmissões apanhada
  se a equipa contactar os 10% / 20% de altas com maior risco.

## Abstract

_A escrever no fim do projeto (problema, dados, modelo escolhido, resultado em recall@k, limitações)._

## Índice

O relatório do projeto é o [`Projeto_Final.ipynb`](Projeto_Final.ipynb):
corre de ponta a ponta e chama o código de `src/` e os resultados em `reports/`.
Os notebooks em `notebooks/` são as bancadas de trabalho, um por fase:

| Notebook | Fase | Tarefas |
|---|---|---|
| [00_profiling](notebooks/00_profiling.ipynb) | Exploração | profiling automático |
| [01_eda](notebooks/01_eda.ipynb) | Exploração | análise exploratória dos dados |
| [02_preprocessing_check](notebooks/02_preprocessing_check.ipynb) | Preparação | mostra e valida o que o `src/` faz |
| [03_baselines_models](notebooks/03_baselines_models.ipynb) | Modelação | baselines e modelos candidatos |
| [04_tuning_selection](notebooks/04_tuning_selection.ipynb) | Modelação | seleção do modelo, ablação, threshold pela capacidade, curva de ganhos |
| [05_interpretation_fairness](notebooks/05_interpretation_fairness.ipynb) | Avaliação | importância por permutação, métricas por idade/raça/sexo |
| [06_deployment](notebooks/06_deployment.ipynb) | Deployment | refit, gravar o modelo (skops), testar a app |

Documentação em `docs/`:

- [problem_framing.md](docs/problem_framing.md) — a decisão que o modelo apoia, os limites e os critérios de sucesso a 3 níveis
- [decisions.md](docs/decisions.md) — registo das decisões (D1–D20) e pressupostos, com a evidência de cada uma
- [data_preparation.md](docs/data_preparation.md) — o que faz `src/data_preparation.py` e como correr (leitura e validação dos dados, descodificação dos IDs)
- [eda_summary.md](docs/eda_summary.md) — resumo dos achados de `notebooks/01_eda.ipynb`
- [dicionario_variaveis.md](docs/dicionario_variaveis.md) — todas as variáveis: significado, exclusão, ajuste e variáveis derivadas
- [anexo_medicamentos.md](docs/anexo_medicamentos.md) — classes, uso e contexto dos 23 medicamentos do dataset
- [features_aprovadas.md](docs/features_aprovadas.md) — colunas conhecidas no momento da alta e as 34 features do modelo (feature engineering, seleção e pré-processamento)
- [modelacao_base.md](docs/modelacao_base.md) — modelação: regras, números de referência e estado das decisões da fase 2
- [figures/](docs/figures/) — figuras da EDA (as de modelação e as páginas do relatório ficam em `reports/figures/`)

## Pré-requisitos

- Python 3.13
- Git

## Configuração (Setup)

1. Clonar o repositório:
   ```
   git clone https://github.com/Just1Student/Bootcamp_Projetos.git
   cd Bootcamp_Projetos
   ```
2. Criar e ativar um ambiente virtual:
   ```
   python3 -m venv .venv
   source .venv/bin/activate     # macOS/Linux
   .venv\Scripts\activate        # Windows
   ```
3. Instalar as dependências:
   ```
   pip install -r requirements.txt
   ```

## Como executar

Todos os comandos correm a partir da raiz do repositório:

- Preparação dos dados: `python -m src.data_preparation` (detalhes em [docs/data_preparation.md](docs/data_preparation.md))
- Testes: `python -m pytest tests/ -v`
- Ordem dos notebooks: `03` (baselines e candidatos) → `04` (previsões out-of-fold, ablação, threshold) → `05` e `06` (leem as OOF do `04`) → `Projeto_Final`
- App (depois de gravar o modelo no `06_deployment`): `python -m src.app`. Mostra o percentil de risco da alta e a ação para 10% e 20% de capacidade
- Notebooks: `jupyter notebook` (ou abrir diretamente no VS Code)
- Profiling automático (`notebooks/00_profiling.ipynb`): o `ydata-profiling` ainda exige pandas < 3, por isso usa um ambiente próprio:
  ```
  python -m venv .venv-profiling
  .venv-profiling\Scripts\python -m pip install -r requirements-profiling.txt
  .venv-profiling\Scripts\python -m ipykernel install --user --name bootcamp-profiling --display-name "Bootcamp (profiling)"
  ```
  Abrir o notebook com o kernel *Bootcamp (profiling)*. O relatório fica em `reports/profiling_raw.html` (não versionado).

## Estrutura do repositório

```
Bootcamp_Projetos/
├── data/raw/          # dados originais (read-only)
├── data/processed/    # dados processados (gerados localmente, não versionados)
├── data/splits/       # split treino/teste e folds por doente (versionado, D13)
├── notebooks/         # bancadas de trabalho: 1 notebook = 1 fase
├── Projeto_Final.ipynb  # relatório final, corre de ponta a ponta
├── src/               # código-fonte (data/, features/, pipeline.py, models/, viz.py para os gráficos IBCS, app.py)
├── models/            # modelo final em models/v1/: model.skops, config.json e model card (gerado localmente, não versionado)
├── reports/           # experiments.csv (comparação atual), experiments_v1.csv (histórico), ablação, capacidade, matrizes de confusão e figures/
├── tests/             # testes (pytest)
├── docs/              # documentação, decisões e figuras da EDA
├── requirements.txt   # dependências Python
├── .gitignore
└── README.md
```

## Equipa

Grupo de 4 elementos — Enterprise Data Science Bootcamp 2026-27, NOVA IMS
- Ana Carapinha
- Filipa Filipe
- Sandra Figueiredo
- Tiago Marques

