# Bootcamp_Projetos

Projeto do grupo para o **Enterprise Data Science Bootcamp (EDSB 2026-27)** — NOVA IMS.

## Visão geral do projeto

O objetivo deste projeto é construir e avaliar um modelo de classificação binária para prever a readmissão de pacientes com diagnóstico de diabetes, com base num dataset clínico de pacientes.

- Dataset: diabetic_data.csv (dados clínicos de pacientes para diagnóstico de diabetes)
- Variável alvo: Readmitted (<30, >30 ou "NO")

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
├── notebooks/         # análise exploratória
├── src/               # código-fonte do projeto
├── tests/             # testes (pytest)
├── docs/              # documentação, decisões e figuras
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

*(preencher com os nomes e número de aluno de todos os elementos do grupo)*
