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
   git clone https://github.com/EDSB26G11/Bootcamp_Projetos.git
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
