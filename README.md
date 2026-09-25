# Bootcamp_Projetos

Projeto do grupo para o **Enterprise Data Science Bootcamp (EDSB 2026-27)** — NOVA IMS.

## Visão geral do projeto

O objetivo deste projeto é construir e avaliar um modelo de classificação binária para prever o diagnóstico de diabetes, com base num dataset clínico de pacientes (ex.: Pima Indians Diabetes Dataset ou equivalente).

- Dataset: a definir/adicionar (dados clínicos de pacientes para diagnóstico de diabetes)
- Variável alvo: diagnóstico de diabetes (positivo/negativo)
- Tópicos relevantes do curso aplicados aqui: tratamento de dados em falta/desequilibrados, feature engineering, avaliação de modelos de classificação

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
   source .venv/bin/activate.     # macOS/Linux
   .venv\Scripts\activate.        # Windows
   ```
3. Instalar as dependências:
   ```
   pip install -r requirements.txt
   ```

## Como executar

- O código do projeto (pré-processamento, modelação, avaliação) deve ser colocado em `/src`.
- Documentação e relatórios devem ser colocados em `/docs`.
- Para correr um script: `python src/nome_do_script.py`
- Para correr notebooks: `jupyter notebook` (ou abrir diretamente no VS Code)

## Estrutura do repositório

```
Bootcamp_Projetos/
├── src/.             # código-fonte do projeto
├── docs/.            # documentação e relatórios
├── requirements.txt. # dependências Python
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
