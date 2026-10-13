# src

Código-fonte do projeto: scripts e/ou notebooks de pré-processamento, feature engineering, modelação e avaliação do modelo.

- `data/load.py` — leitura e validação do `diabetic_data.csv` (T1.1)
- `data/mappings.py` — parser do `IDS_mapping.csv` e descodificação dos IDs (T1.2)
- `data_preparation.py` — orquestra os passos acima e grava `data/processed/diabetic_data_clean.csv`

Correr a partir da raiz do repositório: `python -m src.data_preparation`
