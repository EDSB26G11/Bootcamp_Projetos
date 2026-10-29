# src

Código-fonte do projeto: scripts e/ou notebooks de pré-processamento, feature engineering, modelação e avaliação do modelo.

- `data/load.py` — leitura e validação do `diabetic_data.csv` (T1.1)
- `data/mappings.py` — parser do `IDS_mapping.csv`, descodificação dos IDs (T1.2) e agrupamento dos códigos desconhecidos (D5)
- `data/cleaning.py` — `race` harmonizada por doente (D3) e zeros à esquerda nos códigos ICD-9 (D4)
- `data/population.py` — população elegível (sem encontros impossíveis, D2; sem óbitos, D1; sem `gender` inválido, D7; hospício mantém-se) e target `<30` (D6)
- `data/quality.py` — regras de consistência usadas na EDA (§1.1); só identificam (a exclusão de D2 está em `population.py`)
- `data_preparation.py` — orquestra os passos acima e grava `data/processed/diabetic_data_clean.csv`

Correr a partir da raiz do repositório: `python -m src.data_preparation`
