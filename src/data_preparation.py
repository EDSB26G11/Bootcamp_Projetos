"""
Preparação dos dados do projeto de readmissão hospitalar de diabetes.

Lê os ficheiros originais em data/raw/, aplica as regras descritas no guia
do projeto (IDS_mapping.csv não é uma tabela retangular, "?" representa
dados em falta, diag/age/weight são texto e não números) e grava o
resultado limpo em data/processed/.

A leitura/validação está em src/data/load.py, o parser dos IDs e o
agrupamento dos códigos desconhecidos (D5) em src/data/mappings.py, a
limpeza de race e dos códigos ICD-9 (D3, D4) em src/data/cleaning.py e a
população (D1, D2, D7) e o target (D6) em src/data/population.py e o split
por doente (D13) em src/data/split.py. Correr a partir da raiz do repositório:
    python -m src.data_preparation
"""

import os

from src.data.cleaning import harmonize_race, pad_icd9
from src.data.load import load_diabetic_data, validate_diabetic_data
from src.data.mappings import decode_ids, group_unknown_ids, parse_ids_mapping
from src.data.population import add_target, define_population, exclude_impossible
from src.data.split import assign_splits

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
SPLITS_DIR = "data/splits"


def main() -> None:
    mappings = parse_ids_mapping(f"{RAW_DIR}/IDS_mapping.csv")
    df = load_diabetic_data(f"{RAW_DIR}/diabetic_data.csv")
    validate_diabetic_data(df)
    # D2 primeiro, sobre os dados completos: a regra C5 precisa dos óbitos.
    df = exclude_impossible(df, mappings)
    df = harmonize_race(df)
    df = pad_icd9(df)
    df = decode_ids(df, mappings)
    df = group_unknown_ids(df)
    df = define_population(df)
    df = add_target(df)

    output_path = f"{PROCESSED_DIR}/diabetic_data_clean.csv"
    df.to_csv(output_path, index=False)
    print(f"Guardado: {output_path} ({df.shape[0]} linhas, {df.shape[1]} colunas)")

    splits = assign_splits(df)
    splits_path = f"{SPLITS_DIR}/splits.csv"
    os.makedirs(SPLITS_DIR, exist_ok=True)
    splits.to_csv(splits_path, index=False)
    print(f"Guardado: {splits_path} ({splits['split'].value_counts().to_dict()})")


if __name__ == "__main__":
    main()
