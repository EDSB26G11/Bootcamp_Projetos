"""
Preparação dos dados do projeto de readmissão hospitalar de diabetes.

Lê os ficheiros originais em data/raw/, aplica as regras descritas no guia
do projeto (IDS_mapping.csv não é uma tabela retangular, "?" representa
dados em falta, diag/age/weight são texto e não números) e grava o
resultado limpo em data/processed/.

A leitura/validação está em src/data/load.py, o parser dos IDs em
src/data/mappings.py e a exclusão de óbitos/hospício (T1.4) em
src/data/population.py. Correr a partir da raiz do repositório:
    python -m src.data_preparation
"""

from src.data.load import load_diabetic_data, validate_diabetic_data
from src.data.mappings import decode_ids, parse_ids_mapping
from src.data.population import exclude_dead_hospice

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


def main() -> None:
    mappings = parse_ids_mapping(f"{RAW_DIR}/IDS_mapping.csv")
    df = load_diabetic_data(f"{RAW_DIR}/diabetic_data.csv")
    validate_diabetic_data(df)
    df = decode_ids(df, mappings)
    df = exclude_dead_hospice(df)

    output_path = f"{PROCESSED_DIR}/diabetic_data_clean.csv"
    df.to_csv(output_path, index=False)
    print(f"Guardado: {output_path} ({df.shape[0]} linhas, {df.shape[1]} colunas)")


if __name__ == "__main__":
    main()
