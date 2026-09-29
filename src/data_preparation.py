"""
Preparação dos dados do projeto de readmissão hospitalar de diabetes.

Lê os ficheiros originais em data/raw/, aplica as regras descritas no guia
do projeto (IDS_mapping.csv não é uma tabela retangular, "?" representa
dados em falta, diag/age/weight são texto e não números) e grava o
resultado limpo em data/processed/.
"""

import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


def parse_ids_mapping(path: str) -> dict[str, pd.DataFrame]:
    """Divide o IDS_mapping.csv nas 3 tabelas de lookup que contém.

    O ficheiro tem 3 tabelas (admission_type_id, discharge_disposition_id,
    admission_source_id) separadas por uma linha em branco e um novo
    cabeçalho. Não pode ser lido com um único pd.read_csv().
    """
    with open(path, encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    # As tabelas são separadas por linhas vazias ou só com uma vírgula (",").
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.strip() in ("", ","):
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(line)
    if current:
        blocks.append(current)

    mappings = {}
    for block in blocks:
        header = block[0].split(",")[0]  # ex: "admission_type_id"
        csv_text = "\n".join(block)
        # keep_default_na=False: algumas descrições são literalmente a
        # string "NULL" (uma categoria válida, ex: admission_type_id 6),
        # e o pandas converte "NULL"/"None" para NaN por omissão.
        df = pd.read_csv(pd.io.common.StringIO(csv_text), keep_default_na=False)
        df.iloc[:, 1] = df.iloc[:, 1].str.strip()
        mappings[header] = df

    return mappings


def load_diabetic_data(path: str) -> pd.DataFrame:
    """Lê o diabetic_data.csv preservando os campos que são bandas/códigos.

    "?" é o único marcador de dados em falta usado no ficheiro, por isso é
    convertido para NaN. age, weight e diag_1/2/3 ficam como texto porque
    representam intervalos ou códigos ICD-9, não valores numéricos.
    """
    dtype_as_string = {
        "age": "string",
        "weight": "string",
        "diag_1": "string",
        "diag_2": "string",
        "diag_3": "string",
        "payer_code": "string",
    }
    df = pd.read_csv(path, na_values="?", dtype=dtype_as_string)
    return df


def decode_ids(df: pd.DataFrame, mappings: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Junta as descrições dos 3 códigos (tipo/destino/origem) ao conjunto de dados."""
    df = df.copy()
    for id_col, mapping_df in mappings.items():
        description_col = f"{id_col}_desc"
        lookup = mapping_df.set_index(id_col).iloc[:, 0]
        df[description_col] = df[id_col].map(lookup)
    return df


def main() -> None:
    mappings = parse_ids_mapping(f"{RAW_DIR}/IDS_mapping.csv")
    df = load_diabetic_data(f"{RAW_DIR}/diabetic_data.csv")
    df = decode_ids(df, mappings)

    output_path = f"{PROCESSED_DIR}/diabetic_data_clean.csv"
    df.to_csv(output_path, index=False)
    print(f"Guardado: {output_path} ({df.shape[0]} linhas, {df.shape[1]} colunas)")


if __name__ == "__main__":
    main()
