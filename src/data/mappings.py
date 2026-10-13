"""
Parser do IDS_mapping.csv e descodificação dos IDs (T1.2).
"""

import io

import pandas as pd

EXPECTED_MAPPINGS = (
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
)


def parse_ids_mapping(path: str) -> dict[str, pd.DataFrame]:
    """Divide o IDS_mapping.csv nas 3 tabelas de lookup que contém.

    O ficheiro tem 3 tabelas (admission_type_id, discharge_disposition_id,
    admission_source_id) separadas por uma linha em branco e um novo
    cabeçalho. Não pode ser lido com um único pd.read_csv().
    """
    with open(path, encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    return parse_ids_mapping_lines(lines)


def parse_ids_mapping_lines(lines: list[str]) -> dict[str, pd.DataFrame]:
    """Igual a parse_ids_mapping, mas a partir de linhas já lidas (testável)."""
    # As tabelas são separadas por linhas vazias ou só com uma vírgula (",").
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.strip().rstrip("\r") in ("", ","):
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(line.rstrip("\r"))
    if current:
        blocks.append(current)

    mappings = {}
    for block in blocks:
        header = block[0].split(",")[0]  # ex: "admission_type_id"
        csv_text = "\n".join(block)
        # keep_default_na=False: algumas descrições são literalmente a
        # string "NULL" (uma categoria válida, ex: admission_type_id 6),
        # e o pandas converte "NULL"/"None" para NaN por omissão.
        df = pd.read_csv(io.StringIO(csv_text), keep_default_na=False)
        df.iloc[:, 1] = df.iloc[:, 1].str.strip()
        mappings[header] = df

    if set(mappings) != set(EXPECTED_MAPPINGS) or len(blocks) != len(EXPECTED_MAPPINGS):
        raise ValueError(
            f"IDS_mapping inesperado: tabelas {list(mappings)} "
            f"(esperado {list(EXPECTED_MAPPINGS)})"
        )
    return mappings


def decode_ids(df: pd.DataFrame, mappings: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Junta as descrições dos 3 códigos (tipo/destino/origem) ao conjunto de dados."""
    df = df.copy()
    for id_col, mapping_df in mappings.items():
        description_col = f"{id_col}_desc"
        lookup = mapping_df.set_index(id_col).iloc[:, 0]
        df[description_col] = df[id_col].map(lookup)
    return df
