import os

import pandas as pd
import pytest

from src.data.mappings import decode_ids, parse_ids_mapping_lines, parse_ids_mapping

RAW_MAPPING = "data/raw/IDS_mapping.csv"

SAMPLE = [
    "admission_type_id,description",
    "1,Emergency",
    "6,NULL",
    ",",
    "discharge_disposition_id,description",
    "1,Discharged to home",
    "",
    "admission_source_id,description",
    "7,Emergency Room ",
]


def test_parse_splits_three_tables():
    mappings = parse_ids_mapping_lines(SAMPLE)
    assert list(mappings) == [
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
    ]


def test_null_description_is_kept_as_string():
    mappings = parse_ids_mapping_lines(SAMPLE)
    desc = mappings["admission_type_id"].set_index("admission_type_id")["description"]
    assert desc[6] == "NULL"


def test_descriptions_are_stripped():
    mappings = parse_ids_mapping_lines(SAMPLE)
    assert mappings["admission_source_id"].iloc[0, 1] == "Emergency Room"


def test_wrong_number_of_tables_raises():
    with pytest.raises(ValueError):
        parse_ids_mapping_lines(SAMPLE[:4])


def test_decode_ids_adds_desc_columns_and_keeps_codes():
    mappings = parse_ids_mapping_lines(SAMPLE)
    df = pd.DataFrame(
        {"admission_type_id": [1, 6], "discharge_disposition_id": [1, 1], "admission_source_id": [7, 7]}
    )
    out = decode_ids(df, mappings)
    assert out["admission_type_id_desc"].tolist() == ["Emergency", "NULL"]
    assert out["admission_type_id"].tolist() == [1, 6]


@pytest.mark.skipif(not os.path.exists(RAW_MAPPING), reason="data/raw não disponível")
def test_real_file():
    mappings = parse_ids_mapping(RAW_MAPPING)
    assert len(mappings) == 3
    desc = mappings["admission_type_id"].set_index("admission_type_id")["description"]
    assert desc[6] == "NULL"
