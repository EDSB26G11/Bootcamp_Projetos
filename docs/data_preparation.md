# Preparação dos dados — passo a passo

Este documento explica o que o script `src/data_preparation.py` faz, porquê,
e o que verificámos ao correr sobre os dados reais. Serve para qualquer
elemento da equipa perceber o pipeline sem ter de reler o código linha a linha.

## O problema que este script resolve

Os dados originais em `data/raw/` não podem ser usados diretamente:

1. **`IDS_mapping.csv` não é uma tabela.** Contém 3 tabelas de lookup
   (`admission_type_id`, `discharge_disposition_id`, `admission_source_id`)
   separadas por linhas em branco e novos cabeçalhos. Um `pd.read_csv()`
   direto não funciona.
2. **`diabetic_data.csv` usa `?` para valores em falta.** Sem tratar isto,
   o pandas lê `?` como texto normal em vez de reconhecer que é missing.
3. **Algumas colunas parecem números mas não são.** `age` (ex: `[0-10)`) e
   `weight` (ex: `[75-100)`) são intervalos/bandas; `diag_1/2/3` são códigos
   ICD-9 (ex: `250.83`) que incluem casos especiais (`V`, `E`, decimais).
   Se o pandas as ler como número, perde-se informação ou o formato muda.

## O que o script faz

O código está dividido em módulos: `src/data/mappings.py` (T1.2: parser e
descodificação dos IDs), `src/data/load.py` (T1.1: leitura e validação),
`src/data/cleaning.py` (D3, D4: `race` por doente e códigos ICD-9),
`src/data/population.py` (T1.4/T1.5: população elegível e target),
`src/data/split.py` (T1.7: split por doente) e
`src/data_preparation.py` (só orquestra: carregar → validar → excluir
impossíveis → limpar → descodificar → agrupar desconhecidos → excluir óbitos
e `gender` inválido → target → gravar → split por doente).

### 1. `parse_ids_mapping()` — separar as 3 tabelas de lookup

Lê o `IDS_mapping.csv` linha a linha e corta o ficheiro sempre que encontra
uma linha vazia (ou só com uma vírgula), que é o separador entre as 3
tabelas. Cada bloco resultante é lido como um mini-CSV independente.

**Detalhe importante que descobrimos ao testar:** o código
`admission_type_id = 6` tem como descrição a palavra literal `"NULL"` (é uma
categoria válida do hospital, não um valor em falta). Por omissão, o pandas
interpreta a string `"NULL"` como valor em falta (`NaN`) — o que faria
desaparecer silenciosamente 5.291 categorias válidas. Por isso usamos
`keep_default_na=False` ao ler estas tabelas de lookup.

### 2. `load_diabetic_data()` — carregar o ficheiro principal

Lê `diabetic_data.csv` com `na_values="?"`, para o `?` passar a ser
reconhecido como valor em falta (`NaN`) em qualquer coluna.

As colunas `age`, `weight`, `payer_code`, `diag_1`, `diag_2`, `diag_3` são
forçadas a tipo texto (`string`), para nunca serem interpretadas como número.

Também usa `keep_default_na=False`, pelo mesmo motivo do `IDS_mapping.csv`:
em `A1Cresult` e `max_glu_serum` a string `"None"` significa **teste não
realizado** (84.748 e 96.420 linhas), não dado em falta. Sem esta opção o
pandas convertia-a em `NaN`, misturando "não medido" com "em falta". Só o
`?` é tratado como missing.

`validate_diabetic_data()` confirma depois que o ficheiro lido bate com o
guia do projeto: 101.766 linhas, 71.518 doentes (`patient_nbr` únicos),
`readmitted` só com `<30`, `>30` e `NO`, e os missings esperados em `race`
(2.273) e `weight` (98.569). Se algo não bater, levanta `ValueError` com a
lista de todos os problemas. `describe_dtypes()` devolve o tipo de cada coluna.

### 3. Encontros impossíveis e limpeza de valores

- `exclude_impossible()` remove os 59 encontros (54 doentes) que violam
  uma regra de severidade `erro` de `src/data/quality.py`: recém-nascido
  com idade ≥ 10, doente com 2 `gender`, idade a descer ou a saltar 2+
  bandas, encontro depois de um óbito. Corre **antes** de retirar os óbitos,
  porque a regra C5 precisa deles. Decisão D2.
- `harmonize_race()` dá aos 123 doentes com mais de uma `race` a moda dos
  seus encontros; em empate, a do encontro mais recente. Decisão D3.
- `pad_icd9()` repõe os zeros à esquerda nos códigos ICD-9 numéricos
  (`8` → `008`, `41.1` → `041.1`): 5.732 códigos. Decisão D4.

### 4. `decode_ids()` — juntar as descrições dos códigos

Para cada um dos 3 códigos (`admission_type_id`, `discharge_disposition_id`,
`admission_source_id`), cria uma coluna nova `<coluna>_desc` com a descrição
correspondente (ex: `admission_type_id=1` → `admission_type_id_desc="Emergency"`).
Os códigos originais são mantidos — só acrescentamos as descrições, não
substituímos nada.

Depois, `group_unknown_ids()` põe `Desconhecido` na descrição dos códigos
que querem dizer "desconhecido" (NULL, Not Available, Not Mapped,
Unknown/Invalid): `admission_type_id` 5, 6, 8; `discharge_disposition_id`
18, 25, 26; `admission_source_id` 9, 15, 17, 20, 21. O código original fica
na coluna `_id`. Decisão D5.

### 5. `define_population()` — definir a população elegível

- `exclude_deaths()` remove os encontros com alta por óbito
  (`discharge_disposition_id` 11, 19, 20, 21): 1.652 encontros. Um doente
  que morreu não pode ser readmitido. As altas para hospício (13, 14; 771
  encontros, 43 readmissões) ficam na população. Decisão D1.
- `exclude_invalid_gender()` remove os 3 encontros com `gender`
  `Unknown/Invalid`. Decisão D7.

### 6. `add_target()` — criar o target

Acrescenta `target` = 1 se `readmitted == "<30"`, 0 se `>30` ou `NO`
(11,34% de positivos). `readmitted` fica no ficheiro, mas nunca é feature.
Decisão D6. Todas as decisões estão em [decisions.md](decisions.md).

### 7. Guardar o resultado

O ficheiro final é gravado em `data/processed/diabetic_data_clean.csv`
— **100.052 linhas, 70.412 doentes, 54 colunas** (50 originais + 3 colunas `_desc` + `target`).

### 8. `assign_splits()` — split por doente

Grava `data/splits/splits.csv` com uma linha por encontro: `encounter_id`,
`patient_nbr`, `split` (`train`/`test`) e `fold` (0–4 no treino, -1 no
teste). Usa `StratifiedGroupKFold` por `patient_nbr`, estratificado pelo
`target`, com seed 42: 20% dos doentes para teste e 5 folds de CV no treino.
Nenhum doente aparece em duas partes e todas ficam com 11,34% de positivos.
O teste só se usa uma vez, na avaliação final (T3.6). Decisão D13.

O ficheiro é versionado no git, para toda a equipa usar exatamente o mesmo
split mesmo que a versão do scikit-learn mude.

## Como correr

A partir da raiz do repositório:

```bash
source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m src.data_preparation
```

Deve terminar com:

```text
Guardado: data/processed/diabetic_data_clean.csv (100052 linhas, 54 colunas)
Guardado: data/splits/splits.csv ({'train': 80041, 'test': 20011})
```

## Testes

```bash
python -m pytest tests/ -v
```

`tests/test_mappings.py` cobre o parser do `IDS_mapping.csv` (3 tabelas, `NULL`
preservado como texto, erro se o formato mudar, agrupamento dos desconhecidos) e `tests/test_load.py` cobre a
leitura (`?` → NaN, colunas de texto) e a validação, incluindo uma verificação
contra o ficheiro real em `data/raw/`. `tests/test_population.py` confirma que
só os óbitos e o `gender` inválido são removidos (o hospício fica), que o
target é `<30` contra o resto e que, no ficheiro real, há 59 encontros
impossíveis e ficam 100.052 encontros e 70.412 doentes.
`tests/test_cleaning.py` cobre a moda da `race` (com empate) e os zeros nos
códigos ICD-9. `tests/test_split.py` confirma que nenhum doente aparece em
duas partes, que o split é reprodutível e que, nos dados reais, o teste tem
20% dos encontros e todas as partes a mesma taxa de positivos.

## Aviso para quem for reler `diabetic_data_clean.csv` mais tarde

Se alguém abrir este ficheiro processado com `pd.read_csv()` sem mais
nada, o pandas converte a string `"None"` de `A1Cresult` e `max_glu_serum`
(teste não realizado) em `NaN`, e lê `diag_1–3` como número, perdendo os
zeros à esquerda repostos em D4 (`008` → `8`). As descrições `"NULL"` já não
aparecem (passaram a `Desconhecido`, D5). Para reler corretamente:

```python
text = {c: "string" for c in ["age", "weight", "payer_code", "diag_1", "diag_2", "diag_3"]}
pd.read_csv("data/processed/diabetic_data_clean.csv", keep_default_na=False, na_values="", dtype=text)
```

## Verificações feitas (para dar confiança nos números)

- `race`: 2.273 valores em falta ✔️ (bate certo com o guia do projeto)
- `weight`: 98.569 valores em falta (96,86%) ✔️
- Nenhum código de `admission_type_id`/`discharge_disposition_id`/
  `admission_source_id` ficou sem descrição; os códigos desconhecidos têm
  `Desconhecido` (D5)
