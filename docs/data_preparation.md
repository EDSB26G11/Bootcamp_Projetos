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
`src/data/population.py` (T1.4: população elegível) e `src/data_preparation.py`
(só orquestra: carregar → validar → descodificar → excluir → gravar).

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

### 3. `decode_ids()` — juntar as descrições dos códigos

Para cada um dos 3 códigos (`admission_type_id`, `discharge_disposition_id`,
`admission_source_id`), cria uma coluna nova `<coluna>_desc` com a descrição
correspondente (ex: `admission_type_id=1` → `admission_type_id_desc="Emergency"`).
Os códigos originais são mantidos — só acrescentamos as descrições, não
substituímos nada.

### 4. `exclude_deaths()` — definir a população elegível

Remove os encontros com alta por óbito (`discharge_disposition_id` 11, 19,
20, 21): 1.652 encontros. Um doente que morreu não pode ser readmitido. Os
encontros com alta para hospício (13, 14) ficam, porque o doente pode voltar
ao hospital. Decisão D1 em [decisions.md](decisions.md).

### 5. Guardar o resultado

O ficheiro final é gravado em `data/processed/diabetic_data_clean.csv`
— **100.114 linhas, 70.439 doentes, 53 colunas** (50 originais + 3 colunas `_desc`).

## Como correr

A partir da raiz do repositório:

```bash
source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m src.data_preparation
```

Deve terminar com:

```text
Guardado: data/processed/diabetic_data_clean.csv (100114 linhas, 53 colunas)
```

## Testes

```bash
python -m pytest tests/ -v
```

`tests/test_mappings.py` cobre o parser do `IDS_mapping.csv` (3 tabelas, `NULL`
preservado como texto, erro se o formato mudar) e `tests/test_load.py` cobre a
leitura (`?` → NaN, colunas de texto) e a validação, incluindo uma verificação
contra o ficheiro real em `data/raw/`. `tests/test_population.py` confirma que
só os códigos de óbito são removidos (o hospício fica) e que, no ficheiro real,
ficam 100.114 encontros e 70.439 doentes.

## Aviso para quem for reler `diabetic_data_clean.csv` mais tarde

Se alguém abrir este ficheiro processado com `pd.read_csv()` sem mais
nada, o pandas volta a converter a categoria `"NULL"` (nas colunas
`_desc`) e o `"None"` (teste não realizado) em `NaN`, perdendo a distinção
entre categoria e valor em falta. Para reler corretamente, usar sempre:

```python
from src.data.load import load_clean_data

df = load_clean_data("data/processed/diabetic_data_clean.csv")
```

Só a célula vazia passa a `NaN`, e `age`, `weight`, `payer_code` e
`diag_1/2/3` são forçadas a texto, como na leitura dos dados originais. No
ficheiro completo o pandas já as lê como texto (os códigos `V`/`E` obrigam a
isso), mas numa leitura parcial (`nrows`, *chunks*) os códigos de diagnóstico
podem passar a número (`'276'` → `276.0`), o que estraga o agrupamento ICD-9.

## Verificações feitas (para dar confiança nos números)

- `race`: 2.273 valores em falta ✔️ (bate certo com o guia do projeto)
- `weight`: 98.569 valores em falta (96,86%) ✔️
- Nenhuma categoria de `admission_type_id`/`discharge_disposition_id`/
  `admission_source_id` ficou por mapear além dos códigos "NULL" (que agora
  aparecem corretamente como texto, não como valor em falta)
