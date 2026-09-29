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

### 3. `decode_ids()` — juntar as descrições dos códigos

Para cada um dos 3 códigos (`admission_type_id`, `discharge_disposition_id`,
`admission_source_id`), cria uma coluna nova `<coluna>_desc` com a descrição
correspondente (ex: `admission_type_id=1` → `admission_type_id_desc="Emergency"`).
Os códigos originais são mantidos — só acrescentamos as descrições, não
substituímos nada.

### 4. Guardar o resultado

O ficheiro final é gravado em `data/processed/diabetic_data_clean.csv`
— **101.766 linhas, 53 colunas** (48 originais + 5 colunas de descrição:
as 3 `_desc` mais as bandas/diagnósticos já protegidos como texto).

## Como correr

```bash
source .venv/bin/activate
python src/data_preparation.py
```

Deve terminar com:
```
Guardado: data/processed/diabetic_data_clean.csv (101766 linhas, 53 colunas)
```

## Aviso para quem for reler `diabetic_data_clean.csv` mais tarde

Se alguém abrir este ficheiro processado com `pd.read_csv()` sem mais
nada, o pandas volta a converter a categoria `"NULL"` (nas colunas
`_desc`) em `NaN`, perdendo a distinção entre "categoria NULL" e "sem
mapeamento". Para reler corretamente:

```python
pd.read_csv("data/processed/diabetic_data_clean.csv", keep_default_na=False, na_values="")
```

## Verificações feitas (para dar confiança nos números)

- `race`: 2.273 valores em falta ✔️ (bate certo com o guia do projeto)
- `weight`: 98.569 valores em falta (96,86%) ✔️
- Nenhuma categoria de `admission_type_id`/`discharge_disposition_id`/
  `admission_source_id` ficou por mapear além dos códigos "NULL" (que agora
  aparecem corretamente como texto, não como valor em falta)
