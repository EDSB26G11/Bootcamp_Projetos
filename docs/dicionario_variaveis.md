# Dicionário de variáveis

Todas as colunas de `data/processed/diabetic_data_clean.csv`: as 50 originais
de `diabetic_data.csv` mais as 3 descrições `*_desc` criadas por `decode_ids()`.
Os significados seguem a documentação UCI / Strack et al. (2014). As decisões
são registadas em [decisions.md](decisions.md); esta tabela é o resumo por
variável e deve ser atualizada sempre que uma decisão muda.

## Legenda

- `Y` — decidido ou já implementado no pipeline.
- `Y*` — proposto pela EDA, **ainda por decidir** na tarefa indicada.
- `N` — não.
- **Excluída**: não entra como feature no modelo. A exclusão de *linhas*
  (óbitos, D1) não aparece aqui.
- **Ajuste**: precisa de tratamento antes de ser usada (tipo, missing, recodificação, transformação).
- **Base p/ outras**: serve para construir outra variável, o target ou a população.

## Identificação e target

| Variável | Significado | Tipo | Excluída | Ajuste | Base p/ outras | Detalhe |
|---|---|---|---|---|---|---|
| `encounter_id` | Identificador único do internamento | ID | Y* | N | N | Identificador, não é feature (T1.6) |
| `patient_nbr` | Identificador único do doente | ID | Y* | N | Y | Não é feature (T1.6); agrupa o split por doente (T1.7). Proibido agregar por doente sobre o dataset (P18) |
| `readmitted` | Readmissão: `<30` dias, `>30` dias ou `NO` | Categórica | Y | N | Y* | Nunca é feature (enunciado T1.6); origina o target binário `<30` vs resto (T1.5, P9) |

## Demografia

| Variável | Significado | Tipo | Excluída | Ajuste | Base p/ outras | Detalhe |
|---|---|---|---|---|---|---|
| `race` | Raça (Caucasian, AfricanAmerican, Hispanic, Asian, Other) | Categórica | N | Y* | N | 2,2% missing (`?`): tratar como categoria (T1.9). Uso no modelo vs só fairness em T3.5 |
| `gender` | Género (Female, Male, Unknown/Invalid) | Categórica | N | Y* | N | 3 casos Unknown/Invalid (P12, pendente). Uso no modelo vs só fairness em T3.5 |
| `age` | Idade em bandas de 10 anos (`[0-10)` … `[90-100)`) | Ordinal (texto) | N | Y | Y* | Lida como texto (implementado); converter em ordinal (T1.8) |
| `weight` | Peso em bandas de libras | Ordinal (texto) | Y* | Y | N | 96,9% missing, sem sinal: descartar (T1.9) |

## Admissão, alta e internamento

| Variável | Significado | Tipo | Excluída | Ajuste | Base p/ outras | Detalhe |
|---|---|---|---|---|---|---|
| `admission_type_id` | Tipo de admissão (código; ex.: Emergency, Urgent, Elective) | Categórica (código) | N | Y | Y | Descodificada em `_desc` (implementado). "NULL"/"Not Available"/"Not Mapped" = desconhecido (P4, pendente). Manter só código **ou** descrição (T1.9) |
| `admission_type_id_desc` | Descrição de `admission_type_id` (criada no pipeline) | Categórica | N | Y* | N | Mesma informação que o código; ver acima |
| `discharge_disposition_id` | Destino da alta (código; ex.: casa, SNF, óbito) | Categórica (código) | N | Y | Y | Define a população: óbitos excluídos, hospício mantido (D1, implementado). Proposto também como feature, por ser decidido no momento da alta (T1.6, pendente). NULL/Not Mapped (P4) |
| `discharge_disposition_id_desc` | Descrição de `discharge_disposition_id` (criada no pipeline) | Categórica | N | Y* | N | Ver acima |
| `admission_source_id` | Origem da admissão (código; ex.: urgência, referenciação médica, transferência) | Categórica (código) | N | Y | Y | Descodificada em `_desc`. NULL/Not Available/Not Mapped (P4). Manter código **ou** descrição (T1.9) |
| `admission_source_id_desc` | Descrição de `admission_source_id` (criada no pipeline) | Categórica | N | Y* | N | Ver acima |
| `time_in_hospital` | Dias de internamento (1–14, critério de inclusão) | Numérica | N | N | N | — |
| `payer_code` | Entidade pagadora (ex.: Medicare, seguro, pagamento próprio) | Categórica | Y* | Y | N | Lida como texto; 39,6% missing, sinal fraco (V=0,03). Candidata a sair; testar no modelo (T1.9, T2) |
| `medical_specialty` | Especialidade do médico que admitiu | Categórica | N | Y* | N | 49,1% missing → categoria própria; 73 níveis → agrupar raros (T1.9) |
| `num_lab_procedures` | Nº de análises laboratoriais no internamento | Numérica | N | N | N | — |
| `num_procedures` | Nº de procedimentos (exceto análises) no internamento | Numérica | N | N | N | Máximo 6 parece significar "6 ou mais" (P7) |
| `num_medications` | Nº de medicamentos distintos administrados no internamento | Numérica | N | N | N | — |

## Histórico (ano anterior ao internamento)

| Variável | Significado | Tipo | Excluída | Ajuste | Base p/ outras | Detalhe |
|---|---|---|---|---|---|---|
| `number_outpatient` | Nº de consultas externas no ano anterior | Numérica | N | Y* | N | 84% zeros, skew 8,8: log1p ou truncar no p99 para modelos lineares (T1.8) |
| `number_emergency` | Nº de idas à urgência no ano anterior | Numérica | N | Y* | N | 89% zeros, skew 22,9: idem (T1.8) |
| `number_inpatient` | Nº de internamentos no ano anterior | Numérica | N | Y* | N | Preditor mais forte; 66% zeros, skew 3,6: idem (T1.8) |

## Diagnósticos e análises

| Variável | Significado | Tipo | Excluída | Ajuste | Base p/ outras | Detalhe |
|---|---|---|---|---|---|---|
| `diag_1` | Diagnóstico principal (código ICD-9) | Categórica (código) | Y* | Y | Y* | Lido como texto (implementado). ~700+ códigos: substituir por grupos ICD-9 de Strack + flag de complicações da diabetes (T1.8). O código em bruto sai depois de agrupado |
| `diag_2` | Diagnóstico secundário (ICD-9) | Categórica (código) | Y* | Y | Y* | Idem |
| `diag_3` | Diagnóstico secundário adicional (ICD-9) | Categórica (código) | Y* | Y | Y* | Idem |
| `number_diagnoses` | Nº de diagnósticos registados | Numérica | N | N | N | Máximo 9 parece significar "9 ou mais" (P7) |
| `A1Cresult` | Hemoglobina glicada: `>8`, `>7`, `Norm`, `None` | Categórica | N | Y | N | `"None"` = não medido, mantido como categoria (implementado, P2) |
| `max_glu_serum` | Glicose sérica: `>300`, `>200`, `Norm`, `None` | Categórica | Y* | Y | N | `"None"` mantido como categoria (implementado). 94,7% não medido, sinal fraco: candidata fraca (T1.9) |

## Medicamentos

Valores: `No` (não prescrito), `Steady` (dose mantida), `Up` (aumentada),
`Down` (reduzida). Todos servem de base (`Y*`) para features agregadas (T1.8):
nº de medicamentos prescritos, nº com dose alterada, e/ou indicadores por
classe. Classes e contexto em [anexo_medicamentos.md](anexo_medicamentos.md).

| Variável | Significado | Tipo | Excluída | Ajuste | Base p/ outras | Detalhe |
|---|---|---|---|---|---|---|
| `metformin` | Metformina (biguanida) | Categórica | N | N | Y* | Uso 19,6% |
| `repaglinide` | Repaglinida (meglitinida) | Categórica | N | N | Y* | Uso 1,5% |
| `nateglinide` | Nateglinida (meglitinida) | Categórica | Y* | N | Y* | 99,3% `No` (T1.9) |
| `chlorpropamide` | Clorpropamida (sulfonilureia 1.ª ger.) | Categórica | Y* | N | Y* | 99,9% `No` (T1.9) |
| `glimepiride` | Glimepirida (sulfonilureia 2.ª ger.) | Categórica | N | N | Y* | Uso 5,1% |
| `acetohexamide` | Acetohexamida (sulfonilureia 1.ª ger.) | Categórica | Y* | N | Y* | 1 caso (T1.9) |
| `glipizide` | Glipizida (sulfonilureia 2.ª ger.) | Categórica | N | N | Y* | Uso 12,5% |
| `glyburide` | Gliburida (sulfonilureia 2.ª ger.) | Categórica | N | N | Y* | Uso 10,5% |
| `tolbutamide` | Tolbutamida (sulfonilureia 1.ª ger.) | Categórica | Y* | N | Y* | 99,98% `No` (T1.9) |
| `pioglitazone` | Pioglitazona (glitazona) | Categórica | N | N | Y* | Uso 7,2% |
| `rosiglitazone` | Rosiglitazona (glitazona) | Categórica | N | N | Y* | Uso 6,3% |
| `acarbose` | Acarbose (inibidor alfa-glucosidase) | Categórica | Y* | N | Y* | 99,7% `No` (T1.9) |
| `miglitol` | Miglitol (inibidor alfa-glucosidase) | Categórica | Y* | N | Y* | 99,96% `No` (T1.9) |
| `troglitazone` | Troglitazona (glitazona, retirada em 2000) | Categórica | Y* | N | Y* | 3 casos (T1.9) |
| `tolazamide` | Tolazamida (sulfonilureia 1.ª ger.) | Categórica | Y* | N | Y* | 99,96% `No` (T1.9) |
| `examide` | Não identificado como antidiabético | Categórica | Y* | N | N | Constante (T1.9) |
| `citoglipton` | Não identificado (possível erro por *sitagliptin*, não confirmado) | Categórica | Y* | N | N | Constante (T1.9) |
| `insulin` | Insulina | Categórica | N | N | Y* | Uso 53,4% |
| `glyburide-metformin` | Combinação gliburida + metformina | Categórica | Y* | N | Y* | 99,3% `No` (T1.9) |
| `glipizide-metformin` | Combinação glipizida + metformina | Categórica | Y* | N | Y* | 99,99% `No` (T1.9) |
| `glimepiride-pioglitazone` | Combinação glimepirida + pioglitazona | Categórica | Y* | N | Y* | 1 caso (T1.9) |
| `metformin-rosiglitazone` | Combinação metformina + rosiglitazona | Categórica | Y* | N | Y* | 2 casos (T1.9) |
| `metformin-pioglitazone` | Combinação metformina + pioglitazona | Categórica | Y* | N | Y* | 1 caso (T1.9) |
| `change` | "Houve alteração na medicação" (`Ch`/`No`) | Categórica | Y* | N | N | Não significa só alteração: reproduz-se por "dose alterada ou ≥2 medicamentos" (P20). Redundante com as colunas de medicamentos (T1.8/T1.9) |
| `diabetesMed` | Algum antidiabético prescrito (`Yes`/`No`) | Categórica | Y* | N | N | Igual a "≥1 medicamento prescrito" (P19). Redundante (T1.8/T1.9) |

## Variáveis a construir (propostas, T1.5/T1.8)

| Nova variável | Construída a partir de | Estado |
|---|---|---|
| `target` (`<30` = 1, resto = 0) | `readmitted` | Y* (T1.5, P9) |
| Faixa etária ordinal | `age` | Y* (T1.8) |
| Grupo ICD-9 de cada diagnóstico | `diag_1`, `diag_2`, `diag_3` | Y* (T1.8) |
| Flag de diabetes com complicações (250.x) | `diag_1–3` | Y* (T1.8) |
| Nº de medicamentos prescritos | 23 colunas de medicamentos | Y* (T1.8) |
| Nº de medicamentos com dose alterada (e/ou `Up` e `Down` separados) | 23 colunas de medicamentos | Y* (T1.8) |
| Indicadores por classe terapêutica | 23 colunas de medicamentos | Y* (T1.8), opcional |
| Categoria "Desconhecido" nos IDs | `admission_type_id`, `admission_source_id`, `discharge_disposition_id` | Y* (T1.9, P4) |
| Contagens transformadas (log1p / truncadas) | `number_outpatient/emergency/inpatient` | Y* (T1.8) |
