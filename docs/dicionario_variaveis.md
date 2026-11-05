# Dicionário de variáveis

Todas as colunas de `data/processed/diabetic_data_clean.csv`: as 50 originais
de `diabetic_data.csv`, as 3 descrições `*_desc` criadas por `decode_ids()` e
o `target` (D6).
Os significados seguem a documentação UCI / Strack et al. (2014). As decisões
são registadas em [decisions.md](decisions.md); esta tabela é o resumo por
variável e deve ser atualizada sempre que uma decisão muda.

## Legenda

- `Y` — decidido ou já implementado no pipeline.
- `Y*` — proposto pela EDA, **ainda por decidir** na tarefa indicada.
- Depois de D15, a coluna **Excluída** diz se a variável entra no modelo; a lista final está em [features_aprovadas.md](features_aprovadas.md).
- `N` — não.
- **Excluída**: não entra como feature no modelo. A exclusão de *linhas*
  (óbitos, D1; encontros impossíveis, D2; `gender` inválido, D7) não aparece aqui.
- **Ajuste**: precisa de tratamento antes de ser usada (tipo, missing, recodificação, transformação).
- **Base p/ outras**: serve para construir outra variável, o target ou a população.

## Identificação e target

| Variável | Significado | Tipo | Excluída | Ajuste | Base p/ outras | Detalhe |
|---|---|---|---|---|---|---|
| `encounter_id` | Identificador único do internamento | ID | Y | N | N | Identificador, não é feature (D11). A ordem não pode ser usada como tempo (D12) |
| `patient_nbr` | Identificador único do doente | ID | Y | N | Y | Não é feature (D11); agrupa o split por doente (T1.7). Proibido agregar por doente sobre o dataset (D12) |
| `readmitted` | Readmissão: `<30` dias, `>30` dias ou `NO` | Categórica | Y | N | Y | Nunca é feature (D11); origina o target binário `<30` vs resto (D6, implementado) |
| `target` | 1 se `readmitted == "<30"`, 0 caso contrário (criada no pipeline) | Binária | Y | N | N | É o target (D6, implementado); 11,34% de positivos |

## Demografia

| Variável | Significado | Tipo | Excluída | Ajuste | Base p/ outras | Detalhe |
|---|---|---|---|---|---|---|
| `race` | Raça (Caucasian, AfricanAmerican, Hispanic, Asian, Other) | Categórica | N | Y | N | Harmonizada por doente (D3, implementado). 2,2% missing (`?`): tratar como categoria (T1.9). Uso no modelo vs só fairness em T3.5. **Entra no modelo (D15)** |
| `gender` | Género (Female, Male, Unknown/Invalid) | Categórica | N | Y | N | 3 casos Unknown/Invalid excluídos da população (D7, implementado). Uso no modelo vs só fairness em T3.5. **Entra no modelo (D15)** |
| `age` | Idade em bandas de 10 anos (`[0-10)` … `[90-100)`) | Ordinal (texto) | Y | Y | Y | Lida como texto (implementado); converter em ordinal (T1.8). **Fora do modelo (D15):** substituído por age_band |
| `weight` | Peso em bandas de libras | Ordinal (texto) | Y | Y | N | 96,9% missing, sem sinal: descartar (T1.9). **Fora do modelo (D15):** 96,9% missing, sem sinal |

## Admissão, alta e internamento

| Variável | Significado | Tipo | Excluída | Ajuste | Base p/ outras | Detalhe |
|---|---|---|---|---|---|---|
| `admission_type_id` | Tipo de admissão (código; ex.: Emergency, Urgent, Elective) | Categórica (código) | Y | Y | Y | Descodificada em `_desc` (implementado). "NULL"/"Not Available"/"Not Mapped" agrupados em `Desconhecido` na `_desc` (D5, implementado). Manter só código **ou** descrição (T1.9). **Fora do modelo (D15):** substituído por admission_type_id_desc (D5) |
| `admission_type_id_desc` | Descrição de `admission_type_id` (criada no pipeline) | Categórica | N | Y | N | Mesma informação que o código; ver acima. **Entra no modelo (D15)** |
| `discharge_disposition_id` | Destino da alta (código; ex.: casa, SNF, óbito) | Categórica (código) | Y | Y | Y | Define a população: óbitos excluídos, hospício mantido (D1, implementado). Também é feature, por ser decidido no momento da alta (D10). NULL/Not Mapped → `Desconhecido` (D5). **Fora do modelo (D15):** substituído por discharge_disposition_id_desc (D5) |
| `discharge_disposition_id_desc` | Descrição de `discharge_disposition_id` (criada no pipeline) | Categórica | N | Y | N | Ver acima. **Entra no modelo (D15)** |
| `admission_source_id` | Origem da admissão (código; ex.: urgência, referenciação médica, transferência) | Categórica (código) | Y | Y | Y | Descodificada em `_desc`. NULL/Not Available/Not Mapped → `Desconhecido` (D5). Manter código **ou** descrição (T1.9). **Fora do modelo (D15):** substituído por admission_source_id_desc (D5) |
| `admission_source_id_desc` | Descrição de `admission_source_id` (criada no pipeline) | Categórica | N | Y | N | Ver acima. **Entra no modelo (D15)** |
| `time_in_hospital` | Dias de internamento (1–14, critério de inclusão) | Numérica | N | N | N | —. **Entra no modelo (D15)** |
| `payer_code` | Entidade pagadora (ex.: Medicare, seguro, pagamento próprio) | Categórica | N | Y | N | Lida como texto; 39,6% missing, sinal fraco (V=0,03). Candidata a sair; testar no modelo (T1.9, T2). **Entra no modelo (D15)** |
| `medical_specialty` | Especialidade do médico que admitiu | Categórica | N | Y | N | 49,1% missing → categoria própria; 73 níveis → agrupar raros (T1.9). **Entra no modelo (D15)** |
| `num_lab_procedures` | Nº de análises laboratoriais no internamento | Numérica | N | N | N | —. **Entra no modelo (D15)** |
| `num_procedures` | Nº de procedimentos (exceto análises) no internamento | Numérica | N | N | N | Máximo 6 parece significar "6 ou mais" (P7). **Entra no modelo (D15)** |
| `num_medications` | Nº de medicamentos distintos administrados no internamento | Numérica | N | N | N | —. **Entra no modelo (D15)** |

## Histórico (ano anterior ao internamento)

| Variável | Significado | Tipo | Excluída | Ajuste | Base p/ outras | Detalhe |
|---|---|---|---|---|---|---|
| `number_outpatient` | Nº de consultas externas no ano anterior | Numérica | N | Y | N | 84% zeros, skew 8,8: log1p ou truncar no p99 para modelos lineares (T1.8). **Entra no modelo (D15)** |
| `number_emergency` | Nº de idas à urgência no ano anterior | Numérica | N | Y | N | 89% zeros, skew 22,9: idem (T1.8). **Entra no modelo (D15)** |
| `number_inpatient` | Nº de internamentos no ano anterior | Numérica | N | Y | N | Preditor mais forte; 66% zeros, skew 3,6: idem (T1.8). **Entra no modelo (D15)** |

## Diagnósticos e análises

| Variável | Significado | Tipo | Excluída | Ajuste | Base p/ outras | Detalhe |
|---|---|---|---|---|---|---|
| `diag_1` | Diagnóstico principal (código ICD-9) | Categórica (código) | Y | Y | Y | Lido como texto, com os zeros à esquerda repostos (D4, implementado). Aprovado como feature com pressuposto (D9, P14). ~700+ códigos: substituir por grupos ICD-9 de Strack + flag de complicações da diabetes (T1.8). O código em bruto sai depois de agrupado. **Fora do modelo (D15):** substituído por diag_1_group |
| `diag_2` | Diagnóstico secundário (ICD-9) | Categórica (código) | Y | Y | Y | Idem. **Fora do modelo (D15):** substituído por diag_2_group |
| `diag_3` | Diagnóstico secundário adicional (ICD-9) | Categórica (código) | Y | Y | Y | Idem. **Fora do modelo (D15):** substituído por diag_3_group |
| `number_diagnoses` | Nº de diagnósticos registados | Numérica | N | N | N | Máximo 9 parece significar "9 ou mais" (P7). **Entra no modelo (D15)** |
| `A1Cresult` | Hemoglobina glicada: `>8`, `>7`, `Norm`, `None` | Categórica | N | Y | N | `"None"` = não medido, mantido como categoria (implementado, P2). **Entra no modelo (D15)** |
| `max_glu_serum` | Glicose sérica: `>300`, `>200`, `Norm`, `None` | Categórica | N | Y | N | `"None"` mantido como categoria (implementado). 94,7% não medido, sinal fraco: candidata fraca (T1.9). **Entra no modelo (D15)** |

## Medicamentos

Valores: `No` (não prescrito), `Steady` (dose mantida), `Up` (aumentada),
`Down` (reduzida). Todos servem de base (`Y*`) para features agregadas (T1.8):
nº de medicamentos prescritos, nº com dose alterada, e/ou indicadores por
classe. Classes e contexto em [anexo_medicamentos.md](anexo_medicamentos.md).

| Variável | Significado | Tipo | Excluída | Ajuste | Base p/ outras | Detalhe |
|---|---|---|---|---|---|---|
| `metformin` | Metformina (biguanida) | Categórica | N | N | Y | Uso 19,6%. **Entra no modelo (D15)** |
| `repaglinide` | Repaglinida (meglitinida) | Categórica | N | N | Y | Uso 1,5%. **Entra no modelo (D15)** |
| `nateglinide` | Nateglinida (meglitinida) | Categórica | Y | N | Y | 99,3% `No` (T1.9). **Fora do modelo (D15):** quase constante (>= 99,3% No); só nas contagens |
| `chlorpropamide` | Clorpropamida (sulfonilureia 1.ª ger.) | Categórica | Y | N | Y | 99,9% `No` (T1.9). **Fora do modelo (D15):** quase constante (>= 99,3% No); só nas contagens |
| `glimepiride` | Glimepirida (sulfonilureia 2.ª ger.) | Categórica | N | N | Y | Uso 5,1%. **Entra no modelo (D15)** |
| `acetohexamide` | Acetohexamida (sulfonilureia 1.ª ger.) | Categórica | Y | N | Y | 1 caso (T1.9). **Fora do modelo (D15):** quase constante (>= 99,3% No); só nas contagens |
| `glipizide` | Glipizida (sulfonilureia 2.ª ger.) | Categórica | N | N | Y | Uso 12,5%. **Entra no modelo (D15)** |
| `glyburide` | Gliburida (sulfonilureia 2.ª ger.) | Categórica | N | N | Y | Uso 10,5%. **Entra no modelo (D15)** |
| `tolbutamide` | Tolbutamida (sulfonilureia 1.ª ger.) | Categórica | Y | N | Y | 99,98% `No` (T1.9). **Fora do modelo (D15):** quase constante (>= 99,3% No); só nas contagens |
| `pioglitazone` | Pioglitazona (glitazona) | Categórica | N | N | Y | Uso 7,2%. **Entra no modelo (D15)** |
| `rosiglitazone` | Rosiglitazona (glitazona) | Categórica | N | N | Y | Uso 6,3%. **Entra no modelo (D15)** |
| `acarbose` | Acarbose (inibidor alfa-glucosidase) | Categórica | Y | N | Y | 99,7% `No` (T1.9). **Fora do modelo (D15):** quase constante (>= 99,3% No); só nas contagens |
| `miglitol` | Miglitol (inibidor alfa-glucosidase) | Categórica | Y | N | Y | 99,96% `No` (T1.9). **Fora do modelo (D15):** quase constante (>= 99,3% No); só nas contagens |
| `troglitazone` | Troglitazona (glitazona, retirada em 2000) | Categórica | Y | N | Y | 3 casos (T1.9). **Fora do modelo (D15):** quase constante (>= 99,3% No); só nas contagens |
| `tolazamide` | Tolazamida (sulfonilureia 1.ª ger.) | Categórica | Y | N | Y | 99,96% `No` (T1.9). **Fora do modelo (D15):** quase constante (>= 99,3% No); só nas contagens |
| `examide` | Não identificado como antidiabético | Categórica | Y | N | N | Constante (T1.9). **Fora do modelo (D15):** constante |
| `citoglipton` | Não identificado (possível erro por *sitagliptin*, não confirmado) | Categórica | Y | N | N | Constante (T1.9). **Fora do modelo (D15):** constante |
| `insulin` | Insulina | Categórica | N | N | Y | Uso 53,4%. **Entra no modelo (D15)** |
| `glyburide-metformin` | Combinação gliburida + metformina | Categórica | Y | N | Y | 99,3% `No` (T1.9). **Fora do modelo (D15):** quase constante (>= 99,3% No); só nas contagens |
| `glipizide-metformin` | Combinação glipizida + metformina | Categórica | Y | N | Y | 99,99% `No` (T1.9). **Fora do modelo (D15):** quase constante (>= 99,3% No); só nas contagens |
| `glimepiride-pioglitazone` | Combinação glimepirida + pioglitazona | Categórica | Y | N | Y | 1 caso (T1.9). **Fora do modelo (D15):** quase constante (>= 99,3% No); só nas contagens |
| `metformin-rosiglitazone` | Combinação metformina + rosiglitazona | Categórica | Y | N | Y | 2 casos (T1.9). **Fora do modelo (D15):** quase constante (>= 99,3% No); só nas contagens |
| `metformin-pioglitazone` | Combinação metformina + pioglitazona | Categórica | Y | N | Y | 1 caso (T1.9). **Fora do modelo (D15):** quase constante (>= 99,3% No); só nas contagens |
| `change` | "Houve alteração na medicação" (`Ch`/`No`) | Categórica | Y | N | N | Não significa só alteração: reproduz-se por "dose alterada ou ≥2 medicamentos" (P20). Redundante com as colunas de medicamentos (T1.8/T1.9). **Fora do modelo (D15):** redundante com as colunas de medicamentos (P20) |
| `diabetesMed` | Algum antidiabético prescrito (`Yes`/`No`) | Categórica | Y | N | N | Igual a "≥1 medicamento prescrito" (P19). Redundante (T1.8/T1.9). **Fora do modelo (D15):** igual a n_meds >= 1 (P19) |

## Variáveis construídas (T1.5, T1.8)

| Nova variável | Construída a partir de | Estado |
|---|---|---|
| `target` (`<30` = 1, resto = 0) | `readmitted` | Y (D6, implementado) |
| `age_band` (faixa etária ordinal 0–9) | `age` | Y (D14, implementado) |
| `diag_1–3_group` (grupos de Strack et al.) | `diag_1`, `diag_2`, `diag_3` | Y (D14, implementado) |
| `diabetes_complicated` (250.1–250.9 em algum diagnóstico) | `diag_1–3` | Y (D14, implementado) |
| `n_meds` (nº de medicamentos prescritos) | 23 colunas de medicamentos | Y (D14, implementado) |
| `n_meds_up`, `n_meds_down` (nº com dose aumentada / reduzida) | 23 colunas de medicamentos | Y (D14, implementado) |
| Indicadores por classe terapêutica | 23 colunas de medicamentos | N (opcional, não implementado) |
| Categoria "Desconhecido" nos IDs | `admission_type_id`, `admission_source_id`, `discharge_disposition_id` | Y (D5, implementado nas `_desc`) |
| Indicador "registo de admissão incompleto" | os 3 IDs | Y* (D5: só se o modelo mostrar que ajuda, T2) |
| `n_visits_prior` (total de visitas no ano anterior) | `number_outpatient/emergency/inpatient` | Y (D14, implementado) |
| Contagens com log1p | `number_outpatient/emergency/inpatient`, `n_visits_prior` | Y (D16, no pipeline) |
