# Features aprovadas (T1.6)

Colunas de `data/processed/diabetic_data_clean.csv` que podem entrar no modelo
por serem conhecidas **no momento da alta**. Resulta das decisões D8–D12 em
[decisions.md](decisions.md), fechadas no roadmap a 29/10.

Aprovar aqui quer dizer só "não há fuga de informação". Se cada coluna fica no
modelo final (constantes, redundantes, quase sem variação, código vs.
descrição) decide-se em T1.8/T1.9 e está no
[dicionário de variáveis](dicionario_variaveis.md).

## Resumo

| | Colunas |
|---|---|
| Originais aprovadas | 47 (as 50 originais menos `readmitted`, `encounter_id`, `patient_nbr`) |
| Descrições criadas no pipeline | 3 (`*_desc`; a mesma informação que o código, escolher um dos dois em T1.9) |
| Excluídas por definição (D11) | `readmitted`, `encounter_id`, `patient_nbr` |
| Target | `target` (D6) |

## Aprovadas com pressuposto escrito (D9)

Estes pressupostos têm de aparecer no relatório final.

| Colunas | Pressuposto |
|---|---|
| `diag_1`, `diag_2`, `diag_3`, `number_diagnoses` | Os diagnósticos estão codificados na alta (P14). A codificação ICD-9 pode ser finalizada depois, para faturação |
| `A1Cresult`, `max_glu_serum` | O resultado é conhecido na alta (P15). Pode chegar perto ou depois dela |
| 23 colunas de medicamentos (`metformin` … `metformin-pioglitazone`), `change`, `diabetesMed` | Refletem o internamento e o estado à alta (P16) |

## Aprovadas sem reservas (D10)

| Grupo | Colunas |
|---|---|
| Destino da alta | `discharge_disposition_id` (também define a população, D1) |
| Administrativas | `admission_type_id`, `admission_source_id`, `payer_code`, `medical_specialty` |
| Internamento | `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications` |
| Histórico (ano anterior, P13) | `number_outpatient`, `number_emergency`, `number_inpatient` |
| Demográficas | `age`, `gender`, `race` (uso por fairness decide-se em T3.5) |

`weight` não entrou em nenhuma das decisões da T1.6. É registado no
internamento, por isso não há risco de fuga, mas tem 96,9% de missing e a
proposta é descartá-lo em T1.9. Fica contado nas 47 colunas originais até essa
decisão.

## Regra para variáveis derivadas (D12)

Qualquer feature construída em T1.8 tem de cumprir:

1. **Nada de agregações por doente sobre o dataset completo** (ex.: nº de
   encontros do doente, média de internamentos do doente). Incluem
   internamentos posteriores à alta.
2. **Nada de ordem temporal a partir do `encounter_id`** (ex.: "encontro
   anterior", "nº de encontros até agora"). A ordem não é garantidamente
   temporal (P17).

## Ainda por decidir noutras tarefas

- T1.9: descartar `weight` (96,9% missing), `examide` e `citoglipton`
  (constantes) e os medicamentos quase constantes; código ou `_desc` nos IDs.
- T1.8/T1.9: manter ou não `change` e `diabetesMed` (redundantes com os medicamentos).
- T3.5: usar `race`, `gender` e `age` no modelo ou só para avaliar fairness.
