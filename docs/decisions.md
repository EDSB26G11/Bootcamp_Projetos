# Registo de decisões

Decisões sobre exclusões, target, features e limpeza (T1.4–T1.9). A EDA
(`notebooks/01_eda.ipynb`, resumo em [eda_summary.md](eda_summary.md)) só
quantifica; cada decisão fica aqui registada com a evidência e a tarefa
onde foi tomada.

| # | Tarefa | Decisão | Evidência | Estado |
|---|---|---|---|---|
| D1 | T1.4 | Excluir os encontros com alta por óbito (`discharge_disposition_id` 11, 19, 20, 21) ou para hospício (13, 14). Implementado em `src/data/population.py` | 2 423 encontros (2,4%); 0 readmissões nos óbitos, 43 nos hospícios, onde o objetivo dos cuidados não é evitar a readmissão. Ficam 99 343 encontros, 69 990 doentes, 11,39% de positivos (notebook §5) | Aceite |
| D2 | T1.4 | Excluir os encontros com pelo menos uma regra de severidade `erro` em `src/data/quality.py` (A1–A3 recém-nascido com idade ≥ 10, C1 `gender` inconsistente, C3 idade desce, C4 idade salta 2+ bandas, C5 encontro depois de óbito) | 59 encontros, 54 doentes (0,06%): registos impossíveis, sem forma de saber qual dos campos está errado (notebook §1.1) | Proposta |
| D3 | T1.9 | Harmonizar a `race` por doente com a moda dos seus encontros | 123 doentes (404 encontros) com mais de uma `race` (C2); plausível em dados autorreportados, mas o mesmo doente deve ter uma só categoria no modelo (notebook §1.1) | Proposta |
| D4 | T1.8 | Repor os zeros à esquerda nos códigos ICD-9 numéricos (`8` → `008`) antes de qualquer agrupamento por texto | 5 732 códigos em 5 567 encontros (B1), todos numéricos < 100. O `icd9_group` da §8 converte para número e não é afetado (notebook §1.1) | Proposta |

Estados: `Proposta` · `Aceite` · `Rejeitada`.

## Pressupostos

Pressupostos assumidos em `notebooks/01_eda.ipynb` e em `src/data/`. Não são
decisões, mas os números da EDA dependem deles. Se algum cair, os resultados
indicados têm de ser revistos.

Estados:

- `Confirmado (dados)`: verificado nos próprios dados.
- `Confirmado (doc.)`: está na documentação UCI / Strack et al. (2014).
- `Assumido`: plausível, mas não verificável com os dados.
- `A decidir`: pressuposto provisório da EDA que tem de virar decisão na tarefa indicada.

### Leitura e codificação dos dados

| # | Pressuposto | Onde | Fundamento | Se falhar | Estado | Tarefa |
|---|---|---|---|---|---|---|
| P1 | `?` é o único marcador de valor em falta | `load.py`; notebook §3 | Contagens batem com o guia (race 2 273, weight 98 569) | Missing subestimado noutras colunas | Confirmado (dados) | T1.1 |
| P2 | `"None"` em `A1Cresult`/`max_glu_serum` = teste não realizado, não missing | `load.py`; notebook §3 | Documentação UCI; taxas de <30 diferentes entre medido e não medido | Uma categoria com sinal seria tratada como missing | Confirmado (doc.) | T1.8 |
| P3 | `"NULL"` no `IDS_mapping.csv` é uma descrição válida, não missing | `mappings.py` | É o texto literal do ficheiro (ex.: `admission_type_id` 6) | — | Confirmado (dados) | T1.2 |
| P4 | "NULL", "Not Available" e "Not Mapped" nos IDs são, na prática, desconhecido (≈10%) | notebook §6 | Semântica das descrições | Três categorias "desconhecido" separadas diluem o sinal | A decidir | T1.9 |
| P5 | `age`, `weight`, `diag_1–3` e `payer_code` são texto (bandas/códigos), não números | `load.py` | Formato `[70-80)`, códigos ICD-9 com V/E | Perda de códigos e de zeros à esquerda | Confirmado (dados) | T1.1 |
| P6 | As 23 colunas de medicamentos cobrem todos os antidiabéticos relevantes | notebook §1, §6 | O dataset só tem estas colunas | Contagens de medicamentos incompletas | Assumido | T1.8 |
| P7 | `num_procedures` = 6 significa "6 ou mais" e `number_diagnoses` = 9 significa "9 ou mais" | notebook §6 (comentário) | Acumulação no máximo (4 954 em 6; 48,6% em 9, só 115 > 9) | Valores tratados como exatos | Assumido | T1.8 |
| P8 | `time_in_hospital` só vai de 1 a 14 dias por critério de inclusão | notebook §6 | Documentação UCI | — | Confirmado (doc.) | — |
| P26 | Os diagnósticos repetidos no mesmo encontro (A9) resultam de códigos truncados aos 3 dígitos (ex.: 276.1 e 276.8 → 276), não de erro de registo | notebook §1.1 | 5 492 encontros; 100% dos códigos repetidos não têm casas decimais | Contagens de diagnósticos distintos subestimadas | Assumido | T1.8 |
| P27 | Os encontros sem 250.xx em `diag_1–3` (B4, 62,6%) são diabéticos com o código de diabetes fora dos 3 diagnósticos guardados | notebook §1.1 | Critério de inclusão de Strack et al. (2014) usa todos os diagnósticos (até 16, cf. `number_diagnoses`) | População com não diabéticos | Confirmado (doc.) | — |

### População e target

| # | Pressuposto | Onde | Fundamento | Se falhar | Estado | Tarefa |
|---|---|---|---|---|---|---|
| P9 | Target = `readmitted == "<30"`; `>30` e `NO` juntos como negativos | notebook célula 1 | Objetivo do projeto (readmissão em menos de 30 dias) | Toda a EDA de taxas muda | A decidir | T1.5 |
| P10 | Códigos 11, 19, 20, 21 = óbito; 13, 14 = hospício | notebook §5 (`DEAD_HOSPICE`) | Descrições do `IDS_mapping.csv`; 0 readmissões nos óbitos | — | Confirmado (dados) | T1.4 |
| P11 | A EDA usa todos os 101 766 encontros, **incluindo** os 2 423 de óbito/hospício | notebook §5–§10 | A exclusão só é simulada na §5, o `df` não é filtrado | Associações do destino da alta inflacionadas (V=0,125 em parte pelos óbitos) | Decidido: excluir (D1). A EDA mantém a população completa; os passos seguintes partem de `data/processed/` já sem estes encontros | T1.4 |
| P12 | `gender` Unknown/Invalid (3 casos) não é uma categoria com significado | notebook §1 | n=3 | — | A decidir | T3.5 |
| P28 | `readmitted = NO` com um encontro seguinte do mesmo doente (C6) é readmissão fora da janela de observação, não rótulo errado | notebook §1.1 | 490 de 30 248 encontros com encontro seguinte (1,6%); o dataset não tem datas para confirmar | Alguns negativos podem ser `>30` mal rotulados | Assumido | T1.5 |

### Tempo e disponibilidade na alta

| # | Pressuposto | Onde | Fundamento | Se falhar | Estado | Tarefa |
|---|---|---|---|---|---|---|
| P13 | `number_outpatient/emergency/inpatient` contam visitas no ano **anterior** ao encontro | notebook §6, §10 | Documentação UCI | Fuga de informação no preditor mais forte | Confirmado (doc.) | T1.6 |
| P14 | `diag_1–3` estão disponíveis no momento da alta | notebook §10 | A codificação ICD-9 pode ser finalizada depois (faturação) | Fuga de informação ligeira | Assumido | T1.6 |
| P15 | `A1Cresult` e `max_glu_serum` estão disponíveis no momento da alta | notebook §10 | O resultado pode chegar perto ou depois da alta | Fuga de informação ligeira | Assumido | T1.6 |
| P16 | As colunas de medicamentos refletem o estado à alta | notebook §10 | Documentação UCI (alteração durante o encontro) | — | Assumido | T1.6 |
| P17 | `encounter_id` crescente = ordem temporal dos encontros de um doente | notebook §1 (validação de `change`) | O dataset não tem datas | O teste "mesmos fármacos que no encontro anterior" perde validade | Assumido. **Contrariado em 32 encontros** (notebook §1.1): 18 com idade a descer (C3) e 14 depois de um óbito (C5). Ou o `encounter_id` não é sempre temporal ou o `patient_nbr` junta pessoas diferentes; casos excluídos em D2 | T1.6, T1.7 |
| P18 | O nº de encontros do doente no dataset **não** está disponível na alta | notebook §4 | Inclui encontros futuros; a readmissão gera ela própria um encontro | — | Confirmado (dados) | T1.6 |

### Interpretação de variáveis

| # | Pressuposto | Onde | Fundamento | Se falhar | Estado | Tarefa |
|---|---|---|---|---|---|---|
| P19 | `diabetesMed = "Yes"` sse há pelo menos um medicamento prescrito | notebook §1 | Coincide sem exceções | — | Confirmado (dados) | T1.8 |
| P20 | `change = "Ch"` **não** significa "a medicação mudou": reproduz-se por "dose alterada ou ≥2 colunas ativas" | notebook §1 | Regra exata nos dados; 41% dos `Ch` sem nenhuma dose alterada | Interpretação errada no relatório | Confirmado (dados) | T1.8, T1.9 |
| P21 | Os grupos ICD-9 de Strack et al. (2014) são adequados; V/E entram em "Other" | notebook §8 (`icd9_group`) | Referência publicada sobre este dataset | Perde-se sinal de códigos V/E | Assumido | T1.8 |

### Métodos estatísticos da EDA

| # | Pressuposto | Onde | Fundamento | Se falhar | Estado | Tarefa |
|---|---|---|---|---|---|---|
| P22 | Os intervalos de Wilson (95%) tratam os encontros como independentes | notebook `wilson`/`rate_by` | — | **Não se verifica:** 46% dos encontros são de doentes recorrentes, por isso os IC são algo estreitos demais. As conclusões grandes mantêm-se; diferenças pequenas (≈1 p.p.) devem ser lidas com cautela | Assumido | T3.2 |
| P23 | Missing entra como categoria própria `(missing)` no V de Cramér | notebook §9 | Missing é informativo em `medical_specialty`/`payer_code` | — | Assumido | — |
| P24 | Associação entre numéricas medida com Spearman (monotónica, não linear) | notebook §9 | Variáveis assimétricas, com muitos zeros | Relações não monotónicas não detetadas | Assumido | — |
| P25 | Gráficos de taxa por numérica truncam no p99 e mostram só categorias com n ≥ 100 | notebook §5, §7 | Legibilidade; evitar taxas instáveis | Valores extremos e categorias raras ficam fora dos gráficos (não dos dados) | Assumido | — |
