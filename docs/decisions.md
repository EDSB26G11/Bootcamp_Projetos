# Registo de decisões

Decisões sobre exclusões, target, features e limpeza (T1.4–T1.9). A EDA
(`notebooks/01_eda.ipynb`, resumo em [eda_summary.md](eda_summary.md)) só
quantifica; cada decisão fica aqui registada com a evidência e a tarefa
onde foi tomada. As votações da equipa fazem-se no roadmap partilhado e
cada decisão fechada lá é registada aqui.

| # | Tarefa | Decisão | Evidência | Estado |
|---|---|---|---|---|
| D1 | T1.4 | Excluir os encontros com alta por óbito (`discharge_disposition_id` 11, 19, 20, 21). As altas para hospício (13, 14) **ficam** na população. Implementado em `src/data/population.py` | 1 652 encontros de óbito (1,6%), com 0 readmissões: um doente que morreu não pode ser readmitido. Os 771 encontros de hospício têm 43 readmissões (5,6%), por isso o evento existe e são mantidos. Ficam 100 114 encontros, 70 439 doentes, 11,34% de positivos (notebook §5); depois de D7, 100 111 e 70 436; depois de D2, 100 052 e 70 412. *Revisto:* a versão anterior também excluía o hospício (2 423 encontros) | Aceite |
| D2 | T1.4 | Excluir os encontros com pelo menos uma regra de severidade `erro` em `src/data/quality.py` (A1–A3 recém-nascido com idade ≥ 10, C1 `gender` inconsistente, C3 idade desce, C4 idade salta 2+ bandas, C5 encontro depois de óbito). Corre sobre os dados completos, antes das outras exclusões (C5 precisa dos óbitos). Implementado em `src/data/population.py` (`exclude_impossible`) | 59 encontros, 54 doentes (0,06%): registos impossíveis, sem forma de saber qual dos campos está errado (notebook §1.1) | Aceite (roadmap, 29/10) |
| D3 | T1.9 | Harmonizar a `race` por doente com a moda dos seus encontros. Empate (68 dos 123 doentes): fica a `race` do encontro mais recente (maior `encounter_id`, P17). Implementado em `src/data/cleaning.py` (`harmonize_race`) | 123 doentes (404 encontros) com mais de uma `race` (C2); plausível em dados autorreportados, mas o mesmo doente deve ter uma só categoria no modelo (notebook §1.1) | Aceite (roadmap, 29/10) |
| D4 | T1.8 | Repor os zeros à esquerda nos códigos ICD-9 numéricos (`8` → `008`) antes de qualquer agrupamento por texto. Implementado em `src/data/cleaning.py` (`pad_icd9`) | 5 732 códigos em 5 567 encontros (B1), todos numéricos < 100. O `icd9_group` da §8 converte para número e não é afetado (notebook §1.1) | Aceite (roadmap, 29/10) |
| D5 | T1.9 | Juntar os códigos desconhecidos de cada ID numa categoria `Desconhecido` na coluna `_desc` (`admission_type_id` 5, 6, 8; `discharge_disposition_id` 18, 25, 26; `admission_source_id` 9, 15, 17, 20, 21). Os códigos originais não mudam. Um indicador "registo de admissão incompleto" só entra se o modelo mostrar que ajuda (T2). Implementado em `src/data/mappings.py` (`group_unknown_ids`) | Taxas de <30 parecidas entre subcategorias (8,4%–13,8%, as extremas com n < 350) e perto dos 11,4% do resto. Os desconhecidos aparecem juntos: com tipo desconhecido, a origem também o é em 56,6% dos casos (1,3% com tipo conhecido), o que aponta para falha de registo e não para missing ao acaso. Rejeitadas: imputar (esconde o padrão) e excluir linhas (15,1% dos encontros). Era P4 | Aceite (roadmap, 29/10) |
| D6 | T1.5 | Target binário `target` = 1 se `readmitted == "<30"`, 0 se `>30` ou `NO`. Implementado em `src/data/population.py` (`add_target`) | Responde ao objetivo (readmissão a 30 dias, métrica usada nos hospitais dos EUA). Na população final: 11,34% de positivos. Limitação a escrever no relatório: os `>30` contam como negativos, embora tenham voltado ao hospital. "Qualquer readmissão vs NO" pode servir de análise de sensibilidade. Era P9 | Aceite (roadmap, 29/10) |
| D7 | T1.4 | Excluir os encontros com `gender` = `Unknown/Invalid`. Implementado em `src/data/population.py` (`exclude_invalid_gender`) | 3 encontros, todos `NO`, de 3 doentes sem outro encontro (não há como recuperar o valor). Evita uma categoria com n=3 no modelo e na fairness (T3.5). Era P12 | Aceite (roadmap, 29/10) |
| D8 | T1.6 | "Momento da alta" = tudo o que fica registado até a alta ser dada, incluindo diagnósticos, análises e medicação do internamento, com os pressupostos escritos (P14–P16) | Decisão-base da T1.6; a alternativa (só o que está fechado e validado) deixaria diagnósticos e análises em dúvida. Roadmap 1.6·1 | Aceite (roadmap, 29/10) |
| D9 | T1.6 | Aprovar como features, com pressuposto escrito no relatório: `diag_1–3` e `number_diagnoses` (P14), `A1Cresult` e `max_glu_serum` (P15), as 23 colunas de medicamentos, `change` e `diabetesMed` (P16). Se ficam todas no modelo (redundância de `change`/`diabetesMed`, medicamentos quase constantes) decide-se em T1.8/T1.9 | A codificação ICD-9 e os resultados das análises podem ser finalizados perto ou depois da alta: risco de fuga ligeiro, aceite e documentado. Roadmap 1.6·2–1.6·4 | Aceite (roadmap, 29/10) |
| D10 | T1.6 | Aprovar sem reservas, por serem conhecidas antes ou no momento da alta: destino da alta (`discharge_disposition_id`, também feature além de definir a população), administrativas (`payer_code`, `medical_specialty`, `admission_type_id`, `admission_source_id`), internamento (`time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`), histórico (`number_outpatient/emergency/inpatient`, P13) e demográficas (`age`, `gender`, `race`; o uso por fairness decide-se em T3.5) | Registadas na admissão ou durante o internamento; o histórico refere-se ao ano anterior. Roadmap 1.6·5–1.6·9 | Aceite (roadmap, 29/10) |
| D11 | T1.6 | Excluir das features: `readmitted` (origina o target), `encounter_id` e `patient_nbr` (identificadores; `patient_nbr` só serve para o split, T1.7) | Por definição. Roadmap 1.6·10 | Aceite (roadmap, 29/10) |
| D12 | T1.6 | Proibir (1) agregações por doente calculadas sobre o dataset completo (ex.: nº de encontros do doente) e (2) usar a ordem do `encounter_id` como tempo | (1) incluem internamentos posteriores à alta (P18; taxa de <30 de 4,0% com 1 encontro vs 23,2% com 3+). (2) a ordem não é garantidamente temporal (P17, contrariado em 32 encontros). Roadmap 1.6·11 | Aceite (roadmap, 29/10) |
| D13 | T1.7 | Teste fixo de 20% dos doentes e CV de 5 folds no treino, ambos com `StratifiedGroupKFold` por `patient_nbr` (seed 42). A CV escolhe modelo, threshold e tratamento do desequilíbrio; o teste só se usa uma vez (T3.6). Ids gravados em `data/splits/` | 23,4% dos doentes têm mais de um encontro (46,1% dos encontros). Num split aleatório por encontro, 41% dos encontros de teste teriam o doente no treino. Com o split agrupado e estratificado, cada parte fica com 11,34% de positivos. Alternativas: 70/15/15 fixo (validação mais instável), só o 1.º encontro por doente (muda a população, 8,93% de positivos, usa a ordem do `encounter_id`, D12) | Proposta |

Estados: `Proposta` · `Aceite` · `Rejeitada`.

População final (D1, D2, D7): **100 052 encontros, 70 412 doentes, 11,34% de positivos**.

## Pressupostos

Pressupostos assumidos em `notebooks/01_eda.ipynb` e em `src/data/`. Não são
decisões, mas os números da EDA dependem deles. Se algum cair, os resultados
indicados têm de ser revistos.

Estados:

- `Confirmado (dados)`: verificado nos próprios dados.
- `Confirmado (doc.)`: está na documentação UCI / Strack et al. (2014).
- `Assumido`: plausível, mas não verificável com os dados.
- `A decidir`: pressuposto provisório da EDA que tem de virar decisão na tarefa indicada.
- `Decidido (Dn)`: já virou decisão; ver a linha Dn acima.

### Leitura e codificação dos dados

| # | Pressuposto | Onde | Fundamento | Se falhar | Estado | Tarefa |
|---|---|---|---|---|---|---|
| P1 | `?` é o único marcador de valor em falta | `load.py`; notebook §3 | Contagens batem com o guia (race 2 273, weight 98 569) | Missing subestimado noutras colunas | Confirmado (dados) | T1.1 |
| P2 | `"None"` em `A1Cresult`/`max_glu_serum` = teste não realizado, não missing | `load.py`; notebook §3 | Documentação UCI; taxas de <30 diferentes entre medido e não medido | Uma categoria com sinal seria tratada como missing | Confirmado (doc.) | T1.8 |
| P3 | `"NULL"` no `IDS_mapping.csv` é uma descrição válida, não missing | `mappings.py` | É o texto literal do ficheiro (ex.: `admission_type_id` 6) | — | Confirmado (dados) | T1.2 |
| P4 | "NULL", "Not Available" e "Not Mapped" nos IDs são, na prática, desconhecido (≈10%) | notebook §6 | Semântica das descrições | Três categorias "desconhecido" separadas diluem o sinal | Decidido: juntar (D5) | T1.9 |
| P5 | `age`, `weight`, `diag_1–3` e `payer_code` são texto (bandas/códigos), não números | `load.py` | Formato `[70-80)`, códigos ICD-9 com V/E | Perda de códigos e de zeros à esquerda | Confirmado (dados) | T1.1 |
| P6 | As 23 colunas de medicamentos cobrem todos os antidiabéticos relevantes | notebook §1, §6 | O dataset só tem estas colunas | Contagens de medicamentos incompletas | Assumido | T1.8 |
| P7 | `num_procedures` = 6 significa "6 ou mais" e `number_diagnoses` = 9 significa "9 ou mais" | notebook §6 (comentário) | Acumulação no máximo (4 954 em 6; 48,6% em 9, só 115 > 9) | Valores tratados como exatos | Assumido | T1.8 |
| P8 | `time_in_hospital` só vai de 1 a 14 dias por critério de inclusão | notebook §6 | Documentação UCI | — | Confirmado (doc.) | — |
| P26 | Os diagnósticos repetidos no mesmo encontro (A9) resultam de códigos truncados aos 3 dígitos (ex.: 276.1 e 276.8 → 276), não de erro de registo | notebook §1.1 | 5 492 encontros; 100% dos códigos repetidos não têm casas decimais | Contagens de diagnósticos distintos subestimadas | Assumido | T1.8 |
| P27 | Os encontros sem 250.xx em `diag_1–3` (B4, 62,6%) são diabéticos com o código de diabetes fora dos 3 diagnósticos guardados | notebook §1.1 | Critério de inclusão de Strack et al. (2014) usa todos os diagnósticos (até 16, cf. `number_diagnoses`) | População com não diabéticos | Confirmado (doc.) | — |

### População e target

| # | Pressuposto | Onde | Fundamento | Se falhar | Estado | Tarefa |
|---|---|---|---|---|---|---|
| P9 | Target = `readmitted == "<30"`; `>30` e `NO` juntos como negativos | notebook célula 1 | Objetivo do projeto (readmissão em menos de 30 dias) | Toda a EDA de taxas muda | Decidido (D6) | T1.5 |
| P10 | Códigos 11, 19, 20, 21 = óbito; 13, 14 = hospício | notebook §5 (`DEATHS`, `HOSPICE`) | Descrições do `IDS_mapping.csv`; 0 readmissões nos óbitos, 43 nos hospícios | — | Confirmado (dados) | T1.4 |
| P11 | A EDA usa todos os 101 766 encontros, **incluindo** os 1 652 de óbito | notebook §5–§10 | A exclusão só é simulada na §5, o `df` não é filtrado | Associações do destino da alta inflacionadas (V=0,125 em parte pelos óbitos) | Decidido: excluir os óbitos e manter o hospício (D1). A EDA mantém a população completa; os passos seguintes partem de `data/processed/` já sem os óbitos | T1.4 |
| P12 | `gender` Unknown/Invalid (3 casos) não é uma categoria com significado | notebook §1 | n=3 | — | Decidido: excluir (D7) | T1.4 |
| P28 | `readmitted = NO` com um encontro seguinte do mesmo doente (C6) é readmissão fora da janela de observação, não rótulo errado | notebook §1.1 | 490 de 30 248 encontros com encontro seguinte (1,6%); o dataset não tem datas para confirmar | Alguns negativos podem ser `>30` mal rotulados | Assumido | T1.5 |

### Tempo e disponibilidade na alta

| # | Pressuposto | Onde | Fundamento | Se falhar | Estado | Tarefa |
|---|---|---|---|---|---|---|
| P13 | `number_outpatient/emergency/inpatient` contam visitas no ano **anterior** ao encontro | notebook §6, §10 | Documentação UCI | Fuga de informação no preditor mais forte | Confirmado (doc.); aprovadas (D10) | T1.6 |
| P14 | `diag_1–3` estão disponíveis no momento da alta | notebook §10 | A codificação ICD-9 pode ser finalizada depois (faturação) | Fuga de informação ligeira | Assumido; aceite e a escrever no relatório (D9) | T1.6 |
| P15 | `A1Cresult` e `max_glu_serum` estão disponíveis no momento da alta | notebook §10 | O resultado pode chegar perto ou depois da alta | Fuga de informação ligeira | Assumido; aceite e a escrever no relatório (D9) | T1.6 |
| P16 | As colunas de medicamentos refletem o estado à alta | notebook §10 | Documentação UCI (alteração durante o encontro) | — | Assumido; aceite (D9) | T1.6 |
| P17 | `encounter_id` crescente = ordem temporal dos encontros de um doente | notebook §1 (validação de `change`) | O dataset não tem datas | O teste "mesmos fármacos que no encontro anterior" perde validade | Assumido. **Contrariado em 32 encontros** (notebook §1.1): 18 com idade a descer (C3) e 14 depois de um óbito (C5). Ou o `encounter_id` não é sempre temporal ou o `patient_nbr` junta pessoas diferentes; casos excluídos em D2 | T1.6, T1.7 |
| P18 | O nº de encontros do doente no dataset **não** está disponível na alta | notebook §4 | Inclui encontros futuros; a readmissão gera ela própria um encontro | — | Confirmado (dados); proibido como feature (D12) | T1.6 |

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
