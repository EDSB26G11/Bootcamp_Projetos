# EDA — resumo (T1.3)

Resumo dos achados de `notebooks/01_eda.ipynb`. A EDA só quantifica; as decisões são tomadas nas tarefas indicadas e registadas em `docs/decisions.md`.

## Números-base

- 101 766 encontros, 71 518 doentes, 50 colunas originais.
- Target (`readmitted == "<30"`): **11,16%** de positivos. NO 53,9% · >30 34,9%.
- 23,5% dos doentes têm mais de 1 encontro e somam 46,2% dos encontros.
- 2 423 encontros com alta por óbito ou para hospício (43 positivos).

## Achados principais

1. **Desequilíbrio forte.** Prever sempre "não" dá 88,8% de accuracy. Usar PR-AUC e recall@k (T3.2).
2. **Fuga de informação pelo nº de encontros.** A taxa de <30 é 4,0% em doentes com 1 encontro e 23,2% com 3 ou mais, porque a própria readmissão cria um novo encontro no dataset. Essa contagem **não pode ser feature** (T1.6) e o split tem de ser agrupado por doente (T1.7).
3. **`number_inpatient` é o preditor mais forte**: 8,4% sem internamentos prévios até 39,8% com 6 ou mais. `number_emergency` duplica a média nos positivos.
4. **Destino da alta diferencia muito**: casa 9,3%, SNF 14,7%, reabilitação 27,7%; óbitos 0%.
5. **"None" em `A1Cresult`/`max_glu_serum` é "não medido", não missing**, e tem sinal (A1C não medido 11,4% vs medido 9,9%). Manter como categoria.
6. **`race` e `gender` quase não diferenciam** a taxa (≤1,7 p.p. nas categorias grandes).
7. Contagens de visitas muito assimétricas (66–89% de zeros, máximos até 76).
8. **Consistência (§1.1, `src/data/quality.py`).** 21 regras em 3 camadas (dentro do encontro, formato, entre encontros do doente). **59 encontros (0,06%) são impossíveis**: 19 com admissão/origem/alta de recém-nascido em doentes com 20–100 anos, 14 depois de um óbito, 18 com a idade a descer, 7 de doentes com 2 `gender`. Propõe-se excluí-los (D2). Também: 123 doentes com 2 `race` (D3), 5 732 códigos ICD-9 sem zeros à esquerda (D4), 490 `NO` com encontro seguinte (P28). Domínios, duplicados e regras sexo × diagnóstico estão limpos.

## Variáveis candidatas a descartar

| Variável | Evidência | Proposta | Tarefa |
|---|---|---|---|
| `weight` | 96,9% missing; taxa igual com e sem valor (11,2% vs 11,2%) | Descartar | T1.9 |
| `examide`, `citoglipton` | Constantes | Descartar | T1.9 |
| 13 medicamentos com ≥99,3% `No` (`nateglinide`, `chlorpropamide`, `acetohexamide`, `tolbutamide`, `acarbose`, `miglitol`, `troglitazone`, `tolazamide`, `glyburide-metformin`, `glipizide-metformin`, `glimepiride-pioglitazone`, `metformin-rosiglitazone`, `metformin-pioglitazone`) | Quase sem variação | Descartar; agregar em nº de medicamentos alterados | T1.8, T1.9 |
| `encounter_id`, `patient_nbr` | Identificadores | Não são features (`patient_nbr` só para o split) | T1.6, T1.7 |
| `payer_code` | 39,6% missing, sinal fraco (V=0,03) | Candidata; testar no modelo | T1.9, T2 |
| `medical_specialty` | 49,1% missing, 73 níveis | Manter com `?` como categoria e níveis raros agrupados | T1.9 |
| `max_glu_serum` | 94,7% não medido; sinal fraco (V=0,01) | Candidata fraca | T1.9 |
| `*_id` vs `*_id_desc` | Mesma informação | Manter só uma | T1.9 |

## Figuras

`docs/figures/`: `01_target.png`, `02_missing.png`, `03_encontros_por_doente.png`, `04_taxa_idade_raca.png`, `05_taxa_number_inpatient.png`.
