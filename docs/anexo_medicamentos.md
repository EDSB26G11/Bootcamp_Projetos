# Anexo — Medicamentos do dataset

Referência para as 23 colunas de medicamentos do `diabetic_data.csv` (de
`metformin` a `metformin-pioglitazone`). Cada coluna indica o que aconteceu ao
fármaco no encontro: `No` (não prescrito), `Steady` (dose mantida), `Up` (dose
aumentada) ou `Down` (dose reduzida). Ver `notebooks/01_eda.ipynb`, secções 1 e 6.

## Classes terapêuticas

Todos são antidiabéticos para a diabetes tipo 2, exceto a insulina, que se usa
nos dois tipos. "Uso" = % de encontros com valor diferente de `No` (101 766
encontros).

| Classe | Medicamentos | Como atua | Uso |
|---|---|---|---|
| **Insulina** | insulin | Substitui ou completa a insulina que o corpo produz; única opção na diabetes tipo 1, usada na tipo 2 avançada | 53,44% |
| **Biguanida** | metformin | Reduz a produção de glicose pelo fígado; primeira linha na tipo 2 | 19,64% |
| **Sulfonilureias de 2.ª geração** | glipizide, glyburide, glimepiride | Estimulam o pâncreas a libertar insulina; principal risco é a hipoglicemia | 12,47%, 10,47%, 5,10% |
| **Sulfonilureias de 1.ª geração** | chlorpropamide, tolazamide, tolbutamide, acetohexamide | Mesmo mecanismo; mais antigas, já quase todas substituídas pelas de 2.ª geração | 0,08%, 0,04%, 0,02%, 0,00% |
| **Tiazolidinedionas (glitazonas)** | pioglitazone, rosiglitazone, troglitazone | Aumentam a sensibilidade dos tecidos à insulina | 7,20%, 6,25%, 0,00% |
| **Meglitinidas** | repaglinide, nateglinide | Estimulam a libertação de insulina durante pouco tempo, tomadas às refeições | 1,51%, 0,69% |
| **Inibidores da alfa-glucosidase** | acarbose, miglitol | Atrasam a absorção dos açúcares no intestino | 0,30%, 0,04% |
| **Combinações em dose fixa** | glyburide-metformin, glipizide-metformin, glimepiride-pioglitazone, metformin-rosiglitazone, metformin-pioglitazone | Dois princípios ativos num só comprimido | 0,69%, 0,01%, 0,00%, 0,00%, 0,00% |
| **Não identificados** | examide, citoglipton | Ver nota abaixo | 0,00% |

Os valores 0,00% correspondem a 0–3 encontros: troglitazone 3, metformin-rosiglitazone 2,
acetohexamide, glimepiride-pioglitazone e metformin-pioglitazone 1, examide e citoglipton 0.

## Contexto histórico (dados de 1999–2008)

- **Troglitazona** — retirada do mercado dos EUA pela FDA em março de 2000 por
  toxicidade hepática grave (insuficiência hepática aguda com mortes e
  transplantes). Daí só 3 encontros.
- **Rosiglitazona** — alvo de alertas e restrições nos EUA em 2010, por risco
  cardiovascular, já depois do período do dataset.
- **Sulfonilureias de 1.ª geração** — fármacos antigos, com uso residual no período.
- **Combinações com pioglitazona/rosiglitazona** — entraram no mercado a meio da
  década de 2000, o que explica terem 1–2 encontros. *Datas não confirmadas por fonte.*

## Nota sobre `examide` e `citoglipton`

São constantes (0 doentes). A documentação da UCI só os lista como colunas, sem
os descrever. `citoglipton` não corresponde a nenhum fármaco conhecido e pode ser
um erro de escrita de *sitagliptin* (inibidor da DPP-4, aprovada nos EUA em
2006). `examide` também não foi identificado como antidiabético. **Nenhuma destas
hipóteses está confirmada**: não citar no relatório sem fonte. Para o modelo é
irrelevante, porque ambas saem por serem constantes.

## Implicações para o modelo (T1.8, T1.9)

- Com variação suficiente para usar individualmente: `insulin`, `metformin`,
  `glipizide`, `glyburide`, `pioglitazone`, `rosiglitazone`, `glimepiride`.
- Restantes: descartar e resumir a informação, por exemplo com o nº de
  medicamentos alterados ou indicadores por classe (sulfonilureia, glitazona, etc.).
  Ver [eda_summary.md](eda_summary.md).

## Fontes

- [UCI — Diabetes 130-US Hospitals for Years 1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008)
- [Fairlearn — Diabetes 130-Hospitals Dataset](https://fairlearn.org/v0.9/user_guide/datasets/diabetes_hospital_data.html)
- [NCBI LiverTox — Troglitazone](https://www.ncbi.nlm.nih.gov/books/NBK548114/)
- [AAFP — Clinical Briefs (retirada do Rezulin, 2000)](https://www.aafp.org/pubs/afp/issues/2000/0501/p2887.html)
- [NFB — Rezulin Withdrawn from Market](https://nfb.org/images/nfb/publications/vod/vsum0003.htm)

As classes terapêuticas, os mecanismos de ação e as datas da sitagliptina e das
combinações vêm de conhecimento farmacológico geral, não destas fontes.
