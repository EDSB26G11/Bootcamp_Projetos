---
title: Prioridade de follow-up após a alta (diabetes)
emoji: 📈
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
python_version: '3.12'
app_file: app.py
pinned: false
license: apache-2.0
short_description: Risco de readmissão a 30 dias, para priorizar o follow-up
---

# Prioridade de follow-up após a alta hospitalar (diabetes)

Projeto do grupo **EDSB26_G11** para o Enterprise Data Science Bootcamp 2026-27, NOVA IMS.
Ana Carapinha (20250306), Filipa Filipe (a indicar), Sandra Figueiredo (20251705), Tiago Marques (20251844).

A app estima, no momento da alta, o risco de um doente com diabetes ser readmitido em menos de 30 dias,
e diz onde essa alta fica face às outras e se entra na lista de follow-up com capacidade de 10% ou de 20%
das altas elegíveis.

- **Uma pessoa revê cada caso.** A app propõe; não decide.
- **Não é causal.** Diz quem tem mais risco, não a quem o follow-up evita uma readmissão.
- **Dados:** UCI *Diabetes 130-US hospitals for years 1999–2008* (Strack et al., 2014). Não usar noutras
  populações nem em decisões clínicas.

Modelo: `HistGradientBoostingClassifier` (scikit-learn), gravado com skops. Model card em `models/v1/README.md`.
Código, decisões e relatório: <https://github.com/EDSB26G11/Bootcamp_Projetos>.
