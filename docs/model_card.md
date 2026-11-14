# Model card: readmissão a 30 dias em doentes com diabetes (modelo v1)

Grupo EDSB26_G11, Enterprise Data Science Bootcamp 2026-27, NOVA IMS. Ana Carapinha (20250306), Filipa
Filipe (a indicar), Sandra Figueiredo (20251705), Tiago Marques (20251844).

Ficheiros em `models/v1/`: `model.skops` (pipeline completo), `config.json` (features, cortes, quantis,
versões) e este `README.md`. A fonte deste texto é `docs/model_card.md`, versionado; o `save_model()` copia-o
para `models/v1/README.md`.

> **Estado:** resultados da validação cruzada (previsões out-of-fold do treino). A secção “Avaliação no teste”
> preenche-se depois da avaliação única no teste (T3.6).

## Uso previsto

- **Para quê:** no momento da alta hospitalar, estimar o risco de o doente ser readmitido em menos de 30 dias
  e ordenar as altas elegíveis por esse risco, para a equipa de follow-up contactar primeiro as de maior risco,
  até ao limite da sua capacidade (10% ou 20% das altas elegíveis).
- **Para quem:** a Diretora de Care Quality e a equipa de follow-up de uma rede hospitalar.
- **Como:** o modelo propõe; **uma pessoa revê cada caso** e decide o contacto. A app (`src/app.py`) mostra o
  risco estimado, a posição da alta face às outras e a ação para cada capacidade.
- **Altas elegíveis (D21):** altas para a comunidade sem acompanhamento. As altas para outro hospital ou
  instituição, internamento, apoio domiciliário ou cuidados paliativos aparecem como “fora da lista”.

## Usos excluídos

- Decisões clínicas automáticas, ou recusar cuidados a quem tem risco baixo.
- Afirmar que o follow-up **causa** menos readmissões: o modelo não é causal (ver Limites).
- Explicar o risco de um doente concreto: a importância das variáveis é global, não por doente.
- Usar noutras populações, outros países ou anos recentes sem nova validação (dados dos EUA, 1999–2008).
- Usar o risco como avaliação do desempenho de médicos, serviços ou hospitais.

## Dados

- **Fonte:** UCI *Diabetes 130-US hospitals for years 1999–2008* (Strack et al., 2014,
  DOI 10.1155/2014/781670). Um encontro (internamento) por linha.
- **População (D1–D7):** 101 766 encontros brutos; saem os óbitos (não podem ser readmitidos), 59 encontros
  impossíveis e 3 com `gender` Unknown/Invalid. Ficam 100 052 encontros de 70 412 doentes.
- **Target (D6):** 1 se `readmitted == "<30"`; 0 se `>30` ou `NO`. Positivos: 11,34%.
- **Split (D13):** por doente (`StratifiedGroupKFold`, seed 42): teste com 20% dos doentes (20 011 encontros),
  fechado até à T3.6; treino com 80 041 encontros e 5 folds de validação cruzada. O mesmo doente nunca está
  no treino e na validação.

## Modelo

- `HistGradientBoostingClassifier` do scikit-learn (D17): `learning_rate=0.05`, `max_iter=300`,
  `max_leaf_nodes=31`, `min_samples_leaf=100`, `l2_regularization=1.0`, sem early stopping (a validação interna
  do scikit-learn não separa por doente). Sem tuning.
- **34 features** conhecidas no momento da alta (D8–D15): demografia, tipo, origem e destino da alta,
  especialidade, pagador, duração e atos do internamento, diagnósticos agrupados (Strack et al.), análises,
  medicação e visitas no ano anterior. Sem identificadores e sem agregações por doente (D11–D12).
- **Pré-processamento (D16):** ajustado só no treino; `log1p` nas contagens, escala nas numéricas, one-hot nas
  categóricas com as raras (< 100) agrupadas; valores em falta como categoria própria.
- Retreinado com as 80 041 altas de treino. Gravado com **skops**: o `load_model()` só aceita os tipos revistos.

## Avaliação (validação cruzada, 5 folds por doente)

| Métrica | Modelo v1 | Ordenar por internamentos anteriores | Ao acaso |
|---|---|---|---|
| PR-AUC (todas as altas) | 0,234 ± 0,008 | 0,176 | 0,113 |
| ROC-AUC | 0,677 | 0,607 | 0,50 |
| Recall@10% (altas elegíveis) | 25,0% | 24,0% | 10,0% |
| Recall@20% (altas elegíveis) | 40,1% | 37,6% | 20,0% |
| Readmissões encontradas por 1 000 altas elegíveis, a 10% / 20% | 24 / 38 | 23 / 36 | 10 / 19 |

- **Calibração:** nas altas elegíveis, o risco previsto e o observado diferem no máximo 0,8 pp por decil; Brier
  0,083 contra 0,086 ao acaso. Não foi preciso recalibrar.
- **Cortes por capacidade** (sobre as previsões out-of-fold das altas elegíveis): risco ≥ 0,169 para os 10%
  de maior risco; risco ≥ 0,125 para os 20%.
- **Variáveis que mais pesam** (importância por permutação, perda de PR-AUC): internamentos no ano anterior
  (−0,052) e destino da alta (−0,050); muito atrás, pagador, especialidade e visitas anteriores.
- Para comparação, a literatura neste dataset fica em ROC-AUC 0,65–0,70.

## Avaliação no teste

_Por preencher na T3.6, uma única vez: calibração por decis, recall e precisão a 10% e 20% nas altas
elegíveis, % de altas realmente sinalizadas pelo corte (meta: k ± 1 pp). Se ficar abaixo das metas da D20,
reporta-se assim, sem afinar no teste._

## Equidade

Com 10% de capacidade, nas previsões out-of-fold das altas elegíveis:

- **Sexo:** equilibrado (sinalizadas 9,9% das mulheres e 10,1% dos homens; recall 25,9% e 24,2%).
- **Raça em falta:** só 1,9% destas altas são sinalizadas (recall 2,6%, contra 16–29% nos outros grupos).
- **90–100 anos:** 6,2% sinalizadas (recall 10,1%), apesar da taxa de readmissão mais alta (12,5%).
- Grupos com menos de 100 altas não se reportam.

Quem revê a lista deve conhecer estas diferenças; a monitorização deve seguir a % de sinalizados por grupo.

## Limites

- **Não causal:** diz quem tem mais risco, não a quem o follow-up evita uma readmissão. Medi-lo exige um
  piloto com grupo de controlo.
- **Labels incompletos:** os `>30` contam como negativos, e as readmissões noutros hospitais não ficam
  registadas.
- **Pressupostos de disponibilidade na alta** (P14–P16): diagnósticos, análises e medicação são tratados como
  conhecidos no momento da alta.
- **Dados antigos e sem datas:** não há validação no tempo; a capacidade é uma % das altas, não “por dia”.
- **Sinal fraco:** a vantagem sobre a regra simples é clara a 20% de capacidade e pequena a 10%.
- **Sobreajuste:** PR-AUC 0,379 no treino contra 0,234 na validação.

## Reprodutibilidade

Python 3.13.13, scikit-learn 1.9.1, skops 0.16.0 (as mesmas versões em `requirements.txt`); seed 42.
Treino e gravação: `notebooks/06_deployment.ipynb`. Decisões D1–D23 em `docs/decisions.md`.
