# EDSB 2026-27 — Course & Project Context for Claude Code

> Decision-support context for Claude Code working on the group project of the
> **Enterprise Data Science Bootcamp (EDSB), NOVA IMS, 2026-27** — *Diabetes: Hospital Readmission*, group **G11**.
> Reference it from `CLAUDE.md` with `@EDSB_CONTEXT.md`.
>
> **State as of 11 Nov 2026**, branch `feature/t2-3-kumo-challenger` (uncommitted changes at the time).
> **Source of truth for decisions is `docs/decisions.md`** — if it disagrees with this file, `decisions.md` wins
> and this file must be updated.
>
> **Class sources read:** W1 syllabus slides; W4 and W5 Hugging Face slides; class summaries of 30 Sep (W4),
> 7 Oct (W5), 14 Oct (W6); Class 3 and Class 6–7 teaching repos; project guide PDF.
> **Not read directly:** W2, W7–W8 material (team docs cite W8 IBCS rules — see §3), W9+ (GitHub Actions, Spaces,
> LLMs/agents). Class transcripts are auto-generated and garbled; only summaries and slides are authoritative.

---

## 1. Rules for Claude Code

1. **The team decides; Claude proposes and implements.** Only `Aceite` rows in `decisions.md` are binding.
   `Proposta` / open roadmap items (§4.4) → present options with numbers, don't implement silently.
2. **The test set is locked until T3.6** and is opened **once**, for the single chosen model. All selection,
   tuning, thresholds and fairness use the 5-fold out-of-fold (OOF) predictions on train (D13).
3. **Keep the pipeline runnable end-to-end** (`Projeto_Final.ipynb` runs top to bottom from `src/` + `reports/`).
   One change at a time; run `python -m pytest tests/ -v` after each change.
4. **Log every run** in `reports/experiments.csv` (via `src/models/train.py`, `run_cv`), including failed ones.
5. **Record** each decision in `decisions.md` (task, decision, evidence, status) and each assumption in the P-tables.
6. **Methods policy (D19):** Phase 3 uses only methods taught in class plus what the guide requires. Out:
   XGBoost, SMOTE/undersampling, isotonic/Platt calibration, SHAP, risk tiers, cost-benefit, joblib/pickle.
   (The course itself allows justified alternatives — W1 "justify relevant alternatives" — so D19 is a team
   choice, stricter than the course. Don't reintroduce these without a new decision.)
7. **Persist models with `skops`**, reviewing untrusted types (`sio.get_untrusted_types`) — never pickle/joblib (W5).
8. **Flag uncertainty.** Unconfirmed claims stay labelled (e.g. `examide`/`citoglipton` identity, the `change` rule,
   the 6/9 caps). Never invent meanings for coded fields without a lookup (`payer_code`).

---

## 2. Course at a glance

| Item | Detail |
|---|---|
| Lecturers | Henrique Carreiro, João Azambuja |
| Team (G11) | Ana Carapinha 20250306 · Filipa Filipe *(number missing)* · Sandra Figueiredo 20251705 · Tiago Marques 20251844 |
| Assessment | Project 50% + Exam 50% (exam: slides + recommended readings; T/F, multiple choice, a few open questions; one handwritten A4 sheet, both sides) |
| Status reports | Moodle form, Weeks 3, 6, 9: status, challenges, expected insights, **each member's contribution**, **how AI is used responsibly**, next steps |
| Final | 10-min presentation + 10-min defence + **live demo** in a realistic scenario; every member must explain every decision |
| Calendar (W1) | W1–3 Git/VS Code · W4–5 Hugging Face · W6–8 best practices (imbalance, metrics, presentation/IBCS) · W9–10 end-to-end with **GitHub Actions, HF, Gradio** · W11 **LLMs and agents** · W12 project support (team code freeze) · W13–14 presentations |

**Submission:** Moodle ZIP `EDSB26_G11.zip` with the deck PDF, relevant `.py`/`.ipynb`, other artifacts, and
`EDSB26_G11.csv` (group letter, student number, student name). Final code committed to GitHub (the assessed version).
**Every deliverable lists every student's name and number.** Deck: **max 30 slides incl. cover and appendix**.

**Repository requirements (W1):** README (overview, critical components, how to run, prerequisites); detailed comments
and **docstrings**; `requirements.txt`; `/src`, `/docs`; `.gitignore`; meaningful commit messages and consistent naming.

---

## 3. What each class asks of the project

| Class | Guidance that applies to the project |
|---|---|
| **W1** | Start from the decision (stakeholder, action, baseline, constraints, cost of error). Success on 3 levels: model, operational, business. Evaluate beyond accuracy (imbalance, robustness, explainability, business fit). Responsible AI with human oversight. Reproducible work. |
| **W3** | Python 3.13 + `.venv`; commit both `requirements.txt` (asked) and `requirements-installed.txt` (`pip freeze`, got). `data/raw` read-only; generated files rebuildable and git-ignored; human-written reports committed. **Notebooks committed as `.py` via jupytext, `*.ipynb` ignored.** Black; docstring on every function; one logical change per commit with a descriptive message. Profile data before trusting docs. Human-written reports: problem statement, data audit (feature justification), **AI usage log** — all committed. Repo private, lecturers as collaborators. |
| **W4** | LLM foundations; hallucination sources (cutoff, context overflow, compaction) → verify high-stakes outputs. Agent ≠ LLM. Model-choice checklist: task → licence → size → usage signals → model card → **own test**; pin the revision. |
| **W5** | Adopt a pretrained model by evidence: same folds, simple baseline, metric fit to the problem (PR-AUC under imbalance), costs, written decision record. Pin library commit, weights tag, data version. **A model package = weights/context + `config.json` + Model Card** (intended use, **excluded uses**, data, evaluation, limitations). skops/safetensors, not pickle. Gradio → Spaces. |
| **W6–7** | Train / validation / test; choose on validation only; **test once**. One experiment log, **train + validation rows per model**, named models, code path for reproducibility. Start with a baseline. Lecturer's suggested minimum metrics: **Accuracy, Precision, Recall, F1** (non-prescriptive — pick relevant ones and justify). **Show ≥ 2 confusion matrices (train + validation)**, full table in appendix. Threshold turns a probability into an action. Simple end-to-end pipeline first. Working charts ≠ presentation charts; Plotly for working charts; explain "higher/lower is better" for non-experts. Graded on translating the business problem → model → value, not on the model alone; **start the deck early**. |
| **W8** *(cited in team docs, not read here)* | IBCS reporting: message first, who/what/when, PL (plan) as target, "start from the decision the page supports"; storyline situation → problem → question → answer → evidence → consequences. |
| **W9–W11** *(upcoming)* | GitHub Actions (CI running pytest), HF model repo + Space with Gradio, LLM/agent component with human oversight. |

---

## 4. The project — current state

### 4.1 Question and framing (D20, D22; `docs/problem_framing.md`)
> At discharge, what is the risk that a patient is readmitted in < 30 days, and use that risk to prioritise
> additional follow-up.

Two parts: **(1) risk** — judged by **calibration** (predicted vs observed by decile, Brier, log loss);
**(2) priority** — judged by **PR-AUC and recall/precision in the top k%** of *eligible* discharges (D21).
Stakeholder: Director of Care Quality; a follow-up team with fixed capacity (10% or 20%, still open).
Baseline to beat at the same capacity: **ranking by `number_inpatient`** (random is the floor).
Limits to repeat everywhere: not causal; labels incomplete (other hospitals, `>30` counted negative); 1999–2008, no
dates; human review of every contact; calibration valid only for this population.

### 4.2 Data and population
- Population (D1, D2, D7): **100,052 encounters, 70,412 patients, 11.34% positive**.
  D1 excludes **deaths only** (11, 19, 20, 21); **hospice (13, 14) is kept** (43 readmissions exist).
- Split (D13): fixed test = 20% of patients (20,011 encounters, 14,103 patients) + 5-fold `StratifiedGroupKFold`
  by `patient_nbr` on train (80,041), seed 42; ids in `data/splits/splits.csv`.
- Eligible for follow-up (D21): discharges home without support; 15 discharge codes excluded from the **list**
  (still in training). Train: 52,436 eligible (65.5%), 9.6% positive. All @k metrics, fairness and capacity
  comparisons are computed on eligible discharges only.
- Gotchas still apply: `?` = missing; `"None"` in lab results = not measured (a category); `"NULL"` in ID lookups is a
  valid text; `age`/`weight`/`diag_*`/`payer_code` are text; `change = "Ch"` ≠ "medication changed" (P20).
- Re-read processed data with `pd.read_csv(..., keep_default_na=False, na_values="")`.

### 4.3 Features and model
- 34 features (D15; `docs/features_aprovadas.md`), engineered row-wise only (D14: `age_band`, `diag_1–3_group`,
  `diabetes_complicated`, `n_meds*`, `n_visits_prior`). Forbidden: per-patient aggregates over the whole dataset and
  `encounter_id` order as time (D12).
- Preprocessing (D16): one `ColumnTransformer` fit on train folds (scaler; `log1p`+scaler on visit counts; one-hot
  with `min_frequency=100`) → 160 columns.
- **Model v1 (D17): `HistGradientBoostingClassifier`**, no early stopping. OOF PR-AUC **0.234 ± 0.008**
  (logistic 0.210, RF 0.221, random 0.113). Train PR-AUC 0.379 → **large overfit, tuning pending**.
- Ablation (D18): `payer_code` stays; incomplete-registration flag out; `max_glu_serum` pending.
- Calibration (D22): predicted vs observed within 1 pp per decile on OOF → no recalibration.
- At k = 10% eligible: recall 25.0% vs 24.0% (`number_inpatient`) vs 10% random; at 20%: 40.1% vs 37.6%.
- **Kumo Tabular (D23)**: documented challenger only — ties v1 (PR-AUC 0.233 ± 0.005) at higher cost (GPU, ~14×
  inference time, patient data travels with the model). Not in the app or `models/`. HF Hub used only to publish
  v1 and the Space.
- App: `src/app.py` (Gradio) shows estimated risk vs eligible average, percentile and action; out-of-list discharges
  are labelled. Model saved with skops in `models/v1/` (`model.skops`, `config.json`, README placeholder).

### 4.4 Open items (roadmap, 11 Nov)
| Item | Status |
|---|---|
| Capacity 10% or 20% (F3·1) | open — business decision |
| Targets/PL of D20 (F3·2) | `Proposta` |
| `max_glu_serum` (F3·3, D18) | to confirm (doesn't change the model) |
| Tuning HGB (3–4 combos: `min_samples_leaf`, `max_leaf_nodes`, `learning_rate`) | to do |
| Fairness discussion (`race` missing: recall 7%; age 90–100: 15%) and whether `race` stays as a feature | to do |
| Final test evaluation T3.6 (once) + Model Card filled with test numbers | to do |
| GitHub Actions CI, HF model repo + Space (W9–10) | to do |
| LLM component T5 (W11) with human review | not started |
| Deck, storyline, demo script, rehearsals | to do |

---

## 5. Conflicts and gaps vs the classes (review 11 Nov)

Severity: **High** = visible to graders against an explicit course rule · **Medium** = explicit suggestion not
followed, defensible with a sentence · **Low** = deviation from teaching-repo convention.
*Evidence = what was found in the repo on 11 Nov.*

### 5.1 Conflicts with class rules

| # | Class rule | Project now | Sev. | Suggested fix |
|---|---|---|---|---|
| C1 | **W3:** notebooks committed as `.py` (jupytext), `*.ipynb` in `.gitignore` | 9 `.ipynb` tracked, with outputs (`01_eda` 894 KB, `Projeto_Final` 514 KB, `04` 155 KB); no paired `.py` | High | Pair with `jupytext --set-formats ipynb,py:percent`, commit the `.py`, ignore `*.ipynb`. `Projeto_Final.ipynb` can still go into the Moodle ZIP as a deliverable |
| C2 | **W3 / W1 status report:** AI usage log committed; status report asks about contributions and responsible AI | `docs/status_apoio_ia.md` and `docs/contributions.md` are **git-ignored** | High | Commit a clean `docs/ai_usage.md` (what AI did vs what the team decided) and a contributions summary |
| C3 | **W1:** every deliverable has every student's name **and number** | Filipa's number missing in `EDSB26_G11.csv` (`A_INDICAR`) and README (`_a indicar_`) | High | Fill it in before any submission |
| C4 | **W1/W3:** meaningful commit messages, one change per commit; history shows contributions | Messages like "updates", "decisões 07.11" (twice), "Resumo de 06.11"; 30 of 39 commits from one account; work on a feature branch with uncommitted changes; no PR merges to `main` | Medium | Descriptive messages from now on; merge via reviewed PRs (T0.4); make sure each member's work is visible (commits or contributions doc) |
| C5 | **W6:** ≥ 2 confusion matrices, **train and validation** | Only OOF/validation matrices at k = 10% and 20% (`reports/confusion_k10/k20.csv`) | Medium | Add the in-sample (train) matrix at the same cut — it also makes the overfit (0.379 vs 0.234) visible |
| C6 | **W6:** suggested minimum metrics Accuracy, Precision, Recall, F1 | Log has PR-AUC, ROC-AUC, recall/precision@10/20, log loss, Brier; **no accuracy, no F1** (D22 excludes F1 on purpose) | Medium | Keep D22, but put one line in the deck: why accuracy misleads at 11% positives and why F1 doesn't apply when k is fixed by capacity. Optionally add accuracy/F1 at the capacity cut to the appendix table |
| C7 | **W5:** model package includes a Model Card (intended + **excluded** uses, data, evaluation, limitations) | `models/v1/README.md` is a placeholder, and `models/*` is git-ignored, so the card isn't in GitHub | Medium | Write the card (planned T3.6/T4.1); track it (`!models/v1/README.md` in `.gitignore`, or keep it in `docs/`) and publish with the HF model repo |
| C8 | **W1/W3:** docstring on every function | ~25 public functions/classes lack docstrings: `src/data/quality.py` (13 check functions + `Check`), `src/app.py` (`describe`, `action_table`, `build_app`), `src/models/kumo.py` (5), `src/models/baselines.py` (`fit`, `predict_proba`), `src/models/evaluate.py` (`recall_at_k`, `precision_at_k`), `src/pipeline.py` (`ModelData`), `src/data_preparation.py` (`main`) | Medium | Add one-paragraph docstrings (in, out, purpose) |
| C9 | **W3:** commit `requirements-installed.txt` (pip freeze) next to `requirements.txt` | `requirements.txt` is now pinned (resolved since the 6/11 internal review), but there's no `requirements-installed.txt` | Low | `python -m pip freeze > requirements-installed.txt` in `.venv` and commit |
| C10 | **W3 teaching repo:** raw data stays out of GitHub | `data/raw/diabetic_data.csv` (19 MB) is tracked | Low | Acceptable (under GitHub limits, course-supplied); say so in the README, or ignore it and document where to get it |
| C11 | **W6:** train vs validation gap read as a signal; tune before choosing | HGB train 0.379 vs OOF 0.234, untuned | Medium | Small tuning grid on the same folds before T3.6 (already planned) |

### 5.2 Deviations from the project guide (team kept them on 11 Nov, with justification in `decisions.md`)
Not conflicts with the classes, but explicit guide wording the defence may be asked about:

| Decision | Guide says | Team position | Risk |
|---|---|---|---|
| D7 — exclude the 3 `gender` Unknown/Invalid | "keep the three encounters as unknown; do not infer … or report reliable group metrics" | No inference, no group metrics; 3 rows (all `NO`) removed | **The team's own internal review rated this the hardest to defend** and an easy fix (keep the rows, don't report the group). Changing it requires adding 3 rows to the split without redoing it |
| D3 — `race` harmonised per patient | "preserve source spelling" | Spelling kept; value changed in 404 encounters (0.4%); raw untouched | Low; consider keeping the original column beside the harmonised one |
| D4 — leading zeros restored in ICD-9 | "preserve the raw string before grouping" | Raw file untouched; padding doesn't change groups | Low; same suggestion |

### 5.3 Already aligned with the classes (keep doing)
Test locked until T3.6 (W6) · patient-grouped CV and leak demonstrated with the HF Space recipe (W5/W6) ·
baseline + `number_inpatient` rule compared at the same capacity (W1, W6) · PR-AUC and @k under imbalance (W5) ·
threshold set by capacity, not 0.5 (W6) · experiment log with train/validation rows (W6) · Kumo adoption decided by
evidence with a decision record, pinned commit and costs (W5, mirrors Lab W5_1) · skops with reviewed untrusted
types (W5) · 3-level success criteria and stated limits (W1) · IBCS pages with message titles (W8) ·
`.env` ignored, pinned `requirements.txt`, 61 tests.

---

## 6. Checklist before any modelling or reporting change

1. Is the decision behind it `Aceite`? If not → options + numbers to the team.
2. Feature available at discharge, row-wise, not derived from future encounters (D8–D12)?
3. Same 5 folds (D13), preprocessing fit inside the fold (D16)?
4. Metrics on **eligible** discharges for @k/fairness (D21); calibration on all and eligible (D22)?
5. Run logged in `reports/experiments.csv` with train + validation rows?
6. Test set untouched (until T3.6, once)?
7. Docstring, test, descriptive commit message, `decisions.md` updated, AI support logged?
8. Does `Projeto_Final.ipynb` still run end to end?

---

## 7. To confirm with lecturers / Moodle
- Presentation slots (W13–14) and the status report 3 deadline (W9).
- Whether the live demo must run on HF Spaces or may run locally (W5 said local is an acceptable fallback).
- Content and expectations of W9–W11 (CI, Spaces, LLM/agent component).
