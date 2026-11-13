# LEDGER

Append-only. One entry per experiment or data operation: date, commit, seed, config, result,
failures included. Never edit a past entry; add a correction entry instead.

## 2026-11-12 — Phase 1: data, harness, seal

- **Commit:** first commit of the repository (see `git log`).
- **Seed and salt:** split salt `system-three-v0.1-2026-11-12`; harness tests seed 7, 11, 3, 5; bootstrap seed 20261112.
- **Fetch:** `scripts/01_fetch_data.py`, all 15 sources. Upworthy came from the OSF API (`api.osf.io/v2/nodes/jd64p`), IST from the DataShare REST API, and SWE-bench from a sparse clone of `evaluation/verified/*/results`. Oregon, Camden, and the Michigan voter files were copied from the user's own downloads after click-through or Dataverse download.
- **Failures:**
  - The ISPS page `isps.yale.edu/research/data/d001` returns "Page not found". The voter data came from Yale Dataverse instead, at doi:10.60600/YU/CGMWNW, under CC0.
  - The CRASH-2 public file (`hbiostat.org/data/repo/crash2.rda`, 20,207 rows, 44 columns) has no treatment allocation column. It was not loaded.
- **RouterBench:** the source ships pickle files only. They were converted to parquet in an isolated Modal CPU container (`scripts/modal_convert_routerbench.py`, app `system-three-routerbench-convert`, volume `system-three-data`). Nothing was unpickled locally.
- **Harness on known truth:** `tests/test_harness_truth.py`, 200 replications, n = 2,000, q = 0.2. The 95% cluster-bootstrap interval for policy value per 1,000 actions covered the in-sample exact value within the pre-stated band (85% to 99%) under propensity 0.5, propensity 0.2, and household clusters of 4. The AIPW and IPW means agreed, and AIPW had the smaller standard deviation. Benefit ranking beat risk ranking and random on the fixture where risk and benefit differ.
- **Reproductions before the seal:** `scripts/03_validate_and_seal.py`.

  | Dataset | Reproduced | Published |
  |---|---|---|
  | Camden | 0.006443 | 0.0064 (Table 4) |
  | Michigan, "neighbors" | 0.08131 | 0.08131 (authors' R output) |
  | IST, aspirin | −0.01251 | −0.013 (Lancet 1997, PMID 9174558) |
  | Illinois | 10.83 | 10.8 (QJE replication, Appendix Table A.4a, col 2, weighted by months of coverage) |
  | Hillstrom | 0.07659 | 0.0766 |
  | NSW | 1,794.34 | 1,794 |
  | Criteo, treated share | 0.8500 | 0.85 |

  - Oregon (0.01873) has no verified external target yet.
  - The first Illinois attempt used a strata-weighted difference and gave −25.49. It was replaced by the source specification: no controls, weighted by `covg_0816_0717`.
- **Seal:** test actions and outcomes were written to `../system-three-sealed/` (outside the repository). The SHA-256 hashes are in `results/seal_manifest.json`. No development code reads that folder (`tests/test_seal_guard.py`).
- **Panel:** the six-lens design review ran as workflow `wf_959612c3-83b`. Its outputs are in `notebooks/system-three/panel/` (`02_panel_record.json`, `03_plan_amendment.md`). Of 70 candidate changes, the panel adopted 51, marked 11 to consider, and rejected 8.

## 2026-11-12 — confirmatory-set expansion and first dev run

- **Decisions:** `notebooks/system-three/DECISIONS.md`. The confirmatory set is 11 individually randomized trials with binary outcomes, balanced 5 clinical to 6 behavioral, informational, or financial.
- **GUSTO-I:** loaded from the CRAN `predtools` 0.0.3 file `gusto.rda` (GPL; data source listed as "Internet"). The comparison is accelerated tPA against both streptokinase arms; the tPA plus streptokinase combination arm is excluded. P(tPA) = 1/3. `ttr` is excluded because it is measured after allocation. The relative change in 30-day death reproduces at −0.1374 against −0.14 (NEJM 1993, PMID 8204123).
- **San Francisco official email, June 2010:** loaded from Yale Dataverse doi:10.60600/YU/SCUYUK (CC0). Only 47,463 voters were randomized, 1/3 per arm, clustered by email address (`caseid_first`); 47,309 merged to outcomes. The registrar-email ITT is +0.0054. The published value is not yet verified.
- **StepUp:** downloaded through the OSF view-only link printed in the Nature 2021 data statement. The contrast is all 53 programs pooled against Placebo Control; no arm was selected on outcomes. The outcome is any gym visit in weeks 1 to 4. The propensity is the within-cohort treated share. Participants: 61,293, matching the paper.
- **Failure:** Dryad's dataset and file download APIs return 401 without an account token (ELAIA-1, ELAIA-2, synbiotic sepsis). These need a browser download.
- **First dev run:** `scripts/05_dev_baselines.py` covered Camden, IST, Hillstrom, and Oregon. It fit on context and scored dev only, with q = 0.2. Every paired contrast CI includes 0. The scale model is at or above the benefit learner in IST and Hillstrom. These are smoke-test results only; nothing was tuned on them.

## 2026-11-12 — last four trials, recalibrated scale model, scorer, forecasts

- **Manual downloads by Sanjay:** ELAIA-1, ELAIA-2, and synbiotic sepsis came from Dryad (CC0) through a browser. Karlan–List came from IPA Dataverse after he completed the guestbook.
- **Reproductions:**

  | Trial | Reproduced | Published |
  |---|---|---|
  | ELAIA-1 | +0.0041 | +0.005 (BMJ 2021) |
  | ELAIA-2 | −0.0217 | −0.022 (Nat Commun 2023) |
  | Synbiotic | −40.3% relative | −40% (Nature 2017) |

  Karlan–List: 2.20% against 1.79%. The published value is not yet verified.
- **Synbiotic duplicates:** two Study_IDs appear twice with identical records. All 4,556 rows are kept, matching the published N, with Study_ID as the cluster.
- **Fix to the scale model:** the first version fitted only a treatment log-odds shift on an out-of-fold risk offset. The shift absorbed the risk model's miscalibration: ELAIA-1 implied a 5.4-point harm against a raw +0.4. It now fits logit P = α + γ·logit p0 + β·a (`rankers.scale_fit`). The predicted average effects now track the raw context effects.
- **Forecasts:** `scripts/06_scale_forecast.py` computed scale-predicted gaps on context data only. They are recorded in pre-registration draft v0 and `results/scale_forecast.json`.
- **Forecast holdouts:** the seeded lottery (20261112) stratified by intervention class drew synbiotic, sf_email, and oregon.
- **Scorer:** `scripts/score_sealed.py` is the frozen custodian scorer. It takes the sealed path as an argument. Its synthetic recovery test is in `tests/test_seal_guard.py`.
- **Red-team:** the Fable xhigh adversarial review of the pre-registration ran as workflow `wf_cb641732-837`. Reviewers were not shown the dev results.

## 2026-11-12 — red-team fixes, step 1 (code and moderator table)

- **Red-team:** workflow `wf_cb641732-837` ran the Fable six-lens review of pre-registration v0. It adopted 58 fixes, marked 17 to consider, and rejected 8. The memo is at `notebooks/system-three/preregistration/redteam_v0_revision_memo.md`.
- **Code changes:**
  - Rankers: `early_stopping` is off; deployed scores use fold-averaged p0 and p1; the scale fit is inverse-design-weighted with sandwich SE and step-halving Newton; out-of-fold AUC is reported; cross-fitting is cluster-level (`rankers.folds_for`).
  - The frozen configuration moved to `src/system_three/analysis.py`.
  - Karlan–List uses the design probability 2/3.
  - The seal now carries cluster_id and stratum, and a sealed per-trial `meta.json` hashed in `results/seal_manifest.json`.
  - The power column now uses the pre-specified contrast arm and the two-arm test count.
  - The scorer reads clusters and metadata only from the sealed side, checks predictions against the manifest, enforces a two-sided row match, and records the git commit and the `uv.lock` hash.
- **Pilot failures caught before posting:**
  1. Resampling with replacement duplicated clusters across KFold folds, which biased cross-fitted p0 and β. The bootstrap β interval for ELAIA-2 excluded its own plug-in. Fixed with cluster-level folds and cluster half-sampling, where SE = SD/√2.
  2. Weighted Newton diverged on near-separated half-samples (β about ±1e11). Fixed with step halving.
- **Moderator table:** `scripts/06_scale_forecast.py` ran on Modal (200 half-samples per trial, 0 failed calls). The output is `results/scale_forecast.json`.
- **Sign rule (F06d):** ELAIA-1 and Oregon have adverse context β and are excluded from H1. The confirmatory set has 9 trials.
- **Power (`scripts/07_meta_regression.py simulate`, 3,000 simulations, conservative design SEs):**
  - Weak-form size at slope 0 is 0.027 to 0.053.
  - Power at slope 1 is 0.71 at τ = 0 and 0.48 at τ = 20.
  - Slope information: Hillstrom 0.44, Michigan 0.37, Karlan–List 0.13, GUSTO 0.05.

## 2026-11-12 — pre-registration v1 posted to OSF

- **Verification:** a Fable pre-post review found 9 blocking items. All were fixed before posting:
  - Custodian wording: no one is named, and Sanjay holds the seal until two academic custodians with no Waymark equity take it over.
  - Predictive variance: the text now states it, matching the code.
  - Size range: now taken from the saved simulation.
  - §4a: now discloses the 15-trial dev run.
  - REML check: now backed by `tests/test_meta_regression.py`.
  - Design-weight size: 0.052 to 0.066, saved in `h1_power_simulation.json`.
  - Code coverage: a statement was added. H2, the sign-flip fit, RATE, and the variance floor are now in the hashed code. LOEO and the split-context moderator are to be hashed before unsealing.
  - Table D: labels and formats fixed.
  - Header: the open-items list was removed.
- **Also fixed:** the ELAIA-1 target is now 0.004 (BMJ 2021 abstract: 21.3% vs 20.9%), and the DECISIONS.md reconciliation entry was added.
- **Holdouts (seed 20261113, one per moderator tertile):** IST, StepUp, and Hillstrom.
- **OSF:**
  - Project rs7pm (private) holds the pre-registration, the red-team memo, the manifests, the forecasts, the simulation, the holdouts, the dev results, the dataset table, the code archive, and all 15 prediction parquets.
  - Registration r2vsu (Secondary Data Preregistration template) was registered at 2026-11-12T21:20:07Z, pending admin approval. It becomes public on approval or automatically after 48 hours.
- **Interrupted upload:** the first run lost its connection after 21 of 23 files. `10_osf_post.py` now records the project ID first, skips existing files, and retries. No duplicate project was created.

## 2026-11-12 — frozen secondary analyses and Paper 2 contenders

- **LOEO:** environments are ELAIA-2 (4), synbiotic (2), StepUp (5), and Michigan (5 hashed block groups).
- **Split-context moderators:** stable against the full-context values (Michigan 49.2 vs 49.7; six-city 34.3 vs 34.4; Hillstrom 45.1 vs 56.4; New Haven 25.5 vs 35.4).
- **CausalPFN:** v0.1.5, run on a Modal L4. All 15 trials produced predictions with no missing values. Mean effects are close to the known trial effects (synbiotic −0.037 vs −0.036; Michigan +0.099 vs +0.081). Spearman correlation with the DR-learner ranking ranges from 0.13 to 0.54 (Lenta −0.01).
- **LLM comparator (claude-opus-5-5):**
  - Pilot: 50 of 50 succeeded, $0.10.
  - Full run: 24,786 requests in three batches. All 24,836 requests (pilot plus full run) succeeded with valid estimates and 0 missing answers.
  - Cost is about $90 in total. Per-batch token totals were not saved; the estimate extrapolates from the third batch, $17.59 for 4,786 requests.
  - Requests bypassed the local compression proxy.
- **Scorer:** gained `--allow-subset`. Registration update 1 is drafted (`notebooks/system-three/preregistration/REGISTRATION_UPDATE_1.md`) and will post once r2vsu is approved.

## 2026-11-12 — Deviation 1: pandas 3 text-dtype covariate loss (IST, synbiotic), corrected before unsealing

- **Found:** while checking Do-PFN inputs (IST had 3 features). Under pandas 3, a `dtype == object` check misses `str` columns, so coded covariates coerced to all-missing.
  - IST lost 17 of 20 covariates.
  - Synbiotic lost 5 of 11.
  - A scan of all 19 datasets found no other affected trial.
- **Fix:** `_is_text` in loaders `_base.py`, plus the regression test `tests/test_loaders_text.py`. The seal hashes are unchanged, so the splits and outcomes are identical.
- **Recomputed for IST and synbiotic:** moderators on Modal with pinned uv.lock versions (200 half-samples), dev variances, main predictions, split-context and LOEO predictions, CausalPFN, Do-PFN, and the LLM run (3,363 requests, $5.47; superseded answers kept in `llm/results_superseded.jsonl`).
- **Results:**
  - IST: β changed from −0.055 to −0.081, the moderator from +5.78 to +14.06, and the AUC from 0.662 to 0.773.
  - Synbiotic: unchanged in substance (moderator 0).
  - Power is unchanged: 0.997 at τ = 0, 0.660 at τ = 20, and size 0.024 to 0.029.
- **Verification:** all copied columns in the contender files match the corrected main predictions, and the forecast β equals the deployed β for both trials.
- **Dev read:** the corrected IST dev run produced point estimates (scale-minus-risk +93.8). Only their variance enters the weights, and they are recorded here per §4a.
- **Do-PFN:** all 15 trials ran in an isolated Modal container at commit 90d6743, included at Sanjay's direction despite the missing license. Its levels are inflated (synbiotic CATE −0.11 against a trial effect of −0.036), but only rankings are scored.
- **Registration update 1:** rebuilt with the deviation section.

## 2026-11-12 — Tier A exact-truth predictions (Paper 2)

- **Script:** `scripts/16_tierA_predictions.py`. Only binary-scored evals are used, decided from context: 80 of 86 RouterBench evals. GSM8K and other partial-credit evals are excluded.
- **Features:** a TF-IDF of the prompt reduced to 64 SVD dimensions, plus eval indicators. SWE-bench uses repository indicators only.
- **RouterBench forecasts at 20%:**
  - Mistral-7B to GPT-4: +29.4
  - GPT-3.5 to GPT-4: +1.9
  - Llama-70B to Claude 2: +104.6
  - Mixtral to GPT-4: +4.0
- **SWE-bench:** AUC of p̂0 is 0.43 to 0.46, so it is labeled "no risk signal".
- **Registration update 1:** an addendum adds this layer. It will be re-uploaded to the private OSF `update_1` folder.

## 2026-11-12 — Rose v0, contract scorer, custodian package

- **Rose v0** (`src/system_three/rose.py`; Modal CPU, 30 bags).
  - Conformal factor c is 4.6 to 5.0 on the RouterBench dev splits, so the bag spread understates uncertainty for individual effects.
  - Abstention is 1.2% to 2.1%.
  - All 17 datasets complete, with no missing values.
- **Contract scorer:** `scripts/19_score_contract.py`, with a synthetic coverage test.
- **Custodian package:** `~/Downloads/system-three-custodian-package.zip` (147 MB). It contains the sealed folder, all prediction sets with flat scorer manifests that match the posted hashes, code at 24e0426, and the README.
  - The dry run used fake sealed data (random outcomes) and scored Michigan in about 20 to 25 minutes.
  - When run from the archive, the scorer records the commit as "unknown"; the README covers this.
- **Registration update 2** is written.
