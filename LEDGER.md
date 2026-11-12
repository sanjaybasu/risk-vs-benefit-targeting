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
