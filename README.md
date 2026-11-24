# System Three

Code for the paper "When does ranking people by risk miss the people who benefit? A pre-registered test in 28 randomized trials" (Basu, Huynh, Luo, Rajpurkar; under review, 2026). The pre-registrations are OSF registration r2vsu (wave 1), OSF project rs7pm (wave 2, prediction hashes, and scoring logs), and Zenodo doi:10.5281/zenodo.24487742 (wave 3 health-care uptake, clinical trials, continuous/count outcomes, and cluster-respecting causal forest), and the software archive is deposited on Zenodo (concept DOI https://doi.org/10.5281/zenodo.24480285). The paper's analyses are `scripts/42_final_analyses.py` (waves 1–2 registered results), `43_final_figures.py`, `44_final_tables.py`, `45_rose_interval_detail.py`, `46_response_ranking_predictions.py`, `49_ben_exploratory_analyses.py`, `50_ben_additional_exploratory.py` (waves 1–2 post-unsealing exploratory analyses), `51_wave3_pipeline.py` (pre-registered wave 3 health-care uptake and clinical trials), `52_continuous_count_outcomes.py` (pre-registered 7-trial continuous/count evaluation), `53_cluster_causal_forest.py`, and `54_score_cluster_cf_and_pool28.py` (cluster-respecting honest causal forest and 28-trial pools); `verify_canonical.py` re-derives every number in the manuscript from the canonical results files in `results/`. The calibrated ensemble in the paper is called Rose here.


System Three is a typed contract, a benchmark, and (later) an open model for one question: what changes if we act? A System Three model takes a state, a menu of actions that includes a reference action, a named outcome, and logged context. It returns, for each action, the expected change against the reference, a 90% interval, the probability of harm, a support score, and an abstain flag. The benchmark scores rankings of cases on public randomized experiments and on agent-routing data where both potential outcomes are observed.

## What this is

- `contract/openapi.yaml` holds contract v0.1. It is generated from `src/system_three/contract.py`.
- `src/system_three/loaders/` holds one loader per public dataset, which maps each dataset into a common table.
- `src/system_three/estimators/` and `src/system_three/metrics.py` implement the harness: IPW and AIPW unit scores, a cluster bootstrap, policy value at a fixed budget, RATE (AUTOC and Qini), calibration, coverage, abstention, and cost.
- `scripts/` holds the numbered stages: fetch, validate and seal, and the dataset table.

## Reproducibility and data

The repository holds code only. Data live in `data/` (gitignored), and `DATA.md` lists every source with its license, SHA-256, and reproduction status. Test-split actions and outcomes are sealed outside the repository; `results/seal_manifest.json` holds their hashes. The study team ran the frozen scorer (`scripts/score_sealed.py`) on prediction files whose SHA-256 hashes were posted before scoring, and every scoring run is public on OSF (project rs7pm).

```
make phase1      # fetch, validate against published values, split, seal, write DATA.md
make test        # contract, seal guard, and known-truth harness tests
```

## License

MIT for the code. Each dataset keeps its own terms (see `DATA.md`). Criteo and the Oregon files are evaluation-only.
