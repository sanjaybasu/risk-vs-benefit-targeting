# System Three

Code for the paper "When does ranking people by risk miss the people who benefit? A pre-registered test in 22 randomized trials" (Basu, Huynh, Luo, Rajpurkar). The pre-registrations are OSF registration r2vsu (first wave) and project rs7pm (second wave, prediction hashes, and scoring logs). The paper's analyses are `scripts/42_final_analyses.py` (results), `43_final_figures.py`, `44_final_tables.py`, `45_rose_interval_detail.py`, and `46_response_ranking_predictions.py`; `verify_canonical.py` re-derives every number in the manuscript from `results/final_results.json`. The calibrated ensemble in the paper is called Rose here.


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
