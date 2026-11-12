UV := $(HOME)/.local/bin/uv
PY := $(UV) run --no-sync python

.PHONY: phase1 fetch seal table test openapi

phase1: fetch seal table

fetch:
	$(PY) scripts/01_fetch_data.py

openapi:
	$(PY) scripts/00_build_openapi.py

seal:
	$(PY) scripts/03_validate_and_seal.py

table:
	$(PY) scripts/04_dataset_table.py

test:
	$(PY) -m pytest -q
