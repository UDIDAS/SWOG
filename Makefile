# OAKG AAAI 2027 — reproduce / freeze / verify targets.
# Override the interpreter with:  make PY=python3 <target>
PY ?= python
PAPER_CFG = configs/aaai27_paper.yaml

.PHONY: help smoke-test build-data reproduce-paper freeze registry validate check-anon test clean

help:
	@echo "OAKG make targets:"
	@echo "  make smoke-test       # synthetic demo run (NOT paper results)"
	@echo "  make build-data       # derive the benchmark CSVs from masks -> data/"
	@echo "  make reproduce-paper  # full paper run (assumes data/ is built)"
	@echo "  make freeze           # write paper_snapshot.json + MANIFEST.csv"
	@echo "  make check-anon       # scan tree for anonymity leaks"
	@echo "  make test             # regression test vs the frozen expected tables"

# Clearly-labelled smoke test on synthetic data.
smoke-test:
	$(PY) -m oakg.pipeline

build-data:
	$(PY) -m oakg.build_benchmark --out data

# Full paper reproduction. Requires data/ (make build-data) and, for the native
# tables, the external analysis package (see results/native_cross_dataset/README.md).
reproduce-paper:
	$(PY) -m oakg.pipeline --config $(PAPER_CFG)
	$(PY) -m oakg.strata_results --out results/strata
	$(PY) -m oakg.union_ablation --policy lexicographic --n-boot 10000
	$(PY) -m oakg.qualitative --out results/qualitative --contrast
	$(PY) -m oakg.native_export --out results/native_inputs --policy lexicographic
	@echo ">> Native tables: run the external analysis package on results/native_inputs/"
	$(MAKE) freeze

freeze:
	$(PY) tools/finalization/scripts/freeze_paper_snapshot.py --repo-root . --config $(PAPER_CFG) --out paper_snapshot.json
	$(PY) tools/finalization/scripts/build_manifest.py --repo-root . --mapping tools/finalization/configs/paper_outputs.csv --out MANIFEST.csv

# Emit the supplement [[FILL]] registry (phenotypes, params, seeds, hardware, P/R/mAP).
registry:
	$(PY) -m oakg.export_supplement_registry --data data --out results/audit/supplement_registry.md

# Verify every referenced artifact exists, snapshot hashes match, numbers hold.
validate:
	$(PY) validate_checklist.py

check-anon:
	$(PY) tools/finalization/scripts/check_anonymity.py .

test:
	$(PY) -m pytest tests/ -q
