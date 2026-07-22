# OAKG AAAI 2027 Finalization Tools

Utilities to freeze the paper snapshot, check result consistency, generate a file manifest, scan anonymity, and build an anonymous supplementary ZIP.

## Recommended order

```bash
python scripts/check_result_consistency.py --repo-root /path/to/SWOG --rules configs/consistency_rules.json
python scripts/freeze_paper_snapshot.py --repo-root /path/to/SWOG --config configs/aaai27_paper.yaml --out paper_snapshot.json
python scripts/build_manifest.py --repo-root /path/to/SWOG --mapping configs/paper_outputs.csv --out MANIFEST.csv
python scripts/check_anonymity.py /path/to/SWOG
python scripts/package_openreview_zip.py --repo-root /path/to/SWOG --out OAKG_AAAI27_Anonymous_Supplement.zip
```

Review every warning manually before submission. Adapt CSV column names in the consistency rules when necessary.
