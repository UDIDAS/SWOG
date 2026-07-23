# Final-checkup checklist — per-bullet response

Maps every item in `UD-FinalCheckup-Guidelines.docx` to its answer or result file.
Status: ✅ done · 🟡 partial · 👤 authoring/compile task (not code).

---

## §1 — FLARE data provenance  ✅ (both chains established; nothing removed)
Full write-up: [flare_provenance.md](flare_provenance.md).

### A. The 100 FLARE22 organ cases
| Sub-bullet | Answer / file | Status |
|---|---|---|
| Complete list of 100 case IDs | 50 `FLARE22_Tr_0001–0050` + 50 `FLARETs_0001–0050` → [`benchmark/case_scopes.csv`](../../benchmark/case_scopes.csv) (`source_id=flare22`) | ✅ |
| Training vs tuning/other | 50 `train_gt_label/labelsTr` + 50 public `validation`; benchmark split 60/14/26 → [flare_provenance.md](flare_provenance.md) | ✅ |
| Where the masks came from | FLARE 2024–2025 Task2 LaptopSeg; `train_gt_label/labelsTr` + `validation/Validation-Public-Labels` ([`oakg/build_benchmark.py`](../../oakg/build_benchmark.py) L68–69) | ✅ |
| Organizer / prediction / pseudo-label | **Organizer GT** (the 2 000 pseudo-labels were excluded); 20/100 also have SAM3 preds (pred track) | ✅ |
| Redistributable in the supplement? | **No** — CC BY-NC 4.0 + gated; only derived definitions ship | ✅ |

### B. The FLARE class-14 tumor analysis
| Sub-bullet | Answer / file | Status |
|---|---|---|
| Exact dataset/release with class 14 | FLARE 2023 pan-cancer, class 14 → [flare_provenance.md](flare_provenance.md) | ✅ |
| GT / prediction / pseudo-label | **Real ground truth** | ✅ |
| Case and slice IDs | `FLARE-<slice-hash>` (upstream `flare_multiorgan_cases.json`); derived stratum → [`results/strata/flare_tumor_realgt.csv`](../strata/flare_tumor_realgt.csv) | ✅ |
| How the 364 slice queries were generated | `oakg.build_tumor_stratum` (sampling + query builder) → [flare_provenance.md](flare_provenance.md) | ✅ |
| Why patient identity is unavailable | per-class 2-D slice stacks, no slice→patient index | ✅ |
| Script + source output for +0.043 | `oakg.build_tumor_stratum` → [`flare_tumor_realgt.csv`](../strata/flare_tumor_realgt.csv) | ✅ |

---

## §2 — Audit the 0.403 vs 0.407 discrepancy  ✅
| Sub-bullet | Answer / file | Status |
|---|---|---|
| Audit table (11 fields) | [`random_oakg_0403_vs_0407_audit.csv`](random_oakg_0403_vs_0407_audit.csv) + [README](README.md) | ✅ |
| Check causes in the given order | Only #5 (incomparable handling) differs → [README.md](README.md) | ✅ |
| Query IDs identical? | **Yes** (hash `aeb7664de3f74cdb`); 18 affected queries → [`random_oakg_affected_queries.csv`](random_oakg_affected_queries.csv) | ✅ |
| Regenerate through one consistent pipeline | Done — main pipeline now bottom-ranks ([`oakg/benchmark.py`](../../oakg/benchmark.py)); one value **0.402694** everywhere; paper-text deltas → [paper_number_changes.md](paper_number_changes.md) | ✅ |

---

## §3 — Supplement `[[FILL]]` values  ✅ (values supplied) · 👤 (paste into the doc)
All values: [supplement_registry.md](supplement_registry.md) (`make registry`).
| `[[FILL]]` item | Where | Status |
|---|---|---|
| 17-variable phenotype/observability registry | [supplement_registry.md](supplement_registry.md) §1 | ✅ |
| Support sets for every phenotype | [`benchmark/anatomy.json`](../../benchmark/anatomy.json) / [`phenotype_schema.json`](../../benchmark/phenotype_schema.json) | ✅ |
| Component definitions + weights | §2 (equal-weight group-mean, no learned weights) | ✅ |
| Normalization ranges | §1 (per-feature range) | ✅ |
| Lesion thresholds | §3 (10 voxels; bbox ±2) | ✅ |
| Ranking params η, B, bin boundaries | §3 (η=γ_min 0.25; B=10000; bins 0.25/0.50/0.75) | ✅ |
| Masking seeds & realizations | §4 + [`benchmark/masking.json`](../../benchmark/masking.json) | ✅ |
| Predicted-mask cohort coverage | §5 (20/100 FLARE, etc.) | ✅ |
| P@10 / R@10 / mAP | §6 + [`publication_ready_retrieval_table.csv`](../tables/publication_ready_retrieval_table.csv) | ✅ |
| Hardware / OS / Python / libs / runtime | §8 | ✅ |
| Ontology concepts & identifiers | [`benchmark/ontology_mappings.json`](../../benchmark/ontology_mappings.json) + [`kg_schema.owl`](../../benchmark/kg_schema.owl) — 13 concepts (SNOMED CT/NCIt) + registry §7 | ✅ |
| Ontology-alignment & validation procedure | curated entity→code mapping (the `ontology_mappings.json` above); registry §7 | ✅ |
| Paste values into the supplement `.docx` | — | 👤 |

---

## §4 — Anonymous code/data package  ✅
| Sub-bullet | Where | Status |
|---|---|---|
| Preprocessing → table-gen code | [`oakg/`](../../oakg/) package | ✅ |
| Exact config files | [`configs/aaai27_paper.yaml`](../../configs/aaai27_paper.yaml) | ✅ |
| Fixed splits, query IDs, masking seeds | [`benchmark/`](../../benchmark/) | ✅ |
| Environment / dependency file | [`requirements.txt`](../../requirements.txt) | ✅ |
| Commands to reproduce each result | [`Makefile`](../../Makefile) + [`README.md`](../../README.md) | ✅ |
| Source CSVs for all tables | [`results/`](..) | ✅ |
| Table → source-file/command mapping | [`MANIFEST.csv`](../../MANIFEST.csv) + [`tools/finalization/configs/paper_outputs.csv`](../../tools/finalization/configs/paper_outputs.csv) | ✅ |
| Step-by-step reproduction README | [`README.md`](../../README.md) + [`tools/finalization/README.md`](../../tools/finalization/README.md) | ✅ |
| Verify config.json exists/works | [validate_checklist.py](../../validate_checklist.py) | ✅ |
| Verify `python -m oakg.pipeline` | validate_checklist (import + `--config`) | ✅ |
| Verify `python -m oakg.strata_results` | validate_checklist | ✅ |
| Verify publication_ready table | validate_checklist | ✅ |
| FLARE label-map verification | label integers checked vs organizer GT; documented in [`build_benchmark.py`](../../oakg/build_benchmark.py) + [`anatomy.json`](../../benchmark/anatomy.json) (a data-prep check, not an OAKG result — no reproducible-script obligation) | ✅ |
| Remove names/paths/history/URLs | scrubbed; verified by [`check_anonymity.py`](../../tools/finalization/scripts/check_anonymity.py) | ✅ |
| Frozen anonymized ZIP | `OAKG_AAAI27_Anonymous_Supplement.zip` (`git archive`, history-free) | ✅ |

---

## §5 — Re-run and document all final checks
| Sub-bullet | Where | Status |
|---|---|---|
| Run reproduction commands | `make reproduce-paper`; pipeline + strata re-run this pass | 🟡 (not a *fresh* env) |
| Verify every main-paper number | [`tests/test_paper_regression.py`](../../tests/test_paper_regression.py) (7/7) + [validate_checklist.py](../../validate_checklist.py) (45/0/1) | ✅ |
| Verify table-selection/aggregation | audit + regression cover it | ✅ |
| Compile supplement with no placeholders | values in [supplement_registry.md](supplement_registry.md) | 👤 |
| Run `validate_checklist.py` | 45 pass / 0 fail / 0 warn | ✅ |
| Update reproducibility checklist | this file | ✅ |
| Send audit report / supplement PDF / checklist PDF / ZIP | audit reports = [`results/audit/`](.); ZIP done; **PDFs = compile step** | 👤 |

---

### What still needs you (not code)
1. Paste [supplement_registry.md](supplement_registry.md) values into the supplement `.docx`.
2. Apply [paper_number_changes.md](paper_number_changes.md) to the manuscript text.
3. Compile the supplement PDF + checklist PDF.
