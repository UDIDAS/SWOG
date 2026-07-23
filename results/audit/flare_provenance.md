# FLARE data provenance

Answers the final-checkup doc §1 (A: 100 FLARE22 organ cases; B: class-14 tumor).
**Both provenance chains are established** — no FLARE claim needs to be removed.
The organ masks are organizer ground truth (not pseudo-labels); the tumor masks
are real class-14 ground truth (slice-level, with a documented patient-identity
caveat). The one constraint is redistribution (see the end).

## A. The 100 FLARE22 organ cases

| Question (doc §1.A) | Answer |
|---|---|
| Complete list of 100 case IDs | 50 `FLARE22_Tr_0001…0050` + 50 `FLARETs_0001…0050`. Full list: `benchmark/case_scopes.csv` (rows with `source_id = flare22`) and `data/cases.csv` (`dataset = FLARE`). |
| Which are training vs tuning/other | 50 `FLARE22_Tr_*` = FLARE **train_gt_label** (labelsTr). 50 `FLARETs_*` = FLARE **public validation** set. Benchmark splits (by case-ID hash): 60 train / 14 val / 26 test. |
| Where the reference masks came from | MICCAI **FLARE 2024–2025 Task2 (LaptopSeg)**, HF `FLARE-MedFM/FLARE-Task2-LaptopSeg` (Codabench comp. 2320). Loaded from `train_gt_label/labelsTr/` + `validation/Validation-Public-Labels/` — see `oakg/build_benchmark.py:68-69`. |
| Organizer annotations, predictions, or pseudo-labels? | **Organizer ground truth** for all 100 reference masks. We used *only* the GT directories; the dataset's 2 000 `train_pseudo_label` scans (pseudo-labels from the FLARE22 winning solution) were **deliberately excluded**. Separately, 20 of the 100 cases also have **SAM3 model predictions** (`sam3_delivery/{organ}/*_pred`) used only for the **pred track**. |
| Redistributable in the anonymous supplement? | **No — do not ship raw masks/images.** License is **CC BY-NC 4.0** (dataset README on source) and access is gated on HF; attribution would also break double-blind. We redistribute only our own derived task definitions (`benchmark/`: queries, relevance, scopes, schema), not FLARE pixels. |

Label map (verified against the per-organ organizer GT): liver 1, right-kidney 2,
spleen 3, pancreas 4, left-kidney 13 (`oakg/build_benchmark.py:37`). FLARE22 is
organs-only — **no tumor** in this set. (This is a label-integer mapping check, not
an OAKG result — OAKG performs no segmentation.)

Counts verified on disk: `train_gt_label/labelsTr` = 50, `validation/Validation-Public-Labels`
= 50, `sam3_delivery/{organ}` = 20 pred files each.

## B. The FLARE class-14 tumor stratum

| Question (doc §1.B) | Answer |
|---|---|
| Exact dataset/release with class 14 | **FLARE 2023 pan-cancer** per-class arrays (`/scratch/.../FLARE/class_{1..14}.npy`); class 14 = tumor. A *different* FLARE edition than the organ data above. |
| Ground truth, prediction, or pseudo-label? | **Real ground truth** (`class_14_labels.npy`). Chosen over predicted tumor deliberately (FLARE22 patients have no tumor GT → predictions would be unmeasurable/noisy). |
| Case and slice IDs | `case_id = FLARE-<md5-slice-hash>` (e.g. `FLARE-f622920ac9`); 23 061 slice-cases in `flare_multiorgan_cases.json`. |
| How the 364 slice queries were generated | `oakg.build_tumor_stratum` samples ≤700 tumor + ≤700 non-tumor slices (seed 2027) and runs the standard query/relevance builder → 364 evaluable queries on the cross-organ-tumor phenotype. |
| Why patient identity is unavailable | class-14 is distributed as **per-class 2-D slice stacks** (`class_14_labels.npy` shape 9911×256×256 — no patient/volume axis). `build_flare_multiorgan.py` re-groups slices across class files by CT-content hash, but there is **no slice→patient index**, so cases are slices, not patients. |
| Script + source output for the +0.043 result | `oakg.build_tumor_stratum` (reassembly upstream: `build_flare_multiorgan.py` → `flare_multiorgan_cases.json`) → `results/strata/flare_tumor_realgt.csv`: OAKG − zero-imp = **+0.043 [0.019, 0.066]**. |

Handling (already in the paper): **slice-level, kept separate from the patient-level
corpus, thresholds frozen, only PAIRED deltas reported** (adjacent slices of a case
are correlated, so absolute nDCG is optimistic but paired comparisons are valid).

## Confronting points to state plainly

1. **Two different FLARE data sources.** Organ GT = **FLARE22** annotations (the
   50 GT training cases and the 13-organ label scheme come from FLARE 2022),
   accessed via the **FLARE 2024–2025 Task2 (LaptopSeg)** release; tumor GT =
   **FLARE 2023** pan-cancer (class 14). The "2024–2025" is the download release
   for the FLARE22 organ labels, not a separate dataset. The paper must not imply
   a single unified FLARE source.
2. **Redistribution is constrained** (CC BY-NC 4.0 + gated + anonymity). Raw FLARE
   masks/images are **not** in the supplement; only our derived definitions are.
3. **Tumor is slice-level** (patient identity genuinely lost) — a real limitation,
   already caveated; the stratum is supplementary and reported as paired deltas.
4. Everything else checks out: organ masks are organizer GT (not pseudo-labels),
   the 50 validation cases use the official `Validation-Public-Labels`, and the
   label map is Dice-1.000 verified.
