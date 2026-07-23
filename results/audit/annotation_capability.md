# How observability / annotation-capability is implemented

Answers the PI's question: does the code distinguish **anatomical observation** from
**annotation capability** (e.g. FLARE22 observes the pancreas *organ* but not
pancreatic-*tumor* status)?

## What the code actually does (honest answer)
The MMKG schema represents **both** scopes explicitly — they are just projected onto
one runtime mask:

1. **Anatomical observation scope** — which organs a case observes
   (`available_organs`; surfaced in `benchmark/anatomy.json`).
2. **Annotation-capability scope** — which phenotypes a *source* annotates
   (`DatasetSpec.organs` / `DatasetSpec.tumor_labels` in `oakg/phenotypes.py`;
   FLARE has `tumor_labels = {}` → **no tumor-annotation capability**). Now surfaced
   in `benchmark/annotation_scopes.json`.

`organ_phenotypes` emits a phenotype row **only when both hold** — the organ is
present **and** the source annotates that phenotype. So the runtime mask
`M = ~isnan(X)` (`oakg/data.py`) is the **realized conjunction** of the two scopes,
not a substitute for them:

```
observed(case, phenotype)  ⇔  support_organs ⊆ observed_organs
                             AND  phenotype ∈ source.annotated_phenotypes
```

A FLARE case observes the pancreas *organ* but pancreatic-*tumor* is outside its
annotation-capability scope → `pancreas_tumor_present` is **Unknown** and excluded
from comparison. This is the paper's two-part support formalism, faithfully realized.

## Concrete MSD-Pancreas ↔ FLARE22 example (verified)
| | MSD-Pancreas case | FLARE22 case |
|---|---|---|
| Observation units | pancreas anatomy **+ pancreatic-tumor annotation** | pancreas anatomy **only** |
| `pancreas_present` | observed | observed |
| `pancreas_volume_cm3` | observed | observed |
| **`pancreas_tumor_present`** | **observed** (row exists) | **UNOBSERVED** (no row) |
| Predicate "pancreatic tumor present?" | **T / F** (can answer) | **U** (Unknown) |
| In an MSD↔FLARE pair, is the tumor phenotype compared? | **No** — not jointly observed → excluded from the pairwise similarity |

So the *functional* requirement the paper describes **is met**: a FLARE case is never
treated as tumor-negative; its tumor status is Unknown and excluded from comparison.

## Resolution — bridge option 1 and option 2 (they are the same)
The PI's two options were (1) explicit anatomy + annotation-capability **support units**
and (2) feature availability. The MMKG lets us have **both**: option 1 is the *scope*,
option 2 is its *realization* — and they coincide.

**Explicit support units are now a schema artifact** — `benchmark/support_units.json`
gives every phenotype its two-part support set, e.g.:
- `pancreas_volume_cm3` → anatomy `{pancreas}`, annotation `{}`
- `pancreas_tumor_present` → anatomy `{pancreas}`, annotation `{tumor_annotation:pancreas}`
- `has_tumor` → anatomy `{}`, annotation `{tumor_annotation:*}` (any tumor annotation)

with the two available scopes surfaced in `benchmark/anatomy.json` (anatomy) and
`benchmark/annotation_scopes.json` (annotation capability).

**Verified equivalence.** The explicit support-unit **observability scope**
(units ⊆ case's available units) is a **superset of the runtime mask** `M`, matching it
on **8686 / 8704** (case, phenotype) cells. The **18 residual** cells (0.2%) are all
"scope-observable but value-missing" — pure **measurement validity within scope**,
orthogonal to the anatomy/annotation-capability distinction:
- **13 ×** `liver_tumor_containment` (LiTS) — containment undefined when there is no
  tumor to contain (a conditional phenotype);
- **5 ×** `*_volume_cm3` (FLARE kidney/spleen) — organ in scope but empty segmentation
  → volume unmeasurable.

So: **`M = support-unit scope ∩ per-measurement validity`.** The paper's formalism
(support = anatomy ∩ annotation-capability) is exactly the scope; the mask adds only
data-level measurement validity.

**Faithful paper wording:** *"A phenotype's support comprises anatomical observation
units and annotation-capability units; a phenotype is observed for a case iff its
support units are available (organ observed AND source annotates the phenotype) and the
measurement is defined."* No code/scoring/redesign change; no result change.

Reproduce the verification (needs `data/`): rebuild the support-unit scope and compare
to `Corpus.m_ref_full` — 18/8704 residuals, all measurement-missing as above.
