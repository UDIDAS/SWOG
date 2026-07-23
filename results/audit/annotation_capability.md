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

## Resolution — no redesign, no weaker claim
The earlier framing ("reword down to feature availability") **undersold it**. The MMKG
already carries both scopes; the only thing missing was that the annotation-capability
half lived inside `DatasetSpec` and was **not exposed** in the public schema (which
showed `support_organs` = anatomy only — why the PI saw "only support_organs").

**Fix (done, additive, no rerun, no result change):** surface the annotation-capability
scope as `benchmark/annotation_scopes.json`. The paper can now keep its **full** two-part
formalism, backed by two visible schema artifacts:

- anatomy scope → `benchmark/anatomy.json`
- annotation-capability scope → `benchmark/annotation_scopes.json`

described as: *observability = anatomical observation scope ∩ annotation-capability
scope, realized as feature availability.* No code/scoring/redesign change; the
UNOBSERVED behaviour was already correct.
