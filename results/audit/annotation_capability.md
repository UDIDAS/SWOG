# How observability / annotation-capability is implemented

Answers the PI's question: does the code distinguish **anatomical observation** from
**annotation capability** (e.g. FLARE22 observes the pancreas *organ* but not
pancreatic-*tumor* status)?

## What the code actually does (honest answer)
Observation is represented **per (case, feature)** by a feature-availability mask:

```
M[case, feature] = the phenotype value for (case, feature) exists (not NaN)
```
(`oakg/data.py` — `M = ~np.isnan(X)`.)

Annotation capability is therefore operationalized through **feature availability**,
**not** through explicit, named support units (there is no separate
`pancreatic_tumor_annotation` observation unit in the schema; `support_organs` encodes
anatomy only). A FLARE case simply has **no `pancreas_tumor_present` row** (FLARE22
has no tumor annotation), so that feature is unobserved for it — the correct T/F/U
behaviour, achieved via missingness rather than an explicit capability ontology.

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

## The gap and the two resolutions (PI's call)
The gap is **formalization wording, not a functional bug**: the paper frames support
as an explicit anatomy-**plus-annotation-capability** support set, while the code uses
feature availability to the same effect.

- **Preferred (larger):** extend the schema to explicit support units
  (`pancreas_anatomy`, `pancreatic_tumor_annotation`, …) and rerun. **Not done — a
  redesign should be discussed with the PI first.**
- **Lower-risk (recommended):** revise the paper to match the implementation —
  *"Annotation capability is operationalized through dataset-specific feature
  availability; anatomical breadth through organ observation scopes."* The paper
  should not claim a richer executable capability ontology than the code implements.

No result changes either way — the UNOBSERVED behaviour is already correct.
