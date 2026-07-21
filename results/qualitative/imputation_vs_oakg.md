# Where zero-imputation fails — imputation vs OAKG (hard-distractor)

Query = a broad multi-organ case with a large target organ. Imputation ranks
broad cases high because they share *other* organs — even when their target
organ is small (wrong). OAKG restricts to the shared target evidence and
returns the true matches. ✓ = correct (relevant), ✗ = wrong.

Top-5 relevant rate:  **imputation 0%**  vs  **OAKG 100%**.

### HD1: query `FLARE22_Tr_0006` — large **pancreas** (≥ 78 cm³)

| rank | imputation → | | OAKG → | |
|---|---|---|---|---|
| 1 | FLARETs_0003 (5-organ, panc=69) | ✗ | pancreas_077 (1-organ, panc=90) | ✓ |
| 2 | FLARE22_Tr_0041 (5-organ, panc=70) | ✗ | pancreas_048 (1-organ, panc=91) | ✓ |
| 3 | FLARE22_Tr_0021 (5-organ, panc=65) | ✗ | pancreas_358 (1-organ, panc=89) | ✓ |
| 4 | FLARETs_0049 (5-organ, panc=69) | ✗ | pancreas_357 (1-organ, panc=89) | ✓ |
| 5 | FLARETs_0047 (5-organ, panc=78) | ✗ | pancreas_052 (1-organ, panc=91) | ✓ |

### HD2: query `FLARE22_Tr_0010` — large **pancreas** (≥ 78 cm³)

| rank | imputation → | | OAKG → | |
|---|---|---|---|---|
| 1 | FLARE22_Tr_0028 (5-organ, panc=74) | ✗ | pancreas_207 (1-organ, panc=96) | ✓ |
| 2 | FLARE22_Tr_0038 (5-organ, panc=49) | ✗ | pancreas_145 (1-organ, panc=96) | ✓ |
| 3 | FLARE22_Tr_0011 (5-organ, panc=55) | ✗ | pancreas_327 (1-organ, panc=96) | ✓ |
| 4 | FLARETs_0029 (5-organ, panc=56) | ✗ | pancreas_004 (1-organ, panc=99) | ✓ |
| 5 | FLARETs_0025 (5-organ, panc=68) | ✗ | pancreas_293 (1-organ, panc=99) | ✓ |

### HD3: query `FLARE22_Tr_0014` — large **pancreas** (≥ 78 cm³)

| rank | imputation → | | OAKG → | |
|---|---|---|---|---|
| 1 | FLARETs_0003 (5-organ, panc=69) | ✗ | pancreas_061 (1-organ, panc=147) | ✓ |
| 2 | FLARE22_Tr_0035 (5-organ, panc=76) | ✗ | pancreas_005 (1-organ, panc=150) | ✓ |
| 3 | FLARE22_Tr_0036 (5-organ, panc=60) | ✗ | pancreas_217 (1-organ, panc=146) | ✓ |
| 4 | FLARETs_0019 (5-organ, panc=62) | ✗ | pancreas_096 (1-organ, panc=150) | ✓ |
| 5 | FLARETs_0049 (5-organ, panc=69) | ✗ | pancreas_099 (1-organ, panc=145) | ✓ |

### HD4: query `FLARE22_Tr_0018` — large **pancreas** (≥ 78 cm³)

| rank | imputation → | | OAKG → | |
|---|---|---|---|---|
| 1 | FLARETs_0022 (5-organ, panc=50) | ✗ | pancreas_194 (1-organ, panc=114) | ✓ |
| 2 | FLARE22_Tr_0049 (5-organ, panc=64) | ✗ | pancreas_342 (1-organ, panc=114) | ✓ |
| 3 | FLARE22_Tr_0029 (5-organ, panc=52) | ✗ | pancreas_071 (1-organ, panc=114) | ✓ |
| 4 | FLARE22_Tr_0030 (5-organ, panc=66) | ✗ | pancreas_367 (1-organ, panc=115) | ✓ |
| 5 | FLARE22_Tr_0034 (5-organ, panc=42) | ✗ | pancreas_043 (1-organ, panc=113) | ✓ |

### HD5: query `FLARE22_Tr_0025` — large **pancreas** (≥ 78 cm³)

| rank | imputation → | | OAKG → | |
|---|---|---|---|---|
| 1 | FLARETs_0018 (5-organ, panc=59) | ✗ | pancreas_043 (1-organ, panc=113) | ✓ |
| 2 | FLARE22_Tr_0049 (5-organ, panc=64) | ✗ | pancreas_071 (1-organ, panc=114) | ✓ |
| 3 | FLARETs_0017 (5-organ, panc=66) | ✗ | pancreas_113 (1-organ, panc=112) | ✓ |
| 4 | FLARE22_Tr_0004 (5-organ, panc=78) | ✗ | pancreas_194 (1-organ, panc=114) | ✓ |
| 5 | FLARETs_0025 (5-organ, panc=68) | ✗ | pancreas_342 (1-organ, panc=114) | ✓ |

### HD6: query `FLARE22_Tr_0043` — large **pancreas** (≥ 78 cm³)

| rank | imputation → | | OAKG → | |
|---|---|---|---|---|
| 1 | FLARETs_0018 (5-organ, panc=59) | ✗ | pancreas_051 (1-organ, panc=79) | ✓ |
| 2 | FLARE22_Tr_0049 (5-organ, panc=64) | ✗ | pancreas_234 (1-organ, panc=81) | ✓ |
| 3 | FLARETs_0017 (5-organ, panc=66) | ✗ | pancreas_098 (1-organ, panc=79) | ✓ |
| 4 | FLARE22_Tr_0004 (5-organ, panc=78) | ✗ | pancreas_419 (1-organ, panc=78) | ✓ |
| 5 | FLARE22_Tr_0050 (5-organ, panc=77) | ✗ | pancreas_386 (1-organ, panc=81) | ✓ |
