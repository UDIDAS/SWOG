# Where zero-imputation fails — imputation vs OAKG (hard-distractor)

Query = a broad multi-organ case with a large target organ. Imputation ranks
broad cases high because they share *other* organs — even when their target
organ is small (wrong). OAKG restricts to the shared target evidence and
returns the true matches. ✓ = correct (relevant), ✗ = wrong.

*Relevance is decided on UNROUNDED values; displayed volumes are rounded to 1 decimal, so a value shown equal to the threshold may still be just below it.*

Top-5 relevant rate:  **imputation 0%**  vs  **OAKG 100%**.

### HD1: query `FLARE22_Tr_0006` — large **pancreas** (≥ 78.4 cm³)

| rank | imputation → | | OAKG → | |
|---|---|---|---|---|
| 1 | FLARETs_0003 (5-organ, panc=69.2) | ✗ | pancreas_077 (1-organ, panc=89.5) | ✓ |
| 2 | FLARE22_Tr_0041 (5-organ, panc=69.7) | ✗ | pancreas_048 (1-organ, panc=90.9) | ✓ |
| 3 | FLARE22_Tr_0021 (5-organ, panc=64.8) | ✗ | pancreas_358 (1-organ, panc=89.4) | ✓ |
| 4 | FLARETs_0049 (5-organ, panc=68.8) | ✗ | pancreas_357 (1-organ, panc=89.4) | ✓ |
| 5 | FLARETs_0047 (5-organ, panc=77.6) | ✗ | pancreas_052 (1-organ, panc=91.1) | ✓ |

### HD2: query `FLARE22_Tr_0010` — large **pancreas** (≥ 78.4 cm³)

| rank | imputation → | | OAKG → | |
|---|---|---|---|---|
| 1 | FLARE22_Tr_0028 (5-organ, panc=74.5) | ✗ | pancreas_207 (1-organ, panc=96.2) | ✓ |
| 2 | FLARE22_Tr_0038 (5-organ, panc=48.6) | ✗ | pancreas_145 (1-organ, panc=95.9) | ✓ |
| 3 | FLARE22_Tr_0011 (5-organ, panc=54.9) | ✗ | pancreas_327 (1-organ, panc=95.8) | ✓ |
| 4 | FLARETs_0029 (5-organ, panc=55.8) | ✗ | pancreas_004 (1-organ, panc=98.6) | ✓ |
| 5 | FLARETs_0025 (5-organ, panc=68.1) | ✗ | pancreas_293 (1-organ, panc=98.9) | ✓ |

### HD3: query `FLARE22_Tr_0014` — large **pancreas** (≥ 78.4 cm³)

| rank | imputation → | | OAKG → | |
|---|---|---|---|---|
| 1 | FLARETs_0003 (5-organ, panc=69.2) | ✗ | pancreas_061 (1-organ, panc=147.0) | ✓ |
| 2 | FLARE22_Tr_0035 (5-organ, panc=76.3) | ✗ | pancreas_005 (1-organ, panc=149.7) | ✓ |
| 3 | FLARE22_Tr_0036 (5-organ, panc=59.8) | ✗ | pancreas_217 (1-organ, panc=146.2) | ✓ |
| 4 | FLARETs_0019 (5-organ, panc=62.2) | ✗ | pancreas_096 (1-organ, panc=150.2) | ✓ |
| 5 | FLARETs_0049 (5-organ, panc=68.8) | ✗ | pancreas_099 (1-organ, panc=144.8) | ✓ |

### HD4: query `FLARE22_Tr_0018` — large **pancreas** (≥ 78.4 cm³)

| rank | imputation → | | OAKG → | |
|---|---|---|---|---|
| 1 | FLARETs_0022 (5-organ, panc=50.1) | ✗ | pancreas_194 (1-organ, panc=114.3) | ✓ |
| 2 | FLARE22_Tr_0049 (5-organ, panc=64.0) | ✗ | pancreas_342 (1-organ, panc=114.4) | ✓ |
| 3 | FLARE22_Tr_0029 (5-organ, panc=52.1) | ✗ | pancreas_071 (1-organ, panc=114.1) | ✓ |
| 4 | FLARE22_Tr_0030 (5-organ, panc=66.4) | ✗ | pancreas_367 (1-organ, panc=115.2) | ✓ |
| 5 | FLARE22_Tr_0034 (5-organ, panc=41.5) | ✗ | pancreas_043 (1-organ, panc=113.0) | ✓ |

### HD5: query `FLARE22_Tr_0025` — large **pancreas** (≥ 78.4 cm³)

| rank | imputation → | | OAKG → | |
|---|---|---|---|---|
| 1 | FLARETs_0018 (5-organ, panc=59.4) | ✗ | pancreas_043 (1-organ, panc=113.0) | ✓ |
| 2 | FLARE22_Tr_0049 (5-organ, panc=64.0) | ✗ | pancreas_071 (1-organ, panc=114.1) | ✓ |
| 3 | FLARETs_0017 (5-organ, panc=65.5) | ✗ | pancreas_113 (1-organ, panc=112.3) | ✓ |
| 4 | FLARE22_Tr_0004 (5-organ, panc=78.2) | ✗ | pancreas_194 (1-organ, panc=114.3) | ✓ |
| 5 | FLARETs_0025 (5-organ, panc=68.1) | ✗ | pancreas_342 (1-organ, panc=114.4) | ✓ |

### HD6: query `FLARE22_Tr_0043` — large **pancreas** (≥ 78.4 cm³)

| rank | imputation → | | OAKG → | |
|---|---|---|---|---|
| 1 | FLARETs_0018 (5-organ, panc=59.4) | ✗ | pancreas_051 (1-organ, panc=79.1) | ✓ |
| 2 | FLARE22_Tr_0049 (5-organ, panc=64.0) | ✗ | pancreas_234 (1-organ, panc=80.9) | ✓ |
| 3 | FLARETs_0017 (5-organ, panc=65.5) | ✗ | pancreas_098 (1-organ, panc=78.6) | ✓ |
| 4 | FLARE22_Tr_0004 (5-organ, panc=78.2) | ✗ | pancreas_419 (1-organ, panc=78.5) | ✓ |
| 5 | FLARE22_Tr_0050 (5-organ, panc=76.8) | ✗ | pancreas_386 (1-organ, panc=81.3) | ✓ |
