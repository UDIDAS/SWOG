# OAKG — qualitative retrieval examples

Content-based retrieval on the 512-case benchmark (natural coverage): OAKG's
top-5 for each multi-condition query. ✓ = matches **all** conditions, else the
fraction matched. Regenerate / add your own queries:

```bash
python -m oakg.qualitative        # edit results/qualitative/queries.json first
```

Mean conditions-matched@5 = **0.86**.

| # | Information need | Cond. | OAKG top-5 (dataset · match) |
|---|---|---|---|
| Q1 | Pancreatic tumour, contained, in a large pancreas | 3 | pancreas_124 (Panc·✓); pancreas_070 (Panc·✓); pancreas_364 (Panc·✓); pancreas_293 (Panc·✓); pancreas_102 (Panc·✓) |
| Q2 | Liver tumour, multifocal, high tumour burden | 3 | LiTs-100 (LiTS·✓); LiTs-130 (LiTS·✓); LiTs-116 (LiTS·2/3); LiTs-097 (LiTS·✓); LiTs-016 (LiTS·✓) |
| Q3 | Multi-organ: large spleen & liver, sizeable kidneys | 4 | FLARETs_0013 (FLAR·✓); FLARE22_Tr_0044 (FLAR·✓); FLARETs_0048 (FLAR·✓); FLARE22_Tr_0017 (FLAR·✓); FLARE22_Tr_0018 (FLAR·3/4) |
| Q4 | Multi-organ: sizeable pancreas with balanced kidneys & spleen | 4 | FLARETs_0013 (FLAR·✓); FLARE22_Tr_0044 (FLAR·✓); FLARETs_0048 (FLAR·✓); FLARE22_Tr_0017 (FLAR·✓); FLARE22_Tr_0018 (FLAR·3/4) |
| Q5 | Any tumour, multifocal, above-median burden | 3 | pancreas_087 (Panc·2/3); pancreas_075 (Panc·2/3); pancreas_299 (Panc·2/3); pancreas_089 (Panc·2/3); pancreas_333 (Panc·2/3) |
| Q6 | Pancreatic tumour, contained, low burden (early-stage) | 3 | pancreas_110 (Panc·✓); pancreas_049 (Panc·2/3); pancreas_158 (Panc·2/3); pancreas_125 (Panc·2/3); pancreas_131 (Panc·2/3) |
| Q7 | Liver tumour, solitary lesion, above-median liver volume | 3 | LiTs-095 (LiTS·✓); LiTs-127 (LiTS·2/3); LiTs-112 (LiTS·2/3); LiTs-003 (LiTS·2/3); LiTs-020 (LiTS·1/3) |
| Q8 | Multi-organ with a pancreatic tumour present (cross-source) | 3 | pancreas_124 (Panc·✓); pancreas_070 (Panc·✓); pancreas_364 (Panc·✓); pancreas_293 (Panc·✓); pancreas_102 (Panc·✓) |