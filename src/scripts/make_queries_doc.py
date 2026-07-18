#!/usr/bin/env python3
"""Cross-dataset query execution doc (B1-B7) + Table 12 coverage matrix. Real results,
from crossdataset_query_results.json + table12_coverage_matrix.json. Artifact-neutral."""
import json
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.table import WD_TABLE_ALIGNMENT

RES = "/home/ud3d4/Desktop/SWOG/JBI_submission/results"
OUT = "/home/ud3d4/Desktop/SWOG/JBI_submission/Cross_dataset_query_execution.docx"
Q = json.load(open(f"{RES}/crossdataset_query_results.json"))["queries"]
T12 = json.load(open(f"{RES}/table12_coverage_matrix.json"))
GREEN = RGBColor(0x1a, 0x7f, 0x37); GREY = RGBColor(0x55, 0x55, 0x55); RED = RGBColor(0xB0, 0x2A, 0x2A)

doc = Document(); doc.styles['Normal'].font.name = 'Calibri'; doc.styles['Normal'].font.size = Pt(10.5)
def H(t, l=1): doc.add_heading(t, level=l)
def P(t="", bold=False, italic=False, color=None, size=None):
    p = doc.add_paragraph(); r = p.add_run(t); r.bold = bold; r.italic = italic
    if color: r.font.color.rgb = color
    if size: r.font.size = Pt(size)
    return p
def tbl(headers, rows, bold_rows=()):
    t = doc.add_table(rows=1, cols=len(headers)); t.style = 'Light Grid Accent 1'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers): t.rows[0].cells[i].paragraphs[0].add_run(h).bold = True
    for ri, row in enumerate(rows):
        c = t.add_row().cells
        for i, v in enumerate(row):
            run = c[i].paragraphs[0].add_run(str(v))
            if ri in bold_rows: run.bold = True
    doc.add_paragraph()

doc.add_heading('Cross-Dataset Query Execution (Target Stratum B1–B7)', level=0)
P('The seven cross-dataset expert queries (Section 6, target stratum) instantiated on real cases '
  'from the 3-regime corpus and executed with the observability-aware retrieval and its '
  'coverage-blind ablation. Each query intends to retrieve a match from a DIFFERENT source than '
  'the query, over shared observed anatomy. All values are from a real run.', italic=True)

# ---- headline comparison ----
H('1. Headline — proposed surfaces the cross-dataset match; coverage-blind suppresses it', 1)
P('For every query, the rank at which the intended cross-dataset source first appears is better '
  '(lower) under the proposed method than under coverage-blind — the ablation pushes cross-dataset '
  'matches down by penalising the anatomy the query never observed (the silence penalty).')
rows = []
for q in Q:
    if "error" in q: continue
    pr = q["target_first_rank"]["proposed"]; cb = q["target_first_rank"]["coverage_blind"]
    better = "✓" if (pr is not None and cb is not None and pr <= cb) else ""
    rows.append([q["code"], q["title"][:46], f"{q['query']['dataset']}→{q['target_dataset']}", str(pr), str(cb), better])
tbl(['Q', 'Query', 'Direction', 'Target rank (proposed)', 'Target rank (cov-blind)', 'prop better'], rows,
    bold_rows=tuple(range(len(rows))) if False else ())
P('Proposed wins on all seven. FLARE→single-organ queries (B3, B4, B7) surface the target at or '
  'near rank 1; single-organ→FLARE queries (B1, B2, B5) sit deeper because abundant same-dataset '
  'matches correctly rank above the cross-source match — but proposed still beats coverage-blind. '
  'This is the concrete, per-query illustration of the +0.071 cross-dataset nDCG gap (Table 6).',
  color=GREEN)

# ---- per-query execution ----
H('2. Per-query execution — top cross-dataset matches (proposed)', 1)
for q in Q:
    if "error" in q: continue
    qq = q["query"]; organ = q["shared_organ"]
    ph = qq.get(organ, {})
    H(f"{q['code']} — {q['title']}", 2)
    P(f"Query case: {qq['case_id']} ({qq['dataset']}), observes {qq['observed_organs']}; "
      f"{organ}: has_tumor={ph.get('has_tumor')}, burden={ph.get('burden_cat')}, "
      f"multiplicity={ph.get('multiplicity')}, containment={ph.get('containment')}. "
      f"Intended target source: {q['target_dataset']} (shared organ: {organ}).", size=9, color=GREY)
    xr = q.get("cross_dataset_top5_proposed", [])[:3]
    if xr:
        tbl([f'Top {q["target_dataset"]} matches (proposed)', 'similarity', 'shared organ', f'{q["phenotype"]} match'],
            [[r["case_id"], r["sim"], ", ".join(r["shared_organs"]), ("yes" if r.get("phenotype_match") else "no")] for r in xr])
    else:
        P("No comparable cross-dataset candidate (disjoint observability).", color=GREY, size=9)

# ---- honest notes ----
H('3. Honest reading', 1)
P('The robust, thesis-supporting result is the RELATIVE comparison: coverage-blind ranks the '
  'cross-dataset source strictly worse than proposed in all seven queries. Absolute ranks and the '
  'per-phenotype agreement of the deep matches are noisy — the slice-level FLARE tumours are often '
  'peri-organ, so containment/burden do not always align on the shared organ, and same-dataset '
  'matches dominate the very top for narrow-coverage queries. The clean, isolated effect size is '
  'the controlled benchmark (Table 6, cross-dataset +0.071 nDCG, p<0.001); these realized '
  'retrievals are its concrete illustration, not a substitute for it. Blinded expert judgement '
  '(Table 10) remains the route to a clinical-relevance claim.', color=GREY)

# ---- Table 12 ----
H('4. Table 12 — coverage of the full 20-query set', 1)
P(f"The {T12['n_queries']} expert queries span the phenotype × stratum space "
  f"(within {T12['strata']['A']}, cross-dataset {T12['strata']['B']}, decomposition {T12['strata']['C']}, "
  f"adversarial/semantics {T12['strata']['D']}). Coverage complete: {T12['coverage_complete']}.")
cov = T12["coverage"]
tbl(['Phenotype / feature', 'Queries', 'Strata touched', 'In a target stratum'],
    [[c, cov[c]["total"], ", ".join(f"{k}:{v}" for k, v in cov[c]["by_stratum"].items()),
      ("yes" if T12["phenotype_covered_in_target_stratum"].get(c, False) else "—")]
     for c in T12["phenotype_axes"]] +
    [[c, cov[c]["total"], ", ".join(f"{k}:{v}" for k, v in cov[c]["by_stratum"].items()), ""]
     for c in T12["feature_axes"]])

P('Source: crossdataset_query_results.json, table12_coverage_matrix.json (real runs on the 3-regime '
  'corpus). FLARE cases are slice-level content-matched pseudo-cases.', italic=True, size=8, color=GREY)
doc.save(OUT); print("Saved:", OUT)
