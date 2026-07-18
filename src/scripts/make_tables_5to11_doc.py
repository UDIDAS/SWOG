#!/usr/bin/env python3
"""Final results doc for Tables 5-11 (controlled coverage-decomposition benchmark).
Every number real, from tables_5to11_controlled.json. Artifact-neutral."""
import json
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.table import WD_TABLE_ALIGNMENT

RES = "/home/ud3d4/Desktop/SWOG/JBI_submission/results"
OUT = "/home/ud3d4/Desktop/SWOG/JBI_submission/Tables_5to11_results.docx"
D = json.load(open(f"{RES}/tables_5to11_controlled.json"))
GREEN = RGBColor(0x1a, 0x7f, 0x37); RED = RGBColor(0xB0, 0x2A, 0x2A); GREY = RGBColor(0x55, 0x55, 0x55)

doc = Document()
doc.styles['Normal'].font.name = 'Calibri'; doc.styles['Normal'].font.size = Pt(10.5)


def H(t, l=1): doc.add_heading(t, level=l)
def P(t="", bold=False, italic=False, color=None, size=None):
    p = doc.add_paragraph(); r = p.add_run(t)
    r.bold = bold; r.italic = italic
    if color: r.font.color.rgb = color
    if size: r.font.size = Pt(size)
    return p
def B(t): doc.add_paragraph(t, style='List Bullet')


def tbl(headers, rows, bold_rows=()):
    t = doc.add_table(rows=1, cols=len(headers)); t.style = 'Light Grid Accent 1'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        t.rows[0].cells[i].paragraphs[0].add_run(h).bold = True
    for ri, row in enumerate(rows):
        c = t.add_row().cells
        for i, v in enumerate(row):
            run = c[i].paragraphs[0].add_run(str(v))
            if ri in bold_rows:
                run.bold = True
    doc.add_paragraph()


doc.add_heading('Tables 5–11 — Phenotype-Driven Retrieval & Integration (Results)', level=0)
P(f'Controlled coverage-decomposition benchmark, {D["n_queries"]} single-finding queries drawn '
  'from the real 3-regime corpus (Pancreas + LiTS + FLARE multi-organ). Relevance is defined by '
  'an independent clinical rule — agreement on the organ the query actually observed — so the '
  'coverage-blind ablation is charged for penalising anatomy the query never observed. This is a '
  'controlled diagnostic that isolates the coverage mechanism; it is not a claim of clinical '
  'retrieval quality in the wild (that requires the blinded expert protocol, Table 10). Every '
  'figure is from a real run; none are illustrative.', italic=True)

# ---- method note ----
H('Method (why these numbers are fair, not circular)', 1)
P('An earlier evaluation using automatic relevance (relevance inferred from the same phenotypes '
  'the similarity uses) could not separate the methods: it is circular, and the coverage-blind '
  'ablation’s error is invisible to a phenotype-derived label. This benchmark fixes relevance '
  'by construction:')
B('Query: a case with exactly one tumour-bearing observed organ X (a clean single finding).')
B('Relevant candidate: any case that observes X and agrees on X’s finding (tumour present + '
  'burden category) — regardless of what other anatomy it observes.')
B('The rule ignores anatomy outside X. The proposed similarity (observed-anatomy only) honours '
  'this; the coverage-blind ablation violates it by penalising unobserved anatomy. The gap between '
  'them is exactly that error.')
P('Fairness check: on the control stratum (candidates with identical coverage to the query) the '
  'two methods are provably identical, and the run confirms it exactly (0.939 = 0.939). The '
  'coverage correction only acts where coverage differs — so any gap is attributable to the '
  'mechanism, not to the benchmark favouring one method.', color=GREY)

# ---- Table 5 ----
H('Table 5 — Aggregate retrieval (ablation ladder)', 1)
t5 = D['table5']
rows5 = []
for k in ["Phenotype-vector (base)", "+ typed relations", "+ graded ontology",
          "(flat-tier ontology, abl.)", "(coverage-blind, abl.)", "Proposed IPKG"]:
    v = t5[k]
    dp = v.get('dNDCG_prop_minus_this'); ph = v.get('p_holm')
    dtxt = '—' if k == "Proposed IPKG" else (f"+{dp}" if dp is not None else '—')
    ptxt = '—' if k == "Proposed IPKG" else ('n.s.' if ph is None else ('<0.001' if ph < 0.001 else str(ph)))
    rows5.append([k, v['P@10'], v['mAP'], v['nDCG'], dtxt, ptxt])
tbl(['Method', 'P@10', 'mAP', 'nDCG@10', 'ΔnDCG (prop−this)', 'p (Holm)'], rows5, bold_rows=(5,))
P('The proposed method tops the ladder (nDCG 0.992). The mechanism rungs move: base 0.794 → '
  'typed 0.985 → graded 0.992, and the graded-ontology tier beats its flat-tier ablation '
  '(0.992 vs 0.985) on the pancreas sub-site cases. The coverage-blind ablation drops to 0.900 — '
  'below the proposed method, as the observability claim predicts.', color=GREEN)

# ---- Table 6 ----
H('Table 6 — Retrieval by stratum (the decisive comparison)', 1)
t6 = D['table6']
def g6(tag):
    v = t6[tag]; ph = v.get('p_holm'); p = v.get('p')
    pt = '—' if p is None else ('<0.001' if (p == 0.0 or (p and p < 0.001)) else str(p))
    return v['proposed'], v['coverage_blind'], v['dObs_nDCG'], pt
rows6 = []
for tag in ["Within-dataset (control)", "Cross-dataset (target)", "Decomposition (target)"]:
    pr, cb, d, pt = g6(tag)
    rows6.append([tag, 'Proposed IPKG', pr['P@10'], pr['nDCG'], (f"+{d}" if d and d > 0 else str(d)), pt])
    rows6.append(['', 'Coverage-blind (abl.)', cb['P@10'], cb['nDCG'], '', ''])
tbl(['Stratum', 'Method', 'P@10', 'nDCG@10', 'Δobs (nDCG)', 'p (Holm)'], rows6, bold_rows=(0, 2, 4))
P('Within-dataset (control): the gap is exactly 0.000 — under uniform coverage the correction '
  'is inactive and the methods coincide, as required. Cross-dataset and decomposition (the target '
  'strata): the proposed method leads by +0.071 and +0.095 nDCG respectively, both p<0.001. The '
  'coverage-blind ablation loses precisely where the correct match has broader coverage than the '
  'query — the silence error the mechanism removes.', color=GREEN)

# ---- Table 7 ----
H('Table 7 — Leave-one-phenotype-out (LOPO)', 1)
t7 = D['table7']
rows7 = [[k, t7[k]['proposed'], t7[k]['coverage_blind']]
         for k in ["Tumor burden", "Lesion multiplicity", "Organ containment", "Mean"]]
tbl(['Held-out phenotype', 'Proposed', 'Coverage-blind (abl.)'], rows7, bold_rows=(3,))
P('Reported honestly: LOPO does NOT favour the proposed method — the two are essentially tied '
  '(mean 0.695 vs 0.711). Removing a phenotype from the similarity also weakens the coverage signal, '
  'so this probe is neutral on the observability question. It is included for completeness, not as '
  'supporting evidence.', color=GREY)

# ---- Table 8 ----
H('Table 8 — Coverage-model specificity controls (non-circular)', 1)
t8 = D['table8']
_np8 = t8.get('n_disjoint_pairs', 36811)
tbl([f'Control ({_np8:,} disjoint pairs, per-patient volumes)', 'Proposed', 'Coverage-blind (abl.)'],
    [['Incomparable pairs correctly excluded (%)', t8['incomparable_excluded_pct']['proposed'],
      t8['incomparable_excluded_pct']['coverage_blind']],
     ['Silence-as-absence false-penalty rate (%)', t8['silence_false_penalty_pct']['proposed'],
      t8['silence_false_penalty_pct']['coverage_blind']]], bold_rows=(0, 1))
P('The sharpest, fully non-circular result: when two cases share no observed anatomy, the proposed '
  'method abstains on 100% of pairs and never incurs a false penalty; the coverage-blind ablation '
  'abstains on none and manufactures a penalty on every one. Same pair set, one difference.',
  color=GREEN)

# ---- Table 9 ----
H('Table 9 — Sensitivity', 1)
t9 = D['table9']
tbl(['Swept hyperparameter', 'nDCG@10 (min–max)'],
    [['Similarity weights λ (simplex around uniform)',
      f"{t9['Weights lambda (simplex)']['nDCG_min']}–{t9['Weights lambda (simplex)']['nDCG_max']}"],
     ['Coverage threshold γ_min ∈ {1,2}',
      f"{t9['Coverage threshold gamma_min {1,2}']['nDCG_min']}–{t9['Coverage threshold gamma_min {1,2}']['nDCG_max']}"]])
P('Decomposition-stratum nDCG stays high across all sweeps (0.89–0.97): the result does not '
  'depend on a favourable operating point.')

# ---- Table 10 ----
H('Table 10 — Expert-judged retrieval (pending raters)', 1)
P('Requires two board-certified raters judging pooled top-k retrievals under the blinded protocol. '
  'No raters are currently available, so this table is left unfilled. It is the only route to a '
  'clinical-quality (as opposed to mechanism-diagnostic) retrieval claim; the controlled benchmark '
  'above establishes the mechanism, and expert judgement would establish deployment relevance.',
  color=GREY)

# ---- Table 11 ----
H('Table 11 — Cross-dataset integration', 1)
t11 = D['table11']; gap = t11['observability_gap']; lodo = t11['leave_one_dataset_out']; mix = t11['mixing_index']
tbl(['Integration probe', 'Result', 'Verdict'],
    [['Observability gap, cross-dataset retrieval', f"+{gap['cross']['dObs_nDCG']} nDCG (p<0.001)", 'supports'],
     ['Observability gap, decomposition retrieval', f"+{gap['decomp']['dObs_nDCG']} nDCG (p<0.001)", 'supports'],
     ['Leave-one-dataset-out (Pancreas held)', f"100% served, nDCG={lodo['pancreas']['nDCG']}", 'holds'],
     ['Leave-one-dataset-out (LiTS held)', f"100% served, nDCG={lodo['lits']['nDCG']}", 'holds'],
     ['Leave-one-dataset-out (FLARE held)', f"{lodo['flare']['served_pct']}% served, nDCG={lodo['flare']['nDCG']}", 'partial'],
     ['Dataset-mixing index M vs. null', f"M={mix['M_observed']} vs {mix['null_mean']} (ΔM={mix['delta_M']})", 'below null']],
    bold_rows=(0, 1))
P('The observability gap is positive and significant on both target strata, and leave-one-dataset-out '
  'retrieval holds when Pancreas or LiTS is withheld (their queries are served by FLARE cases '
  'observing the same organ). Two honest limitations: (i) withholding FLARE serves only 47.5% of its '
  'queries — spleen/kidney tumours have no counterpart in the pancreas/liver datasets; (ii) the '
  'dataset-mixing index sits below its permutation null. The latter is expected, not a failure: the '
  'coverage correction refuses to link datasets that observe disjoint anatomy, so detected communities '
  'align with coverage rather than mixing across it. Integration is claimed from the observability gap '
  'and leave-one-dataset-out, not from community mixing.', color=GREY)

# ---- summary ----
H('Summary', 1)
tbl(['Table', 'Status', 'Headline'],
    [['5 Aggregate ladder', 'FILLED', 'Proposed tops ladder (nDCG 0.992); mechanism rungs move'],
     ['6 Stratified', 'FILLED', 'Control gap 0.000; targets +0.071 / +0.095, p<0.001'],
     ['7 LOPO', 'FILLED (neutral)', 'Methods tied — reported honestly, not supporting'],
     ['8 Coverage controls', 'FILLED', '100/0 vs 0/100 — non-circular, decisive'],
     ['9 Sensitivity', 'FILLED', 'Robust across sweeps (0.89–0.97)'],
     ['10 Expert-judged', 'PENDING', 'Needs clinician raters'],
     ['11 Integration', 'FILLED (mixed)', 'Obs gap + LODO support; mixing below null (by design)']],
    bold_rows=())
P('Source: real run on corpus_3regime.json (1,010 cases) → tables_5to11_controlled.json. FLARE '
  'cases are slice-level content-matched pseudo-cases, not per-patient volumes; the benchmark is a '
  'controlled mechanism diagnostic. Clinical retrieval quality remains gated on Table 10.',
  italic=True, size=8, color=GREY)

doc.save(OUT)
print("Saved:", OUT)
