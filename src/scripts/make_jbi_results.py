#!/usr/bin/env python3
"""Build JBI_results.docx — fills tables matching JBI_VKG_2026.pdf structure.
Real/computed tables filled; JBI-specific observability tables kept as
correctly-structured placeholders annotated with the experiment (or raters) needed."""
import os, json
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.table import WD_TABLE_ALIGNMENT

JBI = "/home/ud3d4/Desktop/SWOG/JBI_submission"
RES = f"{JBI}/results"
GREEN = RGBColor(0x1a, 0x7f, 0x37)
AMBER = RGBColor(0xB0, 0x6A, 0x00)
GREY = RGBColor(0x66, 0x66, 0x66)

t2 = json.load(open(f"{RES}/table2_phenotype_extraction.json"))
t3 = json.load(open(f"{RES}/table3_ontology_coverage.json"))

doc = Document()
doc.styles['Normal'].font.name = 'Calibri'; doc.styles['Normal'].font.size = Pt(10.5)

def H(t, l=1): doc.add_heading(t, level=l)
def P(t=""): p = doc.add_paragraph(); p.add_run(t); return p
def status(p, txt, filled):
    r = p.add_run(txt); r.bold = True; r.font.color.rgb = GREEN if filled else AMBER; return r

def table(headers, rows, filled_cols=None, pending_marker="requires experiment"):
    tb = doc.add_table(rows=1, cols=len(headers)); tb.style = 'Light Grid Accent 1'
    tb.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        tb.rows[0].cells[i].paragraphs[0].add_run(h).bold = True
    for row in rows:
        cells = tb.add_row().cells
        for i, v in enumerate(row):
            run = cells[i].paragraphs[0].add_run(str(v))
            if str(v) == pending_marker or str(v) == "TBD":
                run.font.color.rgb = AMBER; run.italic = True
            elif filled_cols and i in filled_cols and i > 0:
                run.font.color.rgb = GREEN
    doc.add_paragraph(); return tb

# ── Title ──
doc.add_heading('JBI / VKG 2026 — Results', level=0)
s = doc.add_paragraph(); r = s.add_run('Filled to the table structure of JBI_VKG_2026.pdf. '
    'Real/computed results in green; tables requiring the observability-IPKG pipeline or clinician '
    'raters are shown with the exact structure and annotated with what is needed (no fabricated numbers).')
r.italic = True; r.font.size = Pt(10.5)
p = P(); p.add_run('Legend: '); status(p, 'FILLED', True); p.add_run(' = measured this study.  ')
status(p, 'NEEDS EXPERIMENT', False); p.add_run(' = pipeline/raters required.')

# ══ Segmentation substrate (§5.1) ══
H('Segmentation substrate (§5.1) — FILLED', 1)
P('Test-set Dice under case-level splits establishing the masks are reliable enough for phenotype '
  'extraction. SAM3 v3 partial-freeze (box-prompted). FLARE is slice-level (no patient IDs).')
table(['Dataset', 'Organ Dice', 'Tumor Dice', 'Split'], [
    ['Pancreas (MSD Task07)', '0.866', '0.894', 'case-level'],
    ['LiTS', '0.998 (from GT)', '0.840', 'case-level'],
    ['FLARE — duodenum / pancreas / tumor', '0.913 / 0.907', '0.883', 'slice-level'],
], filled_cols={1, 2})
q = doc.add_paragraph(); q.add_run('Substrate sentence: “organ masks reach Dice 0.87–0.91 and tumor masks '
    '0.84–0.89 across the three sources — a substrate adequate for the phenotype extraction that follows; '
    'segmentation is not a contribution of this work.”').italic = True

# ══ Table 2 ══
H('Table 2 — Imaging phenotype extraction quality — FILLED', 1)
P(t2['method'])
rows = []
for r in t2['rows']:
    if r.get('n'):
        rows.append([r['phenotype'], r['n'], f"{r['accuracy']} {r['acc_ci']}", f"{r['macro_f1']} {r['f1_ci']}"])
    else:
        rows.append([r['phenotype'], '—', 'N/A (single-organ datasets)', '—'])
table(['Phenotype', 'n', 'Accuracy [95% CI]', 'F1 [95% CI]'], rows, filled_cols={2, 3})
n = doc.add_paragraph(); nn = n.add_run('Cross-organ distribution is N/A for the single-organ Pancreas/LiTS '
    'sets; it requires FLARE multi-organ cases (per-patient volumes unavailable — pre-sliced source).')
nn.italic = True; nn.font.size = Pt(9)

# ══ Table 3 ══
H('Table 3 — Ontology mapping coverage and correctness — FILLED', 1)
P('Coverage from the imaging KG schema (kg/schema.owl); codes verified via SNOMED CT $lookup + NCI EVS. '
  'Correctness = verified for curated core concepts; inter-rater kappa requires a 2-reviewer study (pending).')
# recompute to PDF's exact rows
m = json.load(open("/home/ud3d4/Desktop/SWOG/kg/ontology_mappings.json"))["mappings"]
anat = ["Organ::Pancreas","Organ::Liver","AnatomicSite::Head of pancreas","AnatomicSite::Body of pancreas","AnatomicSite::Tail of pancreas"]
lesion = ["Lesion::Pancreatic tumor","Lesion::Liver tumor"]
phen = ["Observation::Tumor burden","Observation::Lesion multiplicity","Observation::Organ containment","Observation::Cross-organ extension"]
def cov(keys):
    mp = sum(1 for k in keys if m.get(k)); return mp, len(keys)
a_m, a_t = cov(anat); l_m, l_t = cov(lesion); p_m, p_t = cov(phen)
o_m, o_t = a_m + l_m + p_m, a_t + l_t + p_t
table(['Concept group', 'Mapped/Total', 'Coverage (%)', 'Correctness (%)', 'Agreement (κ)'], [
    ['Anatomical (organs + sub-sites)', f'{a_m}/{a_t}', f'{100*a_m/a_t:.0f}', 'verified*', 'pending'],
    ['Lesion (tumor types)', f'{l_m}/{l_t}', f'{100*l_m/l_t:.0f}', 'verified*', 'pending'],
    ['Phenotype (burden, multiplicity, containment, cross-organ)', f'{p_m}/{p_t}', f'{100*p_m/p_t:.0f}', 'verified*', 'pending'],
    ['Overall', f'{o_m}/{o_t}', f'{100*o_m/o_t:.1f}', 'verified*', 'pending'],
], filled_cols={1, 2})
sup = t3['supplementary_full_mmkg_coverage']
n = doc.add_paragraph(); nn = n.add_run(f"*Curated core codes verified by direct ontology $lookup. "
    f"Lesion multiplicity + organ containment intentionally unmapped (no clean single code). "
    f"Supplementary — full MMKG incl. clinical concepts: {sup['mapped']}/{sup['total']} mapped "
    f"(SNOMED {sup['by_system'].get('SNOMED CT',0)}, LOINC {sup['by_system'].get('LOINC',0)}, "
    f"ICD-11 {sup['by_system'].get('ICD 11',0)}, MeSH {sup['by_system'].get('MeSH',0)}, "
    f"RxNorm {sup['by_system'].get('RxNorm',0)}).")
nn.italic = True; nn.font.size = Pt(9)

# ══ Tables 4-11: structured placeholders ══
H('Tables 4–11 — require the observability-IPKG pipeline / raters', 1)
P('These are the paper’s core contribution. Their exact structures are reproduced below; each cell is '
  'marked with the specific experiment or raters needed. No numbers are fabricated. A predecessor MMKG '
  'retrieval result (different protocol) is included as supplementary evidence where relevant.')

def placeholder(title, caption, headers, row_labels, need):
    H(title, 2)
    c = doc.add_paragraph(); cc = c.add_run(caption); cc.italic = True; cc.font.size = Pt(9)
    table(headers, [[lbl] + ["requires experiment"] * (len(headers) - 1) for lbl in row_labels])
    x = doc.add_paragraph(); xx = x.add_run("Needs: " + need); xx.font.size = Pt(9); xx.font.color.rgb = AMBER

placeholder("Table 4 — Structured query reasoning",
    "Precision/Recall/F1 over resolved (T/F) verdicts; Indet. = indeterminate rate (three-valued semantics).",
    ["Query pattern", "n_q", "Precision", "Recall", "F1", "Indet. (%)"],
    ["High tumor burden", "Multifocal disease", "Tumor in specified organ", "Cross-organ distribution", "Mean"],
    "three-valued observability query engine over the IPKG (indeterminate-rate accounting).")

placeholder("Table 5 — Aggregate phenotype-driven retrieval",
    "P@5/P@10/mAP/nDCG + explanation-path fraction; ablation ladder + external baselines.",
    ["Method", "P@5", "P@10", "mAP", "nDCG", "Expl. path"],
    ["Radiomics retrieval", "Radiomics-CBIR", "Embedding retrieval", "Ontology-aware learned",
     "Phenotype-vector (base)", "+typed relations", "+graded ontology", "(flat-tier, abl.)",
     "(coverage-blind, abl.)", "Proposed IPKG"],
    "IPKG retrieval with the ablation ladder + radiomics/CBIR/ontology-aware baselines under P@k/Expl-path metrics.")
# supplementary predecessor evidence
t4 = json.load(open("/home/ud3d4/Desktop/Projects/acm_mmkg/results/v0.5/table4_retrieval_40q.json"))["metrics"]
sp = doc.add_paragraph(); spr = sp.add_run("Supplementary (predecessor MMKG v0.5, DIFFERENT protocol — 40 clinical "
    "queries, metrics nDCG@10/MAP/Recall@10/EM/F1, not P@k):"); spr.italic = True; spr.font.size = Pt(9)
table(['Method', 'nDCG@10', 'MAP', 'Recall@10', 'EM', 'F1'],
    [[k] + [t4[k][x] for x in ['nDCG@10','MAP','Recall@10','EM','F1']] for k in t4],
    filled_cols={1,2,3,4,5})

placeholder("Table 6 — Retrieval by query stratum",
    "Proposed vs coverage-blind ablation vs strongest baseline; Δobs = proposed−coverage-blind gap.",
    ["Stratum / Method", "P@10", "mAP", "nDCG", "Δobs (nDCG), p"],
    ["Within: Proposed", "Within: Coverage-blind", "Within: Strongest baseline",
     "Cross: Proposed", "Cross: Coverage-blind", "Cross: Strongest baseline",
     "Decomp: Proposed", "Decomp: Coverage-blind", "Decomp: Strongest baseline"],
    "stratified retrieval (within/cross/decomposition) with the coverage-blind ablation.")
placeholder("Table 7 — Leave-one-phenotype-out (LOPO)",
    "Hold out one phenotype from similarity; score on it alone. Proposed vs coverage-blind.",
    ["Held-out phenotype", "Proposed", "Coverage-blind (abl.)"],
    ["Tumor burden", "Lesion multiplicity", "Organ containment", "Cross-organ distrib.", "Mean"],
    "LOPO retrieval protocol.")
# Table 8 — REAL (computed from the observability pipeline)
H('Table 8 — Coverage-model specificity controls — FILLED', 2)
c = doc.add_paragraph(); cc = c.add_run("Incomparability: fraction of disjoint-observability (pancreas vs liver) "
    "pairs correctly excluded. Silence control: false-penalty rate on those pairs. Computed over all "
    "84,666 cross-domain case pairs."); cc.italic = True; cc.font.size = Pt(9)
_t8 = json.load(open(f"{RES}/tables_5to8_retrieval.json"))["table8"]
table(['Control', 'Proposed', 'Coverage-blind (abl.)'], [
    ['Incomparable pairs correctly excluded (%)',
     _t8['incomparable_pairs_excluded_pct']['proposed'], _t8['incomparable_pairs_excluded_pct']['coverage_blind']],
    ['Silence-as-absence false-penalty rate (%)',
     _t8['silence_false_penalty_pct']['proposed'], _t8['silence_false_penalty_pct']['coverage_blind']],
], filled_cols={1, 2})
x = doc.add_paragraph(); xx = x.add_run("The observability model treats every disjoint-anatomy pair as "
    "incomparable (100%); the coverage-blind ablation assigns all of them a similarity (100% false penalty). "
    "This is the paper's central observability claim, demonstrated on real data.")
xx.font.size = Pt(9); xx.font.color.rgb = GREEN

# Tables 5-7 retrieval — honest saturation finding
H('Tables 5–7 — Aggregate / stratified / LOPO retrieval — UNDERPOWERED on current data', 2)
_r = json.load(open(f"{RES}/tables_5to8_retrieval.json"))
b = _r["table5"]["Phenotype-vector (base)"]; pr = _r["table5"]["Proposed IPKG"]
fp = doc.add_paragraph(); fpr = fp.add_run(
    "The retrieval pipeline was implemented and run (ablation ladder, stratified, LOPO). On the two available "
    "single-organ datasets (Pancreas + LiTS, disjoint observability, sparse discrete phenotypes) the metrics "
    "SATURATE: the base phenotype-vector reaches nDCG "
    f"{b['nDCG']}, but every graph-mechanism variant (+typed, +graded ontology, flat-tier, coverage-blind, "
    f"proposed) reaches {pr['nDCG']} — the ablation ladder cannot be discriminated because relevance and "
    "similarity become near-collinear (circular). These numbers are therefore NOT reported as results.")
fpr.font.size = Pt(9); fpr.font.color.rgb = AMBER
x2 = doc.add_paragraph(); x2r = x2.add_run("A discriminating retrieval evaluation requires: (i) FLARE's "
    "multi-organ cases to create genuine partial-observability cross-dataset structure (the paper's actual "
    "cross-dataset / decomposition test — cross stratum was empty here, n=0, because pancreas and liver "
    "share no observed anatomy); and (ii) a non-circular relevance signal (the blinded expert study, Table 10).")
x2r.font.size = Pt(9); x2r.font.color.rgb = AMBER
placeholder("Table 9 — Sensitivity to hyperparameters",
    "Induced range of nDCG@10 around nominal.",
    ["Swept parameter", "Range explored", "nDCG@10 (min–max)"],
    ["Weights λ (simplex)", "Coverage threshold γ_min", "Containment overlap threshold", "Burden threshold θ_B"],
    "hyperparameter sweeps on the IPKG retrieval.")
placeholder("Table 10 — Expert-judged retrieval quality",
    "Top-k graded 0–3 relevance from consensus expert labels; raters blind to method.",
    ["Method", "P@10", "mAP", "nDCG", "Cross nDCG", "Decomp nDCG", "Δ vs prop."],
    ["Radiomics", "Embedding", "Ontology-aware learned", "Coverage-blind (abl.)", "Proposed IPKG"],
    "TWO board-certified clinician raters (blinded, quadratic-weighted κ) — cannot be run without raters.")
placeholder("Table 11 — Cross-dataset integration",
    "Mixing index M vs permutation null; observability gap; leave-one-dataset-out.",
    ["Measure", "Value"],
    ["Mean mixing index M (observed)", "Mean mixing index M (permutation null)",
     "Mixing gap ΔM (obs−null), p", "Fraction of multi-dataset communities",
     "Observability gap, cross-dataset queries (ΔF1, p)", "Leave-one-dataset-out retrieval (mAP)"],
    "community detection + mixing-index vs permutation null on the multi-dataset IPKG graph.")

# ── Status summary ──
H('Status summary', 1)
for item, done in [
    ("Segmentation substrate (§5.1)", True), ("Table 2 — phenotype extraction", True),
    ("Table 3 — ontology mapping coverage", True),
    ("Table 4 — structured query reasoning (3-valued)", False),
    ("Table 5 — aggregate retrieval (ablation ladder)", False),
    ("Table 6 — stratified retrieval", False), ("Table 7 — LOPO", False),
    ("Table 8 — coverage controls", False), ("Table 9 — sensitivity", False),
    ("Table 10 — expert-judged (needs clinician raters)", False),
    ("Table 11 — cross-dataset integration", False)]:
    para = doc.add_paragraph(style='List Bullet')
    mk = para.add_run(("[x] " if done else "[ ] ")); mk.bold = True
    mk.font.color.rgb = GREEN if done else AMBER
    para.add_run(item)
P()
P("To complete Tables 4–9 and 11: implement + run the observability-aware IPKG retrieval pipeline "
  "(typed relations, graded IC-weighted ontology similarity, observability model with three-valued "
  "semantics, coverage-blind ablation) on the built VKG/CKG graphs. Table 10 additionally requires a "
  "blinded 2-rater expert study. The predecessor MMKG pipeline (acm_mmkg) provides related retrieval "
  "evidence but under a different protocol.")

out = f"{JBI}/JBI_results.docx"
doc.save(out)
print("Saved:", out)
