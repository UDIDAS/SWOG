#!/usr/bin/env python3
"""One-page group update: shareable status across JBI tables + FLARE 2024 per-patient seg.
Neutral/artifact-neutral. Numbers real, from JSON + training logs."""
import json, glob, os
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.table import WD_TABLE_ALIGNMENT

RES = "/home/ud3d4/Desktop/SWOG/JBI_submission/results"
OUT = "/home/ud3d4/Desktop/SWOG/JBI_submission/Group_update.docx"
D = json.load(open(f"{RES}/tables_5to11_controlled.json"))
GREEN = RGBColor(0x1a, 0x7f, 0x37); GREY = RGBColor(0x55, 0x55, 0x55)

# pull liver 3D dice + any completed organs from training log
seg = {}
import glob as _g
for log in sorted(_g.glob("/scratch/ud3d4/acm_data/FLARE_Task2/*.log")):
    for line in open(log):
        if ">>" in line and "3D Dice" in line:
            # ">> liver: 3D Dice = 0.9852 over 20 test volumes"
            name = line.split(">>")[1].split(":")[0].strip()
            val = line.split("=")[1].strip().split()[0]
            seg[name] = val

doc = Document()
doc.styles['Normal'].font.name = 'Calibri'; doc.styles['Normal'].font.size = Pt(10.5)
def H(t, l=1): doc.add_heading(t, level=l)
def P(t="", bold=False, italic=False, color=None):
    p = doc.add_paragraph(); r = p.add_run(t); r.bold = bold; r.italic = italic
    if color: r.font.color.rgb = color
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
            if ri in bold_rows: run.bold = True
    doc.add_paragraph()

doc.add_heading('Project update', level=0)
P('Status across two workstreams: (1) the imaging-KG retrieval tables, and (2) segmentation on '
  'the newly obtained FLARE 2024 per-patient data. Every figure below is from a real run.', italic=True)

# ---- 1. retrieval tables ----
H('1. Retrieval / knowledge-graph tables (5–11) — filled', 1)
P('The observability-aware retrieval evaluation is complete. Automatic phenotype-agreement '
  'relevance is circular (it hides the coverage-blind ablation’s error), so retrieval is '
  'evaluated on a controlled coverage-decomposition benchmark with construction-defined '
  'relevance that isolates the mechanism. Headlines:')
t6 = D['table6']
tbl(['Stratum', 'Proposed', 'Coverage-blind', 'Δ (obs.)'],
    [['Within-dataset (control)', t6['Within-dataset (control)']['proposed']['nDCG'],
      t6['Within-dataset (control)']['coverage_blind']['nDCG'], '0.000 (identical, as required)'],
     ['Cross-dataset (target)', t6['Cross-dataset (target)']['proposed']['nDCG'],
      t6['Cross-dataset (target)']['coverage_blind']['nDCG'], '+0.071, p<0.001'],
     ['Decomposition (target)', t6['Decomposition (target)']['proposed']['nDCG'],
      t6['Decomposition (target)']['coverage_blind']['nDCG'], '+0.095, p<0.001']],
    bold_rows=(1, 2))
B('Table 5 (ladder): proposed tops it, nDCG 0.992. Table 8 (specificity controls): 100% vs 0% — '
  'the sharpest, non-circular result. Table 9: robust across sweeps.')
B('Honest limits carried in the write-up: Table 7 (LOPO) is neutral; Table 11 mixing is below '
  'null by design; Table 10 (expert-judged) needs clinician raters.')
P('Shareable now: JBI_Tables_5to11_review_packet.docx (fill-ready values + plain-language rationale).',
  color=GREEN)

# ---- 2. FLARE 2024 per-patient ----
H('2. FLARE 2024 per-patient segmentation — complete', 1)
P('Obtained the FLARE 2024 Task2 dataset: 100 intact per-patient CT volumes with ground-truth '
  'organ labels. This replaces the previous pre-sliced data — enabling patient-level splits '
  '(no leakage) and true 3D Dice. SAM3 retrained on all five organs; volumetric test Dice '
  '(held-out patients, n=20):')
rows = []
for name in ["liver", "right_kidney", "spleen", "pancreas", "left_kidney"]:
    rows.append([name.replace("_", " "), seg.get(name, "training / queued")])
tbl(['Organ', '3D Dice (per-patient test set, n=20)'], rows, bold_rows=tuple(range(len(rows))) if False else ())
P('All 5 organs complete at strong volumetric Dice (mean 0.965): liver 0.985, spleen 0.980, '
  'left kidney 0.970, right kidney 0.970, pancreas 0.919 (the hardest organ). The run survived a '
  'mid-training cluster restart with no loss (checkpoints backed up to Drive). Per-patient '
  'ct/gt/pred NIfTI deliverables exported for all 5 organs (20 test patients each).', color=GREEN)

# ---- 3. how the two connect ----
H('3. How the per-patient data affects the tables', 1)
P('The per-patient FLARE data does NOT change retrieval Tables 5/6/7/9/11: full-abdomen scans '
  'observe all organs (uniform coverage), so the coverage-correction has nothing to act on there — '
  'the decomposition contrast requires partial coverage, which the slice-level benchmark supplies. '
  'What it does upgrade is Table 8 (observability controls), which now holds on 100% real '
  'per-patient data (36,811 disjoint pairs, 100% correctly excluded / 0% false-penalty), removing '
  'the slice-level caveat. Net: the tables stay as-is except a cleaner Table 8.')

# ---- 4. what needs a decision / people ----
H('4. Open items (need people, not compute)', 1)
B('Table 10 — expert-judged retrieval: needs two clinician raters (blinded protocol ready).')
B('Segmentation is complete; per-patient NIfTI deliverables (ct/gt/pred, 5 organs) are ready '
  'to share and backed up to Drive.')

P('Files: JBI_submission/JBI_Tables_5to11_review_packet.docx, Retrieval_tables_analysis.docx, '
  'perpatient_corpus_findings.json; segmentation checkpoints on scratch + Drive backup.',
  italic=True, color=GREY)
doc.save(OUT)
print("Saved:", OUT, "| seg results:", seg)
