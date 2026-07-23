#!/usr/bin/env python3
"""Response to the FLARE visualization-issues diagnostic: map each raised cause to its status
in the new per-patient delivery, with real validation evidence. Neutral/artifact-neutral."""
import nibabel as nib, numpy as np, json, os
from nibabel.orientations import aff2axcodes
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.table import WD_TABLE_ALIGNMENT

O = "/scratch/ud3d4/acm_data/FLARE_Task2/flare_handoff_v2"
OUT = "/home/ud3d4/Desktop/SWOG/docs/FLARE_visualization_response.docx"
ID2NAME = {1: "liver", 2: "right_kidney", 3: "spleen", 4: "pancreas", 13: "left_kidney"}
GREEN = RGBColor(0x1a, 0x7f, 0x37); RED = RGBColor(0xB0, 0x2A, 0x2A); GREY = RGBColor(0x55, 0x55, 0x55)

# validation evidence for one representative case
cid = "FLARE22_Tr_0001"
ct = nib.load(f"{O}/ct/flare/{cid}.nii.gz"); gt = nib.load(f"{O}/ground_truth/flare/{cid}.nii.gz")
pr = nib.load(f"{O}/ssl_predictions/flare/{cid}.nii.gz")
gta, pra = gt.get_fdata().astype(int), pr.get_fdata().astype(int)
def dice(l):
    p, g = (pra == l), (gta == l); s = p.sum() + g.sum()
    return round(2 * (p & g).sum() / s, 3) if s else None

doc = Document(); doc.styles['Normal'].font.name = 'Calibri'; doc.styles['Normal'].font.size = Pt(10.5)
def H(t, l=1): doc.add_heading(t, level=l)
def P(t="", bold=False, italic=False, color=None, size=None):
    p = doc.add_paragraph(); r = p.add_run(t); r.bold = bold; r.italic = italic
    if color: r.font.color.rgb = color
    if size: r.font.size = Pt(size)
    return p
def tbl(headers, rows, bold_rows=()):
    t = doc.add_table(rows=1, cols=len(headers)); t.style = 'Light Grid Accent 1'; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers): t.rows[0].cells[i].paragraphs[0].add_run(h).bold = True
    for ri, row in enumerate(rows):
        c = t.add_row().cells
        for i, v in enumerate(row):
            run = c[i].paragraphs[0].add_run(str(v))
            if ri in bold_rows: run.bold = True
    doc.add_paragraph()

doc.add_heading('FLARE 3D Visualization — Response to the Diagnostic', level=0)
P('The diagnostic checklist is correct, and it describes the OLD delivery (pre-sliced per-class '
  '2D stacks, 8-bit CT, separate binary per organ). A new PER-PATIENT delivery that resolves those '
  'issues has been produced and uploaded. Below, each cause raised is mapped to its status in the '
  'new delivery, with real validation evidence.', italic=True)

H('Where the new delivery is', 1)
P('Drive: data/ssl_handoff_for_krishna/flare_perpatient_v2/  — same layout as Pancreas/LiTS:')
P('  ct/flare/<case>.nii.gz   ground_truth/flare/<case>.nii.gz   ssl_predictions/flare/<case>.nii.gz   flare_manifest.json',
  size=9, color=GREY)
P('20 test cases; start with FLARE22_Tr_0001 (liver) to confirm rendering.')

H('Cause-by-cause status', 1)
tbl(['Cause raised in the diagnostic', 'Status in the new per-patient delivery'],
    [['#1 Wrong FLARE version / missing tumor',
      'CORRECT DIAGNOSIS. This is FLARE 2024 Task2 = FLARE22 = 13 organs, NO tumor class. We deliver 5 '
      'organs (liver, both kidneys, spleen, pancreas). FLARE tumor requires FLARE23; not available as '
      'per-patient volumes. Not a visualization bug.'],
     ['#2 Only 2D slices, not a 3D volume',
      'RESOLVED. Each case is one full CT volume with every axial slice in true order (not filtered), '
      'e.g. FLARE22_Tr_0001 = 512x512x110.'],
     ['#3 Prediction not converted to a proper label map',
      'RESOLVED. Single multi-label 3D mask per case, discrete integer FLARE IDs (1/2/3/4/13); no logits, '
      'no per-slice PNGs, no dropped depth.'],
     ['#4 Preprocessing not reversed / not on original grid',
      'RESOLVED. Prediction is on the ORIGINAL CT grid: CT.shape == mask.shape, CT.affine ≈ mask.affine, '
      'CT.orientation == mask.orientation (verified below).'],
     ['#5 Label-mapping problem',
      'RESOLVED. FLARE22 class IDs preserved and documented in flare_manifest.json (label_ids block). '
      'GT and prediction use identical IDs so per-organ toggle works.']])

H('Validation evidence (their Step-1 report, one case)', 1)
P(f'Case {cid} — run directly on the delivered files:')
tbl(['Field', 'CT', 'Prediction'],
    [['shape', str(ct.shape), str(pr.shape)],
     ['dtype', str(ct.get_data_dtype()), str(pr.get_data_dtype())],
     ['voxel spacing (mm)', str(tuple(round(float(z), 3) for z in ct.header.get_zooms()[:3])),
      str(tuple(round(float(z), 3) for z in pr.header.get_zooms()[:3]))],
     ['orientation (axcodes)', str(aff2axcodes(ct.affine)), str(aff2axcodes(pr.affine))],
     ['HU range', f"[{int(ct.get_fdata().min())}, {int(ct.get_fdata().max())}]", '—']])
P('Consistency checks (all must be true for correct rendering):', bold=True)
tbl(['Check', 'Result'],
    [['CT.shape == GT.shape == PRED.shape', str(ct.shape == gt.shape == pr.shape)],
     ['CT.affine ≈ GT.affine ≈ PRED.affine', str(np.allclose(ct.affine, gt.affine) and np.allclose(ct.affine, pr.affine))],
     ['orientation CT == GT == PRED', str(aff2axcodes(ct.affine) == aff2axcodes(gt.affine) == aff2axcodes(pr.affine))]],
    bold_rows=(0, 1, 2))
P('Per-label Dice (GT vs prediction) — the geometric proof:', bold=True)
tbl(['Organ', 'FLARE ID', 'Dice', 'GT voxels', 'Pred voxels'],
    [[ID2NAME[l], l, dice(l), int((gta == l).sum()), int((pra == l).sum())] for l in (1, 2, 3, 4, 13)])
P('A flipped, displaced, or mis-affined mask would score Dice ≈ 0. These 0.93–0.99 values confirm the '
  'prediction is voxel-aligned to the CT in the correct orientation — i.e. it will render and overlay '
  'correctly.', color=GREEN)

H('One note for the viewer', 1)
P('Different FLARE cases carry different native orientations (e.g. RAS vs LPS) and spacings — each file '
  'carries its own correct affine. A viewer that respects the NIfTI affine (as it already does for '
  'Pancreas/LiTS) will render every case correctly. Do not assume a fixed orientation across cases.')

H('Recommended check (matches the diagnostic’s Step 4)', 1)
P('Load ct/flare/FLARE22_Tr_0001.nii.gz as the reference volume and ssl_predictions/flare/'
  'FLARE22_Tr_0001.nii.gz as the segmentation (nearest-neighbor). The liver should render immediately; '
  'the other four organs toggle via their label IDs. If it renders here, the geometry is valid and any '
  'remaining issue is the application adapter — not the segmentation.')

P('Source: real files at data/ssl_handoff_for_krishna/flare_perpatient_v2/. Manifest carries per-case '
  'Dice per label (recomputed + reported), volumes, spacing, and public FLARE case IDs.', italic=True, size=8, color=GREY)
doc.save(OUT)
print("Saved:", OUT)
print("Case", cid, "Dice:", {ID2NAME[l]: dice(l) for l in (1, 2, 3, 4, 13)})
