#!/usr/bin/env python3
"""Generate plan.docx — an end-to-end pipeline + plan document (artifact/role-neutral) suitable for
sharing. Pulls real experiment numbers from results/*.json; segmentation numbers are the current
patient-level results. Output: plan.docx at repo root (*.docx is gitignored — a deliverable, not source)."""
import json
import os

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = "/home/ud3d4/Desktop/SWOG"
R = os.path.join(ROOT, "results")
GREEN, GREY, BLUE = RGBColor(0x1E, 0x7A, 0x3C), RGBColor(0x55, 0x55, 0x55), RGBColor(0x1F, 0x4E, 0x79)


def j(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    A, B, Bv = j("oakg_structured.json"), j("kg_fidelity.json"), j("oakg_evolve.json")
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10.5)

    def h(t, lvl=1):
        doc.add_heading(t, level=lvl)

    def para(t, italic=False, bold=False, color=None, size=None):
        p = doc.add_paragraph(); r = p.add_run(t)
        r.italic, r.bold = italic, bold
        if color: r.font.color.rgb = color
        if size: r.font.size = Pt(size)
        return p

    def bullet(t):
        doc.add_paragraph(t, style="List Bullet")

    def mono(lines):
        p = doc.add_paragraph()
        r = p.add_run("\n".join(lines)); r.font.name = "Consolas"; r.font.size = Pt(9)
        p.paragraph_format.left_indent = Inches(0.2)

    def table(headers, rows):
        tb = doc.add_table(rows=1, cols=len(headers)); tb.style = "Light Grid Accent 1"
        for i, hd in enumerate(headers):
            c = tb.rows[0].cells[i]; c.text = ""; rr = c.paragraphs[0].add_run(hd); rr.bold = True; rr.font.size = Pt(9.5)
        for row in rows:
            cs = tb.add_row().cells
            for i, v in enumerate(row):
                cs[i].text = ""; rr = cs[i].paragraphs[0].add_run(str(v)); rr.font.size = Pt(9.5)

    # ---------------- Title ----------------
    doc.add_heading("Autonomous Abdominal-CT → Imaging Knowledge Graph", level=0)
    para("End-to-end pipeline and project plan", italic=True, color=GREY, size=12)

    # ---------------- Objective ----------------
    h("1. Objective", 1)
    para("Given a new abdominal CT with no annotations, the system automatically (1) segments the liver, "
         "kidneys, and pancreas together with their tumors, (2) converts those masks into an "
         "ontology-grounded, queryable knowledge graph, (3) checks each result is medically plausible "
         "without any ground truth, and (4) retrieves similar prior patients — improving each time a new "
         "case is admitted. The scientific contribution is the downstream reasoning layer: an "
         "observability-aware, self-evolving clinical knowledge graph built end-to-end from label-free "
         "segmentation. Segmentation is treated as infrastructure (built on the SAM3 foundation model), "
         "not as the novelty.")

    # ---------------- End-to-end pipeline ----------------
    h("2. The end-to-end pipeline", 1)
    para("A new, unlabeled scan flows through the pipeline; the knowledge graph both consumes its output "
         "and feeds anatomical knowledge back — the loop is bidirectional:")
    mono([
        "  New abdominal CT  (no annotations)",
        "     |",
        "  1. SEGMENT    organs via SAM3 concept prompts ('liver'/'kidney'/'pancreas'), no box;",
        "                tumors via one generic 'tumor' model  ->  organ + tumor masks",
        "     |            ^",
        "     |            |  (KG anatomical atlas guides the prompt to where an organ sits,",
        "     |            |   and REPAIRS masks: drop spurious blobs, enforce tumor-in-organ)",
        "  2. PHENOTYPE  derive per-organ volume, diameter, centroid, tumor burden, lesion count",
        "     |",
        "  3. KNOWLEDGE GRAPH   write direct, ontology-grounded triples (SNOMED / LOINC / ICD / MeSH)",
        "     |",
        "  4. VALIDATE   score each phenotype against the cohort distribution  ->  flag/repair",
        "                implausible masks as segmentation errors  (GT-free, no labels needed)",
        "     |",
        "  5. RETRIEVE, GROW & GUIDE   find similar patients; admit the case (KG grows);",
        "                the sharpened cohort model + atlas then VALIDATE and SEGMENT the next",
        "                patient better  ->  segmentation feeds the KG, the KG feeds segmentation",
    ])
    para("The knowledge graph is not a passive store: it interprets, vets, and contextualises each new "
         "unlabeled scan, it improves as it grows, and — via the anatomical atlas — it feeds that growing "
         "knowledge back to improve the segmentation that fills it. Adding a dataset (e.g. KiTS for the "
         "kidney) sharpens both the validation and the segmentation priors for that organ.",
         italic=True, color=GREY)

    # ---------------- Data ----------------
    h("3. Data and how it is used", 1)
    para("Data serves two distinct roles — the key to the design:")
    bullet("Image + label pairs → train the segmentation models (you must show a model CTs with correct "
           "masks). Sources: LiTS, MSD Pancreas, KiTS23, and FLARE-Task2.")
    bullet("Labels alone → build the knowledge graph (only numbers are needed — volumes, diameters, "
           "burden). FLARE23’s 1,312 label masks power the graph with no images required.")
    table(["Dataset", "Labels", "Size", "Role"],
          [["LiTS", "liver + liver tumor", "131 patients", "tumor training; liver-tumor model"],
           ["MSD Pancreas", "pancreas + tumor", "281", "tumor training; pancreas models"],
           ["KiTS23", "kidney + tumor", "489", "tumor training; kidney KG records"],
           ["FLARE23 (full)", "13 organs + tumor", "1,312 (labels-only)", "the knowledge graph + tumor training"],
           ["FLARE-Task2", "organs (no tumor)", "100 volumes", "organ segmentation models"]])

    # ---------------- Segmentation ----------------
    h("4. Segmentation models (strictly patient-level results)", 1)
    para("Generic tumor model — one model, prompt \"tumor\", no box, trained on pooled tumors from all four "
         "datasets, strictly patient-level. A controlled incremental re-training (LiTS → +Pancreas → +KiTS, "
         "with FLARE held out entirely) measures cross-dataset generalization: on a completely unseen "
         "dataset (FLARE), held-out Dice climbs 0.35 → 0.41 → 0.51 as datasets are added — more coverage "
         "yields better generalization. The all-four-datasets deployment model is being finalized; "
         "per-dataset in-distribution numbers finalize with it.", bold=False)
    para("Organ models — SAM3, two regimes:")
    bullet("With a localisation box (upper bound): patient-level 3-D Dice 0.92–0.99 (liver 0.985, "
           "spleen 0.980, kidneys 0.970, pancreas 0.919).")
    bullet("Fully autonomous concept prompt (deployment): strong on the liver (0.82 full-volume); small "
           "organs are weaker full-volume and are addressed by KG-guided segmentation (§5) plus per-organ "
           "fine-tuning.")

    # ---------------- KG ----------------
    h("5. The knowledge graph — evolving AND guiding segmentation", 1)
    para("Spans all four training datasets — 2,113 patients (FLARE23 1,312 + KiTS 389 + Pancreas 281 + "
         "LiTS 131). Each patient is a subgraph (case → organ → lesion) with phenotypes as direct triples, "
         "grounded to standard terminologies (SNOMED / LOINC / ICD / MeSH) via a live mapper, open-world. "
         "Roles: retrieval, semantic interoperability, GT-free validation of new patients — and, newly, "
         "GT-free guidance of segmentation.")
    para("KG-guided segmentation (closing the loop). From the cohort the KG builds an anatomical atlas — "
         "per-organ plausible size, diameter, and location, pooled across every dataset that observes the "
         "organ (KiTS sharpens the kidney, LiTS the liver, MSD the pancreas). These priors (a) REPAIR "
         "autonomous masks — drop spurious components, keep the plausible one, enforce tumor-inside-organ "
         "and kidney-pair rules — and (b) GUIDE the prompt to where an organ typically sits, targeting the "
         "small-organ weakness. So the graph does not merely grow: it improves the segmentation that fills "
         "it, and every admitted patient (or dataset) sharpens both.")

    # ---------------- Contribution + experiments ----------------
    h("6. Scientific contribution and evidence", 1)
    para("Problem. Merging many imaging datasets into one graph creates structural partial-observability "
         "— each source labelled different organs, so most patients are only partly described. Standard "
         "retrieval either imputes the missing values (false matches) or compares on the little that is "
         "shared (a fake “perfect match” on one organ).")
    para("Contribution (OAKG). Never impute a missing value; weight every comparison by how much two "
         "patients actually share (a factor γ). This keeps answers correct as the graph scales. Four "
         "experiments, all on real data:")
    rows = []
    if A:
        i = A["levels"].index(0.25) if 0.25 in A["levels"] else 1
        rows.append(["A — retrieval on a merged graph",
                     f"OAKG Precision@10 {A['curve']['oakg']['prec'][i]} vs {A['curve']['masked']['prec'][i]} "
                     f"without the γ weighting (decisive)"])
    if B:
        s = B["summary"]
        rows.append(["B — predicted vs ground-truth KG",
                     f"volume correlation {s['node_fidelity']['volume_pearson_r']}, "
                     f"query ranking {s['query_rank_by_size']['spearman']} (near-identical)"])
    if Bv:
        rows.append(["C — self-evolving GT-free validation",
                     f"joint model AUROC {max(Bv['joint'])} vs {Bv['marginal'][0]} baseline; improves as the graph grows"])
    rows.append(["D — cross-dataset tumor generalization (RUNNING)",
                 "held-out FLARE Dice climbs 0.35 → 0.41 → 0.51 as datasets are added; deployment model finalizing"])
    rows.append(["E — KG-guided segmentation (IN PROGRESS)",
                 "repair implemented + wired in (validated on controlled errors: pancreas 0.98→1.00, "
                 "tumor 0.98→1.00); prompt-guidance + real-mask gain next"])
    table(["Experiment", "Headline result"], rows)
    para("Together: the γ weighting is what makes retrieval correct on a merged graph; a graph built from "
         "autonomous masks answers like one built from ground truth; validation improves as the graph "
         "grows; cross-dataset generalization improves with coverage; and — closing the loop — the graph's "
         "accumulated anatomy improves the segmentation that fills it. The self-evolving property is "
         "therefore bidirectional.", italic=True, color=GREY)

    # ---------------- Status ----------------
    h("7. Current status", 1)
    para("Completed:", bold=True)
    for s in ["Autonomous segmentation + a generic tumor model, and an ontology-grounded knowledge graph "
              "over 2,113 patients across four datasets (FLARE23, KiTS23, MSD Pancreas, LiTS).",
              "The four reasoning-layer experiments (A–C above) — complete and reproducible.",
              "40 full FLARE cases (CT + 13-organ + tumor) extracted for organ segmentation, a predicted "
              "KG, and image reconstruction; KG anatomical atlas built (per-organ priors, all datasets)."]:
        bullet(s)
    para("Experiments running now:", bold=True)
    bullet("D — Strictly patient-level, incremental cross-dataset tumor training. Four warm-started stages "
           "(LiTS → +Pancreas → +KiTS, with FLARE held out entirely, then +FLARE), val AND test held out "
           "by whole patient. Stages 1–3 done — held-out FLARE Dice 0.35 → 0.41 → 0.51 (generalization "
           "improves with coverage); stage 4 (all-four-datasets deployment model) is ~2/3 through and "
           "will report the final per-dataset in-distribution numbers on completion.")
    bullet("E — KG-guided segmentation (closing the loop). The atlas-based repair is implemented and wired "
           "into the pipeline (drop spurious components, flag implausible volumes, remove floating tumor); "
           "validated on controlled errors (pancreas 0.98→1.00, tumor 0.98→1.00). Remaining: the "
           "prompt-guidance half and the Dice gain on real autonomous masks (the 40 full cases), which run "
           "once stage-4 training frees the GPU.")

    # ---------------- Plan ----------------
    h("8. Plan and next steps", 1)
    for i, s in enumerate([
        "Finish + quantify KG-guided segmentation: measure the Dice gain from atlas-based repair and "
        "prompt-guidance on the autonomous masks (especially small organs).",
        "Per-organ fine-tuned concept models to close the remaining small-organ autonomous gap.",
        "Predicted knowledge graph + image reconstruction on the newly extracted full FLARE cases.",
        "Expand training with harder, more diverse tumor-bearing CTs to raise robustness.",
        "Like-for-like comparison against recent SOTA systems; manuscript centred on the OAKG contribution "
        "(benchmark and method framings), targeting a top venue."], 1):
        bullet(f"{i}. {s}")

    out = os.path.join(ROOT, "plan.docx")
    doc.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
