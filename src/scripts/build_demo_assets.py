#!/usr/bin/env python3
"""Demo assets for the MMKG Streamlit app. For a few curated tumor-test cases per dataset, run the organ + tumor
AUSAM models, save a CT + predicted-mask overlay (organ green, tumor red) on the max-tumor slice, and bundle the
predicted + GT phenotypes. -> app/assets/demo/{case}.png + app/assets/demo/demo_cases.json
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
from build_predicted_corpus import CFG, predict_volume, DEV
from eval_ausam_3d import _slice, WIN
from run_pancreas_sam3 import _load_sam3_ckpt, SAM3_MODEL_ID, HF_TOKEN

OUT = "/home/ud3d4/Desktop/SWOG/app/assets/demo"
RES = "/home/ud3d4/Desktop/SWOG/results"
N_PER_DS = 2                                     # curated cases per dataset


def corpus(kind, ds):
    return {r["case_id"]: r for r in json.load(open(f"{RES}/corpus_{kind}_{ds}.json"))["records"]}


def main():
    os.makedirs(OUT, exist_ok=True)
    from transformers import Sam3Processor
    proc = Sam3Processor.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    demo = []
    for ds in ["msd", "lits", "kits"]:
        cfg = CFG[ds]; ax = cfg["ax"]
        pred, gt = corpus("predicted", ds), corpus("gt", ds)
        # curate: cases the model predicted a tumor for (clearest to show), by descending tumor volume
        cand = sorted([c for c, r in pred.items() if r["organs"][cfg["organ"]]["has_tumor"]],
                      key=lambda c: -pred[c]["organs"][cfg["organ"]]["tumor_voxels"])[:N_PER_DS]
        om = _load_sam3_ckpt(cfg["om"], DEV); tm = _load_sam3_ckpt(cfg["tm"], DEV)
        for case in cand:
            try:
                ct, seg, sp = cfg["load"](case)
            except Exception as e:
                print(f"  {case} skip ({type(e).__name__})", flush=True); continue
            omask = predict_volume(om, proc, ct, seg, cfg["olab"], ax, cfg["organ"])
            tmask = predict_volume(tm, proc, ct, seg, cfg["tlab"], ax, "tumor")
            axes = tuple(i for i in range(3) if i != ax)
            z = int((tmask.sum(axis=axes) if tmask.any() else omask.sum(axis=axes)).argmax())
            ctS, oS, tS = _slice(ct, ax, z), _slice(omask, ax, z), _slice(tmask, ax, z)
            lo, hi = WIN; disp = (np.clip(ctS, lo, hi) - lo) / (hi - lo)
            fig, a = plt.subplots(figsize=(4.2, 4.2)); a.imshow(disp, cmap="gray")
            if oS.any():
                a.imshow(np.ma.masked_where(~oS, oS.astype(float)), cmap="Greens", alpha=0.35, vmin=0, vmax=1)
            if tS.any():
                a.imshow(np.ma.masked_where(~tS, tS.astype(float)), cmap="autumn", alpha=0.6, vmin=0, vmax=1)
            a.set_title(f"{case} · {cfg['organ']} (green) + tumor (red)", fontsize=9); a.axis("off")
            fig.tight_layout(); fig.savefig(f"{OUT}/{case}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
            demo.append({"case": case, "dataset": ds, "organ": cfg["organ"], "png": f"{case}.png",
                         "pred": pred[case], "gt": gt.get(case)})
            print(f"  {ds} {case}: saved overlay (slice {z})", flush=True)
        del om, tm
    json.dump(demo, open(f"{OUT}/demo_cases.json", "w"), indent=1)
    print(f"-> {len(demo)} demo cases -> {OUT}/demo_cases.json", flush=True)


if __name__ == "__main__":
    main()
