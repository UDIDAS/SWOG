#!/usr/bin/env python3
"""Summarize the KG-in-training ablation: baseline vs KG-trained, for both models. Reads the four result
JSONs and prints per-organ / per-dataset deltas (the KG's contribution to LEARNING). Writes
results/kg_ablation_summary.json. Safe to run anytime — skips whatever isn't produced yet.
"""
import json
import os

R = "/home/ud3d4/Desktop/SWOG/results"


def load(name):
    p = f"{R}/{name}"
    return json.load(open(p)) if os.path.exists(p) else None


def delta(b, k):
    return None if (b is None or k is None) else round(k - b, 4)


out = {}
print("=" * 66)
print("KG-IN-TRAINING ABLATION  (baseline vs +KG_consistency)")
print("=" * 66)

# ---- ORGAN model ----
ob, ok = load("organ_generic.json"), load("organ_generic_kg.json")
if ob and ok:
    print("\nORGAN model — held-out patient-level test Dice")
    po = {"per_organ": {}, "per_organ_dataset": {}}
    print(f"  {'organ':16s} {'baseline':>9s} {'+KG':>8s} {'Δ':>8s}")
    for o in sorted(set(ob.get("per_organ", {})) | set(ok.get("per_organ", {}))):
        b, k = ob["per_organ"].get(o), ok["per_organ"].get(o)
        d = delta(b, k); po["per_organ"][o] = {"baseline": b, "kg": k, "delta": d}
        print(f"  {o:16s} {b:9.4f} {k:8.4f} {d:+8.4f}")
    for key in sorted(set(ob.get("per_organ_dataset", {})) | set(ok.get("per_organ_dataset", {}))):
        b, k = ob["per_organ_dataset"].get(key), ok["per_organ_dataset"].get(key)
        po["per_organ_dataset"][key] = {"baseline": b, "kg": k, "delta": delta(b, k)}
    out["organ"] = po
else:
    print("\nORGAN model — results not both present yet (baseline:%s kg:%s)" % (bool(ob), bool(ok)))

# ---- TUMOR model (final all-4-dataset stage) ----
tb, tk = load("tumor_incremental.json"), load("tumor_incremental_kg.json")
if tb and tk:
    print("\nTUMOR model — final stage (all 4 datasets), per-dataset test Dice")
    fb = tb["stages"][-1]["per_dataset_test"]; fk = tk["stages"][-1]["per_dataset_test"]
    td = {}
    print(f"  {'dataset':16s} {'baseline':>9s} {'+KG':>8s} {'Δ':>8s}")
    for ds in sorted(set(fb) | set(fk)):
        b, k = fb.get(ds), fk.get(ds); d = delta(b, k); td[ds] = {"baseline": b, "kg": k, "delta": d}
        print(f"  {ds:16s} {b:9.4f} {k:8.4f} {d:+8.4f}")
    out["tumor_final_per_dataset"] = td
else:
    print("\nTUMOR model — results not both present yet (baseline:%s kg:%s)" % (bool(tb), bool(tk)))

json.dump(out, open(f"{R}/kg_ablation_summary.json", "w"), indent=2)
print(f"\n-> {R}/kg_ablation_summary.json")
