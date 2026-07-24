"""Final-submission checklist validator (final-checkup doc §4/§5).

Verifies that every artifact the paper/supplement references actually exists,
that the frozen snapshot hashes still match, that the entry points import, and
that the headline numbers hold. Exit code 0 = all pass, 1 = any failure.

Run:  python validate_checklist.py      (from the repo root)
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
ok = warn = fail = 0


def _p(status: str, msg: str) -> None:
    global ok, warn, fail
    mark = {"OK": "  ✓", "WARN": "  ⚠", "FAIL": "  ✗"}[status]
    print(f"{mark} {msg}")
    if status == "OK":
        ok += 1
    elif status == "WARN":
        warn += 1
    else:
        fail += 1


def check_exists(rel: str) -> None:
    _p("OK" if (ROOT / rel).exists() else "FAIL", f"exists: {rel}")


def main() -> None:
    print("== Referenced artifacts ==")
    for rel in [
        "results/config.json",
        "results/tables/publication_ready_retrieval_table.csv",
        "results/tables/ranking_policy_selection.csv",
        "results/union_ablation/union_ablation_method_summary.csv",
        "results/native_cross_dataset/native_method_summary.csv",
        "results/strata/hard_distractor.csv",
        "results/strata/flare_tumor_realgt.csv",
        "paper_snapshot.json", "MANIFEST.csv",
        "benchmark/queries.json", "benchmark/relevance.csv", "benchmark/case_scopes.csv",
        "benchmark/phenotype_schema.json", "benchmark/anatomy.json", "benchmark/masking.json",
        "benchmark/ontology_mappings.json", "benchmark/kg_schema.owl", "benchmark/annotation_scopes.json", "benchmark/support_units.json",
        "results/audit/annotation_capability.md",
        "results/audit/random_oakg_0403_vs_0407_audit.csv",
        "results/audit/flare_provenance.md",
        "results/audit/supplement_registry.md",
        "configs/aaai27_paper.yaml", "Makefile", "tests/test_paper_regression.py",
        "results/PAPER_TABLES.md", "results/tables/master_nDCG_table.csv", "results/audit/CHANGELOG.md",
        "notebooks/OAKG_FLARE_Evaluation.ipynb", "notebooks/OAKG_Tutorial_Walkthrough.ipynb",
    ]:
        check_exists(rel)

    print("\n== Entry points import ==")
    for mod in ["oakg.pipeline", "oakg.strata_results", "oakg.build_benchmark",
                "oakg.native_export", "oakg.union_ablation", "oakg.qualitative",
                "oakg.export_benchmark_defs", "oakg.export_supplement_registry",
                "oakg.export_master_table", "oakg.export_paper_tables"]:
        try:
            importlib.import_module(mod)
            _p("OK", f"import {mod}")
        except Exception as e:  # noqa: BLE001
            _p("FAIL", f"import {mod}: {e!r}")

    print("\n== Frozen snapshot integrity (SHA-256) ==")
    snap = json.loads((ROOT / "paper_snapshot.json").read_text())
    for tf in snap.get("tracked_files", []):
        path = ROOT / tf["path"]
        if not path.exists():
            _p("FAIL", f"{tf['path']} — missing")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        _p("OK" if digest == tf["sha256"] else "FAIL",
           f"{tf['path']} — {'hash matches' if digest == tf['sha256'] else 'HASH DRIFT'}")

    print("\n== MANIFEST files present ==")
    man = pd.read_csv(ROOT / "MANIFEST.csv")
    missing = [r.output_file for r in man.itertuples(index=False)
               if not (ROOT / r.output_file).exists()]
    _p("OK" if not missing else "FAIL",
       "all manifest outputs present" if not missing else f"missing: {missing}")

    print("\n== Headline numbers ==")
    ps = pd.read_csv(ROOT / "results/tables/ranking_policy_selection.csv")
    lex = float(ps[ps.method == "OAKG-lexicographic [ref]"].mean_metric.iloc[0])
    ua = pd.read_csv(ROOT / "results/union_ablation/union_ablation_method_summary.csv")
    ur = float(ua[(ua.regime == "random") & (ua.method == "OAKG")]["nDCG@10"].iloc[0])
    _p("OK" if abs(lex - 0.402694) < 1e-4 else "FAIL",
       f"random OAKG lexicographic = {lex:.6f} (expected 0.402694)")
    _p("OK" if abs(lex - ur) < 1e-4 else "FAIL",
       f"main == union-ablation random OAKG ({lex:.6f} vs {ur:.6f})")

    print("\n== Cross-file consistency GATE (OAKG-Lexicographic, ref, all 4 regimes) ==")
    # Every main-pipeline file must report the SAME OAKG-Lexicographic value per
    # regime. Under the all-111 convention these aggregations coincide, so any
    # disagreement means a stale or mis-aggregated file -> FAIL.
    pt = pd.read_csv(ROOT / "results/tables/publication_ready_retrieval_table.csv")
    rs = pd.read_csv(ROOT / "results/tables/retrieval_summary.csv")
    mt = pd.read_csv(ROOT / "results/tables/master_nDCG_table.csv")
    for reg in ["uniform", "random", "asymmetric", "dataset_style"]:
        p = round(float(str(pt[(pt.track == "ref") & (pt.masking_regime == reg)
                  & (pt.method == "OAKG-lexicographic [ref]")]["nDCG@10"].iloc[0]).split()[0]), 3)
        s = round(float(rs[(rs.track == "ref") & (rs.masking_regime == reg)
                  & (rs.method == "OAKG-lexicographic [ref]")]["nDCG@10"].mean()), 3)
        m = round(float(mt[mt.method == "OAKG-Lexicographic"][reg].iloc[0]), 3)
        agree = (p == s == m)
        _p("OK" if agree else "FAIL",
           f"{reg}: publication={p} summary={s} master={m}" + ("" if agree else "  <<< DISAGREE"))

    print("\n== OAKG-Union ablation convention GATE (all-111: ablation OAKG == main-pipeline OAKG) ==")
    # The ablation must score abstained queries under the SAME all-111 rule as the
    # main pipeline (oakg.metrics.query_metric_row): a query it cannot serve scores 0.
    # When it does, the ablation OAKG absolute equals the main-pipeline OAKG-Lexicographic
    # per regime. Any drift means the ablation reintroduced a served-only / non-zeroed
    # abstention convention (which previously inflated Δobs in the one-sided regimes).
    ab = pd.read_csv(ROOT / "results/union_ablation/union_ablation_method_summary.csv")
    for reg in ["uniform", "random", "asymmetric", "dataset_style"]:
        a = round(float(ab[(ab.regime == reg) & (ab.method == "OAKG")]["nDCG@10"].iloc[0]), 3)
        m = round(float(mt[mt.method == "OAKG-Lexicographic"][reg].iloc[0]), 3)
        agree = (a == m)
        _p("OK" if agree else "FAIL",
           f"{reg}: ablation OAKG={a} main OAKG-Lex={m}" + ("" if agree else "  <<< CONVENTION DRIFT"))

    print("\n== Supplement-registry sync GATE (§6 nDCG@10 == master, all 4 regimes) ==")
    # The registry §6 table is generated FROM publication_ready, but a stale committed
    # copy can drift. Gate its nDCG@10 against the master so one authoritative value
    # holds in every derived artifact too.
    import re as _re
    reg_ndcg = {}
    for line in (ROOT / "results/audit/supplement_registry.md").read_text().splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) == 7 and cells[1] in ("uniform", "random", "asymmetric", "dataset_style") \
                and all(_re.match(r"^0?\.?\d", cells[i]) for i in (2, 3, 4, 5)):
            reg_ndcg[cells[1]] = round(float(_re.match(r"([0-9.]+)", cells[5]).group(1)), 3)
    for reg in ["uniform", "random", "asymmetric", "dataset_style"]:
        r = reg_ndcg.get(reg)
        m = round(float(mt[mt.method == "OAKG-Lexicographic"][reg].iloc[0]), 3)
        _p("OK" if r == m else "FAIL",
           f"{reg}: registry={r} master={m}" + ("" if r == m else "  <<< STALE REGISTRY"))

    print(f"\n== {ok} passed, {warn} warnings, {fail} failed ==")
    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
