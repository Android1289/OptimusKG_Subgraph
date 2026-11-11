#!/usr/bin/env python3
"""
generate_fyp_report.py : Compiles all GNN training metrics, baseline comparisons, top-K repurposing
breakdowns (% already approved, % clinical trial, % novel repurposed, % with mechanistic paths),
and degree-bias audit numbers into FYP_Experimental_Results_Report.md (and prints to console).

Usage (from C:\\Users\\Abuba\\Desktop\\FYP\\OptimusKG_Subgraph):
    python generate_fyp_report.py
"""
import csv
import glob
import json
import os
import sys
from collections import Counter

import networkx as nx
import pandas as pd

FOLDER = os.environ.get("HYP_FOLDER", "C:/Users/Abuba/Desktop/FYP/OptimusKG_Subgraph/")


def pct_rank_approx(med_rank, n_cand):
    """Percentile rank of the median positive (100% = rank 1, 50% = random)."""
    if not n_cand or n_cand <= 1:
        return 0.0
    return 100.0 * (1.0 - (float(med_rank) - 1.0) / float(n_cand - 1))


def analyze_predictions_csv(csv_path):
    df = pd.read_csv(csv_path)
    total = len(df)
    if total == 0:
        return {}
    known_col = df["already_known_as"].fillna("").astype(str).str.strip()
    mech_col = df["mechanism_summary"].fillna("").astype(str) if "mechanism_summary" in df.columns else pd.Series([""] * total)

    def stats_at_k(k):
        sub_k = known_col.iloc[:k]
        sub_m = mech_col.iloc[:k]
        n_k = len(sub_k)
        strict_c = int((sub_k == "strict").sum())
        trial_c = int((sub_k == "trial-stage").sum())
        lit_c = int((sub_k == "literature").sum())
        known_c = strict_c + trial_c + lit_c
        novel_c = n_k - known_c
        has_direct_gene = int(sub_m.str.contains("shared_genes=").sum())
        has_pathway = int(sub_m.str.contains("pathways\\(").sum())
        has_dis_bridge = int(sub_m.str.contains("disease_bridge=").sum())
        has_any_mech = int((sub_m.str.contains("shared_genes=") | sub_m.str.contains("pathways\\(") | sub_m.str.contains("disease_bridge=")).sum())
        return {
            "k": n_k,
            "strict": strict_c,
            "strict_pct": 100.0 * strict_c / n_k,
            "trial": trial_c,
            "trial_pct": 100.0 * trial_c / n_k,
            "literature": lit_c,
            "lit_pct": 100.0 * lit_c / n_k,
            "known_total": known_c,
            "known_pct": 100.0 * known_c / n_k,
            "novel": novel_c,
            "novel_pct": 100.0 * novel_c / n_k,
            "direct_gene": has_direct_gene,
            "pathway": has_pathway,
            "dis_bridge": has_dis_bridge,
            "explainable": has_any_mech,
            "explainable_pct": 100.0 * has_any_mech / n_k,
        }

    return {
        "df": df,
        "top10": stats_at_k(min(10, total)),
        "top25": stats_at_k(min(25, total)),
        "top50": stats_at_k(min(50, total)),
        "top100": stats_at_k(total),
    }


def main():
    os.chdir(FOLDER if os.path.exists(FOLDER) else ".")
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("# HYPOTHESIS: Experimental Results & Model Evaluation Report")
    out("")
    out("This report is automatically generated from the saved knowledge graph (`.graphml`), evaluation metrics (`results_*.json`), and prediction rankings (`predictions_*.csv`).")
    out("")

    # 1. Graph Summary
    out("## 1. Knowledge Graph Summary (`kg_v14_expanded` vs `kg_v15_mechanisms`)")
    out("")
    out("| Graph Version | Total Nodes | Total Edges | Candidate Drugs | Genes (`GEN`) | Pathways (`PWY`) | Diseases (`DIS`) | Drugs with Target Gene (`DRG-GEN`) |")
    out("|---|---:|---:|---:|---:|---:|---:|---:|")
    for gname in ("kg_v14_expanded.graphml", "kg_v15_mechanisms.graphml"):
        if os.path.exists(gname):
            G = nx.read_graphml(gname)
            nt = Counter(d.get("label", "UNK") for _, d in G.nodes(data=True))
            cands = [n for n, d in G.nodes(data=True) if d.get("label") == "DRG" and d.get("candidate") != "no"]
            with_tgt = sum(
                1 for n in cands
                if any(G.nodes[v].get("label") in ("GEN", "TGT") for _, v in G.out_edges(n))
            )
            pct_tgt = (100.0 * with_tgt / len(cands)) if cands else 0.0
            out(f"| `{gname}` | {G.number_of_nodes():,} | {G.number_of_edges():,} | {len(cands):,} | {nt['GEN']:,} | {nt['PWY']:,} | {nt['DIS']:,} | {with_tgt:,} ({pct_tgt:.1f}%) |")
    out("")

    # 2. Find primary results JSON (prefer kg_v15_mechanisms_neg0p5_cos_seed0.json)
    preferred = [
        "results_kg_v15_mechanisms_neg0p5_cos_seed0.json",
        "results_kg_v15_mechanisms_seed0.json",
        "results_kg_v14_expanded_seed0.json",
    ]
    json_files = [f for f in preferred if os.path.exists(f)]
    for f in sorted(glob.glob("results_*.json")):
        if f not in json_files and "smoke" not in f:
            json_files.append(f)

    if not json_files:
        raise SystemExit("No results_*.json files found. Run train_gnn.py first.")

    out("## 2. Held-Out Validation Accuracy (In-Distribution Disease Link Prediction)")
    out("")
    out("| Run / Model Configuration | Held-Out Positives ($n$) | Candidate Pool ($N$) | MRR | Hits@10 | Hits@50 | Hits@100 | Median Rank | Percentile Rank (Accuracy Proxy) | Gain vs. Popularity MRR | Gain vs. Random H@10 |")
    out("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            res = json.load(f)
        tag = jf.replace("results_", "").replace(".json", "")
        vg = res["validation"]["gnn"]
        vp = res["validation"]["popularity"]
        n_cand = vg["n_candidates"]
        rand_h10 = 10.0 / n_cand
        mrr_gain = vg["MRR"] / vp["MRR"] if vp["MRR"] > 0 else 0.0
        rand_gain = vg["H@10"] / rand_h10 if rand_h10 > 0 else 0.0
        p_rank_g = pct_rank_approx(vg["median_rank"], n_cand)
        p_rank_p = pct_rank_approx(vp["median_rank"], n_cand)
        out(f"| **`{tag}` (GNN)** | {vg['n']} | {n_cand:,} | **{vg['MRR']:.4f}** | **{vg['H@10']*100:.2f}%** | **{vg['H@50']*100:.2f}%** | **{vg['H@100']*100:.2f}%** | **{vg['median_rank']:.1f}** | **{p_rank_g:.2f}%** | **{mrr_gain:.2f}x** | **{rand_gain:.1f}x** |")
        out(f"| `{tag}` (Popularity Baseline) | {vp['n']} | {n_cand:,} | {vp['MRR']:.4f} | {vp['H@10']*100:.2f}% | {vp['H@50']*100:.2f}% | {vp['H@100']*100:.2f}% | {vp['median_rank']:.1f} | {p_rank_p:.2f}% | 1.00x | {(vp['H@10']/rand_h10):.1f}x |")
    out("")

    # 3. Detailed Zero-Shot Test Table for Primary Model
    primary_json = json_files[0]
    with open(primary_json, "r", encoding="utf-8") as f:
        pres = json.load(f)
    ptag = primary_json.replace("results_", "").replace("_seed0.json", "").replace(".json", "")

    out(f"## 3. Zero-Shot Disease Evaluation (`{primary_json}`)")
    out("")
    out("All drug edges for the 12 diseases in the Cutaneous Leishmaniasis, Tuberculosis, and Dengue families were stripped prior to training.")
    out("")
    out("| Disease & Evidence Tier | Known Drugs ($n$) | Model | MRR | Hits@10 | Hits@50 | Hits@100 | Median Rank | Percentile Rank |")
    out("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for split_name, metrics in pres["test"].items():
        g_m = metrics["gnn"]
        p_m = metrics["popularity"]
        nc = g_m["n_candidates"]
        out(f"| **{split_name}** | {g_m['n']} | **GNN** | **{g_m['MRR']:.4f}** | **{g_m['H@10']*100:.1f}%** | **{g_m['H@50']*100:.1f}%** | **{g_m['H@100']*100:.1f}%** | **{g_m['median_rank']:.1f}** | **{pct_rank_approx(g_m['median_rank'], nc):.1f}%** |")
        out(f"| {split_name} | {p_m['n']} | Popularity | {p_m['MRR']:.4f} | {p_m['H@10']*100:.1f}% | {p_m['H@50']*100:.1f}% | {p_m['H@100']*100:.1f}% | {p_m['median_rank']:.1f} | {pct_rank_approx(p_m['median_rank'], nc):.1f}% |")
    out("")

    # 4. Degree-Score Correlation Audit
    if "degree_score_correlation" in pres:
        out("## 4. Degree-Bias Audit (Correlation Between Training Degree and Prediction Score)")
        out("")
        out("| Target Disease | Pearson $r$ | Spearman $\\rho$ | Interpretation |")
        out("|---|---:|---:|---|")
        for dname, corr in pres["degree_score_correlation"].items():
            pr, sr = corr["pearson_r"], corr["spearman_rho"]
            interp = "Low degree bias (mechanistic/structural ranking)" if abs(sr) < 0.35 else "Moderate degree sensitivity"
            out(f"| **{dname}** | `{pr:+.4f}` | `{sr:+.4f}` | {interp} |")
        out("")

    # 5. Breakdown of Top-K Predicted Candidates (% Already Approved vs Trial vs Novel Repurposed & Explainability)
    out("## 5. Composition of Top Predicted Candidates (Rediscovered vs. Novel Repurposed %)")
    out("")
    out("| Disease | Cutoff ($K$) | Rediscovered Approved (`strict`) | Rediscovered Clinical Trial (`trial`) | Rediscovered Literature (`lit`) | **Total Already Known (%)** | **Novel Repurposing Candidates (%)** | **Mechanistically Explainable (%)** |")
    out("|---|---:|---:|---:|---:|---:|---:|---:|")

    disease_csvs = [
        ("Cutaneous Leishmaniasis", f"predictions_cutaneous_leishmaniasis_{ptag}.csv"),
        ("Tuberculosis", f"predictions_tuberculosis_{ptag}.csv"),
        ("Dengue", f"predictions_dengue_{ptag}.csv"),
    ]
    parsed_csvs = {}
    for dname, cpath in disease_csvs:
        if not os.path.exists(cpath):
            continue
        info = analyze_predictions_csv(cpath)
        parsed_csvs[dname] = info
        for tier_key in ("top10", "top25", "top50", "top100"):
            st = info[tier_key]
            out(
                f"| **{dname}** | Top-{st['k']} | "
                f"{st['strict']} ({st['strict_pct']:.0f}%) | "
                f"{st['trial']} ({st['trial_pct']:.0f}%) | "
                f"{st['literature']} ({st['lit_pct']:.0f}%) | "
                f"**{st['known_total']}/{st['k']} ({st['known_pct']:.1f}%)** | "
                f"**{st['novel']}/{st['k']} ({st['novel_pct']:.1f}%)** | "
                f"**{st['explainable']}/{st['k']} ({st['explainable_pct']:.1f}%)** |"
            )
    out("")

    # 6. Top-15 Deduplicated Predictions Table per Disease
    out("## 6. Top-15 Predicted Small-Molecule Candidates per Target Disease")
    out("")
    for dname, info in parsed_csvs.items():
        out(f"### 6.{list(parsed_csvs.keys()).index(dname) + 1} {dname}")
        out("")
        df15 = info["df"].head(15)
        out("| Rank | ChEMBL ID | Drug Name | Score | Train Deg | Known Status | Mechanistic Evidence Path |")
        out("|---:|---|---|---:|---:|---|---|")
        for _, r in df15.iterrows():
            status = str(r.get("already_known_as", "") if pd.notna(r.get("already_known_as")) else "").strip() or "Novel Candidate"
            mech = str(r.get("mechanism_summary", "") if pd.notna(r.get("mechanism_summary")) else "").strip()
            tdeg = int(r["train_deg"]) if "train_deg" in r and pd.notna(r["train_deg"]) else "-"
            out(f"| {int(r['rank'])} | `{r['drug_id']}` | **{r['drug_name']}** | {float(r['score']):.4f} | {tdeg} | `{status}` | {mech} |")
        out("")

    report_path = os.path.join(os.getcwd(), "FYP_Experimental_Results_Report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved complete FYP evaluation report to: {report_path}")


if __name__ == "__main__":
    sys.exit(main())