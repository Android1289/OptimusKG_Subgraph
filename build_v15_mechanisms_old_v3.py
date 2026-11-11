#!/usr/bin/env python3
"""
build_v15_mechanisms.py : Fast offline enrichment + ontology/phase hygiene for HYPOTHESIS KG.

What this script fixes on top of kg_v14_expanded.graphml:
  1) Removes in-subgraph ontology umbrella disease super-hubs (non-test DIS nodes linked to > 80
     of the original 417 genes in kg_v14, such as 'gastrointestinal disease', 'neoplasm', 'cancer',
     'obsolete_infectious disease', 'obsolete_viral disease', 'cardiovascular disease', 'respiratory
     system disease', plus broad symptom buckets 'infection', 'hemorrhage', 'obsolete_pain', 'obsolete_injury')
     and prunes low-score (evidence_score < 0.15) DIS-GEN noise left over from load_kg.py.
  2) Removes Phase <= 1.0 (-1.0, 0.5, 1.0) OpenTargets-only DDI/PK probe edges (11,980 edges such as
     Omeprazole/Midazolam Phase 1 DDI probes) from training diseases, while keeping Phase >= 2.0 and
     DrugCentral edges on specific training diseases so Phase 2/3 drugs receive positive supervision.
     Preserves test-disease (CL, TB, Dengue) ground-truth sets (minus excipients/biologics).
  3) Flags non-therapeutic excipients/diagnostics (Sodium Chloride, Aluminum Hydroxide, Lactose,
     Tuberculin, etc.) AND protein biologics/vaccines without SMILES (Interferons, vaccines, mAbs)
     as candidate='no' so the ranking universe consists of small-molecule therapeutics.
  4) Adds 17,434 pharmacological DRG-GEN edges (excluding metabolic/transporter hubs ENZYME,
     TRANSPORTER, CARRIER), high-confidence DIS-GEN bridge edges, GEN-PWY Reactome links, and
     DRG-DRG (PARENT) salt-to-parent edges from local edges.parquet and nodes.parquet.
  5) Updates node_features.npz in-place for all GEN nodes while preserving RDKit DRG features.

Usage (from C:\\Users\\Abuba\\Desktop\\FYP\\OptimusKG_Subgraph):
    python build_v15_mechanisms.py --in kg_v14_expanded.graphml --out kg_v15_mechanisms.graphml
"""
import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import polars as pl

FOLDER = os.environ.get("HYP_FOLDER", "C:/Users/Abuba/Desktop/FYP/OptimusKG_Subgraph/")
MAIN_DISEASES = {"EFO_0005046", "MONDO_0018076", "EFO_0005547"}
FAMILY_RE = re.compile(r"tubercul|leishman|dengue", re.I)
NON_TARGET_RELS = {"ENZYME", "TRANSPORTER", "CARRIER"}
EXCIPIENT_NAMES = {
    "SODIUM CHLORIDE", "SODIUM CHLORIDE NA 22", "LACTOSE, ANHYDROUS", "LACTOSE",
    "ALUMINUM HYDROXIDE", "TUBERCULIN", "WATER", "STERILE WATER", "DEXTROSE",
    "MANNITOL", "GLYCERIN", "BARIUM SULFATE", "SILVER NITRATE"
}
BIOLOGIC_RE = re.compile(
    r"interferon|vaccine|bcg |immunoglobulin|insulin|filgrastim|sargramostim|"
    r"somatropin|epoetin|darbepoetin|toxoid|coagulation factor|botulinum|[a-z]{3,}mab\b",
    re.I,
)
UMBRELLA_NAMES = {
    "infection", "obsolete_infectious disease", "obsolete_viral disease", "obsolete_bacterial disease",
    "neoplasm", "cancer", "obsolete_carcinoma", "obsolete_adenocarcinoma", "gastrointestinal disease",
    "cardiovascular disease", "respiratory system disease", "obsolete_nervous system disease",
    "obsolete_endocrine system disease", "obsolete_immune system disease", "obsolete_hematologic disease",
    "obsolete_liver disease", "obsolete_kidney disease", "obsolete_mental or behavioural disorder",
    "obsolete_pain", "obsolete_injury", "hemorrhage", "sexually transmitted disease"
}


def text_features_256(texts, dim=256):
    """Hashed character 3-to-5-gram features (L2-normalized), matching build_features.py."""
    X = np.zeros((len(texts), dim), dtype=np.float32)
    for row, raw in enumerate(texts):
        s = f" {str(raw or '').lower().strip()} "
        if len(s) < 3:
            continue
        for n in (3, 4, 5):
            for i in range(len(s) - n + 1):
                gram = s[i : i + n].encode("utf-8")
                h = int.from_bytes(hashlib.blake2b(gram, digest_size=4).digest(), "little")
                idx = h % dim
                sign = 1.0 if (h >> 31) == 0 else -1.0
                X[row, idx] += sign
        norm = np.linalg.norm(X[row])
        if norm > 0:
            X[row] /= norm
    return X


def clean_graphml_attrs(G):
    for _, d in G.nodes(data=True):
        for k in list(d.keys()):
            if d[k] is None or isinstance(d[k], (list, dict, set)):
                d[k] = str(d[k] or "")
    for _, _, d in G.edges(data=True):
        for k in list(d.keys()):
            if d[k] is None or isinstance(d[k], (list, dict, set)):
                d[k] = str(d[k] or "")


def run(in_path, out_path, edges_path, nodes_path, feat_path, min_score=0.20, main_min_score=0.10, max_genes_per_dis=50, max_dis_per_gene=80):
    print(f"Loading graph: {in_path}")
    G = nx.read_graphml(in_path)
    n0, e0 = G.number_of_nodes(), G.number_of_edges()
    print(f"  Start: {n0} nodes, {e0} edges")

    drg_nodes = {n for n, d in G.nodes(data=True) if d.get("label") == "DRG"}
    dis_nodes = {n for n, d in G.nodes(data=True) if d.get("label") == "DIS"}
    gen_nodes = {n for n, d in G.nodes(data=True) if d.get("label") == "GEN"}
    pwy_nodes = {n for n, d in G.nodes(data=True) if d.get("label") == "PWY"}
    test_family = {
        n for n in dis_nodes
        if n in MAIN_DISEASES or FAMILY_RE.search(str(G.nodes[n].get("name") or ""))
    }

    # 1. Flag excipients/diagnostics and structureless protein biologics as candidate='no'
    excip_flagged, bio_flagged = 0, 0
    for n in drg_nodes:
        if G.nodes[n].get("candidate") == "no":
            continue
        nm = str(G.nodes[n].get("name") or "").strip()
        smiles = str(G.nodes[n].get("smiles") or "").strip()
        if nm.upper() in EXCIPIENT_NAMES:
            G.nodes[n]["candidate"] = "no"
            excip_flagged += 1
        elif not smiles and BIOLOGIC_RE.search(nm):
            G.nodes[n]["candidate"] = "no"
            bio_flagged += 1
    print(f"  Flagged {excip_flagged} excipients/diagnostics and {bio_flagged} structureless biologics as candidate='no'")

    # 2. Identify & strip ontology umbrella super-hubs among non-test diseases
    umbrella_dis = set()
    for n in dis_nodes - test_family:
        nm = str(G.nodes[n].get("name") or "").strip().lower()
        gen_deg = sum(1 for _, v in G.out_edges(n) if G.nodes[v].get("label") == "GEN") + sum(
            1 for u, _ in G.in_edges(n) if G.nodes[u].get("label") == "GEN"
        )
        if nm in UMBRELLA_NAMES or gen_deg > 80:
            umbrella_dis.add(n)

    umbrella_edges_removed = 0
    for n in umbrella_dis:
        for u, v in list(G.in_edges(n)) + list(G.out_edges(n)):
            if G.has_edge(u, v):
                G.remove_edge(u, v)
                umbrella_edges_removed += 1
    print(f"  Stripped {umbrella_edges_removed} edges from {len(umbrella_dis)} broad ontology umbrella disease nodes (e.g. gastrointestinal disease, neoplasm, infection)")

    print(f"Reading {edges_path} ...")
    edges_df = pl.read_parquet(edges_path)

    # 3. Drop Phase <= 1.0 OpenTargets-only DDI/PK probe edges on non-test diseases
    ind_df = edges_df.filter(
        (pl.col("label") == "DRG-DIS")
        & (pl.col("relation") == "INDICATION")
        & (pl.col("from").is_in(dis_nodes) | pl.col("to").is_in(dis_nodes))
    )
    phase_lut = {}
    for row in ind_df.iter_rows(named=True):
        u, v = row["from"], row["to"]
        d_id, s_id = (u, v) if u in drg_nodes else (v, u)
        props = row.get("properties")
        if props:
            try:
                p = json.loads(props)
                phase = p.get("highest_clinical_trial_phase")
                srcs = set(p.get("sources", {}).get("direct") or [])
                phase_lut[(d_id, s_id)] = (float(phase) if phase is not None else None, "DRUG_CENTRAL" in srcs)
            except Exception:
                pass

    dropped_phase1 = 0
    for u, v, d in list(G.edges(data=True)):
        if d.get("relation") != "INDICATION" or v in test_family:
            continue
        if (u, v) in phase_lut:
            phase, has_dc = phase_lut[(u, v)]
            if not has_dc and phase is not None and phase < 2.0:
                G.remove_edge(u, v)
                dropped_phase1 += 1
    print(f"  Removed {dropped_phase1} Phase <= 1.0 DDI/PK probe edges from non-test training diseases (Phase >= 2.0 & DrugCentral kept)")

    # 4. Extract pharmacological DRG-GEN edges for our drugs
    dg_df = edges_df.filter(
        (pl.col("label") == "DRG-GEN")
        & (pl.col("from").is_in(drg_nodes) | pl.col("to").is_in(drg_nodes))
        & (~pl.col("relation").is_in(NON_TARGET_RELS))
    )
    drg_gen_pairs = []
    target_genes = set()
    for row in dg_df.iter_rows(named=True):
        u, v, rel = row["from"], row["to"], str(row["relation"] or "TARGETS").upper()
        if u in drg_nodes and v not in drg_nodes:
            d_id, g_id = u, v
        elif v in drg_nodes and u not in drg_nodes:
            d_id, g_id = v, u
        else:
            continue
        drg_gen_pairs.append((d_id, g_id, rel))
        target_genes.add(g_id)
    print(f"  Found {len(drg_gen_pairs)} pharmacological DRG-GEN edges covering {len(target_genes)} target genes")

    # 5. Clean existing low-score DIS-GEN noise & extract high-confidence DIS-GEN edges
    allowed_genes = target_genes | gen_nodes
    dis_g_df = edges_df.filter(
        (pl.col("label") == "DIS-GEN")
        & (
            (pl.col("from").is_in(dis_nodes) & pl.col("to").is_in(allowed_genes))
            | (pl.col("to").is_in(dis_nodes) & pl.col("from").is_in(allowed_genes))
        )
    )

    dis_gen_score_lut = {}
    per_dis_candidates = defaultdict(list)
    for row in dis_g_df.iter_rows(named=True):
        u, v = row["from"], row["to"]
        dis_id, g_id = (u, v) if u in dis_nodes else (v, u)
        if dis_id in umbrella_dis:
            continue
        score = 0.0
        props = row.get("properties")
        if props:
            try:
                p = json.loads(props)
                score = float(p.get("evidence_score") or p.get("disgenet_score") or 0.0)
            except Exception:
                score = 0.0
        dis_gen_score_lut[(dis_id, g_id)] = score
        thresh = main_min_score if dis_id in MAIN_DISEASES else min_score
        if score >= thresh:
            per_dis_candidates[dis_id].append((score, g_id))

    # Prune low-score DIS-GEN edges from initial load_kg.py on non-test diseases
    pruned_dis_gen = 0
    for u, v, d in list(G.edges(data=True)):
        if d.get("label") == "DIS-GEN":
            dis_id, g_id = (u, v) if u in dis_nodes else (v, u)
            if dis_id not in test_family and dis_gen_score_lut.get((dis_id, g_id), 0.0) < 0.15:
                G.remove_edge(u, v)
                pruned_dis_gen += 1
    print(f"  Pruned {pruned_dis_gen} low-confidence (score < 0.15) DIS-GEN edges left over from initial extraction")

    selected_dis_gen = []
    gene_dis_count = Counter()
    for dis_id, items in per_dis_candidates.items():
        items.sort(reverse=True)
        cap = 250 if dis_id in MAIN_DISEASES else max_genes_per_dis
        for score, g_id in items[:cap]:
            selected_dis_gen.append((dis_id, g_id, score))
            gene_dis_count[g_id] += 1

    filtered_dis_gen = [
        (dis_id, g_id, score)
        for dis_id, g_id, score in selected_dis_gen
        if gene_dis_count[g_id] <= max_dis_per_gene or dis_id in MAIN_DISEASES
    ]
    bridge_genes = {g_id for _, g_id, _ in filtered_dis_gen} | gen_nodes

    drug_counts_per_gene = Counter(g_id for _, g_id, _ in drg_gen_pairs)
    multi_drug_genes = {g_id for g_id, c in drug_counts_per_gene.items() if c >= 2}
    genes_to_keep = bridge_genes | multi_drug_genes
    new_gene_ids = genes_to_keep - gen_nodes

    # 6. Load gene names and Reactome pathway xrefs from nodes.parquet
    print(f"Reading {nodes_path} for {len(new_gene_ids)} new genes ...")
    nodes_df = pl.read_parquet(nodes_path).filter(
        (pl.col("label") == "GEN") & pl.col("id").is_in(genes_to_keep)
    )
    gene_meta = {}
    gene_reactome = defaultdict(set)
    for row in nodes_df.iter_rows(named=True):
        gid = row["id"]
        symbol, name = "", ""
        props = row.get("properties")
        if props:
            try:
                p = json.loads(props)
                symbol = str(p.get("symbol") or "")
                name = str(p.get("name") or "")
                for xr in p.get("xrefs") or []:
                    if isinstance(xr, dict) and xr.get("source") == "Reactome" and xr.get("id"):
                        gene_reactome[gid].add(str(xr["id"]))
            except Exception:
                pass
        display = f"{symbol} ({name})".strip() if symbol and name else (symbol or name or gid)
        gene_meta[gid] = (symbol or name or gid, display)

    added_genes = 0
    for gid in sorted(new_gene_ids):
        short_name, desc = gene_meta.get(gid, (gid, gid))
        G.add_node(gid, label="GEN", name=short_name, description=desc, source="OptimusKG_mechanisms")
        added_genes += 1

    added_drg_gen = 0
    for d_id, g_id, rel in drg_gen_pairs:
        if g_id not in genes_to_keep:
            continue
        if not G.has_edge(d_id, g_id):
            G.add_edge(d_id, g_id, label="DRG-GEN", relation=rel, source="OptimusKG_mechanisms")
            added_drg_gen += 1

    added_dis_gen = 0
    for dis_id, g_id, score in filtered_dis_gen:
        if g_id not in genes_to_keep or dis_id in umbrella_dis:
            continue
        if not G.has_edge(dis_id, g_id) and not G.has_edge(g_id, dis_id):
            G.add_edge(dis_id, g_id, label="DIS-GEN", relation="ASSOCIATED_WITH", score=f"{score:.4f}", source="OptimusKG_mechanisms")
            added_dis_gen += 1

    added_gen_pwy = 0
    for gid, pwys in gene_reactome.items():
        if gid not in G:
            continue
        for pid in pwys:
            if pid in pwy_nodes and not G.has_edge(gid, pid):
                G.add_edge(gid, pid, label="GEN-PWY", relation="PARTICIPATES_IN", source="OptimusKG_mechanisms")
                added_gen_pwy += 1

    # 7. Add DRG-DRG PARENT edges (links salts/prodrugs to parent molecules)
    dd_df = edges_df.filter(
        (pl.col("label") == "DRG-DRG")
        & (pl.col("relation") == "PARENT")
        & pl.col("from").is_in(drg_nodes)
        & pl.col("to").is_in(drg_nodes)
    )
    added_drg_parent = 0
    for row in dd_df.iter_rows(named=True):
        u, v = row["from"], row["to"]
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v, label="DRG-DRG", relation="PARENT", source="OptimusKG_mechanisms")
            added_drg_parent += 1

    print(f"\nSummary of changes:")
    print(f"  + {added_genes} new GEN nodes (total GEN: {len(gen_nodes) + added_genes})")
    print(f"  + {added_drg_gen} DRG-GEN edges")
    print(f"  + {added_dis_gen} high-confidence DIS-GEN edges ({pruned_dis_gen} noisy edges pruned)")
    print(f"  + {added_gen_pwy} GEN-PWY edges onto existing Reactome pathways")
    print(f"  + {added_drg_parent} DRG-DRG (PARENT) salt/parent edges")

    for did in ("EFO_0005046", "MONDO_0018076", "EFO_0005547"):
        if did in G:
            g_nbrs = sum(1 for _, v in G.out_edges(did) if G.nodes[v].get("label") == "GEN") + sum(
                1 for u, _ in G.in_edges(did) if G.nodes[u].get("label") == "GEN"
            )
            strict_n = sum(1 for u, _, d in G.in_edges(did, data=True) if G.nodes[u].get("candidate") != "no" and d.get("relation") in ("INDICATION", "OFF_LABEL_USE"))
            trial_n = sum(1 for u, _, d in G.in_edges(did, data=True) if G.nodes[u].get("candidate") != "no" and d.get("relation") == "CLINICAL_TRIAL")
            print(f"  Target {did} ({G.nodes[did].get('name')}): {g_nbrs} GEN edges | strict={strict_n}, trial={trial_n}")

    clean_graphml_attrs(G)
    nx.write_graphml(G, out_path)
    print(f"Saved {out_path} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")

    if os.path.exists(feat_path):
        feats = dict(np.load(feat_path, allow_pickle=False))
        old_gen_X = np.asarray(feats["X_GEN"], dtype=np.float32)
        dim = old_gen_X.shape[1]
        all_gen_ids = sorted(n for n, d in G.nodes(data=True) if d.get("label") == "GEN")
        gen_texts = [
            f"{gid} {G.nodes[gid].get('name') or ''} {G.nodes[gid].get('description') or ''}".strip()
            for gid in all_gen_ids
        ]
        feats["ids_GEN"] = np.array(all_gen_ids)
        feats["X_GEN"] = text_features_256(gen_texts, dim=dim)
        np.savez_compressed(feat_path, **feats)
        print(f"Updated {feat_path} with uniform text features for all {len(all_gen_ids)} GEN nodes (DRG features preserved).")


def main():
    ap = argparse.ArgumentParser(description="Add local DRG-GEN, DIS-GEN, and DRG-DRG PARENT mechanisms + clean umbrella hubs and Phase 1 probes")
    ap.add_argument("--in", dest="infile", default=os.path.join(FOLDER, "kg_v14_expanded.graphml"))
    ap.add_argument("--out", dest="outfile", default=os.path.join(FOLDER, "kg_v15_mechanisms.graphml"))
    ap.add_argument("--edges", default=os.path.join(FOLDER, "edges.parquet"))
    ap.add_argument("--nodes", default=os.path.join(FOLDER, "nodes.parquet"))
    ap.add_argument("--features", default=os.path.join(FOLDER, "node_features.npz"))
    ap.add_argument("--min-score", type=float, default=0.20)
    ap.add_argument("--main-min-score", type=float, default=0.10)
    args = ap.parse_args()
    run(args.infile, args.outfile, args.edges, args.nodes, args.features, args.min_score, args.main_min_score)


if __name__ == "__main__":
    sys.exit(main())