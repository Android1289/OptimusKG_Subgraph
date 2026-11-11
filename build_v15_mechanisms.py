#!/usr/bin/env python3
"""
build_v15_mechanisms.py : Fast offline enrichment of drug-target (DRG-GEN), high-confidence
disease-gene (DIS-GEN), gene-pathway (GEN-PWY), and drug parent (DRG-DRG PARENT) edges from
local edges.parquet and nodes.parquet. Also updates node_features.npz for all GEN nodes
uniformly while preserving your existing DRG (RDKit/ChEMBL) features.

Usage (from C:\\Users\\Abuba\\Desktop\\FYP\\OptimusKG_Subgraph):
    python build_v15_mechanisms.py --in kg_v14_expanded.graphml --out kg_v15_mechanisms.graphml
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import polars as pl

FOLDER = os.environ.get("HYP_FOLDER", "C:/Users/Abuba/Desktop/FYP/OptimusKG_Subgraph/")
MAIN_DISEASES = {"EFO_0005046", "MONDO_0018076", "EFO_0005547"}
# Exclude metabolic/transport/plasma-binding super-hubs (CYP450s, P-gp, Albumin) that link unrelated drugs
NON_TARGET_RELS = {"ENZYME", "TRANSPORTER", "CARRIER"}


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


def run(in_path, out_path, edges_path, nodes_path, feat_path, min_score=0.20, main_min_score=0.10, max_genes_per_dis=50, max_dis_per_gene=120):
    print(f"Loading graph: {in_path}")
    G = nx.read_graphml(in_path)
    n0, e0 = G.number_of_nodes(), G.number_of_edges()
    print(f"  Start: {n0} nodes, {e0} edges")

    drg_nodes = {n for n, d in G.nodes(data=True) if d.get("label") == "DRG"}
    dis_nodes = {n for n, d in G.nodes(data=True) if d.get("label") == "DIS"}
    gen_nodes = {n for n, d in G.nodes(data=True) if d.get("label") == "GEN"}
    pwy_nodes = {n for n, d in G.nodes(data=True) if d.get("label") == "PWY"}

    print(f"Reading {edges_path} ...")
    edges_df = pl.read_parquet(edges_path)

    # 1. Extract pharmacological DRG-GEN edges for our drugs
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

    # 2. Extract high-confidence DIS-GEN edges connecting our 850 diseases to target_genes (or existing gen_nodes)
    allowed_genes = target_genes | gen_nodes
    dis_g_df = edges_df.filter(
        (pl.col("label") == "DIS-GEN")
        & (
            (pl.col("from").is_in(dis_nodes) & pl.col("to").is_in(allowed_genes))
            | (pl.col("to").is_in(dis_nodes) & pl.col("from").is_in(allowed_genes))
        )
    )

    per_dis_candidates = defaultdict(list)
    for row in dis_g_df.iter_rows(named=True):
        u, v = row["from"], row["to"]
        dis_id, g_id = (u, v) if u in dis_nodes else (v, u)
        score = 0.0
        props = row.get("properties")
        if props:
            try:
                p = json.loads(props)
                score = float(p.get("evidence_score") or p.get("disgenet_score") or 0.0)
            except Exception:
                score = 0.0
        thresh = main_min_score if dis_id in MAIN_DISEASES else min_score
        if score >= thresh:
            per_dis_candidates[dis_id].append((score, g_id))

    # Keep top-K highest-scoring genes per disease (keep up to 250 for main test diseases)
    selected_dis_gen = []
    gene_dis_count = Counter()
    for dis_id, items in per_dis_candidates.items():
        items.sort(reverse=True)
        cap = 250 if dis_id in MAIN_DISEASES else max_genes_per_dis
        for score, g_id in items[:cap]:
            selected_dis_gen.append((dis_id, g_id, score))
            gene_dis_count[g_id] += 1

    # Filter out extreme pleiotropic super-hubs across the 850 diseases, EXCEPT always keep edges on MAIN_DISEASES
    filtered_dis_gen = [
        (dis_id, g_id, score)
        for dis_id, g_id, score in selected_dis_gen
        if gene_dis_count[g_id] <= max_dis_per_gene or dis_id in MAIN_DISEASES
    ]
    bridge_genes = {g_id for _, g_id, _ in filtered_dis_gen} | gen_nodes

    # Also keep target genes that are targeted by >= 2 drugs in our graph so drug-drug shared-target paths exist
    drug_counts_per_gene = Counter(g_id for _, g_id, _ in drg_gen_pairs)
    multi_drug_genes = {g_id for g_id, c in drug_counts_per_gene.items() if c >= 2}
    genes_to_keep = bridge_genes | multi_drug_genes
    new_gene_ids = genes_to_keep - gen_nodes

    # 3. Load gene names and Reactome pathway xrefs from nodes.parquet
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

    # Add new gene nodes
    added_genes = 0
    for gid in sorted(new_gene_ids):
        short_name, desc = gene_meta.get(gid, (gid, gid))
        G.add_node(gid, label="GEN", name=short_name, description=desc, source="OptimusKG_mechanisms")
        added_genes += 1

    # Add DRG-GEN edges onto kept genes
    added_drg_gen = 0
    drg_gen_rel_counts = Counter()
    for d_id, g_id, rel in drg_gen_pairs:
        if g_id not in genes_to_keep:
            continue
        if not G.has_edge(d_id, g_id):
            G.add_edge(d_id, g_id, label="DRG-GEN", relation=rel, source="OptimusKG_mechanisms")
            added_drg_gen += 1
            drg_gen_rel_counts[rel] += 1

    # Add DIS-GEN edges
    added_dis_gen = 0
    for dis_id, g_id, score in filtered_dis_gen:
        if g_id not in genes_to_keep:
            continue
        if not G.has_edge(dis_id, g_id) and not G.has_edge(g_id, dis_id):
            G.add_edge(dis_id, g_id, label="DIS-GEN", relation="ASSOCIATED_WITH", score=f"{score:.4f}", source="OptimusKG_mechanisms")
            added_dis_gen += 1

    # Add GEN-PWY edges to existing Reactome pathways in the graph
    added_gen_pwy = 0
    for gid, pwys in gene_reactome.items():
        if gid not in G:
            continue
        for pid in pwys:
            if pid in pwy_nodes and not G.has_edge(gid, pid):
                G.add_edge(gid, pid, label="GEN-PWY", relation="PARTICIPATES_IN", source="OptimusKG_mechanisms")
                added_gen_pwy += 1

    # 4. Add DRG-DRG PARENT edges (links salts/prodrugs to parent molecules)
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

    print(f"\nAdded to graph:")
    print(f"  + {added_genes} new GEN nodes (total GEN: {len(gen_nodes) + added_genes})")
    print(f"  + {added_drg_gen} DRG-GEN edges {dict(drg_gen_rel_counts)}")
    print(f"  + {added_dis_gen} high-confidence DIS-GEN edges")
    print(f"  + {added_gen_pwy} GEN-PWY edges onto existing Reactome pathways")
    print(f"  + {added_drg_parent} DRG-DRG (PARENT) salt/parent edges")

    for did in ("EFO_0005046", "MONDO_0018076", "EFO_0005547"):
        if did in G:
            g_nbrs = sum(1 for _, v in G.out_edges(did) if G.nodes[v].get("label") == "GEN") + sum(
                1 for u, _ in G.in_edges(did) if G.nodes[u].get("label") == "GEN"
            )
            print(f"  Target disease {did} ({G.nodes[did].get('name')}): now has {g_nbrs} GEN edges")

    drugs_with_targets = sum(
        1
        for n in drg_nodes
        if G.nodes[n].get("candidate") != "no"
        and any(G.nodes[v].get("label") in ("GEN", "TGT") for _, v in G.out_edges(n))
    )
    print(f"  Candidate drugs with >= 1 target gene/protein edge: {drugs_with_targets} / {sum(1 for n in drg_nodes if G.nodes[n].get('candidate') != 'no')}")

    clean_graphml_attrs(G)
    nx.write_graphml(G, out_path)
    print(f"Saved {out_path} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")

    # 5. Update node_features.npz in-place for all GEN nodes uniformly so build_features.py does not need to re-fetch ChEMBL
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
    ap = argparse.ArgumentParser(description="Add local DRG-GEN, DIS-GEN, and DRG-DRG PARENT mechanisms from OptimusKG parquet files")
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