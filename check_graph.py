#!/usr/bin/env python3
"""
check_graph.py : pre-training sanity check for the HYPOTHESIS graph.
    python check_graph.py                 # opens the newest kg_v*.graphml it finds
    python check_graph.py some_file.graphml
"""
import os
import sys
from collections import Counter

import networkx as nx

FOLDER = os.environ.get("HYP_FOLDER", "C:/Users/Abuba/Desktop/FYP/OptimusKG_Subgraph/")
NEWEST_FIRST = ["kg_v13_pathways", "kg_v12_mechanisms", "kg_v11_clean", "kg_v10_drugs_fixed", "kg_v9_pathways", "kg_v8_disgenet", "kg_v7_drugs"]
TARGETS = {"EFO_0005046": "Cutaneous Leishmaniasis", "MONDO_0018076": "Tuberculosis (the 'MDR-TB' node)",
           "EFO_0005547": "Dengue"}
POSITIVE = {"INDICATION", "OFF_LABEL_USE", "CLINICAL_TRIAL"}      # usable as link-prediction positives


def pick_file():
    if len(sys.argv) > 1:
        return sys.argv[1]
    for name in NEWEST_FIRST:
        p = FOLDER + name + ".graphml"
        if os.path.exists(p):
            return p
    raise SystemExit("no graph file found, pass the path as an argument")


def main():
    path = pick_file()
    G = nx.read_graphml(path)
    lab = lambda n: G.nodes[n].get("label", "?")
    print(f"File: {path}\n{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print("Node types:", dict(Counter(lab(n) for n in G)))
    print("Edge types:", dict(Counter(d.get("label", "?") for _, _, d in G.edges(data=True))))
    warnings = []

    # ---- drugs ----
    drugs = [n for n in G if lab(n) == "DRG"]
    by_src = Counter(str(G.nodes[n].get("source", "OptimusKG")) for n in drugs)
    gsk = sum(c for s, c in by_src.items() if "GSK" in s)
    print(f"\nDrug nodes: {len(drugs)} | by source: {dict(by_src)}")
    print(f"  with a SMILES structure: {sum(1 for n in drugs if G.nodes[n].get('smiles'))} | "
          f"with only one edge: {sum(1 for n in drugs if G.degree(n) == 1)}")
    if drugs and gsk / len(drugs) > 0.5:
        warnings.append(f"{gsk} of {len(drugs)} drug nodes ({gsk / len(drugs):.0%}) are GSK screening compounds "
                        "with one edge: keep them out of the candidate / negative-sampling pools")

    # ---- positives ----
    pos = [(u, v, d) for u, v, d in G.edges(data=True)
           if lab(u) == "DRG" and lab(v) == "DIS" and d.get("relation") in POSITIVE]
    rel_counts = Counter(d.get("relation") for u, v, d in G.edges(data=True) if lab(u) == "DRG" and lab(v) == "DIS")
    print(f"\nDrug -> disease edges by relation: {dict(rel_counts)}")
    print(f"Usable positives ({'/'.join(sorted(POSITIVE))}): {len(pos)} over "
          f"{len({v for _, v, _ in pos})} diseases")
    print(f"  of which strict (INDICATION/OFF_LABEL_USE only): "
          f"{sum(1 for _, _, d in pos if d.get('relation') in ('INDICATION', 'OFF_LABEL_USE'))}")
    if rel_counts.get("LITERATURE_CANDIDATE") is None and any(
            str(d.get("source", "")).startswith("curated") and d.get("relation") == "INDICATION"
            for _, _, d in G.edges(data=True)):
        warnings.append("hand-curated edges are still labelled INDICATION: run  python enrich_more.py clean")
    for nid, name in TARGETS.items():
        if nid not in G:
            print(f"  {name}: NOT IN GRAPH")
            continue
        rels = Counter(d.get("relation") for u, _, d in G.in_edges(nid, data=True) if lab(u) == "DRG")
        n_pos = sum(c for r, c in rels.items() if r in POSITIVE)
        genes = sum(1 for _, v in G.out_edges(nid) if lab(v) == "GEN")
        print(f"  {name}: {n_pos} positives | drug edges {dict(rels)} | {genes} gene neighbours")
        if n_pos < 15:
            warnings.append(f"{name} has only {n_pos} positive drug edges: use it for zero-shot evaluation, "
                            "not for training, and expect noisy metrics")

    # ---- can a drug -> target -> pathway -> disease path exist? ----
    cand = [n for n in drugs if G.nodes[n].get("candidate") != "no" and "GSK" not in str(G.nodes[n].get("source"))]
    with_gene = [n for n in cand if any(lab(v) in ("GEN", "TGT") for _, v in G.out_edges(n))]
    print(f"\nCandidate drugs with at least one target gene/protein edge: {len(with_gene)} of {len(cand)} "
          f"(drug->gene/target edges in total: {sum(1 for u, v in G.edges if lab(u) == 'DRG' and lab(v) in ('GEN', 'TGT'))})")
    if cand and len(with_gene) / len(cand) < 0.5:
        warnings.append("most candidate drugs have no target edge, so drug -> target -> pathway -> disease "
                        "paths (and the mechanistic explanations) cannot exist yet: run  python enrich_more.py mechanisms")

    # ---- structure ----
    comps = sorted((len(c) for c in nx.weakly_connected_components(G)), reverse=True)
    loops = nx.number_of_selfloops(G)
    print(f"\nWeakly connected components: {len(comps)} | largest holds {comps[0] / G.number_of_nodes():.1%} of nodes "
          f"| isolated nodes: {sum(1 for n in G if G.degree(n) == 0)} | self-loops: {loops}")
    if loops:
        warnings.append(f"{loops} self-loop edges: remove them before training "
                        "(G.remove_edges_from(nx.selfloop_edges(G)))")

    # ---- features ----
    for l, nm in (("DIS", "disease"), ("GEN", "gene"), ("DRG", "drug")):
        nodes = [n for n in G if lab(n) == l]
        if nodes:
            print(f"{nm} nodes with a name: {sum(1 for n in nodes if G.nodes[n].get('name'))} / {len(nodes)}")
    warnings.append("no numeric node features exist yet (names/IDs only): build them before an inductive model")

    print("\nWARNINGS / TO DO BEFORE TRAINING")
    for w in warnings:
        print("  -", w)


if __name__ == "__main__":
    main()