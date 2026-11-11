#!/usr/bin/env python3
"""
build_features.py : numeric node features for the HYPOTHESIS graph (needed for the inductive GNN).

    python build_features.py                 # newest kg_v*.graphml in the folder
    python build_features.py some.graphml

Writes node_features.npz next to the graph with, for every node type T:
    ids_T  (node ids, in order)   X_T  (float32 feature matrix)
Drugs      : 1024-bit Morgan fingerprint of the structure + 1 flag (1 = structure known).
             SMILES come from the graph (GSK compounds) or from ChEMBL (CHEMBL ids).
             Without RDKit it falls back to hashed SMILES character n-grams (weaker, warns).
Everything else (diseases, genes, pathways, phenotypes, targets): hashed character n-grams of the
             name (+ description) = 256 dims. No model download needed; swap in a biomedical text
             encoder later if you want semantic similarity.
"""
import os
import sys
import time

import networkx as nx
import numpy as np
import requests

FOLDER = os.environ.get("HYP_FOLDER", "C:/Users/Abuba/Desktop/FYP/OptimusKG_Subgraph/")
NEWEST_FIRST = ["kg_v14_expanded", "kg_v13_pathways", "kg_v12_mechanisms", "kg_v11_clean", "kg_v10_drugs_fixed"]
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
FP_BITS, TEXT_DIMS = 1024, 256


def fetch_json(url, params=None, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30, headers={"Accept": "application/json"})
            if r.status_code == 200:
                return r.json()
            if 400 <= r.status_code < 500:
                return None
        except Exception:
            pass
        time.sleep(3 * (attempt + 1))
    return None


def chembl_smiles(ids):
    out, ids = {}, list(ids)
    for i in range(0, len(ids), 50):
        data = fetch_json(f"{CHEMBL}/molecule.json", {"molecule_chembl_id__in": ",".join(ids[i:i + 50]), "limit": 100})
        for m in (data or {}).get("molecules", []):
            smi = (m.get("molecule_structures") or {}).get("canonical_smiles")
            if smi:
                out[m["molecule_chembl_id"]] = smi
        time.sleep(0.3)
    return out


def make_fingerprinter():
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdFingerprintGenerator
        RDLogger.DisableLog("rdApp.*")
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)

        def fp(smiles):
            mol = Chem.MolFromSmiles(smiles)
            return None if mol is None else np.array(list(gen.GetFingerprint(mol)), dtype=np.float32)
        return fp, True
    except ImportError:
        from sklearn.feature_extraction.text import HashingVectorizer
        hv = HashingVectorizer(analyzer="char", ngram_range=(2, 4), n_features=FP_BITS, alternate_sign=False, norm="l2")
        print("WARNING: RDKit not installed (python -m pip install rdkit), using hashed SMILES n-grams instead")
        return (lambda smiles: hv.transform([smiles]).toarray()[0].astype(np.float32)), False


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else next(
        (FOLDER + n + ".graphml" for n in NEWEST_FIRST if os.path.exists(FOLDER + n + ".graphml")), None)
    if not path:
        raise SystemExit("no graph file found, pass the path as an argument")
    G = nx.read_graphml(path)
    print(f"Loaded {path}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    from sklearn.feature_extraction.text import HashingVectorizer
    text = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=TEXT_DIMS,
                             alternate_sign=False, norm="l2")
    by_type = {}
    for n, d in G.nodes(data=True):
        by_type.setdefault(d.get("label", "UNK"), []).append(n)

    arrays = {}
    for t, nodes in by_type.items():
        if t == "DRG":
            continue
        docs = [(str(G.nodes[n].get("name") or "") + " " + str(G.nodes[n].get("description") or "")).strip().lower()
                or str(n).lower() for n in nodes]
        arrays[t] = text.transform(docs).toarray().astype(np.float32)
        print(f"  {t}: {len(nodes)} nodes x {TEXT_DIMS} text features")

    drugs = by_type.get("DRG", [])
    smiles = {n: G.nodes[n]["smiles"] for n in drugs if G.nodes[n].get("smiles")}
    need = [n for n in drugs if n not in smiles and str(n).startswith("CHEMBL")]
    print(f"  DRG: {len(drugs)} nodes | SMILES in graph: {len(smiles)} | fetching {len(need)} from ChEMBL")
    smiles.update(chembl_smiles(need))
    fp, has_rdkit = make_fingerprinter()
    X = np.zeros((len(drugs), FP_BITS + 1), dtype=np.float32)
    known = 0
    for i, n in enumerate(drugs):
        vec = fp(smiles[n]) if n in smiles else None
        if vec is not None:
            X[i, :FP_BITS], X[i, FP_BITS] = vec, 1.0
            known += 1
    arrays["DRG"] = X
    print(f"  DRG: structure features for {known} of {len(drugs)} drugs "
          f"({'Morgan fingerprints' if has_rdkit else 'hashed SMILES'}); the rest get zeros + flag 0 "
          "(biologics, vaccines, salts without a SMILES)")

    out = {}
    for t, M in arrays.items():
        out[f"X_{t}"] = M
        out[f"ids_{t}"] = np.array(by_type[t])
    dest = FOLDER + "node_features.npz"
    np.savez_compressed(dest, **out)
    print(f"\nSaved {dest}")


if __name__ == "__main__":
    main()