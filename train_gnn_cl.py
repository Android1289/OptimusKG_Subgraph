#!/usr/bin/env python3
"""
train_gnn.py : train and evaluate the HYPOTHESIS drug-repurposing GNN (heterogeneous GraphSAGE).

    python train_gnn.py --smoke                       # 1) FIRST: tiny synthetic run, checks PyG works (seconds)
    python train_gnn.py                               # 2) newest kg_v15_mechanisms / kg_v14_expanded + node_features.npz
    python train_gnn.py --graph kg_v14_expanded.graphml --epochs 200 --seed 0

Needs node_features.npz built from the SAME graph (python build_features.py).

Setup (decided with the supervisor-facing plan)
  * Positives (training labels): drug -> disease edges INDICATION and OFF_LABEL_USE.
  * Zero-shot test diseases: Cutaneous Leishmaniasis, Dengue, Tuberculosis. The WHOLE disease family
    (every disease whose name contains tubercul / leishman / dengue, e.g. hepatic tuberculosis,
    mucocutaneous leishmaniasis) loses all its drug edges, so near-duplicate diseases cannot leak answers.
  * GSK screening compounds (candidate=no) are dropped. Every other drug is a candidate.
  * Validation = 5% of the remaining positives, used only to pick the best epoch.
  * Test = rank every candidate drug for each test disease; known drugs of that disease are scored with
    "filtered" ranks (other known positives are not counted as mistakes).
  * Two reference points are always printed: a POPULARITY baseline (rank drugs by how many diseases they
    treat in training) and the random expectation. A GNN that does not beat popularity has learned nothing.
Outputs (in the graph folder): results_<tag>_seed<N>.json, predictions_<disease>_<tag>.csv, best_model_<tag>.pt
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import tempfile
import time
from collections import defaultdict

import networkx as nx
import numpy as np

FOLDER = os.environ.get("HYP_FOLDER", "C:/Users/Abuba/Desktop/FYP/OptimusKG_Subgraph/")
NEWEST_FIRST = ["kg_v15_mechanisms", "kg_v14_expanded"]
MAIN = {"EFO_0005046": "Cutaneous Leishmaniasis", "MONDO_0018076": "Tuberculosis", "EFO_0005547": "Dengue"}
FAMILY_RE = re.compile(r"tubercul|leishman|dengue", re.I)
TREATS = {"INDICATION", "OFF_LABEL_USE"}
REL_RENAME = {"CONTRAINDICATION": "contraindicated", "CLINICAL_TRIAL": "trial", "LITERATURE_CANDIDATE": "literature"}
TOP_K_CSV = 100


# ----------------------------------------------------------------------------
# data preparation (numpy only)
# ----------------------------------------------------------------------------
def clean(s):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]+", "_", str(s).lower())).strip("_") or "edge"


def prepare(G, feats, drop_rels=()):
    label = {n: d.get("label", "UNK") for n, d in G.nodes(data=True)}
    dropped = {n for n, d in G.nodes(data=True) if d.get("candidate") == "no"}
    ids, index, X, names = {}, {}, {}, {}
    for t in sorted({label[n] for n in G if n not in dropped}):
        if f"ids_{t}" not in feats or f"X_{t}" not in feats:
            raise SystemExit(f"node_features.npz has no features for node type {t}: rerun build_features.py")
        all_ids = [str(i) for i in feats[f"ids_{t}"]]
        keep = [k for k, i in enumerate(all_ids) if i not in dropped]
        ids[t] = [all_ids[k] for k in keep]
        X[t] = np.asarray(feats[f"X_{t}"], dtype=np.float32)[keep]
        index[t] = {i: k for k, i in enumerate(ids[t])}
        names[t] = [str(G.nodes[i].get("name") or "") if i in G else "" for i in ids[t]]
        graph_ids = {n for n in G if label[n] == t and n not in dropped}
        if set(ids[t]) != graph_ids:
            raise SystemExit(f"node_features.npz does not match this graph for type {t} "
                             f"({len(ids[t])} vs {len(graph_ids)} nodes): rerun  python build_features.py")
    family = {n for n, d in G.nodes(data=True) if label[n] == "DIS" and (
        n in MAIN or FAMILY_RE.search(str(d.get("name") or "")))}
    static, treats = defaultdict(list), []
    heldout = defaultdict(lambda: defaultdict(set))          # disease id -> RELATION -> {drug idx}
    for u, v, d in G.edges(data=True):
        if u in dropped or v in dropped:
            continue
        tu, tv = label[u], label[v]
        rel = str(d.get("relation") or d.get("label") or "edge").upper()
        if tu == "DRG" and tv == "DIS" and v in family:
            heldout[v][rel].add(index["DRG"][u])
            continue
        if tu == "DRG" and tv == "DIS" and rel in TREATS:
            treats.append((index["DRG"][u], index["DIS"][v]))
            continue
        if clean(rel) in drop_rels:
            continue
        static[(tu, REL_RENAME.get(rel, clean(rel)), tv)].append((index[tu][u], index[tv][v]))
    static_np = {}
    for (tu, r, tv), pairs in static.items():
        a = np.unique(np.array(pairs, dtype=np.int64), axis=0).T
        static_np[(tu, r, tv)] = a
        static_np[(tv, "rev_" + r, tu)] = a[::-1].copy()
    pos = np.unique(np.array(treats, dtype=np.int64), axis=0).T if treats else np.zeros((2, 0), dtype=np.int64)
    main = {k: v for k, v in MAIN.items() if k in index["DIS"]}
    return {"ids": ids, "index": index, "X": X, "names": names, "static": static_np, "pos": pos,
            "family": family, "heldout": heldout, "main": main, "n_drug": len(ids["DRG"]),
            "n_dis": len(ids["DIS"]), "graph_labels": label}


def describe(D, G):
    print("\nNodes kept:", {t: len(v) for t, v in D["ids"].items()})
    print("Edge types (with reverse):", len(D["static"]), "| training positives (treats):", D["pos"].shape[1])
    print(f"Candidate drugs: {D['n_drug']} | random chance of a top-10 hit: {10 / D['n_drug']:.2%}")
    print(f"\nTest family: {len(D['family'])} diseases lose ALL drug edges (so they cannot leak answers):")
    for n in sorted(D["family"], key=lambda x: str(G.nodes[x].get("name"))):
        removed = sum(len(s) for s in D["heldout"].get(n, {}).values())
        print(f"   {n:16} {str(G.nodes[n].get('name'))[:48]:48} drug edges removed: {removed}")
    for did, name in D["main"].items():
        h = D["heldout"].get(did, {})
        print(f"   TEST {name}: strict {len(h.get('INDICATION', set()) | h.get('OFF_LABEL_USE', set()))}, "
              f"trial {len(h.get('CLINICAL_TRIAL', set()))}, literature {len(h.get('LITERATURE_CANDIDATE', set()))}")


def split_positives(pos, val_frac, rng):
    perm = rng.permutation(pos.shape[1])
    n_val = max(1, int(val_frac * pos.shape[1]))
    return pos[:, perm[n_val:]], pos[:, perm[:n_val]]


def known_sets(pos_all):
    known = defaultdict(set)
    for d, s in pos_all.T:
        known[int(s)].add(int(d))
    return known


def rank_of(scores, target, exclude):
    """Filtered rank with average handling of ties (1 = best)."""
    s = scores.astype(np.float64).copy()
    if exclude:
        s[np.fromiter(exclude, dtype=np.int64)] = -np.inf
    st = s[target]
    return 1.0 + float((s > st).sum()) + 0.5 * float((s == st).sum() - 1)


def summarize(ranks, n_cand):
    r = np.asarray(ranks, dtype=float)
    return {"n": int(len(r)), "MRR": float((1 / r).mean()), "H@10": float((r <= 10).mean()),
            "H@50": float((r <= 50).mean()), "H@100": float((r <= 100).mean()),
            "median_rank": float(np.median(r)), "n_candidates": int(n_cand)}


def popularity(pos_train, n_drug):
    return np.bincount(pos_train[0], minlength=n_drug).astype(np.float64)


def eval_val(zd, zs, val, known, pop):
    rm, rp = [], []
    for d, s in val.T:
        d, s = int(d), int(s)
        ex = known[s] - {d}
        rm.append(rank_of(zd @ zs[s], d, ex))
        rp.append(rank_of(pop, d, ex))
    return summarize(rm, zd.shape[0]), summarize(rp, zd.shape[0])


def test_sets(D, did):
    h = D["heldout"].get(did, {})
    sets = {"strict (INDICATION/OFF_LABEL)": h.get("INDICATION", set()) | h.get("OFF_LABEL_USE", set()),
            "trial-stage": set(h.get("CLINICAL_TRIAL", set())),
            "literature candidates": set(h.get("LITERATURE_CANDIDATE", set()))}
    known_all = set().union(*sets.values())
    return sets, known_all


def eval_test(zd, zs, D, pop):
    out = {}
    for did, name in D["main"].items():
        s = D["index"]["DIS"][did]
        scores = zd @ zs[s]
        sets, known_all = test_sets(D, did)
        for k, ds in sets.items():
            if not ds:
                continue
            rm = [rank_of(scores, d, known_all - {d}) for d in ds]
            rp = [rank_of(pop, d, known_all - {d}) for d in ds]
            out[f"{name} | {k}"] = {"gnn": summarize(rm, zd.shape[0]), "popularity": summarize(rp, zd.shape[0])}
    return out


def print_table(title, rows):
    print(f"\n{title}")
    print(f"  {'set':46} {'n':>4} {'MRR':>7} {'H@10':>6} {'H@50':>6} {'H@100':>6} {'med.rank':>9}")
    for name, m in rows:
        print(f"  {name:46} {m['n']:>4} {m['MRR']:>7.3f} {m['H@10']:>6.2f} {m['H@50']:>6.2f} "
              f"{m['H@100']:>6.2f} {m['median_rank']:>9.1f}")


def write_predictions(zd, zs, D, G, outdir, tag):
    for did, name in D["main"].items():
        s = D["index"]["DIS"][did]
        scores = zd @ zs[s]
        sets, _ = test_sets(D, did)
        status = {}
        for k, ds in sets.items():
            for d in ds:
                status.setdefault(d, k.split(" ")[0])
        top = np.argsort(-scores)[:TOP_K_CSV]
        path = os.path.join(outdir, f"predictions_{clean(name)}_{tag}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rank", "drug_id", "drug_name", "score", "already_known_as"])
            for r, d in enumerate(top, 1):
                w.writerow([r, D["ids"]["DRG"][d], D["names"]["DRG"][d], f"{scores[d]:.4f}", status.get(int(d), "")])
        novel = sum(1 for d in top if int(d) not in status)
        print(f"  {path}  ({novel} of the top {TOP_K_CSV} are not already known for {name})")


# ----------------------------------------------------------------------------
# torch part (imports are lazy so the data logic above can be tested without torch)
# ----------------------------------------------------------------------------
def build_model(in_dims, edge_types, hidden, layers, dropout):
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import HeteroConv, SAGEConv

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.ModuleDict({t: nn.Linear(d, hidden) for t, d in in_dims.items()})
            self.convs = nn.ModuleList([
                HeteroConv({et: SAGEConv((hidden, hidden), hidden) for et in edge_types}, aggr="mean")
                for _ in range(layers)])
            self.dropout = dropout

        def forward(self, x_dict, edge_index_dict):
            h = {t: F.relu(self.proj[t](x)) for t, x in x_dict.items()}
            for i, conv in enumerate(self.convs):
                out = conv(h, edge_index_dict)
                last = i == len(self.convs) - 1
                h = {t: ((out[t] if last else F.relu(out[t])) + h[t]) if t in out else h[t] for t in h}
                h = {t: F.dropout(v, p=self.dropout, training=self.training) for t, v in h.items()}
            return h

    return Net()


def run_training(D, args, outdir, tag):
    import copy
    import torch
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    pos_train, pos_val = split_positives(D["pos"], args.val_frac, rng)
    known = known_sets(D["pos"])
    pop = popularity(pos_train, D["n_drug"])
    print(f"Positives: {pos_train.shape[1]} train | {pos_val.shape[1]} validation")

    x = {t: torch.from_numpy(a).to(device) for t, a in D["X"].items()}
    static = {et: torch.from_numpy(a).long().to(device) for et, a in D["static"].items()}
    treats_t, rev_treats_t = ("DRG", "treats", "DIS"), ("DIS", "rev_treats", "DRG")
    edge_types = list(static.keys()) + [treats_t, rev_treats_t]
    model = build_model({t: a.shape[1] for t, a in D["X"].items()}, edge_types,
                        args.hidden, args.layers, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    print(f"Edge types: {len(edge_types)} | parameters: {sum(p.numel() for p in model.parameters()):,}")

    def edges_with(treat_edges):
        ei = dict(static)
        t = torch.from_numpy(treat_edges).long().to(device)
        ei[treats_t], ei[rev_treats_t] = t, t.flip(0)
        return ei

    def embed_eval():
        model.eval()
        with torch.no_grad():
            z = model(x, edges_with(pos_train))
        return z["DRG"].cpu().numpy(), z["DIS"].cpu().numpy()

    best, best_state, bad, t0 = -1.0, None, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(pos_train.shape[1])
        k = int(args.mp_frac * len(perm))
        mp, sup = pos_train[:, perm[:k]], pos_train[:, perm[k:]]
        if sup.shape[1] > args.sup_batch:
            sup = sup[:, rng.choice(sup.shape[1], args.sup_batch, replace=False)]
        z = model(x, edges_with(mp))
        sd = torch.from_numpy(sup[0]).long().to(device)
        ss = torch.from_numpy(sup[1]).long().to(device)
        nd = torch.randint(0, D["n_drug"], (sup.shape[1], args.neg), device=device)     # corrupt the drug
        pos_score = (z["DRG"][sd] * z["DIS"][ss]).sum(-1, keepdim=True)
        neg_score = (z["DRG"][nd] * z["DIS"][ss].unsqueeze(1)).sum(-1)
        logits = torch.cat([pos_score, neg_score], dim=1)
        loss = F.cross_entropy(logits, torch.zeros(len(sd), dtype=torch.long, device=device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            zd, zs = embed_eval()
            mv, mp_ = eval_val(zd, zs, pos_val, known, pop)
            print(f"epoch {epoch:4d} | loss {loss.item():.4f} | val MRR {mv['MRR']:.4f} H@10 {mv['H@10']:.3f} "
                  f"(popularity MRR {mp_['MRR']:.4f}) | {time.time() - t0:.0f}s")
            if mv["MRR"] > best:
                best, bad = mv["MRR"], 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
                if bad >= args.patience:
                    print("early stopping")
                    break
    model.load_state_dict(best_state)
    zd, zs = embed_eval()
    return model, zd, zs, pos_val, pos_train, known, pop, best_state


# ----------------------------------------------------------------------------
def synthetic(seed=0):
    """Tiny fake graph + features with the same structure as the real one (for --smoke)."""
    rnd = random.Random(seed)
    G, feats = nx.DiGraph(), {}
    dis = [("EFO_0005046", "cutaneous leishmaniasis"), ("MONDO_0018076", "tuberculosis"),
           ("EFO_0005547", "dengue disease"), ("EFO_X1", "hepatic tuberculosis")] + \
          [(f"DIS{i}", f"disease {i}") for i in range(40)]
    drugs = [f"CHEMBL{i}" for i in range(1, 121)]
    genes = [f"ENSG{i}" for i in range(20)]
    pwys = [f"R-HSA-{i}" for i in range(5)]
    for n, nm in dis:
        G.add_node(n, label="DIS", name=nm)
    for n in drugs:
        G.add_node(n, label="DRG", name=f"drug {n}", candidate="yes")
    G.add_node("TCMDC-1", label="DRG", name="gsk", candidate="no")
    for n in genes:
        G.add_node(n, label="GEN", name=n)
    for n in pwys:
        G.add_node(n, label="PWY", name=n)
    for d in drugs:
        for s in rnd.sample([n for n, _ in dis[4:]], rnd.randint(1, 5)):
            G.add_edge(d, s, relation="INDICATION", label="DRG-DIS")
    for s in ("EFO_0005046", "MONDO_0018076", "EFO_0005547", "EFO_X1"):
        for d in rnd.sample(drugs, 8):
            G.add_edge(d, s, relation="INDICATION", label="DRG-DIS")
        for d in rnd.sample(drugs, 3):
            G.add_edge(d, s, relation="CLINICAL_TRIAL", label="DRG-DIS")
    G.add_edge("TCMDC-1", "MONDO_0018076", relation="ACTIVE_AGAINST", label="DRG-DIS")
    for n, _ in dis:
        for g in rnd.sample(genes, 3):
            G.add_edge(n, g, relation="ASSOCIATED_WITH", label="DIS-GEN")
    for g in genes:
        G.add_edge(g, rnd.choice(pwys), relation="PARTICIPATES_IN", label="GEN-PWY")
    for d in rnd.sample(drugs, 20):
        G.add_edge(d, rnd.choice(genes), relation="TARGETS", label="DRG-GEN")
    by_type = defaultdict(list)
    for n, d in G.nodes(data=True):
        by_type[d["label"]].append(n)
    nprng = np.random.default_rng(seed)
    for t, nodes in by_type.items():
        feats[f"ids_{t}"] = np.array(nodes)
        feats[f"X_{t}"] = nprng.normal(size=(len(nodes), 9 if t == "DRG" else 8)).astype(np.float32)
    return G, feats


def main():
    ap = argparse.ArgumentParser(description="Train/evaluate the HYPOTHESIS GNN")
    ap.add_argument("--graph")
    ap.add_argument("--features")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--neg", type=int, default=32, help="negative drugs per positive")
    ap.add_argument("--mp-frac", type=float, default=0.7, help="share of training positives used as message edges")
    ap.add_argument("--sup-batch", type=int, default=4096, help="positives scored per epoch")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop-rel", default="", help="comma list of relations to remove (ablation), e.g. participates_in,targets")
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic run to test the setup")
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.smoke:
        G, feats = synthetic(args.seed)
        args.epochs, args.hidden, args.eval_every, args.sup_batch, args.neg = 4, 16, 2, 256, 8
        outdir, tag = tempfile.mkdtemp(), "smoke"
        print("SMOKE TEST: synthetic graph, nothing is written to your folder")
    else:
        path = args.graph or next((FOLDER + n + ".graphml" for n in NEWEST_FIRST
                                   if os.path.exists(FOLDER + n + ".graphml")), None)
        if not path:
            raise SystemExit("no graph found: pass --graph kg_v14_expanded.graphml")
        feat_path = args.features or os.path.join(os.path.dirname(os.path.abspath(path)), "node_features.npz")
        print(f"Graph: {path}\nFeatures: {feat_path}")
        G = nx.read_graphml(path)
        feats = dict(np.load(feat_path, allow_pickle=False))
        outdir, tag = os.path.dirname(os.path.abspath(path)), os.path.splitext(os.path.basename(path))[0]

    drop = {clean(r) for r in args.drop_rel.split(",") if r.strip()}
    D = prepare(G, feats, drop)
    describe(D, G)
    if D["pos"].shape[1] < 50 and not args.smoke:
        raise SystemExit("too few training positives, check the graph")

    model, zd, zs, pos_val, pos_train, known, pop, state = run_training(D, args, outdir, tag)
    mv, mp_ = eval_val(zd, zs, pos_val, known, pop)
    print_table("VALIDATION (random held-out positives of non-test diseases)",
                [("GNN", mv), ("popularity baseline", mp_)])
    test = eval_test(zd, zs, D, pop)
    rows = []
    for k, v in test.items():
        rows += [(f"{k} [GNN]", v["gnn"]), (f"{k} [popularity]", v["popularity"])]
    print_table(f"ZERO-SHOT TEST (rank among {D['n_drug']} candidate drugs; random H@10 = {10 / D['n_drug']:.3f})", rows)
    print("\nTop-100 predictions per test disease:")
    write_predictions(zd, zs, D, G, outdir, tag)

    import torch
    torch.save(state, os.path.join(outdir, f"best_model_{tag}.pt"))
    with open(os.path.join(outdir, f"results_{tag}_seed{args.seed}.json"), "w") as f:
        json.dump({"args": vars(args), "validation": {"gnn": mv, "popularity": mp_}, "test": test,
                   "family": sorted(D["family"])}, f, indent=2)
    print(f"\nSaved results_{tag}_seed{args.seed}.json and best_model_{tag}.pt in {outdir}")
    if args.smoke:
        print("SMOKE TEST OK: the pipeline runs end to end. Now run it without --smoke.")


if __name__ == "__main__":
    sys.exit(main())