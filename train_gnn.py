#!/usr/bin/env python3
"""
train_gnn.py : train, evaluate, and audit the HYPOTHESIS drug-repurposing GNN (heterogeneous GraphSAGE).

    python train_gnn.py --smoke                       # 1) FIRST: tiny synthetic run, checks PyG works (seconds)
    python train_gnn.py                               # 2) newest kg_v15_mechanisms / kg_v14_expanded + node_features.npz
    python train_gnn.py --graph kg_v15_mechanisms.graphml --epochs 200 --seed 0
    python train_gnn.py --graph kg_v15_mechanisms.graphml --neg-alpha 0.5 --cosine --seed 0

Built-in validation audit features:
  * Compares GNN against the degree-based popularity baseline on both held-out validation and zero-shot test sets.
  * Computes Pearson r and Spearman rho correlation between drug training degree and GNN prediction score.
  * Deduplicates related salt/stereoisomer formulations (via DRG-DRG PARENT links and base active-ingredient names)
    and writes explicit multi-hop mechanism paths (shared target genes, pathways, and disease bridges) into the CSV.
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
SALT_WORDS = re.compile(
    r"\b(HYDROCHLORIDE|DIHYDROCHLORIDE|SULFATE|PHOSPHATE|ACETATE|SODIUM|POTASSIUM|CALCIUM|"
    r"MAGNESIUM|FUMARATE|MALEATE|MESYLATE|TARTRATE|CITRATE|SUCCINATE|BESYLATE|BROMIDE|"
    r"CHLORIDE|NITRATE|PROPIONATE|VALERATE|ACETONIDE|DIPROPIONATE|BUTYRATE|FUROATE|"
    r"PIVALATE|HEXACETONIDE|ANHYDROUS|MONOHYDRATE|DIHYDRATE|TRIHYDRATE|DISODIUM|FUMARATE)\b",
    re.I,
)
STEREO_PREFIX = re.compile(r"^(DEX|LEVO|ES)(?=[A-Z]{4,})", re.I)
TOP_K_CSV = 100


# ----------------------------------------------------------------------------
# data preparation (numpy only)
# ----------------------------------------------------------------------------
def clean(s):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]+", "_", str(s).lower())).strip("_") or "edge"


def canonical_drug_family(name):
    """Normalizes salt suffixes and enantiomer prefixes (e.g. Esomeprazole -> Omeprazole, Levoketoconazole -> Ketoconazole)."""
    s = str(name or "").upper().strip()
    s = re.sub(r"[,()/+-]", " ", s)
    s = SALT_WORDS.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s in ("ESOMEPRAZOLE", "OMEPRAZOLE"):
        return "OMEPRAZOLE"
    if s in ("DEXLANSOPRAZOLE", "LANSOPRAZOLE"):
        return "LANSOPRAZOLE"
    if s in ("LEVOKETOCONAZOLE", "KETOCONAZOLE"):
        return "KETOCONAZOLE"
    if s in ("ISOTRETINOIN", "ALITRETINOIN", "TRETINOIN", "RETINOL"):
        return "RETINOID_VITAMIN_A"
    if s in ("CHOLECALCIFEROL", "ERGOCALCIFEROL", "CALCITRIOL", "ALFACALCIDOL"):
        return "VITAMIN_D_ANALOG"
    s = STEREO_PREFIX.sub("", s)
    return s or "UNKNOWN"


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
                             f"({len(ids[t])} vs {len(graph_ids)} nodes): rerun build_v15_mechanisms.py or build_features.py")
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


def spearman_corr(x, y):
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    return float((rx @ ry) / denom) if denom > 0 else 0.0


def pearson_corr(x, y):
    vx = np.asarray(x, dtype=np.float64) - np.mean(x)
    vy = np.asarray(y, dtype=np.float64) - np.mean(y)
    denom = np.linalg.norm(vx) * np.linalg.norm(vy)
    return float((vx @ vy) / denom) if denom > 0 else 0.0


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


def explain_drug_disease_path(G, drug_id, dis_id, family_set):
    """Builds a concise human-readable summary of 2-hop target, 3-hop pathway, and disease bridges."""
    d_targets = {v for _, v in G.out_edges(drug_id) if G.nodes[v].get("label") == "GEN"} | {
        u for u, _ in G.in_edges(drug_id) if G.nodes[u].get("label") == "GEN"
    }
    dis_genes = {v for _, v in G.out_edges(dis_id) if G.nodes[v].get("label") == "GEN"} | {
        u for u, _ in G.in_edges(dis_id) if G.nodes[u].get("label") == "GEN"
    }
    direct_genes = sorted(str(G.nodes[g].get("name") or g) for g in (d_targets & dis_genes))

    shared_pwys = set()
    for g1 in d_targets:
        p1 = {v for _, v in G.out_edges(g1) if G.nodes[v].get("label") == "PWY"}
        for g2 in dis_genes:
            p2 = {v for _, v in G.out_edges(g2) if G.nodes[v].get("label") == "PWY"}
            for p in p1 & p2:
                shared_pwys.add(f"{G.nodes[g1].get('name', g1)}->{G.nodes[p].get('name', p)[:28]}<-{G.nodes[g2].get('name', g2)}")

    best_dis_bridges = []
    for _, od in G.out_edges(drug_id):
        if G.nodes[od].get("label") == "DIS" and od not in family_set:
            od_genes = {v for _, v in G.out_edges(od) if G.nodes[v].get("label") == "GEN"} | {
                u for u, _ in G.in_edges(od) if G.nodes[u].get("label") == "GEN"
            }
            ov = od_genes & dis_genes
            if ov:
                best_dis_bridges.append((len(ov), str(G.nodes[od].get("name") or od)))
    best_dis_bridges.sort(reverse=True)

    parts = [f"n_targets={len(d_targets)}"]
    if direct_genes:
        parts.append(f"shared_genes={','.join(direct_genes[:4])}")
    if shared_pwys:
        parts.append(f"pathways({len(shared_pwys)})={'; '.join(sorted(shared_pwys)[:2])}")
    if best_dis_bridges:
        top_b = [f"{nm[:24]}({c}g)" for c, nm in best_dis_bridges[:2]]
        parts.append(f"disease_bridge={', '.join(top_b)}")
    return " | ".join(parts)


def write_predictions(zd, zs, D, G, outdir, tag, pop):
    # Build parent connected components from DRG-DRG PARENT edges so salts/prodrugs map to one group
    parent_uf = {n: n for n in D["ids"]["DRG"]}

    def find_root(x):
        while parent_uf[x] != x:
            parent_uf[x] = parent_uf[parent_uf[x]]
            x = parent_uf[x]
        return x

    for u, v, d in G.edges(data=True):
        if d.get("relation") == "PARENT" and u in parent_uf and v in parent_uf:
            ru, rv = find_root(u), find_root(v)
            if ru != rv:
                parent_uf[ru] = rv

    corr_summary = {}
    print("\nDegree vs. Score Correlation Audit & Top-100 Deduplicated Predictions:")
    for did, name in D["main"].items():
        s = D["index"]["DIS"][did]
        scores = zd @ zs[s]
        r_p = pearson_corr(pop, scores)
        r_s = spearman_corr(pop, scores)
        corr_summary[name] = {"pearson_r": round(r_p, 4), "spearman_rho": round(r_s, 4)}
        print(f"  {name:26} | degree-score Pearson r = {r_p:+.3f}, Spearman rho = {r_s:+.3f}")

        sets, _ = test_sets(D, did)
        status = {}
        for k, ds in sets.items():
            for d in ds:
                status.setdefault(d, k.split(" ")[0])

        # Rank all drugs, deduplicating salt/stereoisomer formulations so top-100 has unique active ingredients
        order = np.argsort(-scores)
        seen_families = set()
        top = []
        for d in order:
            did_str = D["ids"]["DRG"][d]
            dname = D["names"]["DRG"][d]
            fam_key = (find_root(did_str), canonical_drug_family(dname))
            base_key = canonical_drug_family(dname)
            if fam_key in seen_families or base_key in seen_families:
                continue
            seen_families.add(fam_key)
            seen_families.add(base_key)
            top.append(d)
            if len(top) >= TOP_K_CSV:
                break

        path = os.path.join(outdir, f"predictions_{clean(name)}_{tag}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rank", "drug_id", "drug_name", "score", "train_deg", "already_known_as", "mechanism_summary"])
            for r, d in enumerate(top, 1):
                did_str = D["ids"]["DRG"][d]
                dname = D["names"]["DRG"][d]
                mech = explain_drug_disease_path(G, did_str, did, D["family"]) if not os.path.basename(outdir).startswith("tmp") else ""
                w.writerow([r, did_str, dname, f"{scores[d]:.4f}", int(pop[d]), status.get(int(d), ""), mech])
        novel = sum(1 for d in top if int(d) not in status)
        print(f"    -> {path}  ({novel} of top {len(top)} unique drugs are novel for {name})")
    return corr_summary


# ----------------------------------------------------------------------------
# torch part (imports are lazy so the data logic above can be tested without torch)
# ----------------------------------------------------------------------------
def build_model(in_dims, edge_types, hidden, layers, dropout, use_cosine=False):
    import torch
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
            self.use_cosine = use_cosine
            self.logit_scale = nn.Parameter(torch.tensor(2.3026)) if use_cosine else None  # exp(2.3026) ~ 10.0

        def forward(self, x_dict, edge_index_dict):
            h = {t: F.relu(self.proj[t](x)) for t, x in x_dict.items()}
            for i, conv in enumerate(self.convs):
                out = conv(h, edge_index_dict)
                last = i == len(self.convs) - 1
                h = {t: ((out[t] if last else F.relu(out[t])) + h[t]) if t in out else h[t] for t in h}
                h = {t: F.dropout(v, p=self.dropout, training=self.training) for t, v in h.items()}
            if self.use_cosine:
                scale = self.logit_scale.exp().clamp(max=50.0).sqrt()
                h = {t: F.normalize(v, p=2, dim=-1) * scale for t, v in h.items()}
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

    if args.neg_alpha > 0:
        w_deg = np.power(pop + 1.0, args.neg_alpha)
        w_deg /= w_deg.sum()
        w_unif = np.full(D["n_drug"], 1.0 / D["n_drug"], dtype=np.float64)
        neg_probs = torch.from_numpy(0.5 * w_unif + 0.5 * w_deg).float().to(device)
        print(f"Negative sampling: 50% uniform + 50% degree^{args.neg_alpha:.2f} | cosine={args.cosine}")
    else:
        neg_probs = None

    x = {t: torch.from_numpy(a).to(device) for t, a in D["X"].items()}
    static = {et: torch.from_numpy(a).long().to(device) for et, a in D["static"].items()}
    treats_t, rev_treats_t = ("DRG", "treats", "DIS"), ("DIS", "rev_treats", "DRG")
    edge_types = list(static.keys()) + [treats_t, rev_treats_t]
    model = build_model({t: a.shape[1] for t, a in D["X"].items()}, edge_types,
                        args.hidden, args.layers, args.dropout, use_cosine=args.cosine).to(device)
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
        if neg_probs is not None:
            nd = torch.multinomial(neg_probs, sup.shape[1] * args.neg, replacement=True).view(sup.shape[1], args.neg)
        else:
            nd = torch.randint(0, D["n_drug"], (sup.shape[1], args.neg), device=device)
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
    ap.add_argument("--neg-alpha", type=float, default=0.0, help="degree exponent for mixed hard negatives (e.g. 0.5)")
    ap.add_argument("--cosine", action="store_true", help="L2-normalize embeddings with learnable temperature")
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

    drop = {clean(r) for r in args.drop_rel.split(",") if r.strip()}
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
        if drop:
            tag += "_no_" + "_".join(sorted(drop))
        if args.neg_alpha > 0:
            tag += f"_neg{str(args.neg_alpha).replace('.', 'p')}"
        if args.cosine:
            tag += "_cos"

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
    corr_summary = write_predictions(zd, zs, D, G, outdir, tag, pop)

    import torch
    torch.save(state, os.path.join(outdir, f"best_model_{tag}.pt"))
    with open(os.path.join(outdir, f"results_{tag}_seed{args.seed}.json"), "w") as f:
        json.dump({"args": vars(args), "validation": {"gnn": mv, "popularity": mp_}, "test": test,
                   "degree_score_correlation": corr_summary, "family": sorted(D["family"])}, f, indent=2)
    print(f"\nSaved results_{tag}_seed{args.seed}.json and best_model_{tag}.pt in {outdir}")
    if args.smoke:
        print("SMOKE TEST OK: the pipeline runs end to end. Now run it without --smoke.")


if __name__ == "__main__":
    sys.exit(main())