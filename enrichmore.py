#!/usr/bin/env python3
"""
enrich_more.py : continue the HYPOTHESIS knowledge-graph enrichment.

Run from the FYP folder, ONE STEP AT A TIME and in this order:

    python enrich_more.py verify        # check the IDs typed earlier, fix exact matches
    python enrich_more.py opentargets   # clinical-stage drugs + associated genes
    python enrich_more.py targets       # pathogen protein targets + potent clinical-stage compounds

Each step reads the previous graph version and writes a NEW file
(kg_v4_verified, kg_v5_opentargets, kg_v6_targets), so nothing is overwritten
and you can throw a step away by simply not using its output.
"""
import argparse
import os
import time

import networkx as nx
import requests

FOLDER = os.environ.get(
    "HYP_FOLDER",
    "C:/Users/Abuba/Desktop/FYP/OptimusKG_Subgraph/"
)
EDGES_PARQUET = FOLDER + "edges.parquet"        # full OptimusKG edges (used for the hub filter)

FILES = {
    "verify":      (FOLDER + "enriched_subgraph_v3_selective.graphml", FOLDER + "kg_v4_verified.graphml"),
    "opentargets": (FOLDER + "kg_v4_verified.graphml",                 FOLDER + "kg_v5_opentargets.graphml"),
    "targets":     (FOLDER + "kg_v5_opentargets.graphml",              FOLDER + "kg_v6_targets.graphml"),
}

# graph node ids of the three target diseases
CL, MDR, DENGUE = "EFO_0005046", "MONDO_0018076", "EFO_0005547"
DISEASE_NAMES = {
    CL: "Cutaneous Leishmaniasis",
    MDR: "MDR-TB node (OptimusKG calls it 'tuberculosis')",
    DENGUE: "Dengue",
}

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
OT = "https://api.platform.opentargets.org/api/v4/graphql"
HEADERS = {"Accept": "application/json", "User-Agent": "HYPOTHESIS-FYP/1.0"}


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def http_get(url, params=None, retries=3, timeout=30):
    """Returns (status_code, json_or_None). 404 returns immediately; other failures are retried."""
    code = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            code = r.status_code
            if code == 200:
                return code, r.json()
            if code == 404:
                return code, None
            print(f"    HTTP {code}, retrying...")
        except Exception as e:  # network error, bad JSON, ...
            print(f"    request error ({e.__class__.__name__}), retrying...")
        time.sleep(3 * (attempt + 1))
    return code, None


def http_post(url, payload, retries=3, timeout=60):
    code = None
    for attempt in range(retries):
        try:
            r = requests.post(url, json=payload, headers=HEADERS, timeout=timeout)
            code = r.status_code
            if code == 200:
                return code, r.json()
            print(f"    HTTP {code}, retrying...")
        except Exception as e:
            print(f"    request error ({e.__class__.__name__}), retrying...")
        time.sleep(3 * (attempt + 1))
    return code, None


def load(path):
    G = nx.read_graphml(path)
    print(f"Loaded {path}\n  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def _clean(attrs):
    """GraphML cannot store None, lists or dicts."""
    for k, v in list(attrs.items()):
        if v is None:
            attrs[k] = ""
        elif not isinstance(v, (str, int, float, bool)):
            attrs[k] = str(v)


def save(G, path):
    for _, d in G.nodes(data=True):
        _clean(d)
    for _, _, d in G.edges(data=True):
        _clean(d)
    nx.write_graphml(G, path)
    print(f"\nSaved {path}")


def summary(G):
    print(f"\nGraph now: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    for nid, name in DISEASE_NAMES.items():
        if nid in G:
            print(f"  {name}: {G.in_degree(nid)} incoming, {G.out_degree(nid)} outgoing")


def add_node(G, nid, label, name="", **attrs):
    """Adds the node if missing. An existing node is left as it is (only a blank name is filled)."""
    if nid not in G:
        G.add_node(nid, label=label, name=name or "", **attrs)
        return True
    if name and not G.nodes[nid].get("name"):
        G.nodes[nid]["name"] = name
    return False


def add_edge(G, u, v, **attrs):
    """Adds the edge only if it does not exist, so base OptimusKG edges are never overwritten."""
    if G.has_edge(u, v):
        return False
    G.add_edge(u, v, **attrs)
    return True


# ----------------------------------------------------------------------------
# STEP 1: verify the IDs that were typed from memory earlier
# ----------------------------------------------------------------------------
# (id as entered earlier, expected name, disease node, evidence)
CURATED = [
    ("CHEMBL203",     "Ivermectin",           DENGUE, "Approved antiparasitic, dengue clinical trials"),
    ("CHEMBL1200513", "Chloroquine",          DENGUE, "Approved antimalarial, tested in dengue"),
    ("CHEMBL76",      "Chloroquine",          DENGUE, "Dengue NS3 inhibition evidence"),
    ("CHEMBL1643",    "Ribavirin",            DENGUE, "Broad antiviral, dengue evidence"),
    ("CHEMBL374478",  "Celgosivir",           DENGUE, "Phase II dengue clinical trial"),
    ("CHEMBL2110737", "Celgosivir",           DENGUE, "Dengue clinical trial formulation"),
    ("CHEMBL1079",    "Prednisolone",         DENGUE, "Severe dengue case management"),
    ("CHEMBL1200868", "Dexamethasone",        DENGUE, "Dengue clinical use"),
    ("CHEMBL1560",    "Lovastatin",           DENGUE, "Dengue clinical trial, anti-inflammatory"),
    ("CHEMBL862",     "Balapiravir",          DENGUE, "NS5 polymerase inhibitor, dengue Phase II"),
    ("CHEMBL125",     "Miltefosine",          CL,     "Repurposed for CL"),
    ("CHEMBL239129",  "Meglumine antimonate", CL,     "First-line CL treatment"),
    ("CHEMBL4297415", "Antimony",             CL,     "Pentavalent antimonial, standard CL treatment"),
    ("CHEMBL1366",    "Auranofin",            CL,     "Anti-leishmanial activity confirmed"),
    ("CHEMBL1401",    "Nitazoxanide",         CL,     "Broad antiparasitic, CL activity studies"),
    ("CHEMBL1200796", "Cyclophosphamide",     CL,     "Proposed in systems biology study"),
    ("CHEMBL1282",    "Imiquimod",            CL,     "Topical immunomodulator, CL clinical use"),
    ("CHEMBL76",      "Chloroquine",          CL,     "Anti-leishmanial activity in vitro"),
    ("CHEMBL1200699", "Doxycycline",          CL,     "CL adjunct therapy evidence"),
    ("CHEMBL1292",    "Clofazimine",          CL,     "Anti-leishmanial repurposing candidate"),
]
OLD_BULK_TARGETS = [("CHEMBL395", "Dengue NS3"), ("CHEMBL3908", "Dengue NS5"),
                    ("CHEMBL364", "Leishmania major"), ("CHEMBL3952", "Leishmania donovani")]


def names_match(a, b):
    a, b = (a or "").lower().strip(), (b or "").lower().strip()
    return bool(a) and bool(b) and (a in b or b in a)


def find_chembl_by_name(name):
    """Exact (case-insensitive) pref_name match only, so nothing is guessed."""
    _, data = http_get(f"{CHEMBL}/molecule.json", {"pref_name__iexact": name, "limit": 5})
    for m in (data or {}).get("molecules", []):
        if (m.get("pref_name") or "").lower() == name.lower():
            return m["molecule_chembl_id"], m.get("pref_name")
    return None, None


def step_verify(G):
    print("\n[1] What the ChEMBL target IDs used in the discarded bulk fetch really are")
    for tid, claimed in OLD_BULK_TARGETS:
        code, d = http_get(f"{CHEMBL}/target/{tid}.json")
        if d:
            print(f"  {tid}: I called it '{claimed}' | ChEMBL says '{d.get('pref_name')}' "
                  f"({d.get('organism')}, {d.get('target_type')})")
        else:
            print(f"  {tid}: could not look up (HTTP {code})")

    print("\n[2] Curated drug IDs against ChEMBL")
    ok = fixed = unresolved = skipped = 0
    for old_id, name, dis, evidence in CURATED:
        code, mol = http_get(f"{CHEMBL}/molecule/{old_id}.json")
        if mol is None and code != 404:
            skipped += 1
            print(f"  SKIPPED   {old_id} ({name}): ChEMBL not reachable, nothing changed")
            continue
        actual = (mol or {}).get("pref_name")
        if names_match(actual, name):
            ok += 1
            if old_id in G:
                G.nodes[old_id]["id_verified"] = "chembl"
            print(f"  OK        {old_id} = {actual}")
            continue

        print(f"  MISMATCH  {old_id}: expected {name}, ChEMBL says {actual}")
        # remove only OUR curated edge from the wrong node, never base OptimusKG edges
        if G.has_edge(old_id, dis) and str(G.edges[old_id, dis].get("source", "")).startswith("curated"):
            G.remove_edge(old_id, dis)
        if old_id in G and str(G.nodes[old_id].get("source", "")).startswith("curated") and G.degree(old_id) == 0:
            G.remove_node(old_id)

        new_id, new_name = find_chembl_by_name(name)
        if not new_id:
            unresolved += 1
            print(f"            no exact ChEMBL match for '{name}', add it by hand later")
            continue
        tag = "curated_dengue" if dis == DENGUE else "curated_CL"
        add_node(G, new_id, "DRG", new_name, source=tag, id_verified="chembl")
        add_edge(G, new_id, dis, relation="INDICATION", label="DRG-DIS", source=tag,
                 evidence=evidence, id_verified="chembl")
        fixed += 1
        print(f"            -> {new_id} ({new_name})")
    print(f"\n  verified OK: {ok} | fixed: {fixed} | unresolved: {unresolved} | skipped: {skipped}")

    print("\n[3] Leftovers from earlier attempts")
    dg_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("source") == "DisGeNET")
    dg_nodes = [n for n in G.nodes if str(n).startswith("GEN_")]
    lmj = [n for n in G.nodes if str(n).startswith("LmjF.")]
    print(f"  DisGeNET edges in graph: {dg_edges} (nodes with 'GEN_' ids: {len(dg_nodes)})")
    print(f"  Unverified LmjF.* target nodes (replaced in the 'targets' step): {len(lmj)}")


# ----------------------------------------------------------------------------
# STEP 2: Open Targets (clinical-stage drugs and associated genes)
# ----------------------------------------------------------------------------
OT_SEARCH_TERMS = {DENGUE: "dengue", CL: "cutaneous leishmaniasis", MDR: "tuberculosis"}
OT_ID_OVERRIDE = {}          # e.g. {DENGUE: "MONDO_0005502"} if the top search hit is the wrong disease
MIN_PHASE = 2                # keep drugs at clinical phase II or higher
DRUG_TYPES = {"Small molecule"}
MIN_SCORE = 0.2              # minimum Open Targets association score for genes
MAX_GENES = 100              # per disease, after the hub filter
HUB_DEGREE = 500             # same cut-off used when the subgraph was extracted

Q_SEARCH = """query($q: String!) {
  search(queryString: $q, entityNames: ["disease"], page: {index: 0, size: 5}) {
    hits { id name }
  }
}"""
_DRUG_ROW = "phase status mechanismOfAction %s drug { id name }"
Q_DRUGS_FULL = """query($id: String!, $size: Int!, $cursor: String) {
  disease(efoId: $id) { knownDrugs(size: $size, cursor: $cursor) { cursor rows { %s } } }
}""" % (_DRUG_ROW % "drugType")
Q_DRUGS_MIN = """query($id: String!, $size: Int!, $cursor: String) {
  disease(efoId: $id) { knownDrugs(size: $size, cursor: $cursor) { cursor rows { %s } } }
}""" % (_DRUG_ROW % "")
Q_TARGETS = """query($id: String!, $index: Int!, $size: Int!) {
  disease(efoId: $id) {
    associatedTargets(page: {index: $index, size: $size}) {
      rows { score target { id approvedSymbol approvedName } }
    }
  }
}"""


def ot_post(query, variables):
    _, data = http_post(OT, {"query": query, "variables": variables})
    if data and data.get("errors"):
        print("    Open Targets error:", data["errors"][0].get("message"))
    return data


def ot_resolve(term):
    data = ot_post(Q_SEARCH, {"q": term})
    hits = (((data or {}).get("data") or {}).get("search") or {}).get("hits") or []
    for h in hits[:3]:
        print(f"    candidate: {h.get('id')}  {h.get('name')}")
    return hits[0]["id"] if hits else None


def ot_known_drugs(ot_id):
    """Tries the full query, falls back to a minimal one if the schema rejects a field."""
    for q, has_type in ((Q_DRUGS_FULL, True), (Q_DRUGS_MIN, False)):
        rows, cursor, ok = [], None, True
        for _ in range(10):
            data = ot_post(q, {"id": ot_id, "size": 500, "cursor": cursor})
            disease = ((data or {}).get("data") or {}).get("disease")
            if not data or data.get("errors") or disease is None:
                ok = False
                break
            kd = disease.get("knownDrugs") or {}
            rows += kd.get("rows") or []
            cursor = kd.get("cursor")
            if not cursor:
                break
        if ok:
            return rows, has_type
    return [], False


def ot_assoc_genes(ot_id):
    rows = []
    for index in range(3):
        data = ot_post(Q_TARGETS, {"id": ot_id, "index": index, "size": 100})
        d = (((data or {}).get("data") or {}).get("disease") or {}).get("associatedTargets") or {}
        batch = d.get("rows") or []
        rows += batch
        if not batch or (batch[-1].get("score") or 0) < MIN_SCORE:
            break
    return [r for r in rows if (r.get("score") or 0) >= MIN_SCORE]


def hub_degrees(ids):
    """Total degree of each gene in the FULL OptimusKG (same logic as the original hub filter)."""
    import polars as pl
    edges = pl.scan_parquet(EDGES_PARQUET)
    ids = list(ids)
    out_deg = (edges.filter(pl.col("from").is_in(ids)).group_by("from").len()
               .rename({"from": "node", "len": "o"}).collect())
    in_deg = (edges.filter(pl.col("to").is_in(ids)).group_by("to").len()
              .rename({"to": "node", "len": "i"}).collect())
    deg = out_deg.join(in_deg, on="node", how="full", coalesce=True).fill_null(0)
    return {r["node"]: r["o"] + r["i"] for r in deg.iter_rows(named=True)}


def step_opentargets(G):
    for dis, term in OT_SEARCH_TERMS.items():
        print(f"\n=== {DISEASE_NAMES[dis]}  (searching Open Targets for '{term}')")
        ot_id = OT_ID_OVERRIDE.get(dis) or ot_resolve(term)
        if not ot_id:
            print("    no disease found, skipping")
            continue
        print(f"    using {ot_id}")
        if dis not in G:
            print(f"    {dis} is not in the graph, skipping")
            continue

        # ---- clinical-stage drugs ----
        rows, has_type = ot_known_drugs(ot_id)
        if not has_type and rows:
            print("    note: drugType not available, vaccines/biologics were NOT filtered out")
        best = {}
        for r in rows:
            d = r.get("drug") or {}
            did = d.get("id")
            if not did:
                continue
            if has_type and DRUG_TYPES and (r.get("drugType") or "") not in DRUG_TYPES:
                continue
            ph = float(r.get("phase") or 0)
            if did not in best or ph > best[did]["phase"]:
                best[did] = {"name": d.get("name") or "", "phase": ph,
                             "status": r.get("status") or "", "moa": r.get("mechanismOfAction") or ""}
        best = {k: v for k, v in best.items() if v["phase"] >= MIN_PHASE}
        new_d = ann_d = 0
        for did, v in best.items():
            add_node(G, did, "DRG", v["name"], source="OpenTargets")
            rel = "INDICATION" if v["phase"] >= 4 else "CLINICAL_TRIAL"
            if add_edge(G, did, dis, relation=rel, label="DRG-DIS", source="OpenTargets",
                        phase=v["phase"], status=v["status"], mechanism=v["moa"]):
                new_d += 1
            else:
                G.edges[did, dis]["ot_phase"] = v["phase"]     # annotate, never overwrite
                ann_d += 1
        print(f"    drugs (phase >= {MIN_PHASE}): {len(best)} found | {new_d} new edges | {ann_d} existing edges annotated")

        # ---- associated genes (hub genes excluded, as in the original subgraph) ----
        genes = ot_assoc_genes(ot_id)
        if not genes:
            print("    genes: none returned")
            continue
        try:
            deg = hub_degrees([g["target"]["id"] for g in genes])
        except Exception as e:
            print(f"    hub filter failed ({e.__class__.__name__}: {e}); genes skipped to avoid re-adding hubs")
            continue
        kept = [g for g in genes if deg.get(g["target"]["id"], 0) <= HUB_DEGREE]
        hubs = [g for g in genes if deg.get(g["target"]["id"], 0) > HUB_DEGREE]
        kept = kept[:MAX_GENES]
        new_g = ann_g = 0
        for g in kept:
            t = g["target"]
            add_node(G, t["id"], "GEN", t.get("approvedSymbol") or "",
                     description=t.get("approvedName") or "", source="OpenTargets")
            if add_edge(G, dis, t["id"], relation="ASSOCIATED_WITH", label="DIS-GEN",
                        source="OpenTargets", score=g["score"]):
                new_g += 1
            else:
                G.edges[dis, t["id"]]["ot_score"] = g["score"]
                ann_g += 1
        eg = ", ".join(h["target"].get("approvedSymbol") or "?" for h in hubs[:5])
        print(f"    genes (score >= {MIN_SCORE}): {len(genes)} found | {len(hubs)} hub genes dropped"
              f"{' (' + eg + ')' if eg else ''} | {new_g} new edges | {ann_g} existing edges annotated")


# ----------------------------------------------------------------------------
# STEP 3: pathogen protein targets and potent, clinical-stage compounds (ChEMBL)
# ----------------------------------------------------------------------------
# (search text, organism must contain, disease node)
PATHOGEN_QUERIES = [
    ("arginase", "Leishmania", CL),
    ("pteridine reductase", "Leishmania", CL),
    ("trypanothione reductase", "Leishmania", CL),
    ("trypanothione synthetase", "Leishmania", CL),
    ("dihydrofolate reductase", "Leishmania", CL),
    ("N-myristoyltransferase", "Leishmania", CL),
    ("sterol 14-alpha demethylase", "Leishmania", CL),
    ("NS3", "Dengue", DENGUE),
    ("NS5", "Dengue", DENGUE),
    ("envelope protein", "Dengue", DENGUE),
    ("InhA enoyl reductase", "Mycobacterium tuberculosis", MDR),
    ("DprE1", "Mycobacterium tuberculosis", MDR),
    ("MmpL3", "Mycobacterium tuberculosis", MDR),
    ("DNA gyrase", "Mycobacterium tuberculosis", MDR),
    ("cytochrome bc1 QcrB", "Mycobacterium tuberculosis", MDR),
    ("DNA-directed RNA polymerase beta", "Mycobacterium tuberculosis", MDR),
]
MIN_PCHEMBL = 6.0            # IC50/Ki/EC50 <= 1 uM
MIN_CLIN_PHASE = 1           # 0 keeps lab compounds too (much noisier)
MAX_TARGETS_PER_QUERY = 3
MAX_PAGES = 20               # x100 activities per target


def potent_activities(target_id):
    url = f"{CHEMBL}/activity.json"
    params = {"target_chembl_id": target_id, "pchembl_value__gte": MIN_PCHEMBL, "limit": 100}
    best, pages = {}, 0
    while url and pages < MAX_PAGES:
        _, data = http_get(url, params)
        params = None                       # the 'next' url already carries them
        if not data:
            break
        for a in data.get("activities", []):
            mid, p = a.get("molecule_chembl_id"), a.get("pchembl_value")
            if not mid or p is None:
                continue
            p = float(p)
            if mid not in best or p > best[mid][0]:
                best[mid] = (p, a.get("standard_type") or "")
        nxt = (data.get("page_meta") or {}).get("next")
        url = "https://www.ebi.ac.uk" + nxt if nxt else None
        pages += 1
        time.sleep(0.3)
    return best


def clinical_info(mol_ids):
    """molecule id -> (max_phase, pref_name)"""
    out, ids = {}, list(mol_ids)
    for i in range(0, len(ids), 50):
        _, data = http_get(f"{CHEMBL}/molecule.json",
                           {"molecule_chembl_id__in": ",".join(ids[i:i + 50]), "limit": 100})
        for m in (data or {}).get("molecules", []):
            try:
                ph = float(m.get("max_phase") or 0)
            except (TypeError, ValueError):
                ph = 0.0
            out[m["molecule_chembl_id"]] = (ph, m.get("pref_name") or "")
        time.sleep(0.3)
    return out


def step_targets(G):
    old = [n for n, d in G.nodes(data=True) if str(n).startswith("LmjF.") and d.get("source") == "TriTrypDB"]
    G.remove_nodes_from(old)
    print(f"\nRemoved {len(old)} unverified LmjF.* placeholder targets (replaced by ChEMBL-verified ones below)")

    total_t = total_c = 0
    for text, organism, dis in PATHOGEN_QUERIES:
        print(f"\n== {text} | {organism} -> {DISEASE_NAMES[dis]}")
        _, data = http_get(f"{CHEMBL}/target/search.json", {"q": f"{text} {organism}", "limit": 20})
        cands = [t for t in (data or {}).get("targets", [])
                 if organism.lower() in (t.get("organism") or "").lower()
                 and t.get("target_type") == "SINGLE PROTEIN"][:MAX_TARGETS_PER_QUERY]
        if not cands:
            print("    no matching single-protein target in ChEMBL")
            continue
        for t in cands:
            tid, tname = t["target_chembl_id"], t.get("pref_name") or ""
            acts = potent_activities(tid)
            info = clinical_info(acts.keys()) if acts else {}
            keep = {m: acts[m] for m in acts if info.get(m, (0, ""))[0] >= MIN_CLIN_PHASE}
            print(f"    {tid} {tname} ({t.get('organism')}): {len(acts)} potent compounds, "
                  f"{len(keep)} clinical-stage kept")
            if not keep:
                continue
            add_node(G, tid, "TGT", tname, organism=t.get("organism") or "", source="ChEMBL")
            add_edge(G, tid, dis, relation="ASSOCIATED_WITH", label="TGT-DIS", source="ChEMBL")
            total_t += 1
            for mid, (p, stype) in keep.items():
                ph, mname = info[mid]
                add_node(G, mid, "DRG", mname, source="ChEMBL", max_phase=ph)
                if add_edge(G, mid, tid, relation="ACTIVE_AGAINST", label="DRG-TGT",
                            pchembl=p, standard_type=stype, source="ChEMBL"):
                    total_c += 1
    print(f"\nAdded {total_t} targets and {total_c} compound-target edges")


# ----------------------------------------------------------------------------
STEPS = {"verify": step_verify, "opentargets": step_opentargets, "targets": step_targets}


def main():
    ap = argparse.ArgumentParser(description="HYPOTHESIS graph enrichment")
    ap.add_argument("step", choices=STEPS)
    ap.add_argument("--in", dest="infile", help="override the input graph file")
    ap.add_argument("--out", dest="outfile", help="override the output graph file")
    args = ap.parse_args()

    default_in, default_out = FILES[args.step]
    infile, outfile = args.infile or default_in, args.outfile or default_out
    G = load(infile)
    STEPS[args.step](G)
    summary(G)
    save(G, outfile)


if __name__ == "__main__":
    main()