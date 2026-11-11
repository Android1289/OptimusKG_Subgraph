#!/usr/bin/env python3
"""
enrich_more.py : continue the HYPOTHESIS knowledge-graph enrichment.

Run from anywhere, ONE STEP AT A TIME (each step reads the previous graph version
and writes a NEW file, so nothing is overwritten):

    python enrich_more.py verify        # v3_selective -> kg_v4_verified     (already run)
    python enrich_more.py opentargets   # v4 -> kg_v5_opentargets  genes     (already run)
    python enrich_more.py targets       # v5 -> kg_v6_targets      pathogen targets (already run)
    python enrich_more.py drugs         # v6 -> kg_v7_drugs        clinical-stage drugs
    python enrich_more.py review        # prints what was added, saves nothing
    python enrich_more.py disgenet      # v7 -> kg_v8_disgenet     gene-disease links, academic key (NEW)
    python enrich_more.py pathways      # v8 -> kg_v9_pathways     Reactome pathway layer
    python enrich_more.py clean         # relabel curated edges, drop self-loops, flag candidate drugs (NEW)
    python enrich_more.py mechanisms    # drug -> human target gene edges from ChEMBL
    python enrich_more.py expand        # v13 -> kg_v14_expanded   widen the drug pool from edges.parquet (NEW)

Add  --in FILE  /  --out FILE  to override the default file names.
"""
import argparse
import os
import re
import time
from collections import Counter

import networkx as nx
import requests

FOLDER = os.environ.get("HYP_FOLDER", "C:/Users/Abuba/Desktop/FYP/OptimusKG_Subgraph/")
EDGES_PARQUET = FOLDER + "edges.parquet"        # full OptimusKG edges (used for the hub filter)

FILES = {
    "verify":      (FOLDER + "enriched_subgraph_v3_selective.graphml", FOLDER + "kg_v4_verified.graphml"),
    "opentargets": (FOLDER + "kg_v4_verified.graphml",                 FOLDER + "kg_v5_opentargets.graphml"),
    "targets":     (FOLDER + "kg_v5_opentargets.graphml",              FOLDER + "kg_v6_targets.graphml"),
    "drugs":       (FOLDER + "kg_v6_targets.graphml",                  FOLDER + "kg_v7_drugs.graphml"),
    "disgenet":    (FOLDER + "kg_v7_drugs.graphml",                    FOLDER + "kg_v8_disgenet.graphml"),
    "pathways":    (FOLDER + "kg_v8_disgenet.graphml",                 FOLDER + "kg_v9_pathways.graphml"),
    "clean":       (FOLDER + "kg_v10_drugs_fixed.graphml",             FOLDER + "kg_v11_clean.graphml"),
    "mechanisms":  (FOLDER + "kg_v11_clean.graphml",                   FOLDER + "kg_v12_mechanisms.graphml"),
    "expand":      (FOLDER + "kg_v13_pathways.graphml",                FOLDER + "kg_v14_expanded.graphml"),
    "review":      (None, None),            # review opens the newest graph file that exists
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
LAST_ERROR = ""      # text of the last non-200 answer, printed when something is rejected


def http_get(url, params=None, retries=3, timeout=30, headers=None):
    """Returns (status_code, json_or_None). 4xx (except 429) returns immediately; 429/5xx/network errors are retried."""
    global LAST_ERROR
    code = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers or HEADERS, timeout=timeout)
            code = r.status_code
            if code == 200:
                return code, r.json()
            LAST_ERROR = r.text[:200]
            if code == 429:
                wait = int(r.headers.get("Retry-After", 30)) if str(r.headers.get("Retry-After", "")).isdigit() else 30
                print(f"    rate limited (429), waiting {wait}s...")
                time.sleep(wait)
                continue
            if 400 <= code < 500:
                return code, None
            print(f"    HTTP {code}, retrying...")
        except Exception as e:  # network error, bad JSON, ...
            print(f"    request error ({e.__class__.__name__}), retrying...")
        time.sleep(3 * (attempt + 1))
    return code, None


def http_post(url, payload, retries=3, timeout=60):
    """Returns (status_code, json_or_None). A 4xx is NOT retried and its JSON body (the GraphQL error) is kept."""
    code = None
    for attempt in range(retries):
        try:
            r = requests.post(url, json=payload, headers=HEADERS, timeout=timeout)
            code = r.status_code
            if code == 200:
                return code, r.json()
            if 400 <= code < 500:
                try:
                    return code, r.json()
                except Exception:
                    return code, None
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
    print("  node types:", dict(Counter(d.get("label", "?") for _, d in G.nodes(data=True))))
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
# STEP: verify the IDs that were typed from memory earlier
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
# Open Targets helpers (genes and drugs)
# ----------------------------------------------------------------------------
OT_SEARCH_TERMS = {DENGUE: "dengue", CL: "cutaneous leishmaniasis", MDR: "tuberculosis"}
OT_ID_OVERRIDE = {}          # e.g. {DENGUE: "MONDO_0005502"} if the top search hit is the wrong disease
MIN_STAGE = 2.0              # keep drugs at clinical phase II or higher (see STAGE_VALUE)
DRUG_TYPES = {"Small molecule"}
MIN_SCORE = 0.2              # minimum Open Targets association score for genes
MAX_GENES = 100              # per disease, after the hub filter
HUB_DEGREE = 500             # same cut-off used when the subgraph was extracted

Q_SEARCH = """query($q: String!) {
  search(queryString: $q, entityNames: ["disease"], page: {index: 0, size: 5}) {
    hits { id name }
  }
}"""
Q_TARGETS = """query($id: String!, $index: Int!, $size: Int!) {
  disease(efoId: $id) {
    associatedTargets(page: {index: $index, size: $size}) {
      rows { score target { id approvedSymbol approvedName } }
    }
  }
}"""
# 'knownDrugs' was retired in Open Targets release 26.03 (replaced by clinical indications).
DRUG_QUERIES = [
    ("with drugType", """query($id: String!) {
  disease(efoId: $id) { drugAndClinicalCandidates { count rows { maxClinicalStage drug { id name drugType } } } }
}"""),
    ("without drugType", """query($id: String!) {
  disease(efoId: $id) { drugAndClinicalCandidates { count rows { maxClinicalStage drug { id name } } } }
}"""),
]
# Open Targets harmonised clinical stages -> number (Withdrawal counts as Approval, as OT does)
STAGE_VALUE = {"APPROVAL": 4.0, "PHASE_4": 4.0, "PHASE_IV": 4.0, "WITHDRAWAL": 4.0, "PREAPPROVAL": 3.5,
               "PHASE_3": 3.0, "PHASE_III": 3.0, "PHASE_2_3": 2.5, "PHASE_II_III": 2.5,
               "PHASE_2": 2.0, "PHASE_II": 2.0, "PHASE_1_2": 1.5, "PHASE_I_II": 1.5,
               "PHASE_1": 1.0, "PHASE_I": 1.0, "EARLY_PHASE_1": 0.5, "EARLY_PHASE_I": 0.5, "IND": 0.25}


def stage_value(raw):
    key = re.sub(r"[^A-Z0-9]+", "_", str(raw or "").upper()).strip("_")
    return STAGE_VALUE.get(key, 0.0), key


def ot_post(query, variables=None, quiet=False):
    _, data = http_post(OT, {"query": query, "variables": variables or {}})
    if data and data.get("errors") and not quiet:
        print("    Open Targets error:", data["errors"][0].get("message"))
    return data


def ot_resolve(term):
    data = ot_post(Q_SEARCH, {"q": term})
    hits = (((data or {}).get("data") or {}).get("search") or {}).get("hits") or []
    for h in hits[:3]:
        print(f"    candidate: {h.get('id')}  {h.get('name')}")
    return hits[0]["id"] if hits else None


def ot_introspect_disease():
    """Prints the real Disease fields about drugs/clinical data so the query can be corrected."""
    q = ("query($t: String!) { __type(name: $t) { fields { name type { name kind ofType { name kind "
         "ofType { name kind ofType { name kind } } } } } } }")

    def fields_of(type_name):
        d = ot_post(q, {"t": type_name}, quiet=True)
        t = ((d or {}).get("data") or {}).get("__type") or {}
        return t.get("fields") or []

    def named(t):
        while t and not t.get("name") and t.get("ofType"):
            t = t["ofType"]
        return (t or {}).get("name")

    hits = [f for f in fields_of("Disease")
            if any(k in f["name"].lower() for k in ("drug", "clinical", "indication"))]
    print("    >>> Disease fields about drugs/clinical data:", [f["name"] for f in hits] or "none found")
    for f in hits:
        tn = named(f["type"])
        print(f"        {f['name']} -> {tn}: {[x['name'] for x in (fields_of(tn) if tn else [])]}")
    print("    >>> paste the lines above back to me so the query can be fixed")


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


# ----------------------------------------------------------------------------
# STEP: Open Targets associated genes (hub genes excluded)
# ----------------------------------------------------------------------------
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
        genes = ot_assoc_genes(ot_id)
        if not genes:
            print("    genes: none returned")
            continue
        try:
            deg = hub_degrees([g["target"]["id"] for g in genes])
        except Exception as e:
            print(f"    hub filter failed ({e.__class__.__name__}: {e}); genes skipped to avoid re-adding hubs")
            continue
        kept = [g for g in genes if deg.get(g["target"]["id"], 0) <= HUB_DEGREE][:MAX_GENES]
        hubs = [g for g in genes if deg.get(g["target"]["id"], 0) > HUB_DEGREE]
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
# STEP: Open Targets clinical-stage drugs (new schema)
# ----------------------------------------------------------------------------
def step_drugs(G):
    introspected = False
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

        rows, has_type = None, False
        for label, q in DRUG_QUERIES:
            data = ot_post(q, {"id": ot_id}, quiet=True)
            if data and data.get("errors"):
                print(f"    query '{label}' rejected: {data['errors'][0].get('message')}")
                continue
            if data and (data.get("data") or {}).get("disease") is not None:
                dc = data["data"]["disease"].get("drugAndClinicalCandidates") or {}
                rows, has_type = dc.get("rows") or [], (label == "with drugType")
                break
        if rows is None:
            if not introspected:
                ot_introspect_disease()
                introspected = True
            print("    drugs skipped for this disease")
            continue
        if not has_type and rows:
            print("    note: drugType not available, vaccines/biologics were NOT filtered out")

        best, unknown = {}, set()
        for r in rows:
            d = r.get("drug") or {}
            did = d.get("id")
            if not did:
                continue
            if has_type and DRUG_TYPES and (d.get("drugType") or "") not in DRUG_TYPES:
                continue
            val, key = stage_value(r.get("maxClinicalStage"))
            if val == 0.0 and key not in ("", "UNKNOWN", "PRECLINICAL"):
                unknown.add(key)
            if did not in best or val > best[did]["phase"]:
                best[did] = {"name": d.get("name") or "", "phase": val, "stage": key}
        if unknown:
            print(f"    note: unrecognised stage labels treated as 0: {sorted(unknown)}")
        best = {k: v for k, v in best.items() if v["phase"] >= MIN_STAGE}
        new_d = ann_d = 0
        for did, v in best.items():
            add_node(G, did, "DRG", v["name"], source="OpenTargets")
            rel = "INDICATION" if v["phase"] >= 4 else "CLINICAL_TRIAL"
            if add_edge(G, did, dis, relation=rel, label="DRG-DIS", source="OpenTargets",
                        phase=v["phase"], stage=v["stage"]):
                new_d += 1
            else:
                G.edges[did, dis]["ot_phase"] = v["phase"]       # annotate, never overwrite
                ann_d += 1
        print(f"    drugs (stage >= {MIN_STAGE}): {len(best)} found | {new_d} new edges | "
              f"{ann_d} existing edges annotated")


# ----------------------------------------------------------------------------
# STEP: pathogen protein targets and potent, clinical-stage compounds (ChEMBL)
# ----------------------------------------------------------------------------
# (search text, organism must contain, disease node)
PATHOGEN_QUERIES = [
    ("arginase", "Leishmania", CL),
    ("pteridine reductase", "Leishmania", CL),
    ("trypanothione reductase", "Leishmania", CL),
    ("trypanothione synthetase", "Leishmania", CL),
    ("dihydrofolate reductase", "Leishmania", CL),
    ("glycylpeptide N-tetradecanoyltransferase", "Leishmania", CL),   # ChEMBL's name for N-myristoyltransferase
    ("sterol 14-alpha demethylase", "Leishmania", CL),
    ("NS3", "Dengue", DENGUE),
    ("NS5", "Dengue", DENGUE),
    ("InhA enoyl reductase", "Mycobacterium tuberculosis", MDR),
    ("DprE1", "Mycobacterium tuberculosis", MDR),
    ("MmpL3", "Mycobacterium tuberculosis", MDR),
    ("trehalose monomycolate exporter", "Mycobacterium tuberculosis", MDR),
    ("DNA gyrase subunit", "Mycobacterium tuberculosis", MDR),
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

    seen, total_t, total_c = set(), 0, 0
    for text, organism, dis in PATHOGEN_QUERIES:
        print(f"\n== {text} | {organism} -> {DISEASE_NAMES[dis]}")
        _, data = http_get(f"{CHEMBL}/target/search.json", {"q": f"{text} {organism}", "limit": 50})
        cands = [t for t in (data or {}).get("targets", [])
                 if organism.lower() in (t.get("organism") or "").lower()
                 and t.get("target_type") == "SINGLE PROTEIN"][:MAX_TARGETS_PER_QUERY]
        if not cands:
            print("    no matching single-protein target in ChEMBL")
            continue
        for t in cands:
            tid, tname = t["target_chembl_id"], t.get("pref_name") or ""
            if (tid, dis) in seen:
                continue
            seen.add((tid, dis))
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
                print(f"        {mname or mid}  (pChEMBL {p}, max phase {ph})")
    print(f"\nAdded {total_t} targets and {total_c} compound-target edges")


# ----------------------------------------------------------------------------
# STEP: DisGeNET gene-disease associations (free ACADEMIC key = curated sources only)
# ----------------------------------------------------------------------------
# Get the key: create a free account at https://www.disgenet.com (university e-mail, academic plan),
# then copy the API key from your account page. Put it in a file called  .env  next to the graph
# files, containing ONE line:      DISGENET_API_KEY=paste-your-key-here
# (or, for one PowerShell session:  $env:DISGENET_API_KEY = 'paste-your-key-here')
DISGENET = "https://api.disgenet.com/api/v1"
# disease -> (word the returned disease name must contain, fallback UMLS codes)
DG_DISEASES = {
    DENGUE: ("dengue",   ["UMLS_C0011311"]),
    CL:     ("leishman", ["UMLS_C0023281"]),     # 'Leishmaniasis' (broader than cutaneous only)
    MDR:    ("tubercul", ["UMLS_C0041296"]),
}
MIN_DG_SCORE = 0.1           # DisGeNET gene-disease score (0 to 1)
DG_MAX_PAGES = 30


def disgenet_key():
    key = os.environ.get("DISGENET_API_KEY", "").strip()
    if not key:
        try:
            from dotenv import load_dotenv
            load_dotenv(FOLDER + ".env")
            key = os.environ.get("DISGENET_API_KEY", "").strip()
        except ImportError:
            pass
    if not key:
        raise SystemExit(
            "\nNo DisGeNET API key found. Do ONE of these:\n"
            f"  1) create a file named .env in {FOLDER} containing one line:\n"
            "         DISGENET_API_KEY=paste-your-key-here\n"
            "  2) or in PowerShell, before running:  $env:DISGENET_API_KEY = 'paste-your-key-here'\n"
            "Get the key by registering (free, academic plan) at https://www.disgenet.com\n")
    return key


def umls_from_optimuskg(node_ids):
    """UMLS CUIs that OptimusKG stores as cross-references on the disease nodes (best effort)."""
    found = {}
    try:
        import polars as pl
        df = (pl.scan_parquet(FOLDER + "nodes.parquet")
              .filter(pl.col("id").is_in(list(node_ids))).collect())
        for row in df.iter_rows(named=True):
            cuis = re.findall(r"UMLS\W{0,3}(C\d{7})", str(row.get("properties")), flags=re.I)
            found[row["id"]] = sorted({c.upper() for c in cuis})
    except Exception as e:
        print(f"    (UMLS cross-references not read from nodes.parquet: {e.__class__.__name__})")
    return found


def dg_fetch(code, key):
    """All gene-disease rows for one DisGeNET disease code. Returns (rows, status): ok / key / notfound."""
    headers = {"Authorization": key, "Accept": "application/json"}   # raw key, no 'Bearer'
    rows, seen = [], set()
    for page in range(DG_MAX_PAGES):
        status, data = http_get(f"{DISGENET}/gda/summary", {"disease": code, "page_number": page}, headers=headers)
        if status in (401, 403):
            print(f"    key rejected (HTTP {status}): {LAST_ERROR}")
            return rows, "key"
        if data is None:
            print(f"    code {code}: HTTP {status} {LAST_ERROR}")
            return rows, "notfound"
        new = [r for r in (data.get("payload") or [])
               if (r.get("symbolOfGene"), r.get("diseaseUMLSCUI")) not in seen]
        if not new:
            break
        for r in new:
            seen.add((r.get("symbolOfGene"), r.get("diseaseUMLSCUI")))
            rows.append(r)
        time.sleep(0.5)
    return rows, "ok"


Q_TARGET_SEARCH = """query($q: String!) {
  search(queryString: $q, entityNames: ["target"], page: {index: 0, size: 5}) { hits { id name } }
}"""


def ot_gene_id(symbol, cache):
    """Gene symbol -> Ensembl gene id via MyGene.info (fast, reliable), Open Targets search as fallback.
    (MyGene.info lookup contributed by the project owner after Ensembl REST kept timing out.)"""
    if symbol in cache:
        return cache[symbol]

    # 1. Primary lookup: MyGene.info
    try:
        r = requests.get("https://mygene.info/v3/query",
                         params={"q": f"symbol:{symbol}", "species": "human", "fields": "ensembl.gene"},
                         headers=HEADERS, timeout=8)
        if r.status_code == 200:
            for hit in r.json().get("hits", []):
                ens = hit.get("ensembl")
                if isinstance(ens, dict) and ens.get("gene"):
                    cache[symbol] = ens["gene"]
                    return cache[symbol]
                elif isinstance(ens, list) and ens and ens[0].get("gene"):
                    cache[symbol] = ens[0]["gene"]
                    return cache[symbol]
    except Exception:
        pass

    # 2. Fallback: Open Targets search, exact symbol match only
    try:
        data = ot_post(Q_TARGET_SEARCH, {"q": symbol}, quiet=True)
        hits = (((data or {}).get("data") or {}).get("search") or {}).get("hits") or []
        cache[symbol] = next((h["id"] for h in hits if str(h.get("id", "")).startswith("ENSG")
                              and str(h.get("name", "")).upper() == symbol.upper()), None)
    except Exception:
        cache[symbol] = None
    return cache[symbol]


def step_disgenet(G):
    key = disgenet_key()
    cuis = umls_from_optimuskg(DG_DISEASES.keys())
    sym2node = {str(d.get("name")).upper(): n for n, d in G.nodes(data=True)
                if d.get("label") == "GEN" and d.get("name")}
    cache = {}
    for dis, (keyword, fallback) in DG_DISEASES.items():
        print(f"\n=== {DISEASE_NAMES[dis]}")
        if dis not in G:
            print("    not in the graph, skipping")
            continue
        candidates = list(dict.fromkeys(["UMLS_" + c for c in cuis.get(dis, [])] + fallback))
        rows = None
        for code in candidates:
            got, status = dg_fetch(code, key)
            if status == "key":
                raise SystemExit("Stopping: check the key, and that your academic plan includes the API.")
            names = {str(r.get("diseaseName")) for r in got[:50]}
            if got and any(keyword in n.lower() for n in names):
                print(f"    using {code}: {len(got)} gene associations ({sorted(names)[:2]})")
                rows = got
                break
            print(f"    {code}: {len(got)} rows, disease name does not match '{keyword}', skipped")
        if rows is None:
            print("    no usable DisGeNET code for this disease (academic keys only see curated sources)")
            continue

        counts = {t: sum(1 for r in rows if (r.get("score") or 0) >= t) for t in (0.1, 0.2, 0.3)}
        print(f"    genes by score: >=0.1: {counts[0.1]} | >=0.2: {counts[0.2]} | >=0.3: {counts[0.3]} "
              f"(keeping >= {MIN_DG_SCORE})")
        keep = [r for r in rows if (r.get("score") or 0) >= MIN_DG_SCORE and r.get("symbolOfGene")]
        top = sorted(keep, key=lambda r: -(r.get("score") or 0))[:15]
        print("    genes returned:", ", ".join(f"{r['symbolOfGene']} ({r.get('score')})" for r in top))

        mapped, unmapped = [], 0
        for r in keep:
            sym = r["symbolOfGene"]
            nid = sym2node.get(sym.upper()) or ot_gene_id(sym, cache)
            if nid:
                mapped.append((nid, sym, r))
            else:
                unmapped += 1
        new_ids = [n for n, _, _ in mapped if n not in G]
        hubs = set()
        if new_ids:
            try:
                deg = hub_degrees(new_ids)
                hubs = {n for n in new_ids if deg.get(n, 0) > HUB_DEGREE}
            except Exception as e:
                print(f"    hub filter failed ({e.__class__.__name__}: {e}); new genes skipped")
                hubs = set(new_ids)
        new_g = ann_g = 0
        for nid, sym, r in mapped:
            if nid in hubs:
                continue
            add_node(G, nid, "GEN", sym, source="DisGeNET", ncbi_gene_id=r.get("geneNcbiID"))
            if add_edge(G, dis, nid, relation="ASSOCIATED_WITH", label="DIS-GEN", source="DisGeNET",
                        score=r.get("score"), num_pmids=r.get("numPMIDs"), evidence_index=r.get("ei")):
                new_g += 1
            else:
                G.edges[dis, nid]["dg_score"] = r.get("score")
                ann_g += 1
        print(f"    mapped {len(mapped)} of {len(keep)} genes | {unmapped} no Ensembl id | "
              f"{len(hubs)} hub genes dropped | {new_g} new edges | {ann_g} existing edges annotated")


# ----------------------------------------------------------------------------
# STEP: Reactome pathway layer (gene -> pathway)
# ----------------------------------------------------------------------------
REACTOME = "https://reactome.org/ContentService"
MIN_GENES_PER_PATHWAY = 2    # a pathway must link at least this many genes already in the graph
MAX_GENES_PER_PATHWAY = 60   # drops generic pathways ("Immune System" ...) that touch everything


def reactome_pathways(ensg):
    _, data = http_get(f"{REACTOME}/data/mapping/ENSEMBL/{ensg}/pathways", {"species": "9606"})
    time.sleep(0.2)
    return [(p["stId"], p.get("displayName") or "") for p in (data if isinstance(data, list) else [])
            if str(p.get("stId", "")).startswith("R-HSA")]


def step_pathways(G):
    genes = [n for n, d in G.nodes(data=True) if d.get("label") == "GEN" and str(n).startswith("ENSG")]
    print(f"\n{len(genes)} gene nodes to map to Reactome pathways")
    test = reactome_pathways("ENSG00000141510")          # TP53 certainly has pathways
    if not test:
        print(f"Preflight failed: TP53 returned no pathways ({LAST_ERROR}). Nothing was changed; "
              "paste this message back to me.")
        return
    print(f"Preflight OK (TP53 -> {len(test)} pathways, e.g. {test[0][1]})")

    gene2pw = {}
    for i, g in enumerate(genes, 1):
        gene2pw[g] = reactome_pathways(g)
        if i % 100 == 0:
            print(f"  {i}/{len(genes)} genes done")
    names, count = {}, Counter()
    for g, pws in gene2pw.items():
        for sid, nm in pws:
            names[sid] = nm
            count[sid] += 1
    keep = {s for s, c in count.items() if MIN_GENES_PER_PATHWAY <= c <= MAX_GENES_PER_PATHWAY}
    print(f"  {sum(1 for g in gene2pw if gene2pw[g])} genes have pathways | {len(count)} pathways found | "
          f"{len(keep)} kept ({MIN_GENES_PER_PATHWAY}-{MAX_GENES_PER_PATHWAY} graph genes each)")
    new_n = new_e = 0
    for g, pws in gene2pw.items():
        for sid, nm in pws:
            if sid in keep:
                if add_node(G, sid, "PWY", nm, source="Reactome"):
                    new_n += 1
                if add_edge(G, g, sid, relation="PARTICIPATES_IN", label="GEN-PWY", source="Reactome"):
                    new_e += 1
    print(f"  added {new_n} pathway nodes and {new_e} gene-pathway edges")


# ----------------------------------------------------------------------------
# STEP: clean (label integrity, self-loops, candidate flag)
# ----------------------------------------------------------------------------
def step_clean(G):
    n = 0
    for u, v, d in G.edges(data=True):
        if str(d.get("source", "")).startswith("curated") and d.get("relation") == "INDICATION":
            d["original_relation"] = "INDICATION"
            d["relation"] = "LITERATURE_CANDIDATE"
            n += 1
    print(f"\nRelabelled {n} hand-curated edges INDICATION -> LITERATURE_CANDIDATE "
          "(literature or in-vitro evidence, not an approved indication, so not a training positive)")
    loops = list(nx.selfloop_edges(G))
    G.remove_edges_from(loops)
    print(f"Removed {len(loops)} self-loop edges")
    flagged = 0
    for nid, d in G.nodes(data=True):
        if d.get("label") == "DRG":
            screening = "GSK" in str(d.get("source", ""))
            d["candidate"] = "no" if screening else "yes"
            flagged += screening
    print(f"Flagged {flagged} GSK screening compounds candidate=no (every other drug candidate=yes)")


# ----------------------------------------------------------------------------
# STEP: mechanisms (drug -> human target gene, from ChEMBL mechanism-of-action records)
# ----------------------------------------------------------------------------
def human_target_symbols(tid, cache):
    """Gene symbols of one ChEMBL target, only if it is a single HUMAN protein (cached)."""
    if tid not in cache:
        _, t = http_get(f"{CHEMBL}/target/{tid}.json")
        syms = []
        if t and t.get("organism") == "Homo sapiens" and t.get("target_type") == "SINGLE PROTEIN":
            for comp in t.get("target_components") or []:
                for s in comp.get("target_component_synonyms") or []:
                    if s.get("syn_type") == "GENE_SYMBOL" and s.get("component_synonym"):
                        syms.append(s["component_synonym"])
                        break
        cache[tid] = syms
        time.sleep(0.15)
    return cache[tid]


def step_mechanisms(G):
    drugs = [n for n, d in G.nodes(data=True)
             if d.get("label") == "DRG" and str(n).startswith("CHEMBL") and d.get("candidate") != "no"]
    print(f"\n{len(drugs)} candidate drugs with a ChEMBL id to look up")
    tcache, gcache = {}, {}
    with_target = new_genes = new_edges = 0
    for i, mid in enumerate(drugs, 1):
        _, data = http_get(f"{CHEMBL}/mechanism.json", {"molecule_chembl_id": mid, "limit": 50})
        found = False
        for m in (data or {}).get("mechanisms", []):
            tid = m.get("target_chembl_id")
            if not tid:
                continue
            for sym in human_target_symbols(tid, tcache):
                ensg = ot_gene_id(sym, gcache)
                if not ensg:
                    continue
                if add_node(G, ensg, "GEN", sym, source="ChEMBL_mechanism"):
                    new_genes += 1
                if add_edge(G, mid, ensg, relation="TARGETS", label="DRG-GEN", source="ChEMBL",
                            action_type=m.get("action_type"), mechanism=m.get("mechanism_of_action")):
                    new_edges += 1
                found = True
        with_target += found
        time.sleep(0.15)
        if i % 50 == 0:
            print(f"  {i}/{len(drugs)} drugs done")
    print(f"  {with_target} of {len(drugs)} drugs have a human target gene | {new_genes} new gene nodes | "
          f"{new_edges} drug->gene edges added")
    print("  (anti-infectives such as isoniazid act on pathogen proteins, so they correctly get no human target)")


# ----------------------------------------------------------------------------
# STEP: expand (widen the drug pool: every drug with a clinical edge to ANY disease already in the graph)
# ----------------------------------------------------------------------------
def step_expand(G):
    import polars as pl
    dis_ids = [n for n, d in G.nodes(data=True) if d.get("label") == "DIS"]
    before = sum(1 for _, d in G.nodes(data=True) if d.get("label") == "DRG" and d.get("candidate") != "no")
    print(f"\n{len(dis_ids)} disease nodes in the graph; reading drug-disease edges from {EDGES_PARQUET}")
    sel = (pl.scan_parquet(EDGES_PARQUET)
           .filter(pl.col("label") == "DRG-DIS")
           .filter(pl.col("relation").is_in(["INDICATION", "CONTRAINDICATION", "OFF_LABEL_USE"]))
           .filter(pl.col("to").is_in(dis_ids))
           .select(["from", "to", "relation"]).collect())
    print(f"  {sel.height} drug-disease edges found | {sel['from'].n_unique()} distinct drugs")
    new_ids = [d for d in sel["from"].unique().to_list() if d not in G]
    for did in new_ids:
        add_node(G, did, "DRG", "", source="OptimusKG_expansion", candidate="yes")
    new_edges = Counter()
    for row in sel.iter_rows(named=True):
        if add_edge(G, row["from"], row["to"], relation=row["relation"], label="DRG-DIS", source="OptimusKG"):
            new_edges[row["relation"]] += 1
    print(f"  added {len(new_ids)} new drug nodes and {sum(new_edges.values())} edges {dict(new_edges)}")

    need = [n for n in new_ids if str(n).startswith("CHEMBL")]
    named = with_smiles = 0
    for i in range(0, len(need), 50):
        _, data = http_get(f"{CHEMBL}/molecule.json",
                           {"molecule_chembl_id__in": ",".join(need[i:i + 50]), "limit": 100})
        for m in (data or {}).get("molecules", []):
            n = m["molecule_chembl_id"]
            if n not in G:
                continue
            G.nodes[n]["name"] = m.get("pref_name") or ""
            G.nodes[n]["max_phase"] = m.get("max_phase")
            smi = (m.get("molecule_structures") or {}).get("canonical_smiles")
            if smi:
                G.nodes[n]["smiles"] = smi
                with_smiles += 1
            named += bool(m.get("pref_name"))
        time.sleep(0.3)
        if (i // 50) % 10 == 9:
            print(f"  {i + 50}/{len(need)} new drugs described")
    pool = sum(1 for _, d in G.nodes(data=True) if d.get("label") == "DRG" and d.get("candidate") != "no")
    print(f"  names for {named} and SMILES for {with_smiles} of {len(need)} new drugs")
    print(f"  CANDIDATE DRUG POOL IS NOW {pool} drugs (it was {before})")


# ----------------------------------------------------------------------------
# STEP: review (prints what the enrichment added, saves nothing)
# ----------------------------------------------------------------------------
def step_review(G):
    print("\nDrug -> disease edges from curated lists and Open Targets")
    rows = []
    for u, v, d in G.edges(data=True):
        if d.get("label") == "DRG-DIS" and d.get("source") in ("OpenTargets", "curated_dengue", "curated_CL"):
            rows.append((DISEASE_NAMES.get(v, v), str(G.nodes[u].get("name") or u), u,
                         d.get("relation"), d.get("phase", ""), d.get("source")))
    for disease, name, uid, rel, ph, src in sorted(rows):
        print(f"  {disease[:24]:24} | {name[:30]:30} ({uid}) | {rel} | phase {ph} | {src}")
    print("\nCompound -> target edges from ChEMBL")
    for tid, d in G.nodes(data=True):
        if d.get("label") == "TGT" and d.get("source") == "ChEMBL":
            print(f"  {tid} {d.get('name')}")
            for u, _, e in G.in_edges(tid, data=True):
                print(f"      {G.nodes[u].get('name') or u} ({u}) pChEMBL {e.get('pchembl')}")


# ----------------------------------------------------------------------------
STEPS = {"verify": step_verify, "opentargets": step_opentargets, "targets": step_targets,
         "drugs": step_drugs, "disgenet": step_disgenet, "pathways": step_pathways, "clean": step_clean,
         "mechanisms": step_mechanisms, "expand": step_expand, "review": step_review}


def main():
    ap = argparse.ArgumentParser(description="HYPOTHESIS graph enrichment")
    ap.add_argument("step", choices=STEPS)
    ap.add_argument("--in", dest="infile", help="override the input graph file")
    ap.add_argument("--out", dest="outfile", help="override the output graph file")
    args = ap.parse_args()

    default_in, default_out = FILES[args.step]
    infile, outfile = args.infile or default_in, args.outfile or default_out
    if args.step == "review" and not infile:          # newest graph file that exists
        for name in ("expand", "mechanisms", "clean", "pathways", "disgenet", "drugs", "targets", "opentargets", "verify"):
            if os.path.exists(FILES[name][1]):
                infile = FILES[name][1]
                break
    G = load(infile)
    STEPS[args.step](G)
    if args.step == "review":
        return
    summary(G)
    save(G, outfile)


if __name__ == "__main__":
    main()