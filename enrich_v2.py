import networkx as nx
import requests
import time

G = nx.read_graphml("C:/Users/abuba/Desktop/FYP/enriched_subgraph_v2.graphml")
print(f"Starting: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

DENGUE_NODE = "EFO_0005547"
CL_NODE = "EFO_0005046"
TB_NODE = "MONDO_0018076"

# ── PART 1: Dengue — only FDA approved or clinical trial drugs ──
# These are manually curated known antivirals/repurposing candidates
# with evidence from clinical trials or published studies

print("\n── Adding curated Dengue candidates ──")

DENGUE_CANDIDATES = [
    ("CHEMBL203",    "Ivermectin",      "Approved antiparasitic, dengue clinical trials"),
    ("CHEMBL1200513","Chloroquine",      "Approved antimalarial, tested in dengue"),
    ("CHEMBL76",     "Chloroquine",      "Dengue NS3 inhibition evidence"),
    ("CHEMBL1643",   "Ribavirin",        "Broad antiviral, dengue evidence"),
    ("CHEMBL374478", "Celgosivir",       "Phase II dengue clinical trial"),
    ("CHEMBL2110737","Celgosivir HCl",   "Dengue clinical trial formulation"),
    ("CHEMBL1079",   "Prednisolone",     "Dengue severe case management"),
    ("CHEMBL1200868","Dexamethasone",    "Dengue clinical use"),
    ("CHEMBL1560",   "Lovastatin",       "Dengue clinical trial, anti-inflammatory"),
    ("CHEMBL862",    "Balapiravir",      "NS5 polymerase inhibitor, dengue Phase II"),
]

for chembl_id, name, evidence in DENGUE_CANDIDATES:
    if chembl_id not in G.nodes:
        G.add_node(chembl_id,
                   label='DRG',
                   name=name,
                   source='curated_dengue',
                   evidence=evidence)

    if not G.has_edge(chembl_id, DENGUE_NODE):
        G.add_edge(chembl_id, DENGUE_NODE,
                   relation='INDICATION',
                   label='DRG-DIS',
                   source='curated_dengue',
                   evidence=evidence)
        print(f"  Added: {name} → Dengue")

# ── PART 2: CL — manually curated from literature ──
print("\n── Adding curated CL candidates ──")

CL_CANDIDATES = [
    ("CHEMBL125",    "Miltefosine",       "Approved for visceral leishmaniasis, repurposed for CL"),
    ("CHEMBL239129", "Meglumine antimonate", "First-line CL treatment"),
    ("CHEMBL4297415","Antimony cation",   "Pentavalent antimonial, standard CL treatment"),
    ("CHEMBL1366",   "Auranofin",         "Gold compound, anti-leishmanial activity confirmed"),
    ("CHEMBL1401",   "Nitazoxanide",      "Broad antiparasitic, CL activity studies"),
    ("CHEMBL1200796","Cyclophosphamide",  "Immunomodulatory, proposed in systems biology study"),
    ("CHEMBL1282",   "Imiquimod",         "Topical immunomodulator, CL clinical use"),
    ("CHEMBL76",     "Chloroquine",       "Anti-leishmanial activity in vitro"),
    ("CHEMBL1200699","Doxycycline",       "CL adjunct therapy evidence"),
    ("CHEMBL1292",   "Clofazimine",       "Anti-leishmanial repurposing candidate"),
]

for chembl_id, name, evidence in CL_CANDIDATES:
    if chembl_id not in G.nodes:
        G.add_node(chembl_id,
                   label='DRG',
                   name=name,
                   source='curated_CL',
                   evidence=evidence)

    if not G.has_edge(chembl_id, CL_NODE):
        G.add_edge(chembl_id, CL_NODE,
                   relation='INDICATION',
                   label='DRG-DIS',
                   source='curated_CL',
                   evidence=evidence)
        print(f"  Added: {name} → CL")

# ── PART 3: DisGeNET — gene-disease associations for all 3 diseases ──
# Using DisGeNET REST API (free, no login needed for basic queries)
print("\n── Fetching DisGeNET gene-disease associations ──")

DISEASE_MAP = {
    "C0023281": ("EFO_0005046", "Cutaneous Leishmaniasis"),
    "C0041296": ("MONDO_0018076", "Tuberculosis"),
    "C0011311": ("EFO_0005547", "Dengue"),
}

headers = {"accept": "application/json"}

for umls_id, (disease_node, disease_name) in DISEASE_MAP.items():
    print(f"\n  Fetching genes for {disease_name}...")
    url = f"https://www.disgenet.org/api/gda/disease/{umls_id}?source=ALL&format=json&limit=50"

    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            associations = r.json()
            added = 0
            for assoc in associations:
                gene_id = assoc.get('geneid')
                gene_symbol = assoc.get('gene_symbol', '')
                score = assoc.get('score', 0)

                if not gene_id or score < 0.3:  # filter low confidence
                    continue

                ensembl_id = f"ENSG{str(gene_id).zfill(11)}"

                if ensembl_id not in G.nodes:
                    G.add_node(ensembl_id,
                               label='GEN',
                               name=gene_symbol,
                               source='DisGeNET',
                               disgenet_score=str(score))

                if not G.has_edge(ensembl_id, disease_node):
                    G.add_edge(ensembl_id, disease_node,
                               relation='ASSOCIATED_WITH',
                               label='GEN-DIS',
                               source='DisGeNET',
                               score=str(score))
                    added += 1

            print(f"  Added {added} gene-disease edges for {disease_name}")
        else:
            print(f"  HTTP {r.status_code} for {disease_name}")
        time.sleep(1)

    except Exception as e:
        print(f"  Error: {e}")

# ── PART 4: Add Leishmania protein targets from TriTrypDB manually ──
print("\n── Adding key Leishmania protein targets ──")

LEISH_TARGETS = [
    ("LmjF.13.1410", "Leishmania Arginase (LamARG)",
     "Key virulence target, Dabigatran binding evidence"),
    ("LmjF.36.2350", "Leishmania PTR1",
     "Pteridine reductase, drug target"),
    ("LmjF.29.2410", "Leishmania DHFR-TS",
     "Dihydrofolate reductase, antifolate target"),
    ("LmjF.10.0440", "Leishmania TryS",
     "Trypanothione synthetase, validated drug target"),
]

for target_id, name, evidence in LEISH_TARGETS:
    if target_id not in G.nodes:
        G.add_node(target_id,
                   label='TGT',
                   name=name,
                   source='TriTrypDB',
                   evidence=evidence)

    if not G.has_edge(target_id, CL_NODE):
        G.add_edge(target_id, CL_NODE,
                   relation='ASSOCIATED_WITH',
                   label='TGT-DIS',
                   source='TriTrypDB',
                   evidence=evidence)
        print(f"  Added target: {name} → CL")

# ── CLEAN and SAVE ──
print("\nCleaning None values...")
for node, data in G.nodes(data=True):
    for key, value in list(data.items()):
        if value is None:
            G.nodes[node][key] = ''

for u, v, data in G.edges(data=True):
    for key, value in list(data.items()):
        if value is None:
            data[key] = ''

nx.write_graphml(G, "C:/Users/abuba/Desktop/FYP/enriched_subgraph_v3_selective.graphml")
print("Saved as enriched_subgraph_v3_selective.graphml")

print(f"\nFinal: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

print("\n=== TARGET DISEASE CONNECTIONS ===")
for did, name in [("EFO_0005046", "Cutaneous Leishmaniasis"),
                   ("MONDO_0018076", "MDR-TB"),
                   ("EFO_0005547", "Dengue")]:
    if did in G.nodes:
        in_deg = G.in_degree(did)
        out_deg = G.out_degree(did)
        print(f"  {name}: {in_deg} incoming, {out_deg} outgoing")