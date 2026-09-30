import networkx as nx
import requests
import time

G = nx.read_graphml("C:/Users/abuba/Desktop/FYP/enriched_subgraph_v3.graphml")
print(f"Starting: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

DENGUE_NODE = "EFO_0005547"
CL_NODE = "EFO_0005046"

TARGETS = {
    "CHEMBL3908": ("Dengue NS5 polymerase", DENGUE_NODE),
    "CHEMBL364":  ("Leishmania major", CL_NODE),
    "CHEMBL3952": ("Leishmania donovani", CL_NODE),
}

edges_added = 0

for target_chembl_id, (target_name, disease_node) in TARGETS.items():
    print(f"\nFetching: {target_name}...")

    if target_chembl_id not in G.nodes:
        G.add_node(target_chembl_id, label='TGT', name=target_name, source='ChEMBL')

    if not G.has_edge(target_chembl_id, disease_node):
        G.add_edge(target_chembl_id, disease_node,
                   relation='ASSOCIATED_WITH', label='TGT-DIS', source='ChEMBL')

    url = f"https://www.ebi.ac.uk/chembl/api/data/activity?target_chembl_id={target_chembl_id}&limit=100&format=json"
    page = 0

    while url:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                print(f"  HTTP {r.status_code}, stopping")
                break

            data = r.json()
            activities = data.get('activities', [])
            page += 1

            for act in activities:
                compound_id = act.get('molecule_chembl_id')
                if not compound_id:
                    continue

                if compound_id not in G.nodes:
                    G.add_node(compound_id, label='DRG',
                               name=act.get('molecule_pref_name') or '',
                               source='ChEMBL')

                G.add_edge(compound_id, target_chembl_id,
                           relation='ACTIVE_AGAINST', label='DRG-TGT',
                           standard_type=str(act.get('standard_type') or ''),
                           standard_value=str(act.get('standard_value') or ''),
                           standard_units=str(act.get('standard_units') or ''),
                           source='ChEMBL')

                G.add_edge(compound_id, disease_node,
                           relation='ACTIVE_AGAINST', label='DRG-DIS',
                           source='ChEMBL')

                edges_added += 1

            print(f"  Page {page}: {len(activities)} fetched")

            next_url = data.get('page_meta', {}).get('next')
            url = f"https://www.ebi.ac.uk{next_url}" if next_url else None
            time.sleep(0.5)  # slightly longer delay to avoid timeouts

        except Exception as e:
            print(f"  Error on page {page}: {e}")
            print("  Waiting 30 seconds before retrying...")
            time.sleep(30)
            # retry same url
            continue

print("\nCleaning None values...")
for node, data in G.nodes(data=True):
    for key, value in list(data.items()):
        if value is None:
            G.nodes[node][key] = ''

for u, v, data in G.edges(data=True):
    for key, value in list(data.items()):
        if value is None:
            data[key] = ''

print(f"\nNew edges added: {edges_added}")
print(f"Final: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
nx.write_graphml(G, "C:/Users/abuba/Desktop/FYP/enriched_subgraph_v4.graphml")
print("Saved as enriched_subgraph_v4.graphml")