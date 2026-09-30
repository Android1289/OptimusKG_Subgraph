import networkx as nx
import requests
import time

G = nx.read_graphml("C:/Users/abuba/Desktop/FYP/enriched_subgraph_v3_selective.graphml")
print(f"Starting: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Paste your API token here
DISGENET_API_KEY = "7afacaed-1b2a-4e07-9c29-ac6db4b24814"

headers = {
    "accept": "application/json",
    # If the standard Bearer prefix fails with 401/403, change to: "Authorization": DISGENET_API_KEY
    "Authorization": f"Bearer {DISGENET_API_KEY}"
}

DISEASE_MAP = {
    "C0023281": ("EFO_0005046",   "Cutaneous Leishmaniasis"),
    "C0041296": ("MONDO_0018076", "Tuberculosis"),
    "C0011311": ("EFO_0005547",   "Dengue"),
}

for umls_id, (disease_node, disease_name) in DISEASE_MAP.items():
    print(f"\nFetching genes for {disease_name}...")
    url = f"https://www.disgenet.org/api/gda/disease/{umls_id}?source=ALL&format=json&limit=100"

    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            associations = r.json()
            added = 0
            for assoc in associations:
                gene_symbol = assoc.get('gene_symbol', '')
                score = float(assoc.get('score', 0))
                ensembl_ids = assoc.get('ensembl', [])

                if score < 0.3:
                    continue

                # Use Ensembl ID if available, otherwise use gene symbol as ID
                node_id = ensembl_ids[0] if ensembl_ids else f"GEN_{gene_symbol}"

                if node_id not in G.nodes:
                    G.add_node(
                        node_id,
                        label='GEN',
                        name=gene_symbol,
                        source='DisGeNET',
                        disgenet_score=str(score)
                    )

                if not G.has_edge(node_id, disease_node):
                    G.add_edge(
                        node_id, disease_node,
                        relation='ASSOCIATED_WITH',
                        label='GEN-DIS',
                        source='DisGeNET',
                        score=str(score)
                    )
                    added += 1

            print(f"  Added {added} gene-disease edges for {disease_name}")
        else:
            print(f"  HTTP {r.status_code}: {r.text[:150]}")

        time.sleep(1)

    except Exception as e:
        print(f"  Error: {e}")

# Clean None values before export
print("\nCleaning None values...")
for node, data in G.nodes(data=True):
    for key, value in list(data.items()):
        if value is None:
            G.nodes[node][key] = ''

for u, v, data in G.edges(data=True):
    for key, value in list(data.items()):
        if value is None:
            data[key] = ''

output_path = "C:/Users/abuba/Desktop/FYP/enriched_subgraph_v3_selective.graphml"
nx.write_graphml(G, output_path)
print(f"Saved as {output_path}")
print(f"Final: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Connection audit
print("\n=== TARGET DISEASE CONNECTIONS ===")
for did, name in [("EFO_0005046", "Cutaneous Leishmaniasis"),
                  ("MONDO_0018076", "MDR-TB"),
                  ("EFO_0005547", "Dengue")]:
    if did in G.nodes:
        if G.is_directed():
            print(f"  {name}: {G.in_degree(did)} incoming, {G.out_degree(did)} outgoing")
        else:
            print(f"  {name}: {G.degree(did)} connections")