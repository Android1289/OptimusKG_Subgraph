import networkx as nx

# 1. Load the original graph
file_path = "C:/Users/abuba/Desktop/FYP/enriched_subgraph_v2.graphml"
G = nx.read_graphml(file_path)

print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# 2. Check connection counts for the 3 target diseases
target_diseases = [
    ("EFO_0005046", "Cutaneous Leishmaniasis"),
    ("MONDO_0018076", "MDR-TB"),
    ("EFO_0005547", "Dengue")
]

print("\n=== TARGET DISEASE CONNECTIONS ===")
for did, name in target_diseases:
    if did in G.nodes:
        print(f"  {name} ({did}): {G.degree(did)} connections")
    else:
        print(f"  {name} ({did}): Not present in graph")