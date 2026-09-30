import networkx as nx

G = nx.read_graphml("C:/Users/abuba/Desktop/FYP/enriched_subgraph_final.graphml")

# Convert to undirected so hops traverse edges regardless of direction
G_undirected = G.to_undirected()

targets = ["EFO_0005046", "MONDO_0018076", "EFO_0005547"]

for t in targets:
    if t in G_undirected.nodes:
        name = G.nodes[t].get("name", "no name")

        # Computes shortest path distance from node t up to cutoff hops
        lengths = nx.single_source_shortest_path_length(
            G_undirected, t, cutoff=3
        )

        hop_1 = sum(1 for dist in lengths.values() if dist == 1)
        hop_2 = sum(1 for dist in lengths.values() if dist == 2)
        hop_3 = sum(1 for dist in lengths.values() if dist == 3)
        total_within_3 = (
            len(lengths) - 1
        )  # exclude target node itself (dist 0)

        print(f"FOUND: {t} | {name}")
        print(f"  Direct (1-hop): {hop_1}")
        print(f"  Strictly 2-hop: {hop_2}")
        print(f"  Strictly 3-hop: {hop_3}")
        print(f"  Total reachable within 3 hops: {total_within_3}\n")
    else:
        print(f"MISSING: {t}\n")