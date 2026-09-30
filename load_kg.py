import polars as pl
import networkx as nx

nodes = pl.scan_parquet("C:/Users/abuba/Desktop/FYP/nodes.parquet")
edges = pl.scan_parquet("C:/Users/abuba/Desktop/FYP/edges.parquet")

disease_ids = {
    "Cutaneous Leishmaniasis": "EFO_0005046",
    "MDR-TB": "MONDO_0018076",
    "Dengue": "EFO_0005547",
}
disease_id_list = list(disease_ids.values())

DEGREE_CAP = 500

hop1_edges = edges.filter(
    pl.col("from").is_in(disease_id_list) | pl.col("to").is_in(disease_id_list)
).collect()
hop1_nodes = (set(hop1_edges["from"].to_list()) | set(hop1_edges["to"].to_list())) - set(disease_id_list)
print(f"1-hop neighbors: {len(hop1_nodes)}")

neighbor_list = list(hop1_nodes)
out_deg = edges.filter(pl.col("from").is_in(neighbor_list)).group_by("from").len().rename({"from": "node", "len": "out_degree"}).collect()
in_deg = edges.filter(pl.col("to").is_in(neighbor_list)).group_by("to").len().rename({"to": "node", "len": "in_degree"}).collect()
deg = out_deg.join(in_deg, on="node", how="full", coalesce=True).fill_null(0)
deg = deg.with_columns((pl.col("out_degree") + pl.col("in_degree")).alias("total_degree"))

hub_nodes = set(deg.filter(pl.col("total_degree") > DEGREE_CAP)["node"].to_list())
non_hub_neighbors = hop1_nodes - hub_nodes
print(f"Excluded as hubs (degree > {DEGREE_CAP}): {len(hub_nodes)}")
print(f"Non-hub 1-hop neighbors: {len(non_hub_neighbors)}")

CLINICAL_RELATIONS = ["INDICATION", "CONTRAINDICATION", "PARENT", "OFF_LABEL_USE"]
hop2_edges = edges.filter(
    (pl.col("from").is_in(list(non_hub_neighbors)) | pl.col("to").is_in(list(non_hub_neighbors)))
    & pl.col("relation").is_in(CLINICAL_RELATIONS)
).collect()
hop2_nodes = set(hop2_edges["from"].to_list()) | set(hop2_edges["to"].to_list())
print(f"Hop-2 nodes added: {len(hop2_nodes - hop1_nodes)}")

# --- KEY FIX: exclude hub nodes from the FINAL node set ---
subgraph_nodes = (set(disease_id_list) | non_hub_neighbors | hop2_nodes) - hub_nodes
print(f"\nFinal node count (hubs excluded): {len(subgraph_nodes)}")

sub_nodes_df = nodes.filter(pl.col("id").is_in(list(subgraph_nodes))).collect()
sub_edges_df = edges.filter(
    pl.col("from").is_in(list(subgraph_nodes)) & pl.col("to").is_in(list(subgraph_nodes))
).collect()
print(f"Subgraph edges (both endpoints in final set): {sub_edges_df.height}")

G = nx.MultiDiGraph()
for row in sub_nodes_df.iter_rows(named=True):
    G.add_node(row["id"], label=row["label"])
for row in sub_edges_df.iter_rows(named=True):
    G.add_edge(row["from"], row["to"], key=row["relation"], relation=row["relation"], label=row["label"])

print(f"\nFinal MultiDiGraph — Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
nx.write_graphml(G, "C:/Users/abuba/Desktop/FYP/disease_subgraph.graphml")
print("Saved to disease_subgraph.graphml")