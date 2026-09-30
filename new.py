import networkx as nx
from pyvis.network import Network

G = nx.read_graphml("enriched_subgraph_final.graphml")

CATEGORY_COLORS = {
    "GEN": "#4A90E2",
    "DRG": "#50E3C2",
    "DIS": "#E94E77",
    "PHE": "#F5A623",
    "BPO": "#9013FE",
}

net = Network(height="900px", width="100%", directed=True, notebook=False)

# Disable the heavy in-browser physics engine to load instantly
net.set_options("""
{
  "physics": {
    "enabled": false
  },
  "interaction": {
    "dragNodes": true,
    "zoomView": true,
    "hover": true
  }
}
""")

for node, data in G.nodes(data=True):
    node_type = data.get("label", "Unknown")
    color = CATEGORY_COLORS.get(node_type, "#999999")
    net.add_node(
        node,
        label=node,
        title=f"ID: {node}<br>Type: {node_type}",
        color=color,
        size=15 if node_type in ["DRG", "DIS"] else 8,
    )

for u, v, data in G.edges(data=True):
    relation = data.get("relation", "")
    edge_label = data.get("label", "")
    net.add_edge(u, v, title=f"Relation: {relation} ({edge_label})", arrows="to")

net.write_html("graph_visualization.html")
print("Saved! The file will now load instantly without hanging at 15%.")