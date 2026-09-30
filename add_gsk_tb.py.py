import pandas as pd
import networkx as nx

G = nx.read_graphml("C:/Users/abuba/Desktop/FYP/enriched_subgraph.graphml")
print(f"Starting: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

df = pd.read_excel("C:/Users/abuba/Desktop/FYP/gsk_tb.xls")

# Our TB disease node
TB_NODE = "MONDO_0018076"
MDR_TB_NODE = "EFO_0005046"

edges_added = 0

for _, row in df.iterrows():
    compound_id = str(row['DATABase number']).strip()
    gsk_id = str(row['GSKnumber']).strip()
    smiles = str(row['SMILES parent']).strip() if pd.notna(row['SMILES parent']) else ''
    formula = str(row['Formula version']).strip() if pd.notna(row['Formula version']) else ''
    mic90 = str(row['MIC90 BCG (µM)']).strip() if pd.notna(row['MIC90 BCG (µM)']) else ''

    # Add compound node if not already present
    if compound_id not in G.nodes:
        G.add_node(compound_id,
                   label='DRG',
                   name=gsk_id,
                   smiles=smiles,
                   formula=formula,
                   source='GSK_TCAMS_TB')

    # Add edge to TB disease node
    G.add_edge(compound_id, TB_NODE,
               relation='ACTIVE_AGAINST',
               label='DRG-DIS',
               mic90_bcg_um=mic90,
               source='GSK_TCAMS_TB')
    edges_added += 1

# Clean None values
for node, data in G.nodes(data=True):
    for key, value in list(data.items()):
        if value is None:
            G.nodes[node][key] = ''

for u, v, data in G.edges(data=True):
    for key, value in list(data.items()):
        if value is None:
            data[key] = ''

print(f"\nEdges added from GSK TCAMS TB: {edges_added}")
print(f"Final: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

nx.write_graphml(G, "C:/Users/abuba/Desktop/FYP/enriched_subgraph_v2.graphml")
print("Saved as enriched_subgraph_v2.graphml")