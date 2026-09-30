import networkx as nx

G = nx.read_graphml("C:/Users/abuba/Desktop/FYP/enriched_subgraph_final.graphml")

print(f"Graph type: {type(G)}")

for did, name in [("EFO_0005046", "CL"),
                   ("MONDO_0018076", "MDR-TB"),
                   ("EFO_0005547", "Dengue")]:
    if did in G.nodes:
        out_deg = G.out_degree(did) if hasattr(G, 'out_degree') else 'N/A'
        in_deg = G.in_degree(did) if hasattr(G, 'in_degree') else 'N/A'
        total = G.degree(did)
        print(f"\n{name}:")
        print(f"  Total degree: {total}")
        print(f"  In-degree: {in_deg}")
        print(f"  Out-degree: {out_deg}")

# Check specifically if target->disease edges are being counted
print("\nChecking CHEMBL395 -> EFO_0005547:")
print(f"  Edge exists: {G.has_edge('CHEMBL395', 'EFO_0005547')}")
print(f"  CHEMBL395 out-degree: {G.out_degree('CHEMBL395')}")
print(f"  EFO_0005547 in-degree: {G.in_degree('EFO_0005547')}")