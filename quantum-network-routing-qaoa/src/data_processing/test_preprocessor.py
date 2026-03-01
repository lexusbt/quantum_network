from src.data_processing.graph_preprocessor import GraphPreprocessor
import time

preprocessor = GraphPreprocessor()
G = preprocessor.load_snap_graph("ca-GrQc")
G = preprocessor.preprocess_graph(G, "ca-GrQc")

print("\nTesting size 20 subgraph generation...")
start = time.time()

for i in range(5):
    print(f"  Generating subgraph {i+1}/5...")
    subgraph = preprocessor.sample_subgraph(G, size=20, seed=42+i)
    print(f"    ✓ Generated {subgraph.number_of_nodes()} nodes in {time.time()-start:.1f}s total")

print(f"\nTotal time: {time.time()-start:.1f} seconds")