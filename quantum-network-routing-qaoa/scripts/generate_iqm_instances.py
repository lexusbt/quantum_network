"""
Generate small routing instances for IQM Sirius (≤16 qubits)
Simplified version - just QUBO generation for testing
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.graph_preprocessor import GraphPreprocessor
from src.qubo_generation.routing_qubo import RoutingQUBO
import networkx as nx
import numpy as np
import pickle
import pandas as pd

print("="*70)
print("GENERATING IQM-COMPATIBLE ROUTING INSTANCES")
print("="*70)

preprocessor = GraphPreprocessor()

# Create output directory
iqm_dir = Path("instances/iqm")
iqm_dir.mkdir(parents=True, exist_ok=True)

# Load and preprocess graph
print("\nLoading graph...")
G = preprocessor.load_snap_graph('ca-GrQc')
G = preprocessor.preprocess_graph(G, 'ca-GrQc')
print(f"Preprocessed graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Generate instances with different configurations
configs = [
    {'nodes': 4, 'K': 3, 'count': 5},   # ~12 qubits
    {'nodes': 4, 'K': 4, 'count': 5},   # ~16 qubits  
    {'nodes': 5, 'K': 3, 'count': 5},   # ~15 qubits
    {'nodes': 3, 'K': 5, 'count': 5},   # ~15 qubits
]

instances_metadata = []
total_count = 0

print("\nGenerating instances...")

for config in configs:
    n_nodes = config['nodes']
    K = config['K']
    count = config['count']
    
    print(f"\n{n_nodes} nodes × K={K} timesteps (target: ~{n_nodes * K} qubits)")
    
    for i in range(count):
        # Sample subgraph
        subgraph = preprocessor.sample_subgraph(
            G, 
            size=n_nodes, 
            method='random_walk',
            seed=total_count
        )
        
        # Select random source and destination
        nodes = list(subgraph.nodes())
        np.random.seed(total_count)
        s, t = np.random.choice(nodes, size=2, replace=False)
        
        # Get classical optimal path
        try:
            shortest_path = nx.shortest_path(subgraph, s, t)
            classical_optimal = len(shortest_path) - 1  # Number of hops
        except nx.NetworkXNoPath:
            print(f"  ✗ No path, skipping")
            continue
        
        # Generate QUBO
        qubo_gen = RoutingQUBO(penalty_A=0.47, penalty_B=1.0, penalty_C=1.0)
        Q, qubo_metadata = qubo_gen.generate_qubo(
            G=subgraph,
            source=s,
            dest=t,
            K=K,
            edge_weights="uniform"
        )
        
        n_qubits = Q.shape[0]
        
        # Create instance (minimal metadata for testing)
        instance = {
            'qubo_matrix': Q,
            'metadata': {
                'instance_id': f'iqm_{total_count:04d}',
                'dataset_name': 'ca-GrQc',
                'n_nodes': subgraph.number_of_nodes(),
                'n_edges': subgraph.number_of_edges(),
                'n_qubits': n_qubits,
                'K': K,
                'source': s,
                'target': t,
                'classical_optimal_length': classical_optimal,
                'split': 'iqm'
            }
        }
        
        # Save instance
        instance_file = iqm_dir / f"iqm_{total_count:04d}.pkl"
        with open(instance_file, 'wb') as f:
            pickle.dump(instance, f)
        
        # Track metadata
        instances_metadata.append({
            'instance_id': f'iqm_{total_count:04d}',
            'n_nodes': subgraph.number_of_nodes(),
            'n_qubits': n_qubits,
            'K': K,
            'dataset_name': 'ca-GrQc',
            'classical_optimal_length': classical_optimal,
            'split': 'iqm'
        })
        
        print(f"  ✓ iqm_{total_count:04d}: {n_qubits} qubits (optimal: {classical_optimal} hops)")
        total_count += 1

# Save metadata
metadata_df = pd.DataFrame(instances_metadata)
metadata_df.to_csv('instances/iqm/iqm_index.csv', index=False)

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Total instances generated: {total_count}")
print(f"Qubit range: {metadata_df['n_qubits'].min()}-{metadata_df['n_qubits'].max()}")
print(f"Average qubits: {metadata_df['n_qubits'].mean():.1f}")

print("\nQubit distribution:")
qubit_counts = metadata_df['n_qubits'].value_counts().sort_index()
for qubits, count in qubit_counts.items():
    print(f"  {qubits} qubits: {count} instances")

print("\nClassical optimal path lengths:")
opt_counts = metadata_df['classical_optimal_length'].value_counts().sort_index()
for length, count in opt_counts.items():
    print(f"  {length} hops: {count} instances")

print(f"\n✓ Saved to: instances/iqm/")
print(f"✓ Metadata: instances/iqm/iqm_index.csv")
print("\n✓ Ready for IQM Sirius testing!")
print("\nNext step:")
print("  python -m scripts.test_iqm_real_instance")