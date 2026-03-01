"""
Test instance generation with just 5 instances
Run with: python -m scripts.test_instance_generation
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.qubo_generation.instance_generator import InstanceGenerator
import logging

logging.basicConfig(level=logging.INFO)

def main():
    print("\n" + "="*70)
    print("TESTING INSTANCE GENERATION (5 instances)")
    print("Using only downloaded datasets: ca-GrQc, email-Enron, p2p-Gnutella08")
    print("="*70 + "\n")
    
    # Use test config that only has the 3 downloaded datasets
    generator = InstanceGenerator(config_path="configs/datasets.yaml")
    
    # Generate just 5 instances for testing
    instances = generator.generate_all_instances(n_instances=5, seed=42)
    
    print("\n" + "="*70)
    print("✓ TEST COMPLETE!")
    print("="*70)
    print(f"\nGenerated: {len(instances)} instances")
    
    # Show first instance details
    if instances:
        print("\nSample instance:")
        inst = instances[0]
        print(f"  ID: {inst['instance_id']}")
        print(f"  Dataset: {inst['dataset_name']}")
        print(f"  Nodes: {inst['n_nodes']}, Edges: {inst['n_edges']}")
        print(f"  Source: {inst['source']}, Dest: {inst['dest']}, K: {inst['K']}")
        print(f"  Qubits: {inst['n_qubits']}")
        print(f"  Classical optimal: {inst['classical_optimal_length']} hops")
    
    print(f"\nFiles created in: {generator.instances_dir}")
    print("\nTo generate full 200 instances:")
    print("  1. Download missing datasets (ca-CondMat, ca-HepPh)")
    print("  2. Or modify configs/datasets.yaml to disable them")
    print("  3. Run: python -m scripts.generate_instances")

if __name__ == "__main__":
    main()