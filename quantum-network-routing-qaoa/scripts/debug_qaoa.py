"""
Debug QAOA solver on a single instance
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.qaoa_optimization.qaoa_solver import QAOASolver
import pickle
import traceback

# Load first test instance
instance_file = list(Path("instances/test").glob("instance_*.pkl"))[0]
print(f"Testing with: {instance_file}")
print("="*70)

# Load instance
with open(instance_file, 'rb') as f:
    instance = pickle.load(f)

metadata = instance['metadata']
Q = instance['qubo_matrix']

print(f"Instance ID: {metadata['instance_id']}")
print(f"Dataset: {metadata['dataset_name']}")
print(f"Nodes: {metadata['n_nodes']}")
print(f"Qubits: {metadata['n_qubits']}")
print(f"QUBO shape: {Q.shape}")
print(f"Classical optimal: {metadata['classical_optimal_length']} hops")

print("\n" + "="*70)
print("Running QAOA...")
print("="*70)

try:
    solver = QAOASolver(
        backend="qasm_simulator",
        optimizer="COBYLA",
        shots=256,
        max_iterations=20
    )
    
    result = solver.solve_instance(str(instance_file), p=1, verbose=True)
    
    print("\n" + "="*70)
    print("SUCCESS!")
    print("="*70)
    print(f"Best cost: {result['best_cost']}")
    print(f"Approximation ratio: {result.get('approximation_ratio', 'N/A')}")
    print(f"Time: {result['elapsed_time']:.2f}s")
    
except Exception as e:
    print("\n" + "="*70)
    print("ERROR!")
    print("="*70)
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    print("\nFull traceback:")
    traceback.print_exc()