"""Test IQM QAOA on real routing instance"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.qaoa_optimization.iqm_qaoa_solver import IQMQAOASolver
import pandas as pd
import pickle

print("="*70)
print("IQM QAOA - REAL ROUTING INSTANCE TEST")
print("="*70)

# Load IQM instances
iqm_index = pd.read_csv('instances/iqm/iqm_index.csv')

# Select smallest instance for first test
smallest = iqm_index.nsmallest(1, 'n_qubits')
instance_id = smallest.iloc[0]['instance_id']
instance_file = f"instances/iqm/{instance_id}.pkl"

# Load instance details
with open(instance_file, 'rb') as f:
    instance = pickle.load(f)

metadata = instance['metadata']

print(f"\nSelected instance: {instance_id}")
print(f"  Dataset: {metadata['dataset_name']}")
print(f"  Nodes: {metadata['n_nodes']}")
print(f"  Qubits: {metadata['n_qubits']}")
print(f"  Path length K: {metadata['K']}")
print(f"  Classical optimal: {metadata['classical_optimal_length']} hops")

print("\n" + "="*70)
print("CONNECTING TO IQM SIRIUS")
print("="*70)

# Get IQM token
token = input("\nEnter your IQM Resonance token: ").strip()

# Initialize solver
solver = IQMQAOASolver(
    iqm_server_url="https://resonance.meetiqm.com/",
    quantum_computer="sirius",
    token=token,
    shots=1024,
    max_iterations=15
)

print("\n" + "="*70)
print("RUNNING QAOA ON IQM QUANTUM HARDWARE")
print("="*70)
print("This will submit a real quantum job to IQM Sirius")
print("Estimated time: 2-5 minutes\n")

confirm = input("Continue? (y/n): ")
if confirm.lower() != 'y':
    print("Cancelled")
    exit(0)

# Run QAOA
result = solver.solve_instance(instance_file, p=1, verbose=True)

# Results
print("\n" + "="*70)
print("RESULTS FROM IQM QUANTUM HARDWARE")
print("="*70)
print(f"Instance: {result['instance_id']}")
print(f"Dataset: {result['dataset_name']}")
print(f"Problem size: {result['n_qubits']} qubits")
print(f"\nClassical optimal: {result['classical_optimal']} hops")
print(f"QAOA best cost: {result['best_cost']:.4f}")
print(f"Best bitstring: {result['best_bitstring']}")
print(f"\nApproximation ratio: {result['approximation_ratio']:.3f}")
print(f"Iterations: {result['iterations']}")
print(f"Execution time: {result['elapsed_time']:.2f}s")

# Interpret result
if result['approximation_ratio'] <= 1.2:
    print("\n✓ Excellent! QAOA found near-optimal solution")
elif result['approximation_ratio'] <= 1.5:
    print("\n✓ Good! QAOA found reasonable solution")
else:
    print("\n⚠ QAOA solution is suboptimal (expected for noisy quantum hardware)")

# Save results
results_dir = Path("results/iqm")
results_dir.mkdir(parents=True, exist_ok=True)

result_file = results_dir / f"{instance_id}_iqm.pkl"
with open(result_file, 'wb') as f:
    pickle.dump(result, f)

print(f"\n✓ Results saved to: {result_file}")

# Update results summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("✓ Successfully ran QAOA on IQM quantum hardware")
print(f"✓ Tested on routing problem: {metadata['n_nodes']} nodes, K={metadata['K']}")
print(f"✓ Quantum circuit: {result['n_qubits']} qubits, p={result['p_layers']} layer")
print(f"✓ Hardware: IQM Sirius (16-qubit superconducting processor)")

print("\nNext steps:")
print("1. Test on more instances: modify script to loop through iqm_index.csv")
print("2. Compare quantum vs classical results")
print("3. Generate ML training data from quantum results")