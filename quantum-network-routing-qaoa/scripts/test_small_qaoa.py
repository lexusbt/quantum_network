"""
Test QAOA on smallest instances only
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.qaoa_optimization.qaoa_solver import QAOASolver
import pandas as pd
import pickle

print("="*70)
print("TESTING QAOA ON SMALL INSTANCES")
print("="*70)

# Load master index to find small instances
df = pd.read_csv('instances/master_index.csv')
test_df = df[df['split'] == 'test']

# Filter for smallest instances (< 60 qubits)
small_instances = test_df[test_df['n_qubits'] < 60].head(3)

print(f"\nFound {len(small_instances)} small test instances:")
for _, row in small_instances.iterrows():
    print(f"  {row['instance_id']}: {row['n_qubits']} qubits, {row['n_nodes']} nodes")

instance_files = [f"instances/test/{row['instance_id']}.pkl" for _, row in small_instances.iterrows()]

# Initialize solver
solver = QAOASolver(
    backend="qasm_simulator",
    optimizer="COBYLA",
    shots=256,
    max_iterations=20
)

# Solve
print("\nSolving instances...\n")
results = solver.batch_solve(instance_files, p=1, save_results=True, output_dir="results/qaoa_test")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

if len(results) == 0:
    print("\n✗ No instances solved successfully")
else:
    df_results = pd.DataFrame(results)
    print(f"\nInstances solved: {len(results)}/{len(instance_files)}")
    
    if 'approximation_ratio' in df_results.columns:
        print(f"Average approximation ratio: {df_results['approximation_ratio'].mean():.3f}")
    if 'elapsed_time' in df_results.columns:
        print(f"Average time: {df_results['elapsed_time'].mean():.2f}s")
    
    print("\nIndividual results:")
    for r in results:
        print(f"  {r['instance_id']}: ratio={r.get('approximation_ratio', 'N/A'):.3f}, "
              f"time={r.get('elapsed_time', 0):.1f}s")

if __name__ == "__main__":
    pass