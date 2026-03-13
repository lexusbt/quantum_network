"""
Run QAOA on simulator for 10 small instances
Shows the approach works at scale
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.qaoa_optimization.qaoa_solver import QAOASolver
import pandas as pd
import pickle

print("="*70)
print("QAOA SIMULATOR TESTING (For Presentation)")
print("="*70)

# Use main instance set - find smallest instances
master_df = pd.read_csv('instances/master_index.csv')

# Find smallest 10 instances
small_instances = master_df.nsmallest(10, 'n_qubits')

print(f"\nTesting {len(small_instances)} instances on simulator")
print(f"Qubit range: {small_instances['n_qubits'].min()}-{small_instances['n_qubits'].max()}")
print("(Reduced iterations for faster demonstration)")

# Initialize solver
solver = QAOASolver(
    backend="qasm_simulator",
    optimizer="COBYLA",
    shots=256,
    max_iterations=20
)

results = []
failed = []

for idx, row in small_instances.iterrows():
    instance_id = row['instance_id']
    split = row['split']
    instance_file = f"instances/{split}/{instance_id}.pkl"
    
    print(f"\n[{idx+1}/10] Solving {instance_id} ({row['n_qubits']} qubits)...", end=" ")
    
    try:
        result = solver.solve_instance(instance_file, p=1, verbose=False)
        
        # Save
        result_file = Path(f"results/qaoa_simulator/{instance_id}_sim.pkl")
        result_file.parent.mkdir(parents=True, exist_ok=True)
        with open(result_file, 'wb') as f:
            pickle.dump(result, f)
        
        results.append(result)
        
        # Show result
        ratio = result.get('approximation_ratio', 'N/A')
        ratio_str = f"{ratio:.3f}" if isinstance(ratio, (int, float)) else str(ratio)
        print(f"✓ {result['elapsed_time']:.1f}s, ratio={ratio_str}")
        
    except Exception as e:
        failed.append(instance_id)
        print(f"✗ {str(e)[:50]}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Successfully solved: {len(results)}/{len(small_instances)}")

if failed:
    print(f"Failed instances: {len(failed)}")
    for f in failed:
        print(f"  - {f}")

if results:
    df = pd.DataFrame(results)
    
    # Show metrics
    print(f"\nExecution metrics:")
    print(f"  Average time: {df['elapsed_time'].mean():.1f}s")
    print(f"  Total time: {df['elapsed_time'].sum():.1f}s")
    print(f"  Average iterations: {df['iterations'].mean():.1f}")
    
    # Approximation ratios (filter valid ones)
    valid_ratios = df[df['approximation_ratio'].notna() & (df['approximation_ratio'] > 0) & (df['approximation_ratio'] < 10)]
    
    if len(valid_ratios) > 0:
        print(f"\nApproximation ratios:")
        print(f"  Mean: {valid_ratios['approximation_ratio'].mean():.3f}")
        print(f"  Range: {valid_ratios['approximation_ratio'].min():.3f} - {valid_ratios['approximation_ratio'].max():.3f}")
        print(f"  Valid results: {len(valid_ratios)}/{len(df)}")
    
    print("\n✓ Simulator results saved to: results/qaoa_simulator/")
    print("✓ Ready for presentation analysis!")
else:
    print("\n⚠️ No instances solved successfully")
    print("Check instance files and QUBO formulation")