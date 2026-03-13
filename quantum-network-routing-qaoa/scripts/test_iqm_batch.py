"""
Batch test multiple instances on IQM hardware
Run this whenever you want to test more instances
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.qaoa_optimization.iqm_qaoa_solver import IQMQAOASolver
import pandas as pd
import pickle
import time

print("="*70)
print("IQM BATCH TESTING")
print("="*70)

# Load IQM instances and filter for valid instances only
iqm_index = pd.read_csv('instances/iqm/iqm_index.csv')
iqm_index = iqm_index[iqm_index['n_qubits'] <= 16]  # Filter for IQM Sirius capacity
print(f"Valid instances for IQM Sirius (≤16 qubits): {len(iqm_index)}")

# Show available instances
print(f"\nAvailable instances: {len(iqm_index)}")
print("\nInstances by qubit count:")
print(iqm_index.groupby('n_qubits').size())

# Let user choose how many to test
print(f"\nYou have tested: {len(list(Path('results/iqm').glob('*.pkl')))} instances")

num_to_test = input("\nHow many instances to test? (1-10): ").strip()
try:
    num_to_test = int(num_to_test)
    num_to_test = min(max(1, num_to_test), 10)
except:
    num_to_test = 3

# Select instances to test (skip already tested)
tested_ids = [f.stem.replace('_iqm', '') for f in Path('results/iqm').glob('*_iqm.pkl')]
untested = iqm_index[~iqm_index['instance_id'].isin(tested_ids)]

if len(untested) == 0:
    print("\n✓ All instances already tested!")
    exit(0)

instances_to_test = untested.nsmallest(num_to_test, 'n_qubits')

print(f"\nSelected {len(instances_to_test)} instances to test:")
for _, row in instances_to_test.iterrows():
    print(f"  {row['instance_id']}: {row['n_qubits']} qubits, K={row['K']}, optimal={row['classical_optimal_length']} hops")

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
print("RUNNING QAOA ON IQM HARDWARE")
print("="*70)

results_summary = []

for idx, row in instances_to_test.iterrows():
    instance_id = row['instance_id']
    instance_file = f"instances/iqm/{instance_id}.pkl"
    
    print(f"\n[{idx+1}/{len(instances_to_test)}] Testing {instance_id}...")
    
    try:
        result = solver.solve_instance(instance_file, p=1, verbose=True)
        
        # Save result
        result_file = Path(f"results/iqm/{instance_id}_iqm.pkl")
        with open(result_file, 'wb') as f:
            pickle.dump(result, f)
        
        results_summary.append({
            'instance_id': instance_id,
            'n_qubits': result['n_qubits'],
            'best_cost': result['best_cost'],
            'approximation_ratio': result.get('approximation_ratio', 'N/A'),
            'time': result['elapsed_time']
        })
        
        print(f"  ✓ Success! Time: {result['elapsed_time']:.1f}s")
        
        # Small delay between jobs
        time.sleep(2)
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")

# Summary
print("\n" + "="*70)
print("BATCH TESTING COMPLETE")
print("="*70)

if results_summary:
    df = pd.DataFrame(results_summary)
    print(f"\nSuccessfully tested: {len(df)} instances")
    print(f"Total quantum time: {df['time'].sum():.1f}s")
    print(f"Average time per instance: {df['time'].mean():.1f}s")
    
    print("\nResults saved to: results/iqm/")

print("\n✓ Done! Results ready for analysis and presentation.")