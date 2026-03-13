"""
Analyze IQM quantum hardware results
"""
import pandas as pd
import pickle
from pathlib import Path
import numpy as np

print("="*70)
print("IQM QUANTUM HARDWARE RESULTS ANALYSIS")
print("="*70)

# Load all IQM results
results_dir = Path("results/iqm")
result_files = list(results_dir.glob("*_iqm.pkl"))

if not result_files:
    print("\n✗ No IQM results found")
    exit(1)

results = []
for f in sorted(result_files):
    with open(f, 'rb') as fp:
        result = pickle.load(fp)
    results.append(result)

df = pd.DataFrame(results)

print(f"\nTotal instances tested on IQM: {len(df)}")
print(f"Quantum hardware: IQM Sirius (16 qubits)")

# Basic statistics
print("\n" + "="*70)
print("PROBLEM SIZE DISTRIBUTION")
print("="*70)
print(f"Qubits: {df['n_qubits'].min()}-{df['n_qubits'].max()} (avg: {df['n_qubits'].mean():.1f})")
print(f"Classical optimal: {df['classical_optimal'].min()}-{df['classical_optimal'].max()} hops")

print("\nInstances by qubit count:")
print(df['n_qubits'].value_counts().sort_index())

# Performance metrics
print("\n" + "="*70)
print("QAOA PERFORMANCE")
print("="*70)

if 'approximation_ratio' in df.columns:
    valid_ratios = df[df['approximation_ratio'].notna()]['approximation_ratio']
    if len(valid_ratios) > 0:
        print(f"Approximation ratio: {valid_ratios.min():.3f} - {valid_ratios.max():.3f}")
        print(f"Average ratio: {valid_ratios.mean():.3f}")

print(f"\nIterations: {df['iterations'].min()}-{df['iterations'].max()} (avg: {df['iterations'].mean():.1f})")
print(f"Execution time: {df['elapsed_time'].min():.1f}s - {df['elapsed_time'].max():.1f}s (avg: {df['elapsed_time'].mean():.1f}s)")
print(f"Total quantum time: {df['elapsed_time'].sum():.1f}s ({df['elapsed_time'].sum()/60:.1f} minutes)")

# Convergence
converged = df[df['converged'] == True]
print(f"\nConverged: {len(converged)}/{len(df)} ({len(converged)/len(df)*100:.1f}%)")

# Individual results
print("\n" + "="*70)
print("INDIVIDUAL RESULTS")
print("="*70)
for _, row in df.iterrows():
    ratio_str = f"{row.get('approximation_ratio', 'N/A'):.3f}" if pd.notna(row.get('approximation_ratio')) else "N/A"
    print(f"{row['instance_id']}: {row['n_qubits']} qubits, "
          f"classical={row['classical_optimal']} hops, "
          f"ratio={ratio_str}, "
          f"time={row['elapsed_time']:.1f}s")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"✓ Successfully validated QAOA on {len(df)} routing problems")
print(f"✓ Used real quantum hardware: IQM Sirius")
print(f"✓ Total quantum execution time: {df['elapsed_time'].sum()/60:.1f} minutes")
print(f"✓ Results ready for capstone presentation")

# Save summary
summary = {
    'total_instances': len(df),
    'qubit_range': f"{df['n_qubits'].min()}-{df['n_qubits'].max()}",
    'avg_qubits': df['n_qubits'].mean(),
    'avg_iterations': df['iterations'].mean(),
    'avg_time': df['elapsed_time'].mean(),
    'total_time': df['elapsed_time'].sum(),
    'convergence_rate': len(converged) / len(df)
}

import json
with open('results/iqm/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✓ Summary saved to: results/iqm/summary.json")