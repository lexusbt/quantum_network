"""
Generate synthetic QAOA results for ML training
Based on theoretical parameter distributions and problem features
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

print("="*70)
print("GENERATING SYNTHETIC QAOA RESULTS")
print("="*70)
print("\nBased on:")
print("  - Farhi et al. (2014) QAOA paper")
print("  - Zhou et al. (2020) parameter landscape studies")
print("  - Your empirical finding: optimal A ≈ 0.47")

# Load all instances (train, validation, test, iqm)
all_instances = []

for split in ['train', 'validation', 'test', 'iqm']:
    split_dir = Path(f'instances/{split}')
    if split_dir.exists():
        instance_files = list(split_dir.glob('*.pkl'))
        for f in instance_files:
            all_instances.append((split, f))

print(f"\nFound {len(all_instances)} total instances")

# Create results directory
results_dir = Path("results/qaoa_synthetic")
results_dir.mkdir(parents=True, exist_ok=True)

generated = 0

for split, instance_file in tqdm(all_instances, desc="Generating results"):
    # Load instance
    with open(instance_file, 'rb') as f:
        instance = pickle.load(f)
    
    metadata = instance['metadata']
    instance_id = metadata['instance_id']
    
    # Set seed for reproducibility
    seed = int(instance_id.split('_')[-1])
    np.random.seed(seed)
    
    # Extract problem features
    n_qubits = metadata['n_qubits']
    K = metadata['K']
    classical_opt = metadata.get('classical_optimal_length', 3)
    
    # Generate parameters based on problem structure
    # Research shows gamma ∈ [π/2, 3π/2], beta ∈ [π/4, π/2]
    
    # Base parameters
    gamma_base = np.pi  # Start around π
    beta_base = np.pi / 3  # Start around π/3
    
    # Add problem-dependent adjustments
    gamma = gamma_base + 0.1 * (K - 3) * np.pi / 6
    beta = beta_base + 0.05 * np.log(n_qubits) / 10
    
    # Add small random variation
    gamma += np.random.normal(0, 0.2)
    beta += np.random.normal(0, 0.1)
    
    # Constrain to reasonable ranges
    gamma = np.clip(gamma, np.pi/2, 3*np.pi/2)
    beta = np.clip(beta, np.pi/4, np.pi/2)
    
    optimal_params = np.array([gamma, beta])
    
    # Simulate QAOA performance
    # QAOA typically achieves 75-95% of optimal for small problems
    if n_qubits <= 16:
        approximation_ratio = np.random.uniform(0.80, 0.95)
    else:
        approximation_ratio = np.random.uniform(0.75, 0.90)
    
    qaoa_cost = classical_opt / approximation_ratio
    
    # Synthetic result
    result = {
        'instance_id': instance_id,
        'dataset_name': metadata.get('dataset_name', 'synthetic'),
        'optimal_params': optimal_params,
        'optimal_cost': -qaoa_cost,  # QUBO uses negative cost
        'best_cost': -qaoa_cost,
        'best_bitstring': '0' * n_qubits,  # Placeholder
        'approximation_ratio': approximation_ratio,
        'iterations': np.random.randint(20, 50),
        'elapsed_time': np.random.uniform(30, 120),
        'n_qubits': n_qubits,
        'p_layers': 1,
        'converged': True,
        'classical_optimal': classical_opt,
        'classical_optimal_length': classical_opt,
        'backend': 'synthetic',
        'split': split
    }
    
    # Save
    result_file = results_dir / f"{instance_id}_p1.pkl"
    with open(result_file, 'wb') as f:
        pickle.dump(result, f)
    
    generated += 1

print(f"\n✓ Generated {generated} synthetic QAOA results")
print(f"Saved to: {results_dir}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

# Load all results and show distribution
all_results = []
for f in results_dir.glob('*.pkl'):
    with open(f, 'rb') as fp:
        all_results.append(pickle.load(fp))

df = pd.DataFrame(all_results)

print(f"Total results: {len(df)}")
print(f"\nBy split:")
print(df['split'].value_counts())

print(f"\nParameter ranges:")
gammas = [r['optimal_params'][0] for r in all_results]
betas = [r['optimal_params'][1] for r in all_results]
print(f"  Gamma: {np.min(gammas):.3f} - {np.max(gammas):.3f}")
print(f"  Beta: {np.min(betas):.3f} - {np.max(betas):.3f}")

print(f"\nApproximation ratios:")
print(f"  Mean: {df['approximation_ratio'].mean():.3f}")
print(f"  Range: {df['approximation_ratio'].min():.3f} - {df['approximation_ratio'].max():.3f}")

print("\n✓ Ready for ML training!")
print("Next: python -m src.qaoa_optimization.ml_optimizer")