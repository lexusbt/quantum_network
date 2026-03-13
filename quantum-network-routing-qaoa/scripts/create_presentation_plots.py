"""
Generate plots for capstone presentation
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

print("Generating presentation plots...")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
fig_dir = Path("figures")
fig_dir.mkdir(exist_ok=True)

# Plot 1: Instance Size Distribution
print("1. Instance size distribution...")
master_df = pd.read_csv('instances/master_index.csv')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Qubit distribution
master_df['n_qubits'].hist(bins=20, ax=ax1, color='steelblue', edgecolor='black')
ax1.set_xlabel('Number of Qubits', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Instance Size Distribution', fontsize=14, fontweight='bold')
ax1.axvline(16, color='red', linestyle='--', linewidth=2, label='IQM Sirius Limit (16)')
ax1.legend()

# K distribution
master_df['K'].value_counts().sort_index().plot(kind='bar', ax=ax2, color='coral', edgecolor='black')
ax2.set_xlabel('Path Length (K)', fontsize=12)
ax2.set_ylabel('Number of Instances', fontsize=12)
ax2.set_title('Path Length Distribution', fontsize=14, fontweight='bold')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig(fig_dir / 'instance_distribution.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: figures/instance_distribution.png")

# Plot 2: QAOA Convergence (from IQM result)
print("2. QAOA convergence...")
iqm_result_file = list(Path('results/iqm').glob('*_iqm.pkl'))[0]
with open(iqm_result_file, 'rb') as f:
    iqm_result = pickle.load(f)

if 'cost_history' in iqm_result and len(iqm_result['cost_history']) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = range(1, len(iqm_result['cost_history']) + 1)
    costs = iqm_result['cost_history']
    
    ax.plot(iterations, costs, 'o-', linewidth=2, markersize=8, color='darkblue', label='QAOA Cost')
    ax.axhline(iqm_result['best_cost'], color='red', linestyle='--', linewidth=2, label=f"Best Solution: {iqm_result['best_cost']:.2f}")
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Cost Function Value', fontsize=12)
    ax.set_title('QAOA Convergence on IQM Sirius (16 qubits)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'qaoa_convergence.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: figures/qaoa_convergence.png")

# Plot 3: ML Training Results
print("3. ML model performance...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Placeholder - showing the concept
params = ['Gamma (γ)', 'Beta (β)']
mae_values = [0.1501, 0.0959]
r2_values = [-0.2266, -0.1623]

x = np.arange(len(params))
width = 0.35

ax1.bar(x, mae_values, width, label='MAE', color='steelblue', edgecolor='black')
ax1.set_ylabel('Mean Absolute Error', fontsize=12)
ax1.set_title('ML Prediction Error', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(params)

ax2.bar(x, r2_values, width, label='R²', color='coral', edgecolor='black')
ax2.axhline(0, color='red', linestyle='--', linewidth=1)
ax2.set_ylabel('R² Score', fontsize=12)
ax2.set_title('ML Model Performance', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(params)

plt.tight_layout()
plt.savefig(fig_dir / 'ml_performance.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: figures/ml_performance.png")

# Plot 4: Technology Stack Diagram
print("4. Architecture diagram...")
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Draw boxes for architecture
boxes = [
    {'pos': (0.5, 0.9), 'text': 'Network Routing\nProblem', 'color': 'lightcoral'},
    {'pos': (0.5, 0.75), 'text': 'QUBO\nFormulation', 'color': 'lightskyblue'},
    {'pos': (0.25, 0.55), 'text': 'Classical\nSimulator', 'color': 'lightgreen'},
    {'pos': (0.75, 0.55), 'text': 'IQM Sirius\nQuantum Hardware', 'color': 'gold'},
    {'pos': (0.5, 0.35), 'text': 'QAOA\nOptimization', 'color': 'plum'},
    {'pos': (0.5, 0.15), 'text': 'ML Parameter\nPrediction', 'color': 'peachpuff'},
]

for box in boxes:
    rect = plt.Rectangle((box['pos'][0]-0.15, box['pos'][1]-0.05), 0.3, 0.08, 
                          facecolor=box['color'], edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(box['pos'][0], box['pos'][1], box['text'], 
            ha='center', va='center', fontsize=11, fontweight='bold')

# Add arrows
arrows = [
    ((0.5, 0.82), (0.5, 0.8)),
    ((0.5, 0.7), (0.35, 0.6)),
    ((0.5, 0.7), (0.65, 0.6)),
    ((0.35, 0.5), (0.45, 0.4)),
    ((0.65, 0.5), (0.55, 0.4)),
    ((0.5, 0.3), (0.5, 0.23)),
]

for start, end in arrows:
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('Quantum-Enhanced Network Routing Pipeline', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(fig_dir / 'architecture.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: figures/architecture.png")

print("\n✓ All plots generated!")
print(f"Saved to: {fig_dir}/")
print("\nPlots created:")
print("  1. instance_distribution.png - Problem size distribution")
print("  2. qaoa_convergence.png - IQM quantum hardware convergence")
print("  3. ml_performance.png - ML model metrics")
print("  4. architecture.png - Complete system architecture")