"""
Test QAOA solver on generated instances
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.qaoa_optimization.qaoa_solver import QAOASolver
import pandas as pd

def main():
    print("="*70)
    print("TESTING QAOA SOLVER")
    print("="*70)
    
    # Load a few test instances
    instances_dir = Path("instances/test")
    instance_files = list(instances_dir.glob("instance_*.pkl"))[:3]  # Test on 3 instances
    
    if not instance_files:
        print("\n✗ No instances found. Generate instances first:")
        print("  python -m scripts.generate_instances")
        return
    
    print(f"\nFound {len(instance_files)} test instances")
    print(f"Testing QAOA with p=1 layers\n")
    
    # Initialize solver
    solver = QAOASolver(
        backend="qasm_simulator",
        optimizer="COBYLA",
        shots=512,  # Reduced for faster testing
        max_iterations=30
    )
    
    # Solve instances
    results = solver.batch_solve(
        [str(f) for f in instance_files],
        p=1,
        save_results=True,
        output_dir="results/qaoa_test"
    )
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    df = pd.DataFrame(results)
    print(f"\nInstances solved: {len(results)}")
    print(f"Average approximation ratio: {df['approximation_ratio'].mean():.3f}")
    print(f"Average time: {df['elapsed_time'].mean():.2f}s")
    print(f"Average iterations: {df['iterations'].mean():.1f}")
    
    print("\nIndividual results:")
    for r in results:
        print(f"  {r['instance_id']}: ratio={r['approximation_ratio']:.3f}, "
              f"time={r['elapsed_time']:.1f}s, iters={r['iterations']}")

if __name__ == "__main__":
    main()