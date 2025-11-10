"""
Day 2: Experimental Scripts for Parameter Analysis
Run these experiments and record results in your worksheet
"""

import numpy as np
import networkx as nx
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time

# ============================================================================
# EXPERIMENT 1: Vary Penalty Weight A
# ============================================================================

def experiment_penalty_weights(G, demand, A_values=[1.0, 5.0, 10.0, 20.0, 50.0]):
    """Test different penalty weights"""
    print("="*60)
    print("EXPERIMENT 1: Penalty Weight Sensitivity")
    print("="*60)
    
    results = []
    
    for A in A_values:
        print(f"\nTesting A = {A}")
        
        # Construct QUBO with this penalty weight
        Q = construct_qubo_with_penalty(G, demand, A)
        
        # Brute force solve
        best_solution, best_cost = brute_force_solve_simple(Q)
        
        # Check if solution is valid (satisfies flow conservation)
        is_valid = check_flow_conservation(best_solution, G, demand)
        
        results.append({
            'A': A,
            'solution': best_solution,
            'cost': best_cost,
            'valid': is_valid
        })
        
        print(f"  Solution: {best_solution}")
        print(f"  Cost: {best_cost:.4f}")
        print(f"  Valid path: {is_valid}")
    
    # Summary table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'A':<8} {'Solution':<12} {'Cost':<10} {'Valid?'}")
    print("-"*60)
    for r in results:
        print(f"{r['A']:<8.1f} {str(r['solution']):<12} {r['cost']:<10.4f} {'✓' if r['valid'] else '✗'}")
    
    return results

def construct_qubo_with_penalty(network, demand, A):
    """Construct QUBO with specified penalty weight"""
    edges = list(network.edges())
    n = len(edges)
    Q = np.zeros((n, n))
    
    edge_to_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i
    
    # Objective
    for i, (u, v) in enumerate(edges):
        Q[i][i] += -np.log(network[u][v]['fidelity'])
    
    # Flow conservation with specified A
    idx_01 = edge_to_idx.get((0, 1))
    idx_12 = edge_to_idx.get((1, 2))
    
    if idx_01 is not None and idx_12 is not None:
        Q[idx_01][idx_01] += A
        Q[idx_12][idx_12] += A
        Q[idx_01][idx_12] += -2*A
        Q[idx_12][idx_01] += -2*A
    
    return Q

def check_flow_conservation(solution, network, demand):
    """Check if solution forms valid path"""
    edges = list(network.edges())
    active_edges = [edges[i] for i in range(len(solution)) if solution[i] == 1]
    
    # Check if path exists from source to destination
    if len(active_edges) == 0:
        return False
    
    # Check direct path
    if (0, 2) in active_edges or (2, 0) in active_edges:
        return len(active_edges) == 1
    
    # Check two-hop path
    if ((0, 1) in active_edges or (1, 0) in active_edges) and \
       ((1, 2) in active_edges or (2, 1) in active_edges):
        return len(active_edges) == 2
    
    return False

def brute_force_solve_simple(Q):
    """Simple brute force without printing"""
    n = Q.shape[0]
    best_solution = None
    best_cost = float('inf')
    
    for i in range(2**n):
        x = np.array([int(b) for b in format(i, f'0{n}b')])
        cost = x.T @ Q @ x
        if cost < best_cost:
            best_cost = cost
            best_solution = x
    
    return best_solution, best_cost

# ============================================================================
# EXPERIMENT 2: QAOA Parameter Sensitivity
# ============================================================================

def experiment_qaoa_sensitivity(Q, n_trials=5):
    """Test QAOA with different initial parameters"""
    print("\n" + "="*60)
    print("EXPERIMENT 2: QAOA Parameter Sensitivity")
    print("="*60)
    
    backend = AerSimulator()
    results = []
    
    # Get brute force optimum for comparison
    bf_solution, bf_cost = brute_force_solve_simple(Q)
    print(f"\nBrute force optimum: {bf_cost:.4f}\n")
    
    for trial in range(n_trials):
        print(f"Trial {trial+1}/{n_trials}")
        
        # Random initial parameters
        initial_gamma = np.random.uniform(0, np.pi)
        initial_beta = np.random.uniform(0, np.pi)
        
        print(f"  Initial: γ={initial_gamma:.4f}, β={initial_beta:.4f}")
        
        # Optimize
        def objective(params):
            gamma, beta = params
            qc = create_qaoa_simple(Q, gamma, beta)
            job = backend.run(qc, shots=1024)
            counts = job.result().get_counts()
            
            expectation = 0.0
            for bitstring, count in counts.items():
                x = np.array([int(b) for b in bitstring[::-1]])
                cost = x.T @ Q @ x
                expectation += cost * (count / 1024)
            return expectation
        
        result = minimize(objective, [initial_gamma, initial_beta],
                         method='COBYLA',
                         options={'maxiter': 30, 'disp': False})
        
        found_optimum = abs(result.fun - bf_cost) < 0.01
        
        results.append({
            'trial': trial + 1,
            'initial_gamma': initial_gamma,
            'initial_beta': initial_beta,
            'final_gamma': result.x[0],
            'final_beta': result.x[1],
            'final_cost': result.fun,
            'found_optimum': found_optimum,
            'iterations': result.nfev
        })
        
        print(f"  Final: γ={result.x[0]:.4f}, β={result.x[1]:.4f}")
        print(f"  Cost: {result.fun:.4f} {'✓' if found_optimum else '✗'}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    success_rate = sum(r['found_optimum'] for r in results) / n_trials * 100
    print(f"Success rate: {success_rate:.0f}%")
    print(f"Average iterations: {np.mean([r['iterations'] for r in results]):.1f}")
    print(f"\nOptimal parameter ranges:")
    successful = [r for r in results if r['found_optimum']]
    if successful:
        gammas = [r['final_gamma'] for r in successful]
        betas = [r['final_beta'] for r in successful]
        print(f"  γ: [{min(gammas):.4f}, {max(gammas):.4f}]")
        print(f"  β: [{min(betas):.4f}, {max(betas):.4f}]")
    
    return results

def create_qaoa_simple(Q, gamma, beta):
    """Simple QAOA circuit"""
    n = Q.shape[0]
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    
    # Cost
    for i in range(n):
        if abs(Q[i][i]) > 1e-10:
            qc.rz(2 * gamma * Q[i][i], i)
    for i in range(n):
        for j in range(i+1, n):
            if abs(Q[i][j]) > 1e-10:
                qc.cx(i, j)
                qc.rz(2 * gamma * Q[i][j], j)
                qc.cx(i, j)
    
    # Mixer
    for i in range(n):
        qc.rx(2 * beta, i)
    
    qc.measure(range(n), range(n))
    return qc

# ============================================================================
# EXPERIMENT 3: Shot Count Sensitivity
# ============================================================================

def experiment_shot_counts(Q, shot_counts=[512, 1024, 2048, 4096]):
    """Test different shot counts"""
    print("\n" + "="*60)
    print("EXPERIMENT 3: Shot Count Sensitivity")
    print("="*60)
    
    backend = AerSimulator()
    results = []
    
    for shots in shot_counts:
        print(f"\nTesting {shots} shots")
        
        start_time = time.time()
        
        # Run QAOA
        def objective(params):
            gamma, beta = params
            qc = create_qaoa_simple(Q, gamma, beta)
            job = backend.run(qc, shots=shots)
            counts = job.result().get_counts()
            
            expectation = 0.0
            for bitstring, count in counts.items():
                x = np.array([int(b) for b in bitstring[::-1]])
                cost = x.T @ Q @ x
                expectation += cost * (count / shots)
            return expectation
        
        result = minimize(objective, [0.5, 0.5],
                         method='COBYLA',
                         options={'maxiter': 20, 'disp': False})
        
        elapsed = time.time() - start_time
        
        # Get solution frequency
        qc_final = create_qaoa_simple(Q, result.x[0], result.x[1])
        job = backend.run(qc_final, shots=10000)
        counts = job.result().get_counts()
        max_freq = max(counts.values()) / 10000 * 100
        
        results.append({
            'shots': shots,
            'time': elapsed,
            'cost': result.fun,
            'frequency': max_freq
        })
        
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Cost: {result.fun:.4f}")
        print(f"  Solution frequency: {max_freq:.1f}%")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Shots':<8} {'Time(s)':<10} {'Cost':<10} {'Freq(%)'}")
    print("-"*60)
    for r in results:
        print(f"{r['shots']:<8} {r['time']:<10.2f} {r['cost']:<10.4f} {r['frequency']:.1f}")
    
    return results

# ============================================================================
# EXPERIMENT 4: Circuit Analysis
# ============================================================================

def experiment_circuit_analysis(Q, gamma_opt, beta_opt):
    """Analyze circuit complexity"""
    print("\n" + "="*60)
    print("EXPERIMENT 4: Circuit Complexity Analysis")
    print("="*60)
    
    qc = create_qaoa_simple(Q, gamma_opt, beta_opt)
    
    # Remove measurements for depth analysis
    qc_no_measure = qc.copy()
    qc_no_measure.remove_final_measurements(inplace=True)
    
    print("\nCircuit Statistics (p=1):")
    print(f"  Qubits: {qc.num_qubits}")
    print(f"  Classical bits: {qc.num_clbits}")
    print(f"  Total gates: {qc.size()}")
    print(f"  Circuit depth: {qc_no_measure.depth()}")
    print(f"\nGate breakdown:")
    ops = qc.count_ops()
    for gate, count in sorted(ops.items()):
        print(f"  {gate}: {count}")
    
    # Transpile
    backend = AerSimulator()
    qc_transpiled = transpile(qc, backend, optimization_level=3)
    
    print(f"\nAfter transpilation (optimization_level=3):")
    print(f"  Depth: {qc_transpiled.depth()}")
    print(f"  Gates: {qc_transpiled.size()}")
    
    # Hardware feasibility
    nisq_depth_limit = 100
    feasible = qc_transpiled.depth() < nisq_depth_limit
    
    print(f"\nNISQ Hardware Feasibility:")
    print(f"  Depth limit (typical): {nisq_depth_limit}")
    print(f"  Your circuit depth: {qc_transpiled.depth()}")
    print(f"  Feasible? {'✓ YES' if feasible else '✗ NO (too deep)'}")
    
    return {
        'qubits': qc.num_qubits,
        'depth': qc_no_measure.depth(),
        'gates': qc.size(),
        'transpiled_depth': qc_transpiled.depth(),
        'feasible': feasible
    }

# ============================================================================
# EXPERIMENT 5: Network Modification
# ============================================================================

def experiment_network_modifications(base_network):
    """Test how network changes affect optimal solution"""
    print("\n" + "="*60)
    print("EXPERIMENT 5: Network Modification Effects")
    print("="*60)
    
    demand = {'source': 0, 'destination': 2, 'priority': 1.0}
    
    scenarios = []
    
    # Scenario A: Shorter direct path
    print("\nScenario A: Make direct path shorter (4.0 → 2.5 km)")
    G_short = base_network.copy()
    G_short[0][2]['distance'] = 2.5
    update_network_params(G_short)
    
    Q_short = construct_qubo_with_penalty(G_short, demand, A=10.0)
    sol_short, cost_short = brute_force_solve_simple(Q_short)
    path_short = interpret_solution(sol_short, G_short)
    
    print(f"  Solution: {sol_short}")
    print(f"  Path: {path_short}")
    print(f"  Cost: {cost_short:.4f}")
    
    scenarios.append({
        'name': 'Short direct (2.5km)',
        'solution': sol_short,
        'path': path_short,
        'cost': cost_short
    })
    
    # Scenario B: Longer direct path
    print("\nScenario B: Make direct path longer (4.0 → 6.0 km)")
    G_long = base_network.copy()
    G_long[0][2]['distance'] = 6.0
    update_network_params(G_long)
    
    Q_long = construct_qubo_with_penalty(G_long, demand, A=10.0)
    sol_long, cost_long = brute_force_solve_simple(Q_long)
    path_long = interpret_solution(sol_long, G_long)
    
    print(f"  Solution: {sol_long}")
    print(f"  Path: {path_long}")
    print(f"  Cost: {cost_long:.4f}")
    
    scenarios.append({
        'name': 'Long direct (6.0km)',
        'solution': sol_long,
        'path': path_long,
        'cost': cost_long
    })
    
    # Scenario C: Original
    print("\nScenario C: Original network (baseline)")
    Q_orig = construct_qubo_with_penalty(base_network, demand, A=10.0)
    sol_orig, cost_orig = brute_force_solve_simple(Q_orig)
    path_orig = interpret_solution(sol_orig, base_network)
    
    print(f"  Solution: {sol_orig}")
    print(f"  Path: {path_orig}")
    print(f"  Cost: {cost_orig:.4f}")
    
    scenarios.append({
        'name': 'Original (4.0km)',
        'solution': sol_orig,
        'path': path_orig,
        'cost': cost_orig
    })
    
    # Summary
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    for s in scenarios:
        print(f"{s['name']:<20} Path: {s['path']:<15} Cost: {s['cost']:.4f}")
    
    return scenarios

def update_network_params(G):
    """Recalculate network parameters after distance change"""
    alpha = 0.2
    F_0 = 0.99
    
    for u, v in G.edges():
        d = G[u][v]['distance']
        eta = 10 ** (-alpha * d / 10)
        fidelity = F_0 * eta
        G[u][v]['eta'] = eta
        G[u][v]['fidelity'] = fidelity
        G[u][v]['weight'] = -np.log(fidelity)

def interpret_solution(solution, network):
    """Convert binary solution to path description"""
    edges = list(network.edges())
    active_edges = [edges[i] for i in range(len(solution)) if solution[i] == 1]
    
    if len(active_edges) == 1:
        return f"{active_edges[0][0]}->{active_edges[0][1]} (direct)"
    elif len(active_edges) == 2:
        # Find path order
        e1, e2 = active_edges
        if e1[1] == e2[0] or e1[1] == e2[1]:
            return f"{e1[0]}->{e1[1]}->{e2[0] if e2[0] != e1[1] else e2[1]} (2-hop)"
        else:
            return f"{e1[0]}->{e1[1]} + {e2[0]}->{e2[1]} (multi-path)"
    else:
        return f"{len(active_edges)} edges (complex)"

# ============================================================================
# MAIN: Run All Experiments
# ============================================================================

def run_all_experiments():
    """Execute all Day 2 experiments"""
    print("\n" + "="*70)
    print(" "*15 + "DAY 2: COMPREHENSIVE EXPERIMENTS")
    print("="*70)
    
    # Create base network
    print("\nCreating base 3-node network...")
    G = create_base_network()
    demand = {'source': 0, 'destination': 2, 'priority': 1.0}
    
    # Base QUBO
    Q_base = construct_qubo_with_penalty(G, demand, A=10.0)
    
    all_results = {}
    
    # Experiment 1: Penalty weights
    print("\n\nStarting Experiment 1...")
    all_results['penalty'] = experiment_penalty_weights(G, demand)
    
    input("\nPress Enter to continue to Experiment 2...")
    
    # Experiment 2: QAOA sensitivity
    print("\n\nStarting Experiment 2...")
    all_results['qaoa_sensitivity'] = experiment_qaoa_sensitivity(Q_base, n_trials=5)
    
    input("\nPress Enter to continue to Experiment 3...")
    
    # Experiment 3: Shot counts
    print("\n\nStarting Experiment 3...")
    all_results['shot_counts'] = experiment_shot_counts(Q_base)
    
    input("\nPress Enter to continue to Experiment 4...")
    
    # Experiment 4: Circuit analysis
    print("\n\nStarting Experiment 4...")
    # Use optimal parameters from experiment 2
    gamma_opt = 0.5  # Use a reasonable default
    beta_opt = 0.5
    all_results['circuit'] = experiment_circuit_analysis(Q_base, gamma_opt, beta_opt)
    
    input("\nPress Enter to continue to Experiment 5...")
    
    # Experiment 5: Network modifications
    print("\n\nStarting Experiment 5...")
    all_results['network_mods'] = experiment_network_modifications(G)
    
    # Final summary
    print("\n" + "="*70)
    print(" "*20 + "ALL EXPERIMENTS COMPLETE!")
    print("="*70)
    
    print("\nKey Findings:")
    print(f"1. Minimum penalty weight for valid solutions: A ≥ {find_min_penalty(all_results['penalty'])}")
    print(f"2. QAOA success rate: {calculate_success_rate(all_results['qaoa_sensitivity']):.0f}%")
    print(f"3. Recommended shots: {recommend_shots(all_results['shot_counts'])}")
    print(f"4. Circuit depth: {all_results['circuit']['depth']} (feasible: {all_results['circuit']['feasible']})")
    print(f"5. Network sensitivity: Path changes with distance modifications")
    
    return all_results

def create_base_network():
    """Create the standard 3-node test network"""
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2])
    
    edges = [(0, 1, 2.0), (1, 2, 3.0), (0, 2, 4.0)]
    alpha = 0.2
    F_0 = 0.99
    
    for u, v, d in edges:
        eta = 10 ** (-alpha * d / 10)
        fidelity = F_0 * eta
        G.add_edge(u, v, distance=d, eta=eta, fidelity=fidelity, 
                   weight=-np.log(fidelity))
    
    return G

def find_min_penalty(penalty_results):
    """Find minimum penalty weight that gives valid solution"""
    for r in sorted(penalty_results, key=lambda x: x['A']):
        if r['valid']:
            return r['A']
    return None

def calculate_success_rate(qaoa_results):
    """Calculate QAOA success rate"""
    successes = sum(r['found_optimum'] for r in qaoa_results)
    return successes / len(qaoa_results) * 100

def recommend_shots(shot_results):
    """Recommend minimum shots for good results"""
    # Find shots where frequency is high enough (>80%)
    for r in sorted(shot_results, key=lambda x: x['shots']):
        if r['frequency'] > 80:
            return r['shots']
    return 2048  # Default recommendation

# ============================================================================
# QUICK TESTS
# ============================================================================

def quick_test():
    """Quick test to verify everything works"""
    print("Running quick test...")
    
    G = create_base_network()
    demand = {'source': 0, 'destination': 2, 'priority': 1.0}
    Q = construct_qubo_with_penalty(G, demand, A=10.0)
    
    print("\n✓ Network created")
    print(f"✓ QUBO constructed: {Q.shape[0]}x{Q.shape[0]}")
    
    solution, cost = brute_force_solve_simple(Q)
    print(f"✓ Brute force works: solution {solution}, cost {cost:.4f}")
    
    print("\nAll systems operational! Ready for experiments.\n")

# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    print("""
    DAY 2 EXPERIMENTAL SUITE
    ========================
    
    Available functions:
    1. quick_test() - Verify setup
    2. run_all_experiments() - Run complete suite (recommended)
    3. Individual experiments:
       - experiment_penalty_weights(G, demand)
       - experiment_qaoa_sensitivity(Q)
       - experiment_shot_counts(Q)
       - experiment_circuit_analysis(Q, gamma, beta)
       - experiment_network_modifications(G)
    
    Usage:
    >>> quick_test()
    >>> results = run_all_experiments()
    
    OR run individual experiments:
    >>> G = create_base_network()
    >>> demand = {'source': 0, 'destination': 2, 'priority': 1.0}
    >>> experiment_penalty_weights(G, demand)
    """)
    
    # Uncomment to run:
    # quick_test()
    # results = run_all_experiments()
