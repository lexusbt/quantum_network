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
    print(f"  