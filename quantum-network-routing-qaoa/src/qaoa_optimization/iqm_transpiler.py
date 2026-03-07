"""
iqm_transpiler.py
=================
Handles transpilation of QAOA routing circuits to IQM's native gate set.

IQM Native Gates:
  - R(theta, phi)    — single-qubit rotation (replaces RX, RY, RZ)
  - CZ               — two-qubit controlled-Z (replaces CNOT/CX in QAOA cost layer)
  - Measurement

Your QAOA circuits use:
  - RZZ(2*gamma * w_ij)  → decomposed to CZ + RZ (IQM handles this automatically)
  - RX(2*beta)           → R(2*beta, 0) on IQM
  - CX (if any)          → CZ + H sandwich

This module provides:
  1. transpile_qaoa_for_iqm()  — main transpilation function
  2. estimate_circuit_depth()  — circuit depth analysis (important for NISQ)
  3. check_qubit_count()       — validates instance fits on target device
"""

import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


# IQM Sirius qubit counts (update as IQM releases larger devices)
IQM_DEVICE_QUBITS = {
    "sirius":      20,
    "garnet":       20,
    "emerald":      20,
    "deneb":        6,
}


def transpile_qaoa_for_iqm(
    circuit,
    backend,
    optimization_level: int = 1,
    use_iqm_transpiler: bool = True,
) -> any:
    """
    Transpiles a QAOA QuantumCircuit to IQM's native gate set.

    Parameters
    ----------
    circuit           : QuantumCircuit — your QAOA circuit (pre-bound parameters)
    backend           : IQM backend    — target IQM backend from get_backend()
    optimization_level: int            — 0-3, passed to Qiskit transpiler
    use_iqm_transpiler: bool           — use IQM's custom transpiler (recommended)
                                        falls back to standard Qiskit transpile

    Returns
    -------
    transpiled_circuit : QuantumCircuit — IQM-native circuit ready for hardware
    """
    if use_iqm_transpiler:
        try:
            from iqm.qiskit_iqm import transpile_to_IQM
            transpiled = transpile_to_IQM(circuit, backend)
            logger.debug(
                f"IQM transpilation complete — "
                f"depth: {transpiled.depth()}, "
                f"gates: {transpiled.count_ops()}"
            )
            return transpiled
        except Exception as e:
            logger.warning(f"IQM transpiler failed ({e}), falling back to Qiskit transpiler")

    # Standard Qiskit transpiler fallback
    from qiskit import transpile
    transpiled = transpile(circuit, backend, optimization_level=optimization_level)
    logger.debug(f"Qiskit transpilation complete — depth: {transpiled.depth()}")
    return transpiled


def check_qubit_count(n_qubits: int, device: str = "sirius") -> dict:
    """
    Checks whether a QAOA instance fits on the target IQM device.

    For your SNAP-based routing problems, the number of qubits = number of edges
    in the subgraph (one qubit per binary variable y_e in the QUBO).

    Parameters
    ----------
    n_qubits : int  — number of qubits needed (= number of QUBO variables)
    device   : str  — IQM device name

    Returns
    -------
    dict with keys: fits (bool), available_qubits (int), recommendation (str)
    """
    available = IQM_DEVICE_QUBITS.get(device, 20)
    fits = n_qubits <= available

    if fits:
        utilization = n_qubits / available
        recommendation = (
            "✅ Instance fits on hardware."
            if utilization <= 0.8
            else "⚠️ Near qubit limit — consider reducing subgraph size for better results."
        )
    else:
        recommendation = (
            f"❌ Instance too large ({n_qubits} qubits) for {device} ({available} available). "
            f"Reduce subgraph to ≤{available} edges. "
            f"For your SNAP datasets, use smaller subgraph samples or reduce the problem window."
        )

    result = {
        "fits": fits,
        "required_qubits": n_qubits,
        "available_qubits": available,
        "device": device,
        "utilization_pct": round(n_qubits / available * 100, 1),
        "recommendation": recommendation,
    }
    logger.info(recommendation)
    return result


def estimate_circuit_resources(circuit, p_layers: int, n_qubits: int) -> dict:
    """
    Estimates circuit resources before transpilation.
    Helps you decide which instances to send to hardware vs. keep on simulator.

    For QAOA with p layers on n qubits:
      - 2-qubit gates ≈ p * (number of edges in cost Hamiltonian)
      - Circuit depth ≈ p * (graph_diameter + n_qubits)

    Parameters
    ----------
    circuit  : QuantumCircuit — your QAOA circuit
    p_layers : int            — number of QAOA layers
    n_qubits : int            — number of qubits

    Returns
    -------
    dict with depth, gate counts, hardware suitability assessment
    """
    depth = circuit.depth()
    ops = circuit.count_ops()
    two_qubit_gates = ops.get("cx", 0) + ops.get("cz", 0) + ops.get("rzz", 0)

    # IQM Sirius approximate 2-qubit gate fidelity: ~99.5%
    # Estimated fidelity after all 2-qubit gates
    gate_fidelity = 0.995
    est_fidelity = gate_fidelity ** two_qubit_gates

    assessment = (
        "✅ Good for hardware"   if est_fidelity > 0.85 else
        "⚠️ Marginal for hardware — results may be noisy" if est_fidelity > 0.6 else
        "❌ Circuit too deep for current hardware — use p=1 or reduce problem size"
    )

    return {
        "circuit_depth": depth,
        "total_gates": sum(ops.values()),
        "two_qubit_gates": two_qubit_gates,
        "single_qubit_gates": sum(v for k, v in ops.items() if k not in ("cx", "cz", "rzz", "measure")),
        "p_layers": p_layers,
        "n_qubits": n_qubits,
        "estimated_fidelity": round(est_fidelity, 4),
        "hardware_assessment": assessment,
        "gate_counts": ops,
    }


def filter_instances_for_hardware(
    instances: list,
    device: str = "sirius",
    max_depth: int = 50,
    max_qubits: Optional[int] = None,
) -> dict:
    """
    Filters your 200 routing instances into hardware-suitable vs. simulator-only groups.
    Call this before starting a hardware run to avoid wasting quota.

    Parameters
    ----------
    instances  : list  — list of routing problem instance dicts
    device     : str   — IQM device name
    max_depth  : int   — maximum acceptable circuit depth for hardware
    max_qubits : int   — override device qubit limit (optional)

    Returns
    -------
    dict with keys:
      hardware_instances  — instances suitable for IQM hardware
      simulator_instances — instances too large for hardware
      summary             — stats dict
    """
    available_qubits = max_qubits or IQM_DEVICE_QUBITS.get(device, 20)

    hardware_ok = []
    sim_only = []

    for inst in instances:
        n_qubits = inst.get("n_qubits", inst.get("num_variables", 0))
        if n_qubits <= available_qubits:
            hardware_ok.append(inst)
        else:
            sim_only.append(inst)

    summary = {
        "total_instances": len(instances),
        "hardware_suitable": len(hardware_ok),
        "simulator_only": len(sim_only),
        "hardware_pct": round(len(hardware_ok) / len(instances) * 100, 1),
        "device": device,
        "qubit_limit": available_qubits,
    }

    logger.info(
        f"Instance filtering: {len(hardware_ok)}/{len(instances)} suitable for "
        f"{device} ({summary['hardware_pct']}%)"
    )
    return {
        "hardware_instances": hardware_ok,
        "simulator_instances": sim_only,
        "summary": summary,
    }
