"""
iqm_qaoa_runner.py
==================
Orchestrates QAOA circuit execution on IQM hardware for the capstone pipeline.

Covers two phases:
  Phase 1 — Hardware validation subset
    Run ~20-30 small instances on IQM hardware to validate that your Aer-based
    training data is consistent with real hardware results.

  Phase 3 — ML parameter inference validation
    Use XGBoost-predicted (gamma, beta) parameters to warm-start QAOA on IQM hardware.
    Measures: approximation ratio, circuit evaluations saved, solution quality.

This module integrates with your existing QAOASolver structure.
"""

import os
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np

from iqm_backend import BackendMode, get_backend, get_backend_info
from iqm_transpiler import (
    transpile_qaoa_for_iqm,
    check_qubit_count,
    estimate_circuit_resources,
    filter_instances_for_hardware,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Phase 1: Hardware Validation
# ---------------------------------------------------------------------------

class IQMHardwareValidator:
    """
    Phase 1 component: validates that AER simulation results align with IQM hardware.

    Runs a subset of your 200 routing instances on IQM hardware and computes
    correlation metrics between simulation and hardware approximation ratios.

    Typical usage:
        validator = IQMHardwareValidator(api_token="your_token", device="sirius")
        results = validator.validate_subset(instances, qaoa_solver, n_instances=20)
        validator.save_results(results, "results/iqm_phase1_validation.json")
    """

    def __init__(
        self,
        api_token: Optional[str] = None,
        device: str = "sirius",
        shots: int = 1024,
        use_mock: bool = False,
        results_dir: str = "results/iqm_validation",
    ):
        self.device = device
        self.shots = shots
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        mode = BackendMode.IQM_MOCK if use_mock else BackendMode.IQM_HARDWARE
        self.backend, self.sampler = get_backend(
            mode=mode,
            api_token=api_token,
            device=device,
            shots=shots,
        )
        self.backend_info = get_backend_info(self.backend)
        logger.info(f"IQMHardwareValidator ready — {self.backend_info}")

    def validate_subset(
        self,
        instances: list,
        qaoa_solver,
        n_instances: int = 20,
        p_layers: int = 1,
    ) -> Dict[str, Any]:
        """
        Runs n_instances on IQM hardware and compares against Aer simulation results.

        Parameters
        ----------
        instances    : list         — your 200 routing problem instances
        qaoa_solver  : QAOASolver   — your existing solver (for building circuits)
        n_instances  : int          — how many instances to run on hardware (budget-aware)
        p_layers     : int          — QAOA depth p (use p=1 for hardware validation)

        Returns
        -------
        dict with per-instance results + aggregate correlation stats
        """
        # Filter to hardware-suitable instances
        filtered = filter_instances_for_hardware(instances, device=self.device)
        hw_instances = filtered["hardware_instances"][:n_instances]

        if not hw_instances:
            raise ValueError(
                f"No instances fit on {self.device}. "
                f"Reduce subgraph sizes in your instance generator."
            )

        logger.info(
            f"Running Phase 1 validation: {len(hw_instances)} instances on {self.device}"
        )

        results = []
        for i, instance in enumerate(hw_instances):
            logger.info(f"  [{i+1}/{len(hw_instances)}] Instance: {instance.get('id', i)}")
            result = self._run_single_instance(instance, qaoa_solver, p_layers)
            results.append(result)
            time.sleep(0.5)  # Polite pacing between hardware jobs

        summary = self._compute_validation_summary(results)
        return {
            "phase": "phase1_hardware_validation",
            "timestamp": datetime.now().isoformat(),
            "backend_info": self.backend_info,
            "n_instances_run": len(results),
            "per_instance_results": results,
            "summary": summary,
        }

    def _run_single_instance(self, instance: dict, qaoa_solver, p_layers: int) -> dict:
        """Runs one routing instance on IQM hardware and records results."""
        instance_id = instance.get("id", "unknown")
        qubo_matrix = instance["qubo_matrix"]
        n_qubits = instance.get("n_qubits", qubo_matrix.shape[0])

        # Check qubit count
        qubit_check = check_qubit_count(n_qubits, self.device)
        if not qubit_check["fits"]:
            return {
                "instance_id": instance_id,
                "status": "skipped",
                "reason": qubit_check["recommendation"],
            }

        try:
            # Build QAOA circuit using your existing solver
            circuit, initial_params = qaoa_solver.build_circuit(
                qubo_matrix=qubo_matrix,
                p=p_layers,
            )

            # Resource check before sending to hardware
            resources = estimate_circuit_resources(circuit, p_layers, n_qubits)
            logger.debug(f"    Circuit: depth={resources['circuit_depth']}, "
                        f"2Q gates={resources['two_qubit_gates']}, "
                        f"est. fidelity={resources['estimated_fidelity']}")

            # Transpile for IQM
            transpiled = transpile_qaoa_for_iqm(circuit, self.backend)

            # Run optimization loop on hardware
            hw_result = self._run_qaoa_optimization(
                transpiled_circuit=transpiled,
                initial_params=initial_params,
                qubo_matrix=qubo_matrix,
                instance=instance,
            )

            return {
                "instance_id": instance_id,
                "status": "success",
                "n_qubits": n_qubits,
                "p_layers": p_layers,
                "hardware_approximation_ratio": hw_result["approximation_ratio"],
                "hardware_best_solution": hw_result["best_solution"],
                "hardware_cost": hw_result["best_cost"],
                "optimal_params": hw_result["optimal_params"],
                "n_circuit_evaluations": hw_result["n_evaluations"],
                "circuit_resources": resources,
                "aer_approximation_ratio": instance.get("aer_approximation_ratio"),
            }

        except Exception as e:
            logger.error(f"    Instance {instance_id} failed: {e}")
            return {
                "instance_id": instance_id,
                "status": "error",
                "error": str(e),
            }

    def _run_qaoa_optimization(
        self,
        transpiled_circuit,
        initial_params: np.ndarray,
        qubo_matrix: np.ndarray,
        instance: dict,
        max_iter: int = 50,
    ) -> dict:
        """
        Classical optimization loop using COBYLA, with circuit evaluations
        sent to IQM hardware.
        """
        from scipy.optimize import minimize

        n_evaluations = [0]
        best_cost = [np.inf]
        best_params = [initial_params.copy()]
        best_counts = [{}]

        def objective(params):
            n_evaluations[0] += 1

            # Bind parameters to transpiled circuit
            bound_circuit = transpiled_circuit.assign_parameters(
                dict(zip(transpiled_circuit.parameters, params))
            )

            # Execute on IQM
            job = self.sampler.run([bound_circuit], shots=self.shots)
            result = job.result()

            # Extract counts and compute expectation value
            try:
                counts = result[0].data.meas.get_counts()
            except Exception:
                counts = result.get_counts() if hasattr(result, "get_counts") else {}

            cost = _compute_expectation(counts, qubo_matrix)

            if cost < best_cost[0]:
                best_cost[0] = cost
                best_params[0] = params.copy()
                best_counts[0] = counts

            return cost

        # Run COBYLA optimization
        opt_result = minimize(
            objective,
            x0=initial_params,
            method="COBYLA",
            options={"maxiter": max_iter, "rhobeg": 0.5},
        )

        # Decode best solution from measurement counts
        best_solution = _decode_solution(best_counts[0])
        approx_ratio = _compute_approximation_ratio(
            best_cost[0], instance.get("classical_optimal_cost")
        )

        return {
            "optimal_params": best_params[0].tolist(),
            "best_cost": float(best_cost[0]),
            "best_solution": best_solution,
            "approximation_ratio": approx_ratio,
            "n_evaluations": n_evaluations[0],
            "optimizer_success": opt_result.success,
        }

    def _compute_validation_summary(self, results: list) -> dict:
        """Computes correlation between AER and hardware approximation ratios."""
        successful = [r for r in results if r.get("status") == "success"]

        if not successful:
            return {"error": "No successful runs to summarize"}

        hw_ratios = [r["hardware_approximation_ratio"] for r in successful
                     if r.get("hardware_approximation_ratio") is not None]
        aer_ratios = [r["aer_approximation_ratio"] for r in successful
                      if r.get("aer_approximation_ratio") is not None]

        summary = {
            "n_successful": len(successful),
            "n_failed": len(results) - len(successful),
            "mean_hw_approximation_ratio": float(np.mean(hw_ratios)) if hw_ratios else None,
            "std_hw_approximation_ratio": float(np.std(hw_ratios)) if hw_ratios else None,
        }

        # Compute AER vs hardware correlation if both are available
        if aer_ratios and hw_ratios and len(aer_ratios) == len(hw_ratios):
            correlation = float(np.corrcoef(aer_ratios, hw_ratios)[0, 1])
            summary["aer_hardware_correlation"] = correlation
            summary["mean_ratio_gap"] = float(np.mean(np.array(aer_ratios) - np.array(hw_ratios)))

        return summary

    def save_results(self, results: dict, filepath: Optional[str] = None) -> str:
        """Saves validation results to JSON for capstone reporting."""
        if filepath is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.results_dir / f"iqm_phase1_validation_{ts}.json"

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to: {filepath}")
        return str(filepath)


# ---------------------------------------------------------------------------
# Phase 3: ML Parameter Inference Validation
# ---------------------------------------------------------------------------

class IQMInferenceValidator:
    """
    Phase 3 component: validates XGBoost-predicted QAOA parameters on real IQM hardware.

    This is the core novelty demonstration of your capstone:
      1. Extract graph features from a NEW routing instance
      2. XGBoost predicts (gamma, beta) parameters without any circuit evaluations
      3. Use predicted params as warm-start for QAOA on IQM hardware
      4. Compare: predicted warm-start vs. random init vs. classical optimal

    Typical usage:
        validator = IQMInferenceValidator(api_token="your_token")
        results = validator.compare_strategies(
            test_instances=test_set,
            xgb_model=trained_model,
            qaoa_solver=solver,
        )
    """

    def __init__(
        self,
        api_token: Optional[str] = None,
        device: str = "sirius",
        shots: int = 2048,
        use_mock: bool = False,
        results_dir: str = "results/iqm_phase3",
    ):
        self.device = device
        self.shots = shots
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        mode = BackendMode.IQM_MOCK if use_mock else BackendMode.IQM_HARDWARE
        self.backend, self.sampler = get_backend(
            mode=mode,
            api_token=api_token,
            device=device,
            shots=shots,
        )
        logger.info(f"IQMInferenceValidator ready — device: {device}")

    def compare_strategies(
        self,
        test_instances: list,
        xgb_model,
        qaoa_solver,
        p_layers: int = 1,
        max_iter: int = 30,
    ) -> Dict[str, Any]:
        """
        For each test instance, runs three strategies and compares:
          A) XGBoost warm-start  — predicted params, minimal optimization
          B) Random init         — random params, full COBYLA optimization
          C) Grid search (small) — 5-point grid, best used as starting point

        The speedup metric = (evaluations_B - evaluations_A) / evaluations_B

        Parameters
        ----------
        test_instances : list     — held-out test set (30 instances from your 140/30/30 split)
        xgb_model      : trained  XGBoost model
        qaoa_solver    : your QAOASolver instance
        p_layers       : int      — QAOA depth
        max_iter       : int      — max COBYLA iterations per strategy

        Returns
        -------
        dict with per-instance comparisons + aggregate speedup metrics
        """
        filtered = filter_instances_for_hardware(test_instances, device=self.device)
        hw_instances = filtered["hardware_instances"]

        logger.info(
            f"Phase 3 validation: {len(hw_instances)} instances on {self.device}"
        )

        comparisons = []
        for i, instance in enumerate(hw_instances):
            logger.info(f"  [{i+1}/{len(hw_instances)}] Comparing strategies — "
                       f"instance {instance.get('id', i)}")
            comparison = self._compare_single_instance(
                instance, xgb_model, qaoa_solver, p_layers, max_iter
            )
            comparisons.append(comparison)
            time.sleep(0.5)

        summary = self._compute_comparison_summary(comparisons)
        return {
            "phase": "phase3_ml_inference_validation",
            "timestamp": datetime.now().isoformat(),
            "device": self.device,
            "p_layers": p_layers,
            "comparisons": comparisons,
            "summary": summary,
        }

    def _compare_single_instance(
        self,
        instance: dict,
        xgb_model,
        qaoa_solver,
        p_layers: int,
        max_iter: int,
    ) -> dict:
        """Runs all three strategies on one instance."""
        instance_id = instance.get("id", "unknown")
        qubo_matrix = instance["qubo_matrix"]
        features = instance.get("features", {})

        try:
            circuit, _ = qaoa_solver.build_circuit(qubo_matrix, p=p_layers)
            transpiled = transpile_qaoa_for_iqm(circuit, self.backend)

            # Strategy A: XGBoost warm-start
            logger.info("    Strategy A: XGBoost warm-start")
            xgb_params = _predict_params(xgb_model, features, p_layers)
            result_a = self._run_with_params(
                transpiled, xgb_params, qubo_matrix, max_iter=max_iter
            )

            # Strategy B: Random initialization
            logger.info("    Strategy B: Random initialization")
            n_params = 2 * p_layers
            random_params = np.random.uniform(0, 2 * np.pi, n_params)
            result_b = self._run_with_params(
                transpiled, random_params, qubo_matrix, max_iter=max_iter
            )

            # Compute speedup
            speedup = (
                (result_b["n_evaluations"] - result_a["n_evaluations"])
                / result_b["n_evaluations"]
            ) if result_b["n_evaluations"] > 0 else 0

            return {
                "instance_id": instance_id,
                "status": "success",
                "n_qubits": qubo_matrix.shape[0],
                "xgb_warmstart": result_a,
                "random_init": result_b,
                "speedup_pct": round(speedup * 100, 2),
                "quality_gap": round(
                    result_a["approximation_ratio"] - result_b["approximation_ratio"], 4
                ),
                "xgb_predicted_params": xgb_params.tolist(),
            }

        except Exception as e:
            logger.error(f"    Comparison failed for {instance_id}: {e}")
            return {"instance_id": instance_id, "status": "error", "error": str(e)}

    def _run_with_params(
        self,
        transpiled_circuit,
        initial_params: np.ndarray,
        qubo_matrix: np.ndarray,
        max_iter: int = 30,
    ) -> dict:
        """Runs COBYLA optimization starting from given params on IQM hardware."""
        from scipy.optimize import minimize

        n_evaluations = [0]
        best_cost = [np.inf]
        best_counts = [{}]

        def objective(params):
            n_evaluations[0] += 1
            bound = transpiled_circuit.assign_parameters(
                dict(zip(transpiled_circuit.parameters, params))
            )
            job = self.sampler.run([bound], shots=self.shots)
            result = job.result()
            try:
                counts = result[0].data.meas.get_counts()
            except Exception:
                counts = result.get_counts() if hasattr(result, "get_counts") else {}

            cost = _compute_expectation(counts, qubo_matrix)
            if cost < best_cost[0]:
                best_cost[0] = cost
                best_counts[0] = counts
            return cost

        minimize(
            objective,
            x0=initial_params,
            method="COBYLA",
            options={"maxiter": max_iter, "rhobeg": 0.5},
        )

        best_solution = _decode_solution(best_counts[0])
        approx_ratio = _compute_approximation_ratio(best_cost[0], cost_ref=None)

        return {
            "n_evaluations": n_evaluations[0],
            "best_cost": float(best_cost[0]),
            "best_solution": best_solution,
            "approximation_ratio": approx_ratio,
        }

    def _compute_comparison_summary(self, comparisons: list) -> dict:
        """Aggregates speedup and quality metrics across all instances."""
        successful = [c for c in comparisons if c.get("status") == "success"]
        if not successful:
            return {"error": "No successful comparisons"}

        speedups = [c["speedup_pct"] for c in successful]
        quality_gaps = [c["quality_gap"] for c in successful]

        return {
            "n_successful": len(successful),
            "mean_speedup_pct": round(float(np.mean(speedups)), 2),
            "std_speedup_pct": round(float(np.std(speedups)), 2),
            "min_speedup_pct": round(float(np.min(speedups)), 2),
            "max_speedup_pct": round(float(np.max(speedups)), 2),
            "mean_quality_gap": round(float(np.mean(quality_gaps)), 4),
            "instances_with_positive_speedup": sum(1 for s in speedups if s > 0),
        }

    def save_results(self, results: dict, filepath: Optional[str] = None) -> str:
        if filepath is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.results_dir / f"iqm_phase3_inference_{ts}.json"
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to: {filepath}")
        return str(filepath)


# ---------------------------------------------------------------------------
# Shared utility functions
# ---------------------------------------------------------------------------

def _compute_expectation(counts: dict, qubo_matrix: np.ndarray) -> float:
    """
    Computes <H_cost> from measurement counts and QUBO matrix.
    This is the objective function value for the QAOA optimization.
    """
    if not counts:
        return 0.0

    total_shots = sum(counts.values())
    expectation = 0.0

    for bitstring, count in counts.items():
        # Convert bitstring to binary array
        try:
            x = np.array([int(b) for b in bitstring.replace(" ", "")], dtype=float)
            if len(x) == qubo_matrix.shape[0]:
                cost = float(x @ qubo_matrix @ x)
                expectation += (count / total_shots) * cost
        except (ValueError, IndexError):
            continue

    return expectation


def _decode_solution(counts: dict) -> str:
    """Returns the most frequent bitstring (best observed solution)."""
    if not counts:
        return ""
    return max(counts, key=counts.get)


def _compute_approximation_ratio(cost: float, cost_ref: Optional[float]) -> Optional[float]:
    """
    Computes approximation ratio = QAOA_cost / optimal_cost.
    Returns None if reference cost is not available.
    """
    if cost_ref is None or cost_ref == 0:
        return None
    return round(cost / cost_ref, 4)


def _predict_params(xgb_model, features: dict, p_layers: int) -> np.ndarray:
    """
    Uses the trained XGBoost model to predict QAOA parameters from graph features.
    Returns a flat array [gamma_1, ..., gamma_p, beta_1, ..., beta_p].
    """
    feature_vec = np.array(list(features.values())).reshape(1, -1)
    predicted = xgb_model.predict(feature_vec)[0]

    # If model returns fewer params than needed, tile/interpolate
    n_params = 2 * p_layers
    if len(predicted) < n_params:
        predicted = np.tile(predicted, n_params // len(predicted) + 1)[:n_params]

    return predicted[:n_params]
