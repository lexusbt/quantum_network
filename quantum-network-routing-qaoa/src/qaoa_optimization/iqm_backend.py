"""
iqm_backend.py
==============
Backend abstraction layer for the QAOA routing capstone.
Supports seamless switching between:
  - AerSimulator       (local simulation, development)
  - IQM Mock/Facade    (IQM noise model simulation, pre-hardware testing)
  - IQM Sirius         (real IQM quantum hardware via Resonance)

Usage:
    from iqm_backend import get_backend, BackendMode

    # Development (no API key needed)
    backend, sampler = get_backend(BackendMode.SIMULATOR)

    # Pre-hardware validation with IQM noise model
    backend, sampler = get_backend(BackendMode.IQM_MOCK, api_token="your_token")

    # Real hardware
    backend, sampler = get_backend(BackendMode.IQM_HARDWARE, api_token="your_token")
"""

import os
import logging
from enum import Enum, auto
from typing import Optional, Tuple, Any

logger = logging.getLogger(__name__)


class BackendMode(Enum):
    """Execution modes for the QAOA routing pipeline."""
    SIMULATOR   = auto()   # AerSimulator — bulk training data generation (Phase 1)
    IQM_MOCK    = auto()   # IQM Facade — noisy simulation w/ IQM topology (pre-hardware)
    IQM_HARDWARE = auto()  # IQM Sirius real device via Resonance (Phase 1 validation + Phase 3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_backend(
    mode: BackendMode = BackendMode.SIMULATOR,
    api_token: Optional[str] = None,
    device: str = "sirius",
    shots: int = 1024,
) -> Tuple[Any, Any]:
    """
    Factory function that returns (backend, sampler) for the requested mode.

    Parameters
    ----------
    mode        : BackendMode — which execution environment to use
    api_token   : str         — IQM Resonance API token (required for IQM modes)
                               Falls back to IQM_TOKEN env variable if not provided.
    device      : str         — IQM device name (default: "sirius")
                               Use "sirius:mock" for mock/facade backend.
    shots       : int         — number of measurement shots

    Returns
    -------
    (backend, sampler) tuple — backend for transpilation info,
                               sampler for circuit execution
    """
    if mode == BackendMode.SIMULATOR:
        return _build_aer_backend(shots)

    # Resolve API token
    token = api_token or os.environ.get("IQM_TOKEN")
    if not token:
        raise ValueError(
            "IQM API token required. Pass api_token= or set the IQM_TOKEN "
            "environment variable. Get your token at https://resonance.meetiqm.com"
        )

    if mode == BackendMode.IQM_MOCK:
        return _build_iqm_mock_backend(token, device, shots)
    elif mode == BackendMode.IQM_HARDWARE:
        return _build_iqm_hardware_backend(token, device, shots)
    else:
        raise ValueError(f"Unknown BackendMode: {mode}")


def get_backend_info(backend) -> dict:
    """
    Returns key topology/calibration info from a backend.
    Useful for logging experiment metadata alongside results.
    """
    info = {"backend_name": getattr(backend, "name", str(backend))}
    try:
        # IQM backends expose this
        info["num_qubits"] = backend.num_qubits
        info["coupling_map"] = str(backend.coupling_map)
    except AttributeError:
        pass
    return info


# ---------------------------------------------------------------------------
# Private builders
# ---------------------------------------------------------------------------

def _build_aer_backend(shots: int):
    """Local AerSimulator — no API key, used for bulk training data generation."""
    try:
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import SamplerV2 as AerSampler
    except ImportError:
        raise ImportError(
            "qiskit-aer not found. Install with: pip install qiskit-aer"
        )

    backend = AerSimulator(method="statevector")
    sampler = AerSampler()
    sampler.options.default_shots = shots

    logger.info("✅ AerSimulator backend initialized (local simulation)")
    return backend, sampler


def _build_iqm_mock_backend(token: str, device: str, shots: int):
    """
    IQM Facade backend — combines mock remote execution with local noisy simulation.
    Uses IQM's actual qubit topology and calibration data without spending hardware quota.
    Perfect for validating transpilation and circuit structure before hardware runs.
    """
    _check_iqm_installed()
    from iqm.qiskit_iqm import IQMProvider

    # Append :mock if not already specified
    mock_device = device if device.endswith(":mock") else f"{device}:mock"

    provider = IQMProvider(
        url="https://resonance.meetiqm.com/",
        token=token,
    )
    backend = provider.get_backend(mock_device)
    sampler = _build_iqm_sampler(backend, shots)

    logger.info(f"✅ IQM Mock backend initialized — device: {mock_device}")
    return backend, sampler


def _build_iqm_hardware_backend(token: str, device: str, shots: int):
    """
    Real IQM quantum hardware via Resonance.
    Used for Phase 1 validation subset and Phase 3 ML-parameter inference validation.
    """
    _check_iqm_installed()
    from iqm.qiskit_iqm import IQMProvider

    provider = IQMProvider(
        url="https://resonance.meetiqm.com/",
        token=token,
    )
    backend = provider.get_backend(device)
    sampler = _build_iqm_sampler(backend, shots)

    logger.info(f"✅ IQM Hardware backend initialized — device: {device}")
    _log_hardware_info(backend)
    return backend, sampler


def _build_iqm_sampler(backend, shots: int):
    """
    Builds a Qiskit SamplerV2 primitive targeting the IQM backend.
    IQM uses the standard Qiskit primitives interface, so this is straightforward.
    """
    from qiskit.primitives import StatevectorSampler
    from qiskit_ibm_runtime import SamplerV2

    # IQM backends work with the standard Qiskit runtime sampler
    try:
        from qiskit_ibm_runtime import SamplerV2 as RuntimeSampler
        sampler = RuntimeSampler(backend)
        sampler.options.default_shots = shots
    except Exception:
        # Fallback: use backend.run() directly (wrapped in IQMSamplerFallback)
        sampler = IQMSamplerFallback(backend, shots)

    return sampler


def _check_iqm_installed():
    """Checks that iqm-client[qiskit] is installed and gives a helpful error if not."""
    try:
        import iqm.qiskit_iqm  # noqa: F401
    except ImportError:
        raise ImportError(
            "IQM Qiskit adapter not found.\n"
            "Install with: pip install 'iqm-client[qiskit]>=33.0.1'\n"
            "Documentation: https://docs.meetiqm.com/iqm-client/user_guide_qiskit.html"
        )


def _log_hardware_info(backend):
    """Logs calibration and topology info for the hardware backend."""
    try:
        logger.info(f"   Qubits available: {backend.num_qubits}")
        logger.info(f"   Coupling map: {backend.coupling_map}")
    except Exception:
        pass  # Not all backends expose these attrs


# ---------------------------------------------------------------------------
# Fallback sampler wrapper (used if qiskit-ibm-runtime is not installed)
# ---------------------------------------------------------------------------

class IQMSamplerFallback:
    """
    Minimal sampler wrapper for IQM backends that uses backend.run() directly.
    Provides the same .run(circuit, shots) interface as SamplerV2.
    """
    def __init__(self, backend, shots: int = 1024):
        self.backend = backend
        self.shots = shots
        self.options = type("Options", (), {"default_shots": shots})()

    def run(self, circuits, shots: Optional[int] = None, **kwargs):
        from qiskit import transpile
        from iqm.qiskit_iqm import transpile_to_IQM

        _shots = shots or self.shots
        if not isinstance(circuits, list):
            circuits = [circuits]

        results = []
        for circuit in circuits:
            try:
                transpiled = transpile_to_IQM(circuit, self.backend)
            except Exception:
                transpiled = transpile(circuit, self.backend, optimization_level=1)

            job = self.backend.run(transpiled, shots=_shots)
            result = job.result()
            results.append(result.get_counts())

        return IQMFallbackResult(results)


class IQMFallbackResult:
    """Mimics the SamplerV2 result interface for downstream compatibility."""
    def __init__(self, counts_list):
        self._counts_list = counts_list

    def get_counts(self, idx: int = 0):
        return self._counts_list[idx]

    def quasi_dists(self, idx: int = 0, shots: int = 1024):
        counts = self._counts_list[idx]
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}
