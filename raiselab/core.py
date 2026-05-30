# ==============================================================================
# Title       : Cached Scalable dvqe Framework for Monolithic and Distributed Execution
# Author      : Milad Hasanzadeh
# Revision    : Cached parameterized circuits + sampling energy + SPSA optimizer
#
# Main changes in this version:
# 1. Builds the parameterized ansatz circuit once.
# 2. Builds the DISKIT topology once.
# 3. Remaps the distributed circuit once.
# 4. Adds measurements once.
# 5. Tries to transpile the measured parameterized circuit once.
# 6. During optimization, only binds new parameter values and runs the simulator.
# 7. Uses sampling-based QUBO energy, not density matrices.
# 8. Uses SPSA, not finite-difference Adam.
# 9. Keeps the main solver name lowercase: dvqe(...)
# ==============================================================================

import random
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from qiskit import Aer, QuantumCircuit, ClassicalRegister, transpile
from qiskit.circuit import Parameter, QuantumRegister

from diskit import Topology, CircuitRemapper

warnings.filterwarnings("ignore")

sim = Aer.get_backend("qasm_simulator")


# ====================================================
# QUBO UTILITIES
# ====================================================

def qubo_cost(z: Union[np.ndarray, List[int]], Q: np.ndarray, q_linear: np.ndarray) -> float:
    """Compute QUBO objective z^T Q z + q^T z."""
    z = np.asarray(z, dtype=int)
    return float(z.T @ Q @ z + q_linear @ z)


def qubo_to_pauli_hamiltonian(Q: np.ndarray, q_linear: np.ndarray) -> List[Tuple[float, str]]:
    """
    QUBO-to-Pauli conversion kept for compatibility and inspection.

    Important: the scalable solver below does not build full Pauli matrices.
    Energy is computed from measured bitstrings and QUBO costs.
    """
    n = len(q_linear)
    term_dict = defaultdict(float)

    for i in range(n):
        q = q_linear[i]
        term_dict["I"] += q / 2.0
        term_dict[f"Z{i}"] += -q / 2.0

    for i in range(n):
        for j in range(i, n):
            qij = Q[i, j]
            if abs(qij) < 1e-12:
                continue

            if i == j:
                term_dict["I"] += qij / 2.0
                term_dict[f"Z{i}"] += -qij / 2.0
            else:
                term_dict["I"] += qij / 4.0
                term_dict[f"Z{i}"] += -qij / 4.0
                term_dict[f"Z{j}"] += -qij / 4.0
                term_dict[f"Z{i}Z{j}"] += qij / 4.0

    return [
        (float(coeff), pauli)
        for pauli, coeff in sorted(term_dict.items())
        if abs(coeff) > 1e-12
    ]


# ====================================================
# PARAMETERIZED ANSATZ GENERATION
# ====================================================

def create_monolithic_parameterized_ansatz(
    num_qubits: int,
    depth: int,
) -> Tuple[QuantumCircuit, List[Parameter]]:
    """
    Create a symbolic monolithic RX-RY ansatz.

    The circuit is built once. Numerical values are inserted later using
    assign_parameters or bind_parameters.
    """
    qc = QuantumCircuit(num_qubits)
    parameters = []
    p = 0

    for d in range(depth):
        for q in range(num_qubits):
            theta_ry = Parameter(f"theta_{p}")
            p += 1

            theta_rx = Parameter(f"theta_{p}")
            p += 1

            parameters.append(theta_ry)
            parameters.append(theta_rx)

            qc.ry(theta_ry, q)
            qc.rx(theta_rx, q)

        if d < depth - 1:
            for i in reversed(range(1, num_qubits)):
                qc.cx(i - 1, i)

    return qc, parameters


def create_distributed_parameterized_ansatz(
    qregs: List[QuantumRegister],
    depth: int = 1,
    num_qpus: int = 2,
    cross_qpu_entangle: bool = True,
) -> Tuple[QuantumCircuit, List[Parameter]]:
    """
    Create a symbolic distributed ansatz on QPU registers.

    Communication registers whose names start with 'com_' are not directly
    parameterized by the ansatz.
    """
    qc = QuantumCircuit(*qregs)

    system_qregs = [reg for reg in qregs if not reg.name.startswith("com_")]
    all_qubits = [q for reg in system_qregs for q in reg]

    parameters = []
    p = 0

    for d in range(depth):
        for q in all_qubits:
            theta_ry = Parameter(f"theta_{p}")
            p += 1

            theta_rx = Parameter(f"theta_{p}")
            p += 1

            parameters.append(theta_ry)
            parameters.append(theta_rx)

            qc.ry(theta_ry, q)
            qc.rx(theta_rx, q)

        if d < depth - 1:
            for i in range(len(all_qubits) - 1, 0, -1):
                qc.cx(all_qubits[i - 1], all_qubits[i])

    return qc, parameters


# ====================================================
# NUMERIC ANSATZ FUNCTIONS KEPT FOR COMPATIBILITY
# ====================================================

def bind_parameter_values(
    circuit: QuantumCircuit,
    parameters: List[Parameter],
    values: Union[List[float], np.ndarray],
) -> QuantumCircuit:
    """Bind numeric parameter values to a parameterized circuit."""
    values = np.asarray(values, dtype=float)

    if len(values) != len(parameters):
        raise ValueError(f"Expected {len(parameters)} parameters, got {len(values)}.")

    bind_dict = {p: float(v) for p, v in zip(parameters, values)}

    try:
        return circuit.assign_parameters(bind_dict, inplace=False)
    except AttributeError:
        return circuit.bind_parameters(bind_dict)


def create_monolithic_ansatz(
    num_qubits: int,
    depth: int,
    param_values: Union[List[float], np.ndarray],
) -> QuantumCircuit:
    """Create numeric monolithic RX-RY ansatz. Kept for compatibility."""
    qc, parameters = create_monolithic_parameterized_ansatz(num_qubits, depth)
    return bind_parameter_values(qc, parameters, param_values)


def create_ansatz_on_topology_regs(
    qregs: List[QuantumRegister],
    depth: int = 1,
    num_qpus: int = 2,
    cross_qpu_entangle: bool = True,
    parametric: bool = False,
    param_values: Optional[Union[List[float], np.ndarray]] = None,
) -> QuantumCircuit:
    """Create distributed ansatz. Kept for compatibility with old code."""
    qc, parameters = create_distributed_parameterized_ansatz(
        qregs=qregs,
        depth=depth,
        num_qpus=num_qpus,
        cross_qpu_entangle=cross_qpu_entangle,
    )

    if parametric or param_values is None:
        return qc

    return bind_parameter_values(qc, parameters, param_values)


def remap_with_diskit(ansatz_qc: QuantumCircuit, topology: Topology) -> QuantumCircuit:
    """Apply DISKIT circuit remapping."""
    remapper = CircuitRemapper(topology)
    return remapper.remap_circuit(ansatz_qc)


# ====================================================
# SAMPLING AND ENERGY UTILITIES
# ====================================================

def get_computational_qubits(circuit: QuantumCircuit, mode: str, n: int):
    """Return only computational QUBO qubits; exclude communication qubits."""
    if mode == "distributed":
        comp_qubits = []
        for reg in circuit.qregs:
            if not reg.name.startswith("com_"):
                comp_qubits.extend(reg)
        return comp_qubits[:n]

    return list(circuit.qubits[:n])


def add_qubo_measurements_once(
    circuit: QuantumCircuit,
    mode: str,
    n: int,
) -> QuantumCircuit:
    """Add measurements to a copy of the circuit once."""
    measured = circuit.copy()
    creg = ClassicalRegister(n, "qubo")
    measured.add_register(creg)

    comp_qubits = get_computational_qubits(measured, mode, n)

    if len(comp_qubits) < n:
        raise ValueError(f"Need {n} computational qubits, found {len(comp_qubits)}")

    for i in range(n):
        measured.measure(comp_qubits[i], creg[i])

    return measured


def normalize_counts_to_qubo_bits(counts_raw: Dict[str, int], n: int) -> Dict[str, int]:
    """
    Convert Qiskit count strings to QUBO bitstrings ordered as z[0], ..., z[n-1].
    """
    histogram = defaultdict(int)

    for bitstring, count in counts_raw.items():
        clean = bitstring.replace(" ", "")
        bits_lsb_first = clean[::-1]
        qubo_bits = bits_lsb_first[:n]
        histogram[qubo_bits] += count

    return dict(histogram)


def estimate_energy_from_histogram(
    histogram: Dict[str, int],
    Q: np.ndarray,
    q_linear: np.ndarray,
) -> float:
    """Estimate expected QUBO objective value from sampled bitstrings."""
    total = sum(histogram.values())

    if total <= 0:
        raise RuntimeError("Empty histogram. Cannot estimate energy.")

    energy = 0.0

    for bstring, count in histogram.items():
        z = np.array([int(b) for b in bstring], dtype=int)
        energy += (count / total) * qubo_cost(z, Q, q_linear)

    return float(energy)


# ====================================================
# TOPOLOGY HELPER
# ====================================================

def prepare_topology(mode: str, n: int, qpu_qubit_config: Optional[List[int]]):
    """Prepare DISKIT topology for distributed mode."""
    if mode == "monolithic":
        return None, 1, [n]

    if qpu_qubit_config is None:
        raise ValueError("qpu_qubit_config is required for distributed mode.")

    if sum(qpu_qubit_config) < n:
        raise ValueError(
            f"QPU config provides insufficient qubits: need {n}, got {sum(qpu_qubit_config)}"
        )

    cumulative = 0
    adjusted = []

    for q in qpu_qubit_config:
        if cumulative + q <= n:
            adjusted.append(q)
            cumulative += q
        else:
            adjusted.append(n - cumulative)
            break

    adjusted = [q for q in adjusted if q > 0]
    num_qpus = len(adjusted)

    topo = Topology()
    topo.create_qmap(num_qpus, adjusted)

    return topo, num_qpus, adjusted


# ====================================================
# CACHED CIRCUIT EVALUATOR
# ====================================================

class CachedCircuitEvaluator:
    """
    Builds the quantum circuit infrastructure once and reuses it.

    Cached once:
    - topology
    - QPU registers
    - symbolic ansatz
    - DISKIT remapped circuit
    - measured circuit
    - transpiled measured circuit, when possible
    - parameter order

    Repeated every energy call:
    - bind parameter values
    - run simulator
    - compute sampled QUBO energy
    """

    def __init__(
        self,
        mode: str,
        Q: np.ndarray,
        q_linear: np.ndarray,
        depth: int,
        qpu_qubit_config: Optional[List[int]],
        backend=None,
        transpile_once: bool = True,
        transpile_optimization_level: int = 0,
    ):
        self.mode = mode
        self.Q = np.asarray(Q, dtype=float)
        self.q_linear = np.asarray(q_linear, dtype=float)
        self.depth = depth
        self.n = len(self.q_linear)
        self.backend = backend if backend is not None else sim
        self.transpile_once = transpile_once
        self.transpile_optimization_level = transpile_optimization_level

        if self.Q.shape != (self.n, self.n):
            raise ValueError(f"Q must have shape {(self.n, self.n)}, got {self.Q.shape}.")

        self.topo, self.num_qpus, self.qpu_qubit_config = prepare_topology(
            mode=self.mode,
            n=self.n,
            qpu_qubit_config=qpu_qubit_config,
        )

        self.base_circuit, self.parameters = self._build_base_parameterized_circuit()

        self.measured_circuit = add_qubo_measurements_once(
            self.base_circuit,
            self.mode,
            self.n,
        )

        self.run_circuit_template = self._try_transpile_once(self.measured_circuit)

    @property
    def num_params(self) -> int:
        return len(self.parameters)

    def _build_base_parameterized_circuit(self) -> Tuple[QuantumCircuit, List[Parameter]]:
        if self.mode == "monolithic":
            return create_monolithic_parameterized_ansatz(self.n, self.depth)

        if self.mode == "distributed":
            qregs = self.topo.get_regs()

            ansatz_qc, parameters = create_distributed_parameterized_ansatz(
                qregs=qregs,
                depth=self.depth,
                num_qpus=self.num_qpus,
            )

            remapped_qc = remap_with_diskit(ansatz_qc, self.topo)
            return remapped_qc, parameters

        raise ValueError("mode must be 'monolithic' or 'distributed'.")

    def _try_transpile_once(self, circuit: QuantumCircuit) -> QuantumCircuit:
        if not self.transpile_once:
            return circuit

        try:
            return transpile(
                circuit,
                backend=self.backend,
                optimization_level=self.transpile_optimization_level,
            )
        except Exception:
            return circuit

    def bind(self, param_values: Union[List[float], np.ndarray]) -> QuantumCircuit:
        return bind_parameter_values(
            self.run_circuit_template,
            self.parameters,
            param_values,
        )

    def counts(
        self,
        param_values: Union[List[float], np.ndarray],
        num_shots: int = 1024,
    ) -> Dict[str, int]:
        bound_circuit = self.bind(param_values)
        result = self.backend.run(bound_circuit, shots=num_shots).result()
        counts_raw = result.get_counts()
        return normalize_counts_to_qubo_bits(counts_raw, self.n)

    def energy(
        self,
        param_values: Union[List[float], np.ndarray],
        num_shots: int = 1024,
    ) -> float:
        histogram = self.counts(param_values, num_shots=num_shots)
        return estimate_energy_from_histogram(histogram, self.Q, self.q_linear)

    def energy_and_histogram(
        self,
        param_values: Union[List[float], np.ndarray],
        num_shots: int = 1024,
    ) -> Tuple[float, Dict[str, int]]:
        histogram = self.counts(param_values, num_shots=num_shots)
        energy = estimate_energy_from_histogram(histogram, self.Q, self.q_linear)
        return energy, histogram

    def circuit_with_bound_parameters(
        self,
        param_values: Union[List[float], np.ndarray],
    ) -> QuantumCircuit:
        """Return final bound circuit using the cached measured circuit template."""
        return self.bind(param_values)

    def circuit_without_measurements(
        self,
        param_values: Union[List[float], np.ndarray],
    ) -> QuantumCircuit:
        """Return final bound circuit without the measurement-added wrapper."""
        return bind_parameter_values(
            self.base_circuit,
            self.parameters,
            param_values,
        )


# ====================================================
# BACKWARD-COMPATIBLE ENERGY FUNCTION
# ====================================================

def compute_sampling_energy(
    param_values: Union[List[float], np.ndarray],
    Q: np.ndarray,
    q_linear: np.ndarray,
    mode: str,
    depth: int,
    n: int,
    topo: Optional[Topology] = None,
    num_qpus: int = 1,
    num_shots: int = 1024,
    backend=None,
    return_circuit: bool = False,
):
    """
    Backward-compatible one-shot sampling energy.

    This rebuilds a temporary evaluator, so dvqe(...) does not use this function
    inside its optimization loop. It is kept for external compatibility.
    """
    if mode == "monolithic":
        qpu_config = [n]
    else:
        qpu_config = [n]

    evaluator = CachedCircuitEvaluator(
        mode=mode,
        Q=Q,
        q_linear=q_linear,
        depth=depth,
        qpu_qubit_config=qpu_config,
        backend=backend,
        transpile_once=True,
    )

    energy, histogram = evaluator.energy_and_histogram(
        param_values,
        num_shots=num_shots,
    )

    circuit = evaluator.circuit_without_measurements(param_values)

    if return_circuit:
        return energy, circuit, histogram

    return energy


# ====================================================
# SPSA OPTIMIZER
# ====================================================

class SPSAOptimizer:
    """
    SPSA optimizer.

    It uses two energy evaluations per iteration, independent of the number of
    variational parameters.
    """

    def __init__(
        self,
        a: float = 0.1,
        c: float = 0.15,
        alpha: float = 0.602,
        gamma: float = 0.101,
        A: float = 10.0,
        seed: Optional[int] = None,
    ):
        self.a = a
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        self.A = A
        self.rng = np.random.default_rng(seed)

    def gains(self, k: int) -> Tuple[float, float]:
        ak = self.a / ((k + 1 + self.A) ** self.alpha)
        ck = self.c / ((k + 1) ** self.gamma)
        return ak, ck

    def step(self, theta: np.ndarray, k: int, energy_function):
        theta = np.asarray(theta, dtype=float)
        ak, ck = self.gains(k)

        delta = self.rng.choice([-1.0, 1.0], size=theta.shape)

        theta_plus = np.mod(theta + ck * delta, 2 * np.pi)
        theta_minus = np.mod(theta - ck * delta, 2 * np.pi)

        e_plus = energy_function(theta_plus)
        e_minus = energy_function(theta_minus)

        gradient_estimate = ((e_plus - e_minus) / (2.0 * ck)) * delta
        update = -ak * gradient_estimate

        theta_new = np.mod(theta + update, 2 * np.pi)

        return theta_new, update, float(e_plus), float(e_minus), theta_plus, theta_minus


# ====================================================
# REDUCED-COST METAHEURISTIC INITIALIZERS
# ====================================================

def black_hole_optimize_vqe(
    evaluator: CachedCircuitEvaluator,
    num_params: int,
    N: int = 4,
    max_iter: int = 8,
    num_shots: int = 256,
) -> np.ndarray:
    """Reduced-cost Black Hole initialization using cached circuit evaluator."""
    N = max(1, int(N))
    max_iter = max(0, int(max_iter))

    stars = np.random.uniform(0, 2 * np.pi, size=(N, num_params))
    energies = np.array([
        evaluator.energy(s, num_shots=num_shots)
        for s in stars
    ])

    for _ in range(max_iter):
        bh_index = int(np.argmin(energies))
        bh = stars[bh_index].copy()
        bh_energy = energies[bh_index]

        r_event = abs(bh_energy) / (np.sum(np.abs(energies)) + 1e-8)

        for i in range(N):
            if i == bh_index:
                continue

            stars[i] = np.mod(
                stars[i] + random.random() * (bh - stars[i]),
                2 * np.pi,
            )

            if np.linalg.norm(stars[i] - bh) < r_event:
                stars[i] = np.random.uniform(0, 2 * np.pi, size=num_params)

            energies[i] = evaluator.energy(stars[i], num_shots=num_shots)

    return stars[int(np.argmin(energies))]


def gwo_optimize_vqe(
    evaluator: CachedCircuitEvaluator,
    num_params: int,
    N: int = 5,
    max_iter: int = 8,
    num_shots: int = 256,
) -> np.ndarray:
    """Reduced-cost Grey Wolf Optimizer initialization using cached circuit evaluator."""
    N = max(5, int(N))
    max_iter = max(0, int(max_iter))

    wolves = np.random.uniform(0, 2 * np.pi, size=(N, num_params))
    energies = np.array([
        evaluator.energy(w, num_shots=num_shots)
        for w in wolves
    ])

    for t in range(max_iter):
        sorted_idx = np.argsort(energies)

        alpha = wolves[sorted_idx[0]].copy()
        beta = wolves[sorted_idx[1]].copy()
        delta = wolves[sorted_idx[2]].copy()

        a = 2.0 - 2.0 * (t / max_iter) if max_iter > 0 else 0.0

        for i in range(N):
            r1 = np.random.rand(num_params)
            r2 = np.random.rand(num_params)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            X1 = alpha - A1 * np.abs(C1 * alpha - wolves[i])

            r1 = np.random.rand(num_params)
            r2 = np.random.rand(num_params)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            X2 = beta - A2 * np.abs(C2 * beta - wolves[i])

            r1 = np.random.rand(num_params)
            r2 = np.random.rand(num_params)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            X3 = delta - A3 * np.abs(C3 * delta - wolves[i])

            wolves[i] = np.mod((X1 + X2 + X3) / 3.0, 2 * np.pi)
            energies[i] = evaluator.energy(wolves[i], num_shots=num_shots)

    return wolves[int(np.argmin(energies))]


def abc_optimize_vqe(
    evaluator: CachedCircuitEvaluator,
    num_params: int,
    N: int = 5,
    max_iter: int = 8,
    limit: int = 3,
    num_shots: int = 256,
) -> np.ndarray:
    """Reduced-cost Artificial Bee Colony initialization using cached circuit evaluator."""
    N = max(5, int(N))
    max_iter = max(0, int(max_iter))
    limit = max(1, int(limit))

    food = np.random.uniform(0, 2 * np.pi, size=(N, num_params))
    fitness = np.array([
        evaluator.energy(f, num_shots=num_shots)
        for f in food
    ])
    trial = np.zeros(N)

    for _ in range(max_iter):
        for i in range(N):
            k = np.random.choice([j for j in range(N) if j != i])
            phi = np.random.uniform(-1, 1, size=num_params)
            v = np.mod(food[i] + phi * (food[i] - food[k]), 2 * np.pi)

            energy = evaluator.energy(v, num_shots=num_shots)

            if energy < fitness[i]:
                food[i] = v
                fitness[i] = energy
                trial[i] = 0
            else:
                trial[i] += 1

        weights = fitness.max() - fitness + 1e-8
        probs = weights / weights.sum() if weights.sum() > 0 else np.ones(N) / N

        for _bee in range(N):
            i = np.random.choice(N, p=probs)
            k = np.random.choice([j for j in range(N) if j != i])
            phi = np.random.uniform(-1, 1, size=num_params)
            v = np.mod(food[i] + phi * (food[i] - food[k]), 2 * np.pi)

            energy = evaluator.energy(v, num_shots=num_shots)

            if energy < fitness[i]:
                food[i] = v
                fitness[i] = energy
                trial[i] = 0
            else:
                trial[i] += 1

        for i in range(N):
            if trial[i] >= limit:
                food[i] = np.random.uniform(0, 2 * np.pi, size=num_params)
                fitness[i] = evaluator.energy(food[i], num_shots=num_shots)
                trial[i] = 0

    return food[int(np.argmin(fitness))]


# ====================================================
# FINAL SOLUTION DECODING
# ====================================================

def best_solution_from_histogram(
    histogram: Dict[str, int],
    Q: np.ndarray,
    q_linear: np.ndarray,
    top_k: int = 100,
) -> Tuple[np.ndarray, float]:
    """Return best QUBO solution among the top-k most frequent samples."""
    sorted_counts = sorted(histogram.items(), key=lambda item: -item[1])[:top_k]

    best_cost = float("inf")
    z_best = None

    for bstring, _count in sorted_counts:
        z = np.array([int(b) for b in bstring], dtype=int)
        cost = qubo_cost(z, Q, q_linear)

        if cost < best_cost:
            best_cost = cost
            z_best = z

    if z_best is None:
        raise RuntimeError("No valid QUBO solution found from measured bitstrings.")

    return z_best, float(best_cost)


def sample_final_solution(
    final_circuit: QuantumCircuit,
    Q: np.ndarray,
    q_linear: np.ndarray,
    qubo_indices: Optional[List[int]] = None,
    qpu_qubit_config: Optional[List[int]] = None,
    mode: str = "monolithic",
    num_shots: int = 4000,
    top_k: int = 100,
    return_histogram: bool = True,
):
    """
    Backward-compatible final sampler.

    This function does not use the cached evaluator. dvqe(...) uses the cached
    evaluator directly for final sampling.
    """
    n = len(q_linear) if qubo_indices is None else len(qubo_indices)

    measured = add_qubo_measurements_once(final_circuit, mode, n)
    result = sim.run(measured, shots=num_shots).result()
    counts_raw = result.get_counts()

    histogram = normalize_counts_to_qubo_bits(counts_raw, n)

    z_best, best_cost = best_solution_from_histogram(
        histogram,
        Q,
        q_linear,
        top_k=top_k,
    )

    if return_histogram:
        return z_best, best_cost, histogram

    return z_best, best_cost


# ====================================================
# PARAMETER INITIALIZATION
# ====================================================

def initialize_parameters(
    init_type: int,
    evaluator: CachedCircuitEvaluator,
    warm_start_population: int,
    warm_start_iters: int,
    warm_start_shots: int,
) -> np.ndarray:
    """Initialize ansatz parameters using cached evaluator."""
    num_params = evaluator.num_params

    if init_type == 1:
        return np.random.uniform(0, 2 * np.pi, size=num_params)

    if init_type == 2:
        return black_hole_optimize_vqe(
            evaluator=evaluator,
            num_params=num_params,
            N=warm_start_population,
            max_iter=warm_start_iters,
            num_shots=warm_start_shots,
        )

    if init_type == 3:
        return gwo_optimize_vqe(
            evaluator=evaluator,
            num_params=num_params,
            N=max(5, warm_start_population),
            max_iter=warm_start_iters,
            num_shots=warm_start_shots,
        )

    if init_type == 4:
        return abc_optimize_vqe(
            evaluator=evaluator,
            num_params=num_params,
            N=max(5, warm_start_population),
            max_iter=warm_start_iters,
            num_shots=warm_start_shots,
        )

    raise ValueError("Invalid init_type. Use 1=random, 2=BH, 3=GWO, 4=ABC.")


# ====================================================
# MAIN LOWERCASE SOLVER
# ====================================================

def dvqe(
    mode: str,
    Q: np.ndarray,
    q_linear: np.ndarray,
    init_type: int,
    depth: int,
    lr: float,
    max_iters: int,
    qpu_qubit_config: Optional[List[int]],
    rel_tol: float,
    num_shots: int = 1024,
    final_shots: int = 4000,
    spsa_c: float = 0.15,
    spsa_A: float = 10.0,
    warm_start_population: int = 4,
    warm_start_iters: int = 8,
    warm_start_shots: int = 256,
    seed: Optional[int] = None,
    verbose: bool = True,
    transpile_once: bool = True,
    transpile_optimization_level: int = 0,
):
    """
    Cached scalable DVQE solver with lowercase name: dvqe(...)

    Parameters:
    - mode: "monolithic" or "distributed"
    - Q: QUBO matrix
    - q_linear: linear QUBO vector
    - init_type: 1=random, 2=Black Hole, 3=GWO, 4=ABC
    - depth: ansatz depth
    - lr: SPSA learning-rate scale
    - max_iters: number of SPSA iterations
    - qpu_qubit_config: QPU qubit allocation for distributed mode
    - rel_tol: stopping tolerance based on max parameter update

    Returns:
    - z_best
    - final_circuit_without_measurements
    - hist
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if mode not in ["monolithic", "distributed"]:
        raise ValueError("Invalid mode. Use 'monolithic' or 'distributed'.")

    Q = np.asarray(Q, dtype=float)
    q_linear = np.asarray(q_linear, dtype=float)
    n = len(q_linear)

    if Q.shape != (n, n):
        raise ValueError(f"Q must have shape {(n, n)}, but got {Q.shape}.")

    top_k = 100 if n <= 7 else int(np.floor(n / 1.5) * 100)

    evaluator = CachedCircuitEvaluator(
        mode=mode,
        Q=Q,
        q_linear=q_linear,
        depth=depth,
        qpu_qubit_config=qpu_qubit_config,
        backend=sim,
        transpile_once=transpile_once,
        transpile_optimization_level=transpile_optimization_level,
    )

    param_values = initialize_parameters(
        init_type=init_type,
        evaluator=evaluator,
        warm_start_population=warm_start_population,
        warm_start_iters=warm_start_iters,
        warm_start_shots=warm_start_shots,
    )

    optimizer = SPSAOptimizer(
        a=lr,
        c=spsa_c,
        A=spsa_A,
        seed=seed,
    )

    best_energy = float("inf")
    best_params = param_values.copy()

    def energy_function(theta):
        return evaluator.energy(theta, num_shots=num_shots)

    for it in range(max_iters):
        old_params = param_values.copy()

        param_values, update, e_plus, e_minus, theta_plus, theta_minus = optimizer.step(
            theta=param_values,
            k=it,
            energy_function=energy_function,
        )

        # No third energy call. Use the already evaluated SPSA points.
        if e_plus <= e_minus:
            current_energy = e_plus
            current_best_candidate = theta_plus
        else:
            current_energy = e_minus
            current_best_candidate = theta_minus

        if current_energy < best_energy:
            best_energy = current_energy
            best_params = current_best_candidate.copy()

        max_update = float(np.max(np.abs(param_values - old_params)))

        if verbose:
            print(
                f"Iter {it + 1:4d} | energy={current_energy:.6f} | "
                f"best={best_energy:.6f} | max_update={max_update:.3e}"
            )

        if it > 2 and max_update < rel_tol:
            if verbose:
                print(f"Converged at iteration {it + 1}.")
            break

    final_energy, hist = evaluator.energy_and_histogram(
        best_params,
        num_shots=final_shots,
    )

    z_best, cost_best = best_solution_from_histogram(
        hist,
        Q,
        q_linear,
        top_k=top_k,
    )

    final_circuit = evaluator.circuit_without_measurements(best_params)

    if verbose:
        print(f"Final estimated energy: {final_energy:.6f}")
        print(f"Best sampled QUBO cost: {cost_best:.6f}")
        print(f"Best sampled bitstring: {z_best}")

    return z_best, final_circuit, hist


# Optional backward-compatible alias. The recommended solver name is dvqe(...).
DVQE = dvqe