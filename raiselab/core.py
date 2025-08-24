# ==============================================================================
# Title       : General VQE Framework for Monolithic and Distributed Execution
# Author      : Milad Hasanzadeh
# Email       : e.mhasanzadeh1377@yahoo.com
# Affiliation : Department of Electrical and Computer Engineering,
#               Louisiana State University, Baton Rouge, LA, USA
# Date        : July 26, 2025
#
# Description :
# This Python module implements a modular hybrid Variational 
# Quantum Eigensolver (VQE) framework capable of solving binary optimization 
# problems in both monolithic and distributed quantum architectures.
#
# Inputs :
# - Q : numpy.ndarray
#       Symmetric QUBO matrix of shape (n, n), where n is the number of binary variables.
# - c : numpy.ndarray or list
#       Linear QUBO coefficients (bias vector) of shape (n,).
# - init_type : int
#       Initialization strategy: 1=random, 2=Black Hole, 3=GWO, 4=ABC
# - depth : int
#       Number of layers in the parameterized ansatz circuit.
# - lr : float
#       Learning rate used in the Adam optimizer.
# - max_iters : int
#       Maximum number of quantum optimization iterations.
# - qpu_qubit_config : list of int
#       Number of qubits per QPU (used only in distributed mode).
# - rel_tol : float
#       Relative tolerance for convergence based on max parameter update.
# - mode : str
#       Mode of execution: "monolithic" or "distributed".
#
# Outputs :
# - z_best : numpy.ndarray
#       Final optimized binary solution (bitstring) with lowest QUBO cost observed.
# - final_circuit : qiskit.QuantumCircuit
#       The final ansatz quantum circuit used to evaluate the solution.
#
# Functional Overview :
# - Constructs parameterized ansatz circuits for both centralized and distributed layouts.
# - Supports custom topologies via QPU configuration (multi-region QPU partitioning).
# - Defines QUBO → Hamiltonian conversion for Pauli-Z operators.
# - Implements several metaheuristic-based initializers: Black Hole, GWO, ABC.
# - Supports gradient-based parameter updates using Adam optimizer.
# - Integrates circuit evaluation, energy expectation, and decoding routines.
# - Tracks feasible bitstring candidates and selects the lowest-cost solution.
# - Visualizes the final ansatz circuit for analysis and debugging.
#
# Key Features :
# - Dual-mode architecture: 'monolithic' and 'distributed' execution via a single interface.
# - Modular components for ansatz construction, energy computation, and optimization.
# - Interoperable with any QUBO-formulated problem (linear or quadratic).
# - Clear grouping of utility functions for extensibility and readability.
# - Compatible with classical-to-quantum hybrid optimization pipelines (e.g., ADMM+VQE).
#
# Requirements :
# - Python 3.8+
# - Qiskit 0.39.0
# - qiskit-aer 0.11.0
# - qiskit-terra 0.22.0
# - qiskit-optimization 0.4.0
# - diskit 0.1 (for circuit distribution and remapping)
#
# License :
# This module is intended for academic and research purposes only.
# Redistribution or commercial use is not permitted without prior written consent.
# Proper citation and acknowledgment must be provided when used in published work.
#
# Copyright (c) 2025, Milad Hasanzadeh
# ==============================================================================


# === Standard Library Imports ===
import math
import random
import warnings
from collections import defaultdict
from typing import List, Tuple, Union, Optional, Dict

# === Scientific & Plotting Libraries ===
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# === Qiskit Core Imports ===
from qiskit import (
    Aer, assemble, transpile,
    QuantumCircuit, execute
)
from qiskit.circuit import Parameter
from qiskit.visualization import plot_bloch_vector, plot_histogram
from qiskit.quantum_info import (
    Statevector, DensityMatrix, partial_trace, state_fidelity
)

# === DISKIT Imports (Distributed Quantum Toolkit) ===
from diskit import Topology, CircuitRemapper

# === Jupyter Display Utilities ===
from IPython.display import display

# === Warning Filter ===
warnings.filterwarnings("ignore")

# === Define a global simulator instance ===
sim = Aer.get_backend("qasm_simulator")

# ====================================================
# ⚙️ QUBO → PAULI HAMILTONIAN MAPPING
# ====================================================


def qubo_to_pauli_hamiltonian(Q: np.ndarray, q_linear: np.ndarray) -> List[Tuple[float, str]]:
    """
    Converts a full QUBO Hamiltonian (Q, q) into Pauli-Z operator form.

    H(z) = z.T @ Q @ z + q.T @ z
         = Σ Q_ij z_i z_j + Σ q_i z_i
         = transformed into Σ coeff * PauliString (Z_i, Z_i Z_j, etc.)

    Parameters:
    ----------
    Q : np.ndarray of shape (n, n)
        Symmetric QUBO matrix for quadratic terms.
    q_linear : np.ndarray of shape (n,)
        Linear QUBO coefficients.

    Returns:
    -------
    pauli_terms : List[Tuple[float, str]]
        List of (coefficient, Pauli string) suitable for VQE or QAOA.
    """

    n = len(q_linear)
    term_dict = defaultdict(float)

    # === Linear terms ===
    for i in range(n):
        q = q_linear[i]
        term_dict["I"] += q / 2.0
        term_dict[f"Z{i}"] += -q / 2.0

    # === Quadratic terms ===
    for i in range(n):
        for j in range(i, n):
            qij = Q[i, j]
            if abs(qij) < 1e-8:
                continue

            if i == j:
                # z_i^2 = z_i (because z_i ∈ {0,1})
                term_dict["I"] += qij / 2.0
                term_dict[f"Z{i}"] += -qij / 2.0
            else:
                # z_i z_j term
                term_dict["I"] += qij / 4.0
                term_dict[f"Z{i}"] += -qij / 4.0
                term_dict[f"Z{j}"] += -qij / 4.0
                term_dict[f"Z{i}Z{j}"] += qij / 4.0

    # Clean and sort
    return [(float(coeff), pauli) for pauli, coeff in sorted(term_dict.items()) if abs(coeff) > 1e-8]

# ====================================================
# 🧠 GRADIENT CALCULATION FUNCTIONS (Monolithic + Distributed)
# ====================================================


def finite_difference_monolithic_gradient(param_values, pauli_terms, depth, n, epsilon=1e-3):
    def grad_i(i):
        shift = np.zeros_like(param_values)
        shift[i] = epsilon

        qc_plus = create_monolithic_ansatz(n, depth, param_values + shift)
        qc_minus = create_monolithic_ansatz(n, depth, param_values - shift)

        e_plus = compute_energy_monolithic(qc_plus, pauli_terms)
        e_minus = compute_energy_monolithic(qc_minus, pauli_terms)

        return (e_plus - e_minus) / (2 * epsilon)

    grads = Parallel(n_jobs=-1)(delayed(grad_i)(i) for i in range(len(param_values)))
    return np.array(grads)


def finite_difference_gradient_parallel(
    param_values: np.ndarray,
    topo,
    pauli_terms,
    depth: int,
    num_qpus: int,
    qubits_per_qpu: int,
    epsilon: float = 1e-3
) -> np.ndarray:
    """
    Computes the gradient of the energy expectation with respect to each parameter
    using central finite differences, in parallel.

    Parameters:
    ----------
    param_values : np.ndarray
        Current parameter values of the VQE ansatz (1D array).
    topo : Topology
        DISKIT Topology object representing the distributed QPU configuration.
    pauli_terms : list
        List of Pauli operators used to construct the Hamiltonian.
    depth : int
        Circuit depth (number of layers) in the variational ansatz.
    num_qpus : int
        Number of QPUs used in the distributed simulation.
    qubits_per_qpu : int
        Number of logical qubits assigned to each QPU.
    epsilon : float, optional
        Finite difference step size (default is 1e-3).

    Returns:
    -------
    grads : np.ndarray
        The gradient vector of the energy expectation w.r.t. parameters.
    """

    def grad_i(i: int) -> float:
        """Computes the i-th component of the gradient using central difference."""
        shift = np.zeros_like(param_values)
        shift[i] = epsilon

        e_plus = compute_energy_expectation(
            param_values + shift, topo, pauli_terms, depth, num_qpus, qubits_per_qpu
        )
        e_minus = compute_energy_expectation(
            param_values - shift, topo, pauli_terms, depth, num_qpus, qubits_per_qpu
        )

        return (e_plus - e_minus) / (2 * epsilon)

    # Use joblib to parallelize gradient computation over all parameters
    grads = Parallel(n_jobs=-1)(
        delayed(grad_i)(i) for i in range(len(param_values))
    )
    
    return np.array(grads)

# ====================================================
# 🧠 ANSATZ GENERATION FUNCTIONS
# ====================================================

def create_monolithic_ansatz(
    num_qubits: int,
    depth: int,
    param_values: list
) -> QuantumCircuit:
    """
    Creates a layered monolithic variational ansatz circuit with RX-RY rotations
    and chain-like CX entanglement (bottom to top) between adjacent qubits.

    Parameters:
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Number of layers in the variational circuit.
    param_values : list or np.ndarray
        Flat list of rotation parameters, of length 2 * num_qubits * depth.

    Returns:
    -------
    qc : QuantumCircuit
        The constructed parameterized ansatz quantum circuit.
    """

    qc = QuantumCircuit(num_qubits)
    param_index = 0  # Index for accessing parameters

    for d in range(depth):
        # === Rotation Layer: Apply RY and RX to every qubit ===
        for q in range(num_qubits):
            theta_ry = param_values[param_index]
            theta_rx = param_values[param_index + 1]
            param_index += 2

            qc.ry(theta_ry, q)
            qc.rx(theta_rx, q)

        # === Entanglement Layer: Chain CXs (skip after final layer) ===
        if d < depth - 1:
            for i in reversed(range(1, num_qubits)):
                qc.cx(i - 1, i)

    return qc


from qiskit.circuit import QuantumRegister, QuantumCircuit, Parameter

def create_ansatz_on_topology_regs(
    qregs: list,
    depth: int = 1,
    num_qpus: int = 2,
    cross_qpu_entangle: bool = True,
    parametric: bool = False,
    param_values: list = None
) -> QuantumCircuit:
    """
    Constructs a distributed variational ansatz circuit on a given list of 
    quantum registers, optionally using symbolic parameters or fixed values.

    Each layer consists of:
    - Rotation layer (RY and RX per qubit)
    - Optional entanglement (CX) between adjacent qubits in the flattened layout

    Parameters:
    ----------
    qregs : list of QuantumRegister
        List of quantum registers, typically representing subsystems or QPUs.
        Communication registers (e.g., com_*) are ignored.
    depth : int, optional
        Number of repeated layers in the ansatz (default is 1).
    num_qpus : int, optional
        Number of logical QPUs (default is 2). Included for interface consistency.
    cross_qpu_entangle : bool, optional
        If True, entanglement across QPUs is allowed. Currently not used in logic.
    parametric : bool, optional
        If True, use symbolic Parameters for θ and ϕ rotations.
    param_values : list, optional
        List of numerical parameter values for all RY and RX gates.
        Length must equal 2 * num_qubits * depth.

    Returns:
    -------
    qc : QuantumCircuit
        A quantum circuit with layered variational ansatz applied.
    """

    qc = QuantumCircuit(*qregs)

    # Filter only system (computational) registers
    system_qregs = [reg for reg in qregs if not reg.name.startswith("com_")]

    # Flatten all system qubits
    all_qubits = [q for reg in system_qregs for q in reg]
    num_qubits = len(all_qubits)
    total_params_needed = 2 * depth * num_qubits  # 2 params per qubit per layer

    # Validate provided parameters
    if param_values is not None and len(param_values) != total_params_needed:
        raise ValueError(
            f"[create_ansatz_on_topology_regs] Expected {total_params_needed} parameters, got {len(param_values)}"
        )

    param_index = 0      # For real-valued parameters
    theta_index = 0      # For symbolic parameters (naming)

    for d in range(depth):
        # --- Rotation Layer (RY + RX) ---
        for q in all_qubits:
            if parametric:
                theta_ry = Parameter(f"θ{theta_index}")
                theta_rx = Parameter(f"ϕ{theta_index}")
                theta_index += 1
            elif param_values is not None:
                theta_ry = param_values[param_index]
                theta_rx = param_values[param_index + 1]
                param_index += 2
            else:
                theta_ry = np.pi / 8
                theta_rx = np.pi / 8

            qc.ry(theta_ry, q)
            qc.rx(theta_rx, q)

        # --- Skip entanglement after final layer ---
        if d == depth - 1:
            break

        # --- Entanglement Layer (linear CXs) ---
        for i in range(len(all_qubits) - 1, 0, -1):
            qc.cx(all_qubits[i - 1], all_qubits[i])

    return qc




def remap_with_diskit(ansatz_qc: QuantumCircuit, topology: Topology) -> QuantumCircuit:
    """
    Applies DISKIT's circuit remapper to distribute a monolithic circuit
    onto the specified topology of QPUs.

    Parameters:
    ----------
    ansatz_qc : QuantumCircuit
        The monolithic quantum circuit to be remapped.
    topology : diskit.Topology
        The distributed system topology specifying how qubits are divided.

    Returns:
    -------
    QuantumCircuit
        A remapped (distributed) version of the original circuit.
    """
    remapper = CircuitRemapper(topology)
    return remapper.remap_circuit(ansatz_qc)


def get_reduced_state(circuit: QuantumCircuit, sim=sim) -> DensityMatrix:
    """
    Simulates a circuit and returns the reduced density matrix by tracing out
    all communication qubits (registers starting with 'com_').

    Parameters:
    ----------
    circuit : QuantumCircuit
        The full quantum circuit including system and communication qubits.
    sim : Backend, optional
        The simulator backend (default is Aer simulator).

    Returns:
    -------
    DensityMatrix
        The reduced density matrix after tracing out communication qubits.
    """
    # Copy circuit to avoid side effects and save final statevector
    circuit_copy = circuit.copy()
    circuit_copy.save_statevector()
    job = assemble(circuit_copy)
    state = sim.run(job).result().get_statevector()

    # Find communication qubits (usually used for teleportation/EPR sharing)
    comm_qubits = [
        q for reg in circuit.qregs if reg.name.startswith("com_") for q in reg
    ]
    comm_indices = [circuit.find_bit(q).index for q in comm_qubits]

    # Trace out communication qubits to get reduced state
    return partial_trace(state, comm_indices)

# ====================================================
# ⚡ ENERGY EVALUATION FUNCTIONS (DISTRIBUTED & MONOLITHIC)
# ====================================================


def compute_energy_monolithic(qc, pauli_terms):
    qc_copy = qc.copy()
    qc_copy.save_statevector()
    sv = sim.run(assemble(qc_copy)).result().get_statevector()
    energy = 0.0
    for coeff, term in pauli_terms:
        if term == "I":
            energy += coeff
        else:
            op = [np.eye(2)] * qc.num_qubits
            if 'Z' in term:
                labels = term.split('Z')
                positions = [int(k) for k in labels[1:] if k != '']
                for pos in positions:
                    op[pos] = np.array([[1, 0], [0, -1]])
            full_op = op[0]
            for k in range(1, len(op)):
                full_op = np.kron(full_op, op[k])
            energy += coeff * np.real(np.vdot(sv.data, full_op @ sv.data))

    return energy

def compute_energy_expectation(
    param_values: Union[np.ndarray, List[float]],
    topo,
    pauli_terms: List[Tuple[float, str]],
    depth: int,
    num_qpus: int,
    qubits_per_qpu: int,
    return_circuit: bool = False
) -> Union[float, Tuple[float, QuantumCircuit]]:
    """
    Computes the energy expectation ⟨H⟩ = ⟨ψ|H|ψ⟩ of a distributed circuit
    under the given parameter values and topology using DISKIT.

    Parameters:
    ----------
    param_values : array-like
        Flat list of ansatz parameters (length = 2 * num_qubits * depth).
    topo : diskit.Topology
        The distributed system topology used to generate QPU registers.
    pauli_terms : list of (float, str)
        Hamiltonian terms as (coefficient, PauliString), e.g., [(0.5, "Z0Z1"), (-0.2, "I")].
    depth : int
        Number of ansatz layers.
    num_qpus : int
        Number of QPUs (used to set up the registers).
    qubits_per_qpu : int
        Number of qubits per QPU.
    return_circuit : bool, optional
        If True, also return the remapped distributed circuit.

    Returns:
    -------
    energy : float
        The expected energy ⟨ψ|H|ψ⟩ given the current ansatz parameters.

    Optional:
    --------
    If return_circuit is True:
        (energy, distributed_circuit) is returned.
    """

    # === Step 1: Build parameterized ansatz circuit on QPU registers ===
    qregs = topo.get_regs()
    ansatz_qc = create_ansatz_on_topology_regs(
        qregs,
        depth=depth,
        num_qpus=num_qpus,
        param_values=param_values
    )

    # === Step 2: Remap circuit using DISKIT to distributed layout ===
    dist_qc = remap_with_diskit(ansatz_qc, topo)

    # === Step 3: Simulate and get reduced state (excluding communication qubits) ===
    reduced_dm: DensityMatrix = get_reduced_state(dist_qc, sim)

    # === Step 4: Compute ⟨H⟩ = Σ c_i ⟨ψ|P_i|ψ⟩ ===
    energy = 0.0
    for coeff, term in pauli_terms:
        if term == "I":
            energy += coeff
            continue

        # Construct full n-qubit operator from tensor product of I/Z
        num_qubits = reduced_dm.num_qubits
        ops = [np.eye(2)] * num_qubits

        # Parse indices from Pauli string like "Z0Z2"
        z_indices = [int(idx) for idx in term.replace('Z', ' ').split() if idx != '']
        for idx in z_indices:
            ops[idx] = np.array([[1, 0], [0, -1]])  # Pauli-Z

        # Build full Pauli operator via Kronecker product
        full_op = ops[0]
        for op in ops[1:]:
            full_op = np.kron(full_op, op)

        # Compute trace(ρ * H)
        energy += coeff * np.real(np.trace(reduced_dm.data @ full_op))

    return (energy, dist_qc) if return_circuit else energy

# ====================================================
# 🔁 CLASSICAL OPTIMIZER UTILITIES
# ====================================================

class AdamOptimizer:
    """
    Implements the Adam optimization algorithm for parameter updates.
    
    This optimizer is commonly used in variational quantum algorithms like VQE
    to update ansatz parameters based on gradients of the energy function.

    Attributes:
    ----------
    lr : float
        Learning rate (step size).
    beta1 : float
        Exponential decay rate for the first moment estimate (momentum).
    beta2 : float
        Exponential decay rate for the second moment estimate (RMS).
    epsilon : float
        Small constant to avoid division by zero.
    m : np.ndarray
        First moment vector (initialized during first step).
    v : np.ndarray
        Second moment vector (initialized during first step).
    t : int
        Time step counter (starts from 0).
    """

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment (mean of gradients)
        self.v = None  # Second moment (uncentered variance of gradients)
        self.t = 0     # Iteration counter

    def step(self, grads: np.ndarray) -> np.ndarray:
        """
        Computes the Adam update step for the given gradient.

        Parameters:
        ----------
        grads : np.ndarray
            The gradient of the objective function w.r.t. parameters.

        Returns:
        -------
        update : np.ndarray
            The computed parameter update to apply.
        """

        if self.m is None:
            # Initialize first and second moment vectors on first call
            self.m = np.zeros_like(grads)
            self.v = np.zeros_like(grads)

        # Update time step
        self.t += 1

        # Update biased moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Compute parameter update
        update = -self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return update

        
# ====================================================
# 🌌 METAHEURISTIC-BASED VQE INITIALIZATION STRATEGIES
# ====================================================


def black_hole_optimize_vqe(
    topo,
    pauli_terms,
    depth,
    num_qpus,
    num_params,
    N=10,
    max_iter=100,
    mode="distributed"
):
    low, high = 0.0, 2 * np.pi
    stars = np.random.uniform(low, high, size=(N, num_params))
    energies = np.zeros(N)
    n = num_params // (2 * depth)

    for i in range(N):
        if mode == "distributed":
            energies[i], _ = compute_energy_expectation(stars[i], topo, pauli_terms, depth, num_qpus, num_params, return_circuit=True)
        else:
            qc = create_monolithic_ansatz(n, depth, stars[i])
            energies[i] = compute_energy_monolithic(qc, pauli_terms)

    for it in range(max_iter):
        bh_index = np.argmin(energies)
        bh = stars[bh_index].copy()
        bh_energy = energies[bh_index]
        r_event = bh_energy / (np.sum(energies) + 1e-8)

        for i in range(N):
            if i == bh_index:
                continue
            stars[i] += random.uniform(0, 1) * (bh - stars[i])
            stars[i] = np.clip(stars[i], low, high)

            if np.linalg.norm(stars[i] - bh) < r_event:
                stars[i] = np.random.uniform(low, high, size=num_params)

            if mode == "distributed":
                energies[i], _ = compute_energy_expectation(stars[i], topo, pauli_terms, depth, num_qpus, num_params, return_circuit=True)
            else:
                qc = create_monolithic_ansatz(n, depth, stars[i])
                energies[i] = compute_energy_monolithic(qc, pauli_terms)

    return stars[np.argmin(energies)]

def gwo_optimize_vqe(
    topo,
    pauli_terms,
    depth,
    num_qpus,
    num_params,
    N=10,
    max_iter=100,
    mode="distributed"
):
    low, high = 0.0, 2 * np.pi
    wolves = np.random.uniform(low, high, size=(N, num_params))
    energies = np.zeros(N)
    n = num_params // (2 * depth)

    for i in range(N):
        if mode == "distributed":
            energies[i], _ = compute_energy_expectation(wolves[i], topo, pauli_terms, depth, num_qpus, num_params, return_circuit=True)
        else:
            qc = create_monolithic_ansatz(n, depth, wolves[i])
            energies[i] = compute_energy_monolithic(qc, pauli_terms)

    for t in range(max_iter):
        sorted_idx = np.argsort(energies)
        alpha, beta, delta = wolves[sorted_idx[:3]]
        a = 2 - 2 * (t / max_iter)

        for i in range(N):
            X1 = X2 = X3 = np.zeros(num_params)
            for j in range(num_params):
                for leader, X in zip([alpha, beta, delta], [X1, X2, X3]):
                    r1, r2 = random.random(), random.random()
                    A = 2 * a * r1 - a
                    C = 2 * r2
                    D = abs(C * leader[j] - wolves[i][j])
                    X[j] = leader[j] - A * D
            wolves[i] = np.clip((X1 + X2 + X3) / 3, low, high)

            if mode == "distributed":
                energies[i], _ = compute_energy_expectation(wolves[i], topo, pauli_terms, depth, num_qpus, num_params, return_circuit=True)
            else:
                qc = create_monolithic_ansatz(n, depth, wolves[i])
                energies[i] = compute_energy_monolithic(qc, pauli_terms)

    return wolves[np.argmin(energies)]

def abc_optimize_vqe(
    topo,
    pauli_terms,
    depth,
    num_qpus,
    num_params,
    N=10,
    max_iter=100,
    limit=5,
    mode="distributed"
):
    low, high = 0.0, 2 * np.pi
    food = np.random.uniform(low, high, size=(N, num_params))
    fitness = np.zeros(N)
    trial = np.zeros(N)
    n = num_params // (2 * depth)

    for i in range(N):
        if mode == "distributed":
            fitness[i], _ = compute_energy_expectation(food[i], topo, pauli_terms, depth, num_qpus, num_params, return_circuit=True)
        else:
            qc = create_monolithic_ansatz(n, depth, food[i])
            fitness[i] = compute_energy_monolithic(qc, pauli_terms)

    for it in range(max_iter):
        for i in range(N):
            k = np.random.choice([j for j in range(N) if j != i])
            phi = np.random.uniform(-1, 1, size=num_params)
            v = np.clip(food[i] + phi * (food[i] - food[k]), low, high)

            if mode == "distributed":
                energy, _ = compute_energy_expectation(v, topo, pauli_terms, depth, num_qpus, num_params, return_circuit=True)
            else:
                qc = create_monolithic_ansatz(n, depth, v)
                energy = compute_energy_monolithic(qc, pauli_terms)

            if energy < fitness[i]:
                food[i] = v
                fitness[i] = energy
                trial[i] = 0
            else:
                trial[i] += 1

        prob = fitness.max() - fitness + 1e-8
        prob /= prob.sum()

        for _ in range(N):
            i = np.random.choice(N, p=prob)
            k = np.random.choice([j for j in range(N) if j != i])
            phi = np.random.uniform(-1, 1, size=num_params)
            v = np.clip(food[i] + phi * (food[i] - food[k]), low, high)

            if mode == "distributed":
                energy, _ = compute_energy_expectation(v, topo, pauli_terms, depth, num_qpus, num_params, return_circuit=True)
            else:
                qc = create_monolithic_ansatz(n, depth, v)
                energy = compute_energy_monolithic(qc, pauli_terms)

            if energy < fitness[i]:
                food[i] = v
                fitness[i] = energy
                trial[i] = 0
            else:
                trial[i] += 1

        for i in range(N):
            if trial[i] >= limit:
                food[i] = np.random.uniform(low, high, size=num_params)
                if mode == "distributed":
                    fitness[i], _ = compute_energy_expectation(food[i], topo, pauli_terms, depth, num_qpus, num_params, return_circuit=True)
                else:
                    qc = create_monolithic_ansatz(n, depth, food[i])
                    fitness[i] = compute_energy_monolithic(qc, pauli_terms)
                trial[i] = 0

    return food[np.argmin(fitness)]

# ===========================================
# 🧠 SOLUTION DECODING AND SELECTION UTILITIES
# ===========================================

def sample_final_solution(
    final_circuit: QuantumCircuit,
    Q: np.ndarray,
    q_linear: np.ndarray,
    qubo_indices: list,
    qpu_qubit_config: list,
    mode: str = "monolithic",
    num_shots: int = 4000,
    top_k: int = 100,
    return_histogram: bool = True,
) -> Tuple[np.ndarray, float, Optional[Dict[str, int]]]:
    """
    Samples from the final circuit (distributed or monolithic) and returns
    the best binary solution based on QUBO cost. Communication qubits are
    excluded from measurement and histogram processing.

    Parameters:
    - final_circuit: full circuit (with or without communication qubits)
    - Q, q_linear: QUBO cost matrices
    - qubo_indices: indices of qubits used in QUBO (length = n)
    - qpu_qubit_config: list of #qubits per QPU (only used in distributed)
    - mode: "distributed" or "monolithic"

    Returns:
    - z_best: best binary solution
    - cost_best: corresponding QUBO cost
    - histogram: measured counts (only for computational bits)
    """
    n = len(qubo_indices)
    measured_circuit = final_circuit.copy()

    if mode == "distributed":
        # === 1. Collect only computational qubits ===
        comp_qubits = []
        for reg in measured_circuit.qregs:
            if not reg.name.startswith("com_"):
                comp_qubits.extend(reg)

        # === 2. Add classical register to match comp_qubits ===
        from qiskit import ClassicalRegister
        creg = ClassicalRegister(n, "creg")
        measured_circuit.add_register(creg)

        # === 3. Measure ONLY computational qubits ===
        for i in range(n):
            measured_circuit.measure(comp_qubits[i], creg[i])

    else:  # Monolithic
        if not measured_circuit.cregs:
            measured_circuit.measure_all()

    # === 4. Execute full circuit (including com_ qubits, but only comp measured) ===
    backend = Aer.get_backend("qasm_simulator")
    result = execute(measured_circuit, backend=backend, shots=num_shots).result()
    counts_raw = result.get_counts()

    # === 5. Histogram: only keep creg values (not full bitstring) ===
    # Qiskit returns a bitstring with bits from all cregs + creg ordering reversed
    histogram = {}
    for bitstring, count in counts_raw.items():
        bits = bitstring.replace(" ", "")[::-1]  # LSB first
        trimmed = bits[-n:]                     # only last n bits → computational
        histogram[trimmed] = count

    # === 6. Evaluate QUBO cost on top_k most frequent ===
    sorted_counts = sorted(histogram.items(), key=lambda x: -x[1])[:top_k]
    best_cost = float("inf")
    z_best = None

    for bstring, _ in sorted_counts:
        z = np.array([int(b) for b in bstring])
        cost = z.T @ Q @ z + q_linear @ z
        if cost < best_cost:
            best_cost = cost
            z_best = z

    if z_best is None:
        raise RuntimeError("No valid QUBO solution found from measured bitstrings.")

    return (z_best, best_cost, histogram) if return_histogram else (z_best, best_cost)


# ===========================================
# 🚀 VQE MAIN EXECUTION CONTROLLER
# ===========================================

    

def DVQE(
    mode: str,                           # "monolithic" or "distributed"
    Q: np.ndarray,                      # Full QUBO matrix (not yet used, for future)
    q_linear: np.ndarray,              # Linear QUBO coefficients
    init_type: int,                    # 1=random, 2=BH, 3=GWO, 4=ABC
    depth: int,                        # Ansatz circuit depth
    lr: float,                         # Learning rate
    max_iters: int,                    # Number of optimization steps
    qpu_qubit_config: list,           # QPUs configuration (used if mode = distributed)
    rel_tol: float                     # Convergence threshold
) -> np.ndarray:
    """
    General-purpose VQE runner that supports both monolithic and distributed modes.

    Parameters:
    - mode: "monolithic" or "distributed"
    - Q, q_linear: QUBO matrix and linear coefficients
    - init_type: 1=random, 2=BH, 3=GWO, 4=ABC
    - depth: ansatz depth
    - lr: learning rate for Adam
    - max_iters: max iterations
    - qpu_qubit_config: list of ints for QPU qubit allocation (used in distributed)
    - rel_tol: max |Δparam| stopping condition

    Returns:
    - z_best: best binary solution found
    """

    n = len(q_linear)
    num_params = 2 * n * depth
    numsol = 100
    if n > 7:
        numsol = int(np.floor(n / 1.5) * 100)


    pauli_terms = qubo_to_pauli_hamiltonian(Q, q_linear)

    if mode not in ["monolithic", "distributed"]:
        raise ValueError("Invalid mode. Use 'monolithic' or 'distributed'.")

    if mode == "distributed":
        if sum(qpu_qubit_config) < n:
            raise ValueError(f"QPU config provides insufficient qubits: need {n}, got {sum(qpu_qubit_config)}")

        # Adjust config to fit n qubits exactly
        cumulative, adjusted = 0, []
        for q in qpu_qubit_config:
            if cumulative + q <= n:
                adjusted.append(q)
                cumulative += q
            else:
                adjusted.append(n - cumulative)
                break
        qpu_qubit_config = [q for q in adjusted if q > 0]
        num_qpus = len(qpu_qubit_config)

        # Build topology
        topo = Topology()
        topo.create_qmap(num_qpus, qpu_qubit_config)
    else:
        topo = None  # So that the call to the optimizer is well-formed
        num_qpus = 1  # dummy placeholder for argument consistency

    # Initialize parameters
    if init_type == 1:
        param_values = np.random.uniform(0, 2 * np.pi, size=num_params)
    elif init_type == 2:
        param_values = black_hole_optimize_vqe(topo, pauli_terms, depth, num_qpus, num_params, mode=mode)
    elif init_type == 3:
        param_values = gwo_optimize_vqe(topo, pauli_terms, depth, num_qpus, num_params, mode=mode)
    elif init_type == 4:
        param_values = abc_optimize_vqe(topo, pauli_terms, depth, num_qpus, num_params, mode=mode)
    else:
        raise ValueError("Invalid init_type")

    # Setup optimizer and memory
    opt = AdamOptimizer(lr=lr)
    feasible_solutions = []

    for it in range(max_iters):
        if mode == "distributed":
            energy, dist_qc = compute_energy_expectation(param_values, topo, pauli_terms, depth, num_qpus, num_params, return_circuit=True)
            reduced_state = get_reduced_state(dist_qc, sim)
            final_state = reduced_state.data[:, 0]
        else:
            # Monolithic mode
            mono_qc = create_monolithic_ansatz(n, depth, param_values)
            energy = compute_energy_monolithic(mono_qc, pauli_terms)
            final_state = Statevector(mono_qc).data  # You could simulate instead

        # Update params
        if mode == "distributed":
            grads = finite_difference_gradient_parallel(param_values, topo, pauli_terms, depth, num_qpus, num_params)
        else:
            grads = finite_difference_monolithic_gradient(param_values, pauli_terms, depth, n)

        update = opt.step(grads)
        param_values += update

        if it > 2 and np.abs(update).max() < rel_tol:
            break

    final_circuit = dist_qc if mode == "distributed" else mono_qc

    z_best, cost_best, hist = sample_final_solution(
        final_circuit=final_circuit,
        Q=Q,
        q_linear=q_linear,
        qubo_indices=list(range(len(q_linear))),  # or just: list(range(n))
        qpu_qubit_config=qpu_qubit_config,
        mode=mode,
        num_shots=1000,
        top_k=numsol,
        return_histogram=True
    )





    return z_best, final_circuit, hist
