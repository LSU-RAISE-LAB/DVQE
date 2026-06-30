# Raiselab: DVQE and QQP Optimization Software

![License](https://img.shields.io/badge/license-Academic-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Qiskit](https://img.shields.io/badge/Qiskit-0.39.0-purple)

> A modular Python package for **distributed VQE-based QUBO solving** and **QP/LP-to-QUBO reformulation**.

---

## Overview

This repository provides two related optimization tools developed by the Resilient Advanced Infrastructure & Systems Engineering Lab at Louisiana State University:

1. **DVQE**: a distributed variational quantum eigensolver for solving quadratic unconstrained binary optimization problems.
2. **QQP**: a sequential quadratic-programming-to-QUBO framework for solving bounded constrained quadratic and linear programs through local one-bit QUBO reformulations.

Together, DVQE and QQP provide an end-to-end workflow in which binary QUBO problems can be solved directly using variational quantum circuits, while bounded constrained QP/LP problems can first be reformulated into local QUBO subproblems and then solved using DVQE or a classical QUBO backend.

---

## Main Capabilities

### DVQE: Distributed VQE for QUBO Problems

DVQE solves QUBO problems of the form

```math
\min_{z \in \{0,1\}^{n}} z^T Q z + q^T z.
```

The solver supports both monolithic and distributed quantum-circuit execution. In monolithic mode, the ansatz is built on a single quantum processor model. In distributed mode, the ansatz is partitioned across multiple QPUs using distributed circuit remapping.

DVQE evaluates candidate solutions directly from measured bitstrings and selects the best sampled solution according to the original QUBO objective.

### QQP: QP/LP-to-QUBO Reformulation

QQP solves bounded linearly constrained QP/LP problems of the form

```math
\begin{aligned}
\min_x \quad & x^T A x + b^T x + c \\
\text{s.t.} \quad & Gx = r, \\
& Hx \le h, \\
& \ell \le x \le u.
\end{aligned}
```

QQP transforms the original continuous problem into a normalized bounded form, handles equality and inequality constraints through a Powell-Hestenes-Rockafellar augmented-Lagrangian fixed-region construction, and solves the resulting bounded QP subproblems through sequential local one-bit QUBO refinements.

Each local QUBO can be solved using the DVQE backend or a classical QUBO backend.

---

## Features

* **DVQE QUBO Solver**: Solves binary QUBO problems using a variational quantum workflow.
* **Monolithic and Distributed Execution**: Supports both single-QPU and multi-QPU circuit execution models.
* **Two-Stage Variational Training**: Combines metaheuristic warm-start initialization with sampling-based variational refinement.
* **Metaheuristic Initialization**: Supports Black Hole optimization, Grey Wolf Optimization, and Artificial Bee Colony initialization.
* **Sampled QUBO Energy Evaluation**: Evaluates QUBO objectives directly from measured bitstrings.
* **CVaR Energy Mode**: Supports low-cost-sample-focused energy estimation for QUBO optimization.
* **QUBO-to-Pauli Support**: Provides Pauli Hamiltonian compatibility for QUBO models.
* **Distributed Circuit Remapping**: Uses DISKIT-style distributed remapping for multi-QPU circuit execution.
* **QQP Solver for Bounded QP/LP Problems**: Reformulates constrained continuous optimization problems into local QUBO subproblems.
* **PHR Fixed-Region Augmented Lagrangian**: Handles equality and inequality constraints without introducing slack variables.
* **Sequential One-Bit QUBO Refinement**: Avoids large multi-bit discretization by using repeated local binary refinements.
* **Classical Backend Option**: Allows local QUBO subproblems to be solved using a classical MIQP backend when available.
* **Modular Design**: Separates QUBO solving, circuit construction, training, scaling, constraint handling, and decoding.

---

## Installation

To install the package directly from GitHub:

```bash
pip install git+https://github.com/LSU-RAISE-LAB/DVQE.git
```

After installation, both solvers can be imported as:

```python
from raiselab import dvqe, qqp
```

---

## Basic DVQE Usage

```python
import numpy as np
from raiselab import dvqe

Q = np.array([
    [1.0, -2.0],
    [-2.0, 1.0]
])

q = np.array([0.0, 0.0])

z_best, final_circuit, hist = dvqe(
    mode="monolithic",
    Q=Q,
    q_linear=q,
    init_type=2,
    depth=1,
    lr=0.08,
    max_iters=30,
    num_shots=1024,
    final_shots=4000,
    energy_mode="cvar",
    cvar_alpha=0.2
)

print("Best binary solution:", z_best)
print("Final histogram:", hist)
```

---

## Basic QQP Usage

```python
import numpy as np
from raiselab import qqp

A = np.array([
    [1.0, 0.0],
    [0.0, 1.0]
])

b = np.array([-2.0, -4.0])
c = 0.0

# Inequality: x1 + x2 <= 2.5
H = np.array([[1.0, 1.0]])
h = np.array([2.5])

lb = np.array([0.0, 0.0])
ub = np.array([3.0, 3.0])

x_sol, f_val, al_iters, info = qqp(
    A=A,
    b=b,
    c=c,
    H=H,
    h=h,
    lb=lb,
    ub=ub,
    qubo_solver="dvqe",
    mode="monolithic",
    init_type=2,
    depth=1,
    lr=0.08,
    max_iters=30,
    num_shots=128,
    final_shots=1000,
    warm_start_population=4,
    warm_start_iters=5,
    warm_start_shots=64,
    energy_mode="cvar",
    cvar_alpha=0.2
)

print("Continuous solution:", x_sol)
print("Objective value:", f_val)
print("Augmented-Lagrangian iterations:", al_iters)
print("Final residual:", info["final_residual_norm_inf"])
```

---

## Distributed DVQE Example

```python
from raiselab import dvqe

z_best, final_circuit, hist = dvqe(
    mode="distributed",
    Q=Q,
    q_linear=q,
    init_type=2,
    depth=1,
    lr=0.08,
    max_iters=30,
    qpu_qubit_config=[2, 2],
    num_shots=1024,
    final_shots=4000,
    energy_mode="cvar",
    cvar_alpha=0.2
)
```

The input `qpu_qubit_config` defines the QPU partition used for distributed execution.

---

## Optional Classical QUBO Backend

QQP can also solve local QUBO subproblems using a classical MIQP backend:

```python
x_sol, f_val, al_iters, info = qqp(
    A=A,
    b=b,
    c=c,
    H=H,
    h=h,
    lb=lb,
    ub=ub,
    qubo_solver="classical"
)
```

The classical backend requires a working Gurobi installation and license. Gurobi is not required when using `qubo_solver="dvqe"`.

---

## Package Interface

The main public interface is:

```python
from raiselab import dvqe, qqp
```

* `dvqe(...)`: solves QUBO problems using monolithic or distributed variational quantum execution.
* `qqp(...)`: solves bounded constrained QP/LP problems through sequential QUBO reformulation.

---

## Citation

If you use this software in your research, please cite:

> M. Hasanzadeh and A. Kargarian, “Two-stage Variational Quantum Eigensolver Software for Distributed QUBO Solving and Quadratic Programming.”

For the earlier DVQE preprint, please cite:

> M. Hasanzadeh and A. Kargarian, “Distributed Implementation of Variational Quantum Eigensolver to Solve QUBO Problems,” https://arxiv.org/abs/2508.17471

---

## Authors

**Milad Hasanzadeh**
Department of Electrical and Computer Engineering
Louisiana State University
Email: [mhasa42@lsu.edu](mailto:mhasa42@lsu.edu)

**Amin Kargarian**
Department of Electrical and Computer Engineering
Louisiana State University
Email: [kargarian@lsu.edu](mailto:kargarian@lsu.edu)

---

## License

Academic and research use only.

---

## Release Information

Release date: August 2025

Repository: https://github.com/LSU-RAISE-LAB/DVQE.git
