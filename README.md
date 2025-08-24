# Raiselab: Resilient Advanced Infrastructure & Systems Engineering Lab

![License](https://img.shields.io/badge/license-Academic-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Qiskit](https://img.shields.io/badge/Qiskit-0.39.0-purple)

> A modular hybrid **Variational Quantum Eigensolver (VQE)** framework for solving binary optimization problems using both **monolithic** and **distributed quantum** architectures.

---

## 🔬 Overview

Raiselab offers **DVQE** that is a general-purpose Variational Quantum Eigensolver built with extensibility, hybridization, and modularity in mind. It allows researchers to solve any QUBO (Quadratic Unconstrained Binary Optimization) problem using:

- Monolithic quantum circuits (single QPU)
- Distributed quantum circuits (multi-QPU, using entangled teleportation logic via DISKIT

This package supports different quantum initialization methods and advanced circuit distribution schemes.

---

## 🌟 Features

- ✅ **Monolithic & Distributed Execution**: Easily switch between single-QPU and multi-QPU modes.
- 🧠 **Hybrid Optimizers**: Combines gradient-based Adam optimizer with swarm-inspired metaheuristics initialization for ansatz parameters such as Black Hole, GWO, and ABC.
- 🔁 **TeleGate Execution**: Integrates DISKIT for simulating entangled circuit distribution across QPUs.
- 🛠 **Flexible Ansatz Construction**: Layered, parameterized quantum circuits with adjustable depth.
- 🔄 **Circuit Remapping**: Automatically distributes circuits based on QPU partitioning configuration.
- 🔬 **QUBO to Pauli Hamiltonian**: Translates binary quadratic problems into quantum Hamiltonians.
- 📉 **Real-Time Convergence Tracking**: Monitor parameter updates and early-stop based on tolerance.
- 🧪 **Top-k Bitstring Evaluation**: Selects final solution by comparing energy from most probable outcomes.
- 📈 **Visualization Support**: Render final circuits via Qiskit tools for debugging and analysis.
- 🧩 **Modular Design**: Each function is clearly grouped (initialization, circuit build, energy evaluation, decoding).

---

## 📦 Installation

To install the package directly from GitHub:

```bash
pip install git+https://github.com/LSU-RAISE-LAB/DVQE.git
