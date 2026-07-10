import numpy as np
import time
import contextlib
import io

from raiselab.core import dvqe


# ============================================================
# BASIC HELPERS
# ============================================================

def _objective(A, b, c, x):
    """
    Compute f(x) = x^T A x + b^T x + c.
    """
    return float(x.T @ A @ x + b.T @ x + c)


def _symmetrize(A):
    """
    Symmetrize A because x^T A x only depends on the symmetric part.
    """
    return 0.5 * (A + A.T)


def _scale_qubo(Q, q):
    """
    Scale QUBO coefficients for DVQE numerical stability.
    Scaling does not change the minimizer.
    """
    scale = max(
        float(np.max(np.abs(Q))) if Q.size > 0 else 0.0,
        float(np.max(np.abs(q))) if q.size > 0 else 0.0,
        1.0
    )

    return Q / scale, q / scale, scale


# ============================================================
# FIRST MAPPING: x-PROBLEM TO alpha-PROBLEM
# ============================================================

def _build_alpha_qup(A, b, c, x, lb, ub, rho):
    """
    Build the normalized alpha-space quadratic problem.

    This is the same mapping as the original implementation:
        x_candidate = x + diag(ub - lb) alpha

    but it avoids explicitly forming the diagonal matrix diag(ub-lb).
    That makes the function faster without changing the mathematics.
    """

    var_range = ub - lb
    safe_range = np.maximum(var_range, 1e-12)

    # Equivalent to:
    # R = diag(var_range)
    # A_alpha = R.T @ A @ R
    # b_alpha = 2 R.T @ A @ x + R.T @ b
    A_alpha = A * np.outer(var_range, var_range)
    b_alpha = var_range * (2.0 * (A @ x) + b)
    c_alpha = _objective(A, b, c, x)

    alpha_lb = np.maximum(-rho, (lb - x) / safe_range)
    alpha_ub = np.minimum(rho, (ub - x) / safe_range)

    # Numerical protection
    alpha_lb = np.minimum(alpha_lb, alpha_ub)

    return A_alpha, b_alpha, c_alpha, alpha_lb, alpha_ub, var_range

# ============================================================
# SECOND MAPPING: alpha-PROBLEM TO LOCAL ONE-BIT QUBO
# ============================================================

def _build_local_alpha_qubo(
    A_alpha,
    b_alpha,
    c_alpha,
    alpha,
    alpha_delta,
    alpha_lb,
    alpha_ub
):
    """
    Build one local n-bit QUBO in alpha-space.

    This is the same one-bit local mapping as the original implementation:
        alpha_candidate = y + P z
    with P = diag(p), but it avoids explicitly forming P.

    The resulting QUBO is:
        z.T Q_local z + q_local.T z + constant
    """

    d_plus = np.minimum(alpha_delta, alpha_ub - alpha)
    d_minus = np.minimum(alpha_delta, alpha - alpha_lb)

    d_plus = np.maximum(d_plus, 0.0)
    d_minus = np.maximum(d_minus, 0.0)

    y = alpha - d_minus
    p = d_plus + d_minus

    # Equivalent to:
    # P = diag(p)
    # Q_local = P.T @ A_alpha @ P
    # q_local = 2 P.T @ A_alpha @ y + P.T @ b_alpha
    Q_local = A_alpha * np.outer(p, p)
    q_local = p * (2.0 * (A_alpha @ y) + b_alpha)

    constant = float(y.T @ A_alpha @ y + b_alpha.T @ y + c_alpha)

    return Q_local, q_local, constant, y, p, d_plus, d_minus

def _decode_alpha_solution(z, y, p):
    """
    Decode DVQE binary solution into alpha candidate.
    """
    z = np.asarray(z, dtype=float)
    return y + p * z



# ============================================================
# CLASSICAL NON-BRUTE-FORCE QUBO SOLVER
# ============================================================

def _qubo_energy(Q, q, z):
    """
    Compute QUBO energy:
        E(z) = z^T Q z + q^T z
    where z is binary.
    """
    z = np.asarray(z, dtype=float)
    return float(z.T @ Q @ z + q.T @ z)


def _solve_qubo_classical_miqp_gurobi(
    Q,
    q,
    time_limit=None,
    mip_gap=0.0,
    verbose=False,
    threads=None
):
    """
    Solve the QUBO as a classical binary MIQP using Gurobi.

    QUBO:
        min_z z.T Q z + q.T z
        s.t.  z_i in {0,1}

    Notes
    -----
    - This fully replaces the old classical local-search QUBO solver.
    - With mip_gap=0.0 and no time limit, Gurobi attempts to prove global
      optimality of each local QUBO.
    - If a time limit is provided, Gurobi may return the best feasible
      solution found within the time limit.
    """

    import gurobipy as gp
    from gurobipy import GRB

    Q = np.asarray(Q, dtype=float)
    q = np.asarray(q, dtype=float)

    n = q.shape[0]

    if Q.shape != (n, n):
        raise ValueError("Q must have shape (n,n).")

    if mip_gap < 0:
        raise ValueError("mip_gap must be nonnegative.")

    Q = _symmetrize(Q)

    model = gp.Model("local_qubo_miqp")

    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.MIPGap = float(mip_gap)

    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)

    if threads is not None:
        model.Params.Threads = int(threads)

    z = model.addVars(n, vtype=GRB.BINARY, name="z")

    obj = gp.QuadExpr()

    # Diagonal quadratic terms satisfy z_i^2 = z_i for binary variables.
    for i in range(n):
        obj += float(Q[i, i] + q[i]) * z[i]

    # Off-diagonal terms of z.T Q z are 2 Q_ij z_i z_j for i < j.
    for i in range(n):
        for j in range(i + 1, n):
            coef = 2.0 * float(Q[i, j])
            if coef != 0.0:
                obj += coef * z[i] * z[j]

    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    allowed_statuses = {GRB.OPTIMAL, GRB.SUBOPTIMAL}

    if time_limit is not None:
        allowed_statuses.add(GRB.TIME_LIMIT)

    if model.Status not in allowed_statuses:
        raise RuntimeError(f"Gurobi MIQP failed with status {model.Status}.")

    if model.SolCount == 0:
        raise RuntimeError("Gurobi MIQP did not return a feasible binary solution.")

    z_sol = np.array([round(z[i].X) for i in range(n)], dtype=int)
    energy = _qubo_energy(Q, q, z_sol)

    gap = None
    try:
        gap = float(model.MIPGap)
    except Exception:
        gap = None

    info = {
        "method": "gurobi_miqp",
        "status": int(model.Status),
        "objective_solver": float(model.ObjVal),
        "qubo_energy": energy,
        "mip_gap_requested": float(mip_gap),
        "mip_gap_returned": gap,
        "time_limit": time_limit,
        "threads": threads,
        "num_binary_variables": n,
    }

    return z_sol, energy, info
    
# ============================================================
# MAIN SOLVER: SCALE-ADAPTIVE DQUP
# ============================================================

def dqup(
    A,
    b=None,
    c=0.0,
    x0=None,
    lb=None,
    ub=None,

    # Outer x-space settings
    rho0=0.1,
    rho_decay=0.4,
    rho_tol=1e-9,
    obj_tol=1e-10,
    grad_tol=None,
    max_outer_iters=5,

    # NEW outer early-stop setting
    max_consecutive_outer_rejections=4,

    # Inner alpha-space settings
    alpha0=None,
    alpha_step0=None,
    alpha_step_fraction=1.0,
    alpha_step_decay=0.1,
    alpha_step_tol=1e-8,
    max_inner_iters=20,

    # NEW inner early-stop setting
    max_consecutive_inner_rejections=20,

    alpha_warm_start=True,
    alpha_warm_start_scale=0.5,

    # DVQE settings
    mode="distributed",
    init_type=2,
    depth=1,
    lr=0.08,
    max_iters=30,
    qpu_qubit_config=None,
    rel_tol=1e-4,
    num_shots=128,
    final_shots=1000,
    warm_start_population=4,
    warm_start_iters=5,
    warm_start_shots=64,
    seed=None,
    backend=None,
    energy_mode="cvar",
    cvar_alpha=0.2,

    # QUBO solver selection
    qubo_solver="dvqe",          # "dvqe" or "classical"

    # Classical QUBO solver settings
    # In this modified version, "classical" means Gurobi MIQP.
    classical_time_limit=None,
    classical_mip_gap=0.0,
    classical_threads=None,

    # Old local-search arguments are accepted only for backward compatibility.
    # They are ignored because the classical solver is now Gurobi MIQP.
    classical_num_restarts=None,
    classical_max_sweeps=None,

    # Printing controls
    verbose=False,
    silent=True,

    # History controls
    store_full_history=False,
    store_z_history=False,

    # Output settings
    return_info=True
):
    """
    DQUP: Scale-adaptive sequential one-bit QUBO-based solver for bounded
    continuous quadratic unconstrained programming.

    Solves approximately:

        min_x  x^T A x + b^T x + c
        s.t.   lb <= x <= ub

    Main idea:
    ----------
    At outer iteration k, transform the x-problem into a normalized
    alpha-problem:

        x_candidate = x^k + diag(ub-lb) alpha

    where alpha is bounded in a trust region:

        alpha_i in [alpha_lb_i, alpha_ub_i]
        alpha_lb_i >= -rho
        alpha_ub_i <=  rho

    Then solve the alpha-problem using an inner sequential one-bit DVQE loop.

    Each inner QUBO call solves an n-bit QUBO, where n is the number of
    continuous variables. Therefore, the method avoids n*K full discretization.

    New early-stop behavior:
    ------------------------
    - max_consecutive_inner_rejections:
        Stop the inner alpha loop after repeated QUBO candidates fail
        to improve the alpha objective.

    - max_consecutive_outer_rejections:
        Stop the outer x-loop after repeated outer candidates fail to improve
        the original x-space objective.

    Returns
    -------
    If return_info=True:
        x_best, f_best, total_outer_iterations, info

    Else:
        x_best, f_best, total_outer_iterations
    """

    # ------------------------------------------------------------
    # Input processing
    # ------------------------------------------------------------

    A = np.asarray(A, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")

    n = A.shape[0]
    A = _symmetrize(A)

    if b is None:
        b = np.zeros(n, dtype=float)
    else:
        b = np.asarray(b, dtype=float)

    if b.shape != (n,):
        raise ValueError("b must have shape (n,).")

    if lb is None or ub is None:
        raise ValueError("lb and ub must be provided.")

    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    if lb.shape != (n,) or ub.shape != (n,):
        raise ValueError("lb and ub must have shape (n,).")

    if np.any(ub <= lb):
        raise ValueError("Every upper bound must be greater than lower bound.")

    var_range = ub - lb

    if x0 is None:
        x = 0.5 * (lb + ub)
    else:
        x = np.asarray(x0, dtype=float)

    if x.shape != (n,):
        raise ValueError("x0 must have shape (n,).")

    x = np.clip(x, lb, ub)

    if qpu_qubit_config is None:
        qpu_qubit_config = [n]

    rho = float(rho0)

    if rho <= 0:
        raise ValueError("rho0 must be positive.")

    if not (0 < rho_decay < 1):
        raise ValueError("rho_decay must be between 0 and 1.")

    if max_consecutive_outer_rejections is not None:
        if max_consecutive_outer_rejections <= 0:
            raise ValueError(
                "max_consecutive_outer_rejections must be positive or None."
            )

    if not (0 < alpha_step_decay < 1):
        raise ValueError("alpha_step_decay must be between 0 and 1.")

    if max_consecutive_inner_rejections is not None:
        if max_consecutive_inner_rejections <= 0:
            raise ValueError(
                "max_consecutive_inner_rejections must be positive or None."
            )

    if not (0 < alpha_warm_start_scale <= 1):
        raise ValueError("alpha_warm_start_scale must be in (0, 1].")
    # ------------------------------------------------------------
    # QUBO solver checks
    # ------------------------------------------------------------

    qubo_solver = qubo_solver.lower()

    if qubo_solver not in ["dvqe", "classical"]:
        raise ValueError("qubo_solver must be either 'dvqe' or 'classical'.")

    if classical_mip_gap < 0:
        raise ValueError("classical_mip_gap must be nonnegative.")

    if classical_time_limit is not None and classical_time_limit <= 0:
        raise ValueError("classical_time_limit must be positive or None.")

    if classical_threads is not None and classical_threads <= 0:
        raise ValueError("classical_threads must be positive or None.")
    # ------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------

    f_current = _objective(A, b, c, x)

    trajectory = [x.copy()]
    objective_history = [f_current]
    rho_history = [rho]
    accepted_outer_history = []
    outer_status_history = []

    alpha_solution_history = []
    alpha_initial_history = []
    alpha_objective_history = []
    alpha_step_history = []
    inner_status_history = []
    z_history = []

    local_qubo_history = []
    dvqe_runtime_history = []

    accepted_outer_updates = 0
    rejected_outer_updates = 0
    total_dvqe_calls = 0

    start_total = time.time()
    previous_alpha = None

    # NEW: count repeated rejected outer updates
    consecutive_outer_rejections = 0

    # ============================================================
    # OUTER LOOP IN x-SPACE
    # ============================================================

    for outer_iter in range(1, max_outer_iters + 1):

        # --------------------------------------------------------
        # Outer stopping checks
        # --------------------------------------------------------

        if rho <= rho_tol:
            outer_status_history.append("stopped: rho tolerance reached")
            break

        if grad_tol is not None:
            grad = 2.0 * A @ x + b
            if np.linalg.norm(grad, ord=np.inf) <= grad_tol:
                outer_status_history.append("stopped: gradient tolerance reached")
                break

        # --------------------------------------------------------
        # Build alpha-QUP around current x
        # --------------------------------------------------------

        A_alpha, b_alpha, c_alpha, alpha_lb, alpha_ub, var_range = _build_alpha_qup(
            A=A,
            b=b,
            c=c,
            x=x,
            lb=lb,
            ub=ub,
            rho=rho
        )

        # If trust region is numerically zero, shrink rho and continue.
        alpha_width = alpha_ub - alpha_lb

        if np.max(alpha_width) <= 1e-14:
            rho *= rho_decay
            rho_history.append(rho)
            rejected_outer_updates += 1
            consecutive_outer_rejections += 1
            outer_status_history.append("rejected: alpha region too small, rho reduced")

            if (
                max_consecutive_outer_rejections is not None
                and consecutive_outer_rejections >= max_consecutive_outer_rejections
            ):
                outer_status_history.append("stopped: consecutive outer rejections")
                break

            continue

        # --------------------------------------------------------
        # Initialize alpha for inner loop
        # --------------------------------------------------------

        if alpha0 is not None:
            alpha = np.asarray(alpha0, dtype=float)
            if alpha.shape != (n,):
                raise ValueError("alpha0 must have shape (n,).")

        elif alpha_warm_start and previous_alpha is not None:
            alpha = alpha_warm_start_scale * previous_alpha

        else:
            alpha = np.zeros(n, dtype=float)

        alpha = np.clip(alpha, alpha_lb, alpha_ub)
        alpha_initial_history.append(alpha.copy())

        if alpha_step0 is None:
            alpha_delta = alpha_step_fraction * alpha_width
        else:
            alpha_step0_arr = np.asarray(alpha_step0, dtype=float)

            if alpha_step0_arr.ndim == 0:
                alpha_delta = np.ones(n, dtype=float) * float(alpha_step0_arr)
            elif alpha_step0_arr.shape == (n,):
                alpha_delta = alpha_step0_arr.copy()
            else:
                raise ValueError("alpha_step0 must be None, scalar, or shape (n,).")

        alpha_delta = np.minimum(alpha_delta, alpha_width)
        alpha_delta = np.maximum(alpha_delta, 0.0)

        f_alpha_current = float(alpha.T @ A_alpha @ alpha + b_alpha.T @ alpha + c_alpha)

        inner_objectives = [f_alpha_current]
        inner_steps = [alpha_delta.copy()]
        inner_statuses = []

        # NEW: count repeated rejected inner DVQE-QUBO refinements
        consecutive_inner_rejections = 0

        # ========================================================
        # INNER LOOP IN alpha-SPACE
        # ========================================================

        for inner_iter in range(1, max_inner_iters + 1):

            relative_alpha_step = np.max(
                alpha_delta / np.maximum(alpha_width, 1e-12)
            )

            if relative_alpha_step <= alpha_step_tol:
                inner_statuses.append("stopped: alpha step tolerance reached")
                break

            # ----------------------------------------------------
            # Build local one-bit QUBO in alpha-space
            # ----------------------------------------------------

            Q_local, q_local, constant, y, p, d_plus, d_minus = _build_local_alpha_qubo(
                A_alpha=A_alpha,
                b_alpha=b_alpha,
                c_alpha=c_alpha,
                alpha=alpha,
                alpha_delta=alpha_delta,
                alpha_lb=alpha_lb,
                alpha_ub=alpha_ub
            )

            if store_full_history:
                local_qubo_history.append({
                    "outer_iter": outer_iter,
                    "inner_iter": inner_iter,
                    "Q": Q_local.copy(),
                    "q_linear": q_local.copy(),
                    "constant": constant,
                    "alpha": alpha.copy(),
                    "alpha_y": y.copy(),
                    "alpha_p": p.copy(),
                    "d_plus": d_plus.copy(),
                    "d_minus": d_minus.copy(),
                    "rho": rho
                })

            # DVQE benefits from coefficient scaling. Gurobi MIQP can solve
            # the original local QUBO coefficients directly.
            if qubo_solver.lower() == "dvqe":
                Q_train, q_train, scale = _scale_qubo(Q_local, q_local)
            else:
                Q_train, q_train, scale = Q_local, q_local, 1.0

            # ----------------------------------------------------
            # Solve local alpha-QUBO using selected QUBO solver
            # ----------------------------------------------------

            qubo_start = time.time()

            try:
                qubo_seed = None
                if seed is not None:
                    qubo_seed = seed + 1000 * outer_iter + inner_iter

                if qubo_solver.lower() == "dvqe":

                    if silent:
                        buffer = io.StringIO()
                        with contextlib.redirect_stdout(buffer):
                            z_qubo, final_circuit, hist = dvqe(
                                mode=mode,
                                Q=Q_train,
                                q_linear=q_train,
                                init_type=init_type,
                                depth=depth,
                                lr=lr,
                                max_iters=max_iters,
                                qpu_qubit_config=qpu_qubit_config,
                                rel_tol=rel_tol,
                                num_shots=num_shots,
                                final_shots=final_shots,
                                warm_start_population=warm_start_population,
                                warm_start_iters=warm_start_iters,
                                warm_start_shots=warm_start_shots,
                                seed=qubo_seed,
                                verbose=False,
                                backend=backend,
                                energy_mode=energy_mode,
                                cvar_alpha=cvar_alpha
                            )
                    else:
                        z_qubo, final_circuit, hist = dvqe(
                            mode=mode,
                            Q=Q_train,
                            q_linear=q_train,
                            init_type=init_type,
                            depth=depth,
                            lr=lr,
                            max_iters=max_iters,
                            qpu_qubit_config=qpu_qubit_config,
                            rel_tol=rel_tol,
                            num_shots=num_shots,
                            final_shots=final_shots,
                            warm_start_population=warm_start_population,
                            warm_start_iters=warm_start_iters,
                            warm_start_shots=warm_start_shots,
                            seed=qubo_seed,
                            verbose=verbose,
                            backend=backend,
                            energy_mode=energy_mode,
                            cvar_alpha=cvar_alpha
                        )

                    solver_info = {
                        "solver": "dvqe",
                        "hist": hist,
                    }

                elif qubo_solver.lower() == "classical":

                    z_qubo, qubo_energy, classical_info = _solve_qubo_classical_miqp_gurobi(
                        Q=Q_train,
                        q=q_train,
                        time_limit=classical_time_limit,
                        mip_gap=classical_mip_gap,
                        verbose=False if silent else verbose,
                        threads=classical_threads
                    )

                    final_circuit = None
                    hist = None

                    solver_info = {
                        "solver": "classical",
                        "method": "gurobi_miqp",
                        "qubo_energy": qubo_energy,
                        "classical_info": classical_info,
                    }

                else:
                    raise ValueError(
                        "qubo_solver must be either 'dvqe' or 'classical'."
                    )

                qubo_runtime = time.time() - qubo_start
                dvqe_runtime_history.append(qubo_runtime)
                total_dvqe_calls += 1

                z_qubo = np.asarray(z_qubo, dtype=int)

                if z_qubo.shape != (n,):
                    raise ValueError(
                        f"QUBO solver returned solution with shape {z_qubo.shape}, expected {(n,)}."
                    )

                if store_z_history:
                    z_history.append({
                        "outer_iter": outer_iter,
                        "inner_iter": inner_iter,
                        "z": z_qubo.copy(),
                        "qubo_solver": qubo_solver,
                        "solver_info": solver_info,
                    })

                # ------------------------------------------------
                # Decode QUBO solution into alpha candidate
                # ------------------------------------------------

                alpha_candidate = _decode_alpha_solution(z_qubo, y, p)
                alpha_candidate = np.clip(alpha_candidate, alpha_lb, alpha_ub)

                f_alpha_candidate = float(
                    alpha_candidate.T @ A_alpha @ alpha_candidate
                    + b_alpha.T @ alpha_candidate
                    + c_alpha
                )

                improvement_alpha = f_alpha_current - f_alpha_candidate

                # ------------------------------------------------
                # Inner acceptance / rejection
                # ------------------------------------------------

                if improvement_alpha > obj_tol:
                    alpha = alpha_candidate.copy()
                    f_alpha_current = f_alpha_candidate
                    inner_status = "accepted"

                    # NEW: reset because alpha improved
                    consecutive_inner_rejections = 0

                else:
                    alpha_delta = alpha_step_decay * alpha_delta
                    inner_status = "rejected: alpha step reduced"

                    # NEW: count failed alpha refinements
                    consecutive_inner_rejections += 1

                inner_objectives.append(f_alpha_current)
                inner_steps.append(alpha_delta.copy())
                inner_statuses.append(inner_status)

                # NEW: stop inner loop after repeated failed DVQE-QUBO refinements
                if (
                    max_consecutive_inner_rejections is not None
                    and consecutive_inner_rejections >= max_consecutive_inner_rejections
                ):
                    inner_statuses.append("stopped: consecutive inner rejections")
                    break

                if verbose and not silent:
                    print(
                        f"Outer {outer_iter:3d} | Inner {inner_iter:3d} | "
                        f"f_alpha={f_alpha_current:.10f} | "
                        f"cand={f_alpha_candidate:.10f} | "
                        f"imp={improvement_alpha:.3e} | "
                        f"status={inner_status} | "
                        f"max(alpha_step/width)={relative_alpha_step:.3e} | "
                        f"inner_rej={consecutive_inner_rejections}"
                    )

            except Exception as e:
                inner_statuses.append(
                    f"failed at outer {outer_iter}, inner {inner_iter}: {str(e)}"
                )
                break

        else:
            inner_statuses.append("stopped: max inner iterations reached")

        # --------------------------------------------------------
        # End of inner loop: alpha is the best scale vector found
        # --------------------------------------------------------

        alpha_solution_history.append(alpha.copy())
        alpha_objective_history.append(inner_objectives)
        alpha_step_history.append(inner_steps)
        inner_status_history.append(inner_statuses)

        # Candidate in original x-space
        x_candidate = x + var_range * alpha
        x_candidate = np.clip(x_candidate, lb, ub)

        f_candidate = _objective(A, b, c, x_candidate)
        improvement_outer = f_current - f_candidate

        # --------------------------------------------------------
        # Outer global acceptance / rejection
        # --------------------------------------------------------

        if improvement_outer > obj_tol:
            x = x_candidate.copy()
            f_current = f_candidate
            accepted_outer = True
            accepted_outer_updates += 1
            outer_status = "accepted"

            # NEW: reset because x improved
            consecutive_outer_rejections = 0

            # Save successful alpha to warm-start the next outer iteration
            previous_alpha = alpha.copy()

        else:
            rho = rho_decay * rho
            accepted_outer = False
            rejected_outer_updates += 1
            outer_status = "rejected: rho reduced"

            # NEW: count failed outer update
            consecutive_outer_rejections += 1

            # Optional: weaken the previous alpha after failed outer update
            if previous_alpha is not None:
                previous_alpha = alpha_warm_start_scale * previous_alpha

        trajectory.append(x.copy())
        objective_history.append(f_current)
        rho_history.append(rho)
        accepted_outer_history.append(accepted_outer)
        outer_status_history.append(outer_status)

        if verbose and not silent:
            print(
                f"OUTER {outer_iter:3d} | "
                f"f={f_current:.10f} | "
                f"candidate={f_candidate:.10f} | "
                f"improvement={improvement_outer:.3e} | "
                f"accepted={accepted_outer} | "
                f"rho={rho:.3e} | "
                f"dvqe_calls={total_dvqe_calls} | "
                f"outer_rej={consecutive_outer_rejections}"
            )

        # NEW: stop outer loop after repeated failed outer updates
        if (
            max_consecutive_outer_rejections is not None
            and consecutive_outer_rejections >= max_consecutive_outer_rejections
        ):
            outer_status_history.append("stopped: consecutive outer rejections")
            break

    else:
        outer_status_history.append("stopped: max outer iterations reached")

    # ------------------------------------------------------------
    # Return result
    # ------------------------------------------------------------

    total_runtime = time.time() - start_total
    total_outer_iterations = len(accepted_outer_history)

    info = {
        "trajectory": trajectory,
        "objective_history": objective_history,
        "rho_history": rho_history,
        "accepted_outer_history": accepted_outer_history,
        "outer_status_history": outer_status_history,

        "alpha_initial_history": alpha_initial_history,
        "alpha_solution_history": alpha_solution_history,
        "alpha_objective_history": alpha_objective_history,
        "alpha_step_history": alpha_step_history,
        "inner_status_history": inner_status_history,

        "z_history": z_history,
        "local_qubo_history": local_qubo_history,
        "dvqe_runtime_history": dvqe_runtime_history,

        "accepted_outer_updates": accepted_outer_updates,
        "rejected_outer_updates": rejected_outer_updates,
        "total_outer_iterations": total_outer_iterations,
        "total_dvqe_calls": total_dvqe_calls,
        "total_runtime_sec": total_runtime,

        "final_rho": rho,
        "qpu_qubit_config": qpu_qubit_config,
        "mode": mode,
        "silent": silent,

        # QUBO solver settings
        "qubo_solver": qubo_solver,
        "classical_method": "gurobi_miqp",
        "classical_time_limit": classical_time_limit,
        "classical_mip_gap": classical_mip_gap,
        "classical_threads": classical_threads,
        "store_full_history": store_full_history,

        # NEW saved settings
        "alpha_warm_start": alpha_warm_start,
        "alpha_warm_start_scale": alpha_warm_start_scale,
        "max_consecutive_inner_rejections": max_consecutive_inner_rejections,
        "max_consecutive_outer_rejections": max_consecutive_outer_rejections,
    }

    if return_info:
        return x, f_current, total_outer_iterations, info

    return x, f_current, total_outer_iterations



# ============================================================
# PHR FIXED-REGION HELPERS
# ============================================================

def _phr_augmented_objective(
    A,
    b,
    c,
    x,
    G,
    r,
    H,
    h,
    lambda_eq,
    lambda_ineq,
    mu
):
    """
    Evaluate the Powell-Hestenes-Rockafellar augmented Lagrangian.

    Original QP:
        min f(x) = x^T A x + b^T x + c

        s.t. Gx = r
             Hx <= h

    Let:
        e(x) = Gx - r
        g(x) = Hx - h

    The PHR augmented Lagrangian is:

        L_mu(x, lambda_eq, lambda_ineq)
        = f(x)
          + lambda_eq^T e(x)
          + mu/2 ||e(x)||^2
          + 1/(2 mu) (
                ||max(0, lambda_ineq + mu g(x))||^2
                - ||lambda_ineq||^2
            )

    The maximum is componentwise.
    """
    value = _objective(A, b, c, x)

    if G.shape[0] > 0:
        eq_residual = G @ x - r
        value += float(lambda_eq.T @ eq_residual)
        value += 0.5 * mu * float(eq_residual.T @ eq_residual)

    if H.shape[0] > 0:
        ineq_residual = H @ x - h
        shifted = lambda_ineq + mu * ineq_residual
        positive_shifted = np.maximum(shifted, 0.0)

        value += (
            float(positive_shifted.T @ positive_shifted)
            - float(lambda_ineq.T @ lambda_ineq)
        ) / (2.0 * mu)

    return float(value)


def _build_phr_fixed_region_qp(
    A,
    b,
    c,
    G,
    r,
    H,
    h,
    lambda_eq,
    lambda_ineq,
    mu,
    region_mask
):
    """
    Build one ordinary QP corresponding to a fixed PHR region.

    For a selected PHR region P:

        P = { i : lambda_i + mu (H_i x - h_i) > 0 }

    the inequality contribution for i in P is:

        lambda_i (H_i x - h_i)
        + mu/2 (H_i x - h_i)^2

    while a row outside P contributes only a constant.

    Therefore, for a fixed region, the PHR AL is exactly a quadratic
    function of x. DQUP can solve this n-variable bounded QP directly.

    The returned constant includes the PHR term:
        - ||lambda_ineq||^2 / (2 mu)
    so the returned QP objective equals the fixed-region PHR expression.
    """
    n = A.shape[0]

    region_mask = np.asarray(region_mask, dtype=bool)

    if region_mask.shape != (H.shape[0],):
        raise ValueError("region_mask must have shape (number of inequalities,).")

    # Equality contribution:
    #
    # lambda_eq^T(Gx-r) + mu/2 ||Gx-r||^2
    #
    A_aug = A.copy()
    b_aug = b.copy()
    c_aug = float(c)

    if G.shape[0] > 0:
        A_aug += 0.5 * mu * (G.T @ G)
        b_aug += G.T @ lambda_eq - mu * (G.T @ r)

        c_aug += (
            -float(lambda_eq.T @ r)
            + 0.5 * mu * float(r.T @ r)
        )

    # PHR inequality contribution for the currently selected region.
    active_indices = np.where(region_mask)[0]

    if active_indices.size > 0:
        H_region = H[active_indices, :]
        h_region = h[active_indices]
        lambda_region = lambda_ineq[active_indices]

        A_aug += 0.5 * mu * (H_region.T @ H_region)
        b_aug += (
            H_region.T @ lambda_region
            - mu * (H_region.T @ h_region)
        )

        c_aug += (
            -float(lambda_region.T @ h_region)
            + 0.5 * mu * float(h_region.T @ h_region)
        )

    # This constant does not affect the x minimizer, but including it
    # makes the fixed-region objective equal to the PHR expression.
    if H.shape[0] > 0:
        c_aug -= 0.5 * float(lambda_ineq.T @ lambda_ineq) / mu

    return (
        _symmetrize(A_aug),
        b_aug,
        float(c_aug),
        active_indices
    )


# ============================================================
# MAIN SOLVER: FIXED-REGION PHR DQP
# ============================================================

def dqp(
    A,
    b=None,
    c=0.0,

    # Equality constraints: Gx = r
    G=None,
    r=None,

    # Inequality constraints: Hx <= h
    H=None,
    h=None,

    x0=None,
    lb=None,
    ub=None,

    # Augmented-Lagrangian settings
    lambda0=None,
    mu0=150.0,
    mu_min=1,
    mu_max=1e8,
    constraint_tol=1e-6,
    stationarity_tol=None,
    max_al_iters=20,
    min_al_iters=2,

    # Fixed PHR-region settings
    phr_region_tol=1e-12,
    max_region_iters=10,

    # Mu update settings
    update_mu=True,
    mu_update_rule="primal_dual_balance",
    balance_factor=10.0,
    mu_increase=2.0,
    mu_decrease=1.5,
    allow_mu_decrease=True,
    primal_stall_ratio=0.90,

    # Printing controls
    verbose=False,

    # Output settings
    return_info=True,

    # Passed into dqup()
    **dqup_kwargs
):
    """
    Fixed-region Powell-Hestenes-Rockafellar (PHR) augmented-
    Lagrangian DQP solver for bounded QPs with linear equalities
    and inequalities.

    Solves approximately:

        min_x  x^T A x + b^T x + c

        s.t.   Gx = r
               Hx <= h
               lb <= x <= ub

    No slack variables are introduced. Every call to DQUP has the
    original dimension n.

    PHR inequality formulation:
    ---------------------------
    Let:

        g(x) = Hx - h <= 0

    and let lambda_ineq >= 0. The PHR AL term is:

        1/(2 mu) [
            ||max(0, lambda_ineq + mu g(x))||^2
            - ||lambda_ineq||^2
        ]

    This function is piecewise quadratic. At a fixed region P:

        P = {i : lambda_i + mu g_i(x) > 0},

    it becomes one ordinary quadratic function of x. This solver
    repeatedly:

        1) chooses P from the current x,
        2) builds the corresponding fixed-region QP,
        3) solves that QP with DQUP,
        4) recomputes P,
        5) repeats until P is stable or max_region_iters is reached.

    Multiplier updates:
    -------------------
    Equality multipliers are unrestricted:

        lambda_eq <- lambda_eq + mu (Gx-r)

    Inequality multipliers are projected:

        lambda_ineq <- max(0, lambda_ineq + mu (Hx-h))

    Notes:
    ------
    A fixed-region DQUP subproblem is a QP, but global convergence
    still depends on sufficiently accurate DQUP solves, sensible mu
    updates, and successful PHR-region stabilization.
    """

    # ------------------------------------------------------------
    # Input processing
    # ------------------------------------------------------------

    A = np.asarray(A, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")

    n = A.shape[0]
    A = _symmetrize(A)

    if b is None:
        b = np.zeros(n, dtype=float)
    else:
        b = np.asarray(b, dtype=float)

    if b.shape != (n,):
        raise ValueError("b must have shape (n,).")

    if lb is None or ub is None:
        raise ValueError("lb and ub must be provided.")

    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    if lb.shape != (n,) or ub.shape != (n,):
        raise ValueError("lb and ub must have shape (n,).")

    if np.any(ub <= lb):
        raise ValueError("Every upper bound must be greater than lower bound.")

    # ------------------------------------------------------------
    # Equality constraints Gx = r
    # ------------------------------------------------------------

    if (G is None) != (r is None):
        raise ValueError("G and r must be both provided or both None.")

    has_eq = G is not None and r is not None

    if has_eq:
        G = np.asarray(G, dtype=float)
        r = np.asarray(r, dtype=float)

        if G.ndim != 2 or G.shape[1] != n:
            raise ValueError("G must have shape (m, n).")

        m = G.shape[0]

        if r.shape != (m,):
            raise ValueError("r must have shape (m,).")

    else:
        G = np.zeros((0, n), dtype=float)
        r = np.zeros(0, dtype=float)
        m = 0

    # ------------------------------------------------------------
    # Inequality constraints Hx <= h
    # ------------------------------------------------------------

    if (H is None) != (h is None):
        raise ValueError("H and h must be both provided or both None.")

    has_ineq = H is not None and h is not None

    if has_ineq:
        H = np.asarray(H, dtype=float)
        h = np.asarray(h, dtype=float)

        if H.ndim != 2 or H.shape[1] != n:
            raise ValueError("H must have shape (p, n).")

        p = H.shape[0]

        if h.shape != (p,):
            raise ValueError("h must have shape (p,).")

    else:
        H = np.zeros((0, n), dtype=float)
        h = np.zeros(0, dtype=float)
        p = 0

    if m == 0 and p == 0:
        raise ValueError(
            "At least one of Gx=r or Hx<=h must be provided."
        )

    # ------------------------------------------------------------
    # Initial x
    # ------------------------------------------------------------

    if x0 is None:
        x = 0.5 * (lb + ub)
    else:
        x = np.asarray(x0, dtype=float)

    if x.shape != (n,):
        raise ValueError("x0 must have shape (n,).")

    x = np.clip(x, lb, ub)

    # ------------------------------------------------------------
    # Settings validation
    # ------------------------------------------------------------

    mu = float(mu0)

    if mu <= 0:
        raise ValueError("mu0 must be positive.")

    if mu_min <= 0:
        raise ValueError("mu_min must be positive.")

    if mu_max < mu_min:
        raise ValueError("mu_max must be greater than or equal to mu_min.")

    if max_al_iters <= 0:
        raise ValueError("max_al_iters must be positive.")

    if min_al_iters < 1:
        raise ValueError("min_al_iters must be at least 1.")

    if max_region_iters <= 0:
        raise ValueError("max_region_iters must be positive.")

    phr_region_tol = float(phr_region_tol)

    if phr_region_tol < 0:
        raise ValueError("phr_region_tol must be nonnegative.")

    if balance_factor <= 1:
        raise ValueError("balance_factor must be greater than 1.")

    if mu_increase <= 1:
        raise ValueError("mu_increase must be greater than 1.")

    if mu_decrease <= 1:
        raise ValueError("mu_decrease must be greater than 1.")

    if not (0.0 < primal_stall_ratio <= 1.0):
        raise ValueError("primal_stall_ratio must be in (0, 1].")

    # Do not allow duplicate DQUP arguments.
    for key in ["A", "b", "c", "x0", "lb", "ub", "return_info"]:
        dqup_kwargs.pop(key, None)

    # ------------------------------------------------------------
    # Multiplier initialization
    # ------------------------------------------------------------

    total_constraints = m + p

    if lambda0 is None:
        lambda_eq = np.zeros(m, dtype=float)
        lambda_ineq = np.zeros(p, dtype=float)

    else:
        lambda0 = np.asarray(lambda0, dtype=float)

        if lambda0.shape != (total_constraints,):
            raise ValueError("lambda0 must have shape (m+p,).")

        lambda_eq = lambda0[:m].copy()
        lambda_ineq = np.maximum(0.0, lambda0[m:m + p])

    # ------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------

    start_total = time.time()

    x_history = [x.copy()]

    lambda_eq_history = [lambda_eq.copy()]
    lambda_ineq_history = [lambda_ineq.copy()]
    lambda_history = [np.concatenate([lambda_eq, lambda_ineq])]

    mu_history = [mu]

    original_objective_history = [_objective(A, b, c, x)]
    phr_augmented_objective_history = [
        _phr_augmented_objective(
            A, b, c, x, G, r, H, h, lambda_eq, lambda_ineq, mu
        )
    ]

    eq_residual_history = []
    ineq_residual_history = []
    ineq_violation_history = []

    eq_residual_norm_inf_history = []
    ineq_residual_norm_inf_history = []
    ineq_violation_inf_history = []
    primal_residual_history = []
    dual_residual_history = []
    stationarity_norm_inf_history = []
    complementarity_norm_inf_history = []

    # PHR fixed-region histories.
    phr_region_mask_history = []
    phr_region_indices_history = []
    phr_region_count_history = []
    phr_region_stable_history = []
    phr_region_iterations_history = []
    phr_region_info_history = []

    # Compatibility aliases for previous runner code.
    active_set_history = []
    active_indices_history = []
    num_active_ineq_history = []

    dqup_info_history = []
    dqup_iterations_history = []
    dqup_runtime_history = []

    mu_update_history = []
    status_history = []

    best_feasible_x = None
    best_feasible_f = np.inf

    previous_primal_residual = np.inf

    # ============================================================
    # OUTER PHR AUGMENTED-LAGRANGIAN LOOP
    # ============================================================

    for al_iter in range(1, max_al_iters + 1):

        x_region = x.copy()
        region_stable = (p == 0)
        final_region_mask = np.zeros(p, dtype=bool)
        final_region_indices = np.zeros(0, dtype=int)

        final_dqup_info = None
        final_dqup_outer_iters = 0
        total_region_runtime = 0.0
        region_records = []

        # ========================================================
        # INNER FIXED-REGION LOOP
        # ========================================================

        for region_iter in range(1, max_region_iters + 1):

            if p > 0:
                g_region = H @ x_region - h
                shifted_region = lambda_ineq + mu * g_region

                region_mask = shifted_region > phr_region_tol
            else:
                g_region = np.zeros(0, dtype=float)
                shifted_region = np.zeros(0, dtype=float)
                region_mask = np.zeros(0, dtype=bool)

            (
                A_region,
                b_region,
                c_region,
                region_indices
            ) = _build_phr_fixed_region_qp(
                A=A,
                b=b,
                c=c,
                G=G,
                r=r,
                H=H,
                h=h,
                lambda_eq=lambda_eq,
                lambda_ineq=lambda_ineq,
                mu=mu,
                region_mask=region_mask
            )

            dqup_start = time.time()

            x_candidate, fixed_region_objective, dqup_outer_iters, dqup_info = dqup(
                A=A_region,
                b=b_region,
                c=c_region,
                x0=x_region,
                lb=lb,
                ub=ub,
                return_info=True,
                **dqup_kwargs
            )

            dqup_runtime = time.time() - dqup_start
            total_region_runtime += dqup_runtime

            x_candidate = np.clip(x_candidate, lb, ub)

            if p > 0:
                g_candidate = H @ x_candidate - h
                shifted_candidate = lambda_ineq + mu * g_candidate
                candidate_region_mask = shifted_candidate > phr_region_tol
            else:
                g_candidate = np.zeros(0, dtype=float)
                shifted_candidate = np.zeros(0, dtype=float)
                candidate_region_mask = np.zeros(0, dtype=bool)

            region_records.append({
                "region_iteration": region_iter,
                "region_mask_before_solve": region_mask.copy(),
                "region_indices_before_solve": region_indices.copy(),
                "shifted_before_solve": shifted_region.copy(),
                "x_before_solve": x_region.copy(),
                "x_after_solve": x_candidate.copy(),
                "ineq_residual_after_solve": g_candidate.copy(),
                "shifted_after_solve": shifted_candidate.copy(),
                "region_mask_after_solve": candidate_region_mask.copy(),
                "region_stable": bool(
                    np.array_equal(region_mask, candidate_region_mask)
                ),
                "fixed_region_objective": float(fixed_region_objective),
                "dqup_info": dqup_info,
                "dqup_outer_iterations": int(dqup_outer_iters),
                "dqup_runtime_sec": float(dqup_runtime),
            })

            x_region = x_candidate.copy()
            final_region_mask = region_mask.copy()
            final_region_indices = region_indices.copy()
            final_dqup_info = dqup_info
            final_dqup_outer_iters = int(dqup_outer_iters)

            if np.array_equal(region_mask, candidate_region_mask):
                region_stable = True
                break

        # Accept the final point generated by the fixed-region loop.
        x_before_al = x.copy()
        x = x_region.copy()

        # Recompute the PHR region at the accepted point.
        if p > 0:
            ineq_residual = H @ x - h
            ineq_violation = np.maximum(ineq_residual, 0.0)

            shifted_final_before_update = (
                lambda_ineq + mu * ineq_residual
            )

            final_region_mask_at_x = (
                shifted_final_before_update > phr_region_tol
            )

            final_region_indices_at_x = np.where(
                final_region_mask_at_x
            )[0]

        else:
            ineq_residual = np.zeros(0, dtype=float)
            ineq_violation = np.zeros(0, dtype=float)
            shifted_final_before_update = np.zeros(0, dtype=float)
            final_region_mask_at_x = np.zeros(0, dtype=bool)
            final_region_indices_at_x = np.zeros(0, dtype=int)

        if m > 0:
            eq_residual = G @ x - r
        else:
            eq_residual = np.zeros(0, dtype=float)

        f_original = _objective(A, b, c, x)

        phr_augmented_objective = _phr_augmented_objective(
            A, b, c, x, G, r, H, h, lambda_eq, lambda_ineq, mu
        )

        eq_residual_norm_inf = (
            float(np.linalg.norm(eq_residual, ord=np.inf))
            if m > 0 else 0.0
        )

        ineq_residual_norm_inf = (
            float(np.linalg.norm(ineq_residual, ord=np.inf))
            if p > 0 else 0.0
        )

        ineq_violation_inf = (
            float(np.linalg.norm(ineq_violation, ord=np.inf))
            if p > 0 else 0.0
        )

        primal_residual = max(
            eq_residual_norm_inf,
            ineq_violation_inf
        )

        # --------------------------------------------------------
        # Multiplier updates
        # --------------------------------------------------------

        if m > 0:
            lambda_eq_new = lambda_eq + mu * eq_residual
        else:
            lambda_eq_new = np.zeros(0, dtype=float)

        if p > 0:
            lambda_ineq_new = np.maximum(
                0.0,
                lambda_ineq + mu * ineq_residual
            )
        else:
            lambda_ineq_new = np.zeros(0, dtype=float)

        # --------------------------------------------------------
        # KKT-related diagnostics
        # --------------------------------------------------------

        grad_x = 2.0 * A @ x + b

        stationarity = grad_x.copy()

        if m > 0:
            stationarity += G.T @ lambda_eq_new

        if p > 0:
            stationarity += H.T @ lambda_ineq_new

        stationarity_norm_inf = float(
            np.linalg.norm(stationarity, ord=np.inf)
        )

        if p > 0:
            complementarity_norm_inf = float(
                np.linalg.norm(
                    lambda_ineq_new * ineq_residual,
                    ord=np.inf
                )
            )
        else:
            complementarity_norm_inf = 0.0

        # This is an AL-change diagnostic, not an exact KKT dual
        # residual. It uses the final fixed PHR region.
        if m == 0 and final_region_indices_at_x.size == 0:
            dual_residual = np.inf
        else:
            dx = x - x_before_al
            dual_vec = np.zeros(n, dtype=float)

            if m > 0:
                dual_vec += mu * (G.T @ (G @ dx))

            if final_region_indices_at_x.size > 0:
                H_final_region = H[final_region_indices_at_x, :]
                dual_vec += mu * (
                    H_final_region.T @ (H_final_region @ dx)
                )

            dual_residual = float(
                np.linalg.norm(dual_vec, ord=np.inf)
            )

        # --------------------------------------------------------
        # Convergence test
        # --------------------------------------------------------

        eq_satisfied = eq_residual_norm_inf <= constraint_tol
        ineq_satisfied = ineq_violation_inf <= constraint_tol
        constraint_satisfied = eq_satisfied and ineq_satisfied

        if stationarity_tol is None:
            stationarity_satisfied = True
        else:
            stationarity_satisfied = (
                stationarity_norm_inf <= stationarity_tol
            )

        if constraint_satisfied and f_original < best_feasible_f:
            best_feasible_x = x.copy()
            best_feasible_f = f_original

        # --------------------------------------------------------
        # Store histories
        # --------------------------------------------------------

        x_history.append(x.copy())

        # Histories store the multipliers that produced this AL step.
        lambda_eq_history.append(lambda_eq.copy())
        lambda_ineq_history.append(lambda_ineq.copy())
        lambda_history.append(np.concatenate([lambda_eq, lambda_ineq]))

        mu_history.append(mu)

        original_objective_history.append(f_original)
        phr_augmented_objective_history.append(phr_augmented_objective)

        eq_residual_history.append(eq_residual.copy())
        ineq_residual_history.append(ineq_residual.copy())
        ineq_violation_history.append(ineq_violation.copy())

        eq_residual_norm_inf_history.append(eq_residual_norm_inf)
        ineq_residual_norm_inf_history.append(ineq_residual_norm_inf)
        ineq_violation_inf_history.append(ineq_violation_inf)

        primal_residual_history.append(primal_residual)
        dual_residual_history.append(dual_residual)
        stationarity_norm_inf_history.append(stationarity_norm_inf)
        complementarity_norm_inf_history.append(complementarity_norm_inf)

        phr_region_mask_history.append(final_region_mask_at_x.copy())
        phr_region_indices_history.append(
            final_region_indices_at_x.copy()
        )
        phr_region_count_history.append(
            int(final_region_indices_at_x.size)
        )
        phr_region_stable_history.append(bool(region_stable))
        phr_region_iterations_history.append(len(region_records))
        phr_region_info_history.append(region_records)

        # Compatibility aliases.
        active_set_history.append(final_region_mask_at_x.copy())
        active_indices_history.append(
            final_region_indices_at_x.copy()
        )
        num_active_ineq_history.append(
            int(final_region_indices_at_x.size)
        )

        dqup_info_history.append(final_dqup_info)
        dqup_iterations_history.append(final_dqup_outer_iters)
        dqup_runtime_history.append(total_region_runtime)

        # --------------------------------------------------------
        # Stop if converged
        # --------------------------------------------------------

        if (
            al_iter >= min_al_iters
            and constraint_satisfied
            and stationarity_satisfied
        ):
            status_history.append("converged")

            mu_update_history.append({
                "iteration": al_iter,
                "mu_before": mu,
                "mu_after": mu,
                "status": "not updated: converged",
                "primal_residual": primal_residual,
                "dual_residual": dual_residual,
                "eq_residual_inf": eq_residual_norm_inf,
                "ineq_violation_inf": ineq_violation_inf,
                "phr_region_count": int(final_region_indices_at_x.size),
                "phr_region_indices": final_region_indices_at_x.copy(),
                "region_stable": bool(region_stable),
                "region_iterations": len(region_records),
            })

            lambda_eq = lambda_eq_new.copy()
            lambda_ineq = lambda_ineq_new.copy()
            break

        # --------------------------------------------------------
        # Accept multiplier update
        # --------------------------------------------------------

        lambda_eq = lambda_eq_new.copy()
        lambda_ineq = lambda_ineq_new.copy()

        # --------------------------------------------------------
        # Mu update
        # --------------------------------------------------------

        mu_before = mu
        mu_update_status = "not updated"

        stalled_primal = (
            np.isfinite(previous_primal_residual)
            and primal_residual > constraint_tol
            and primal_residual >= (
                primal_stall_ratio * previous_primal_residual
            )
        )

        if update_mu and mu_update_rule == "primal_dual_balance":

            if stalled_primal:
                mu = min(mu_increase * mu, mu_max)
                mu_update_status = "increased: primal residual stalled"

            elif np.isfinite(dual_residual):

                if primal_residual > balance_factor * dual_residual:
                    mu = min(mu_increase * mu, mu_max)
                    mu_update_status = "increased: primal dominates"

                elif (
                    allow_mu_decrease
                    and dual_residual > balance_factor * primal_residual
                ):
                    mu = max(mu / mu_decrease, mu_min)
                    mu_update_status = "decreased: dual dominates"

                else:
                    mu_update_status = "unchanged: balanced"

            else:
                mu_update_status = "unchanged: no AL penalty rows"

        elif not update_mu:
            mu_update_status = "unchanged: update_mu=False"

        else:
            raise ValueError(
                f"Unknown mu_update_rule: {mu_update_rule}. "
                "Supported: 'primal_dual_balance'."
            )

        mu_update_history.append({
            "iteration": al_iter,
            "mu_before": mu_before,
            "mu_after": mu,
            "status": mu_update_status,
            "primal_residual": primal_residual,
            "dual_residual": dual_residual,
            "eq_residual_inf": eq_residual_norm_inf,
            "ineq_residual_inf": ineq_residual_norm_inf,
            "ineq_violation_inf": ineq_violation_inf,
            "phr_region_count": int(final_region_indices_at_x.size),
            "phr_region_indices": final_region_indices_at_x.copy(),
            "region_stable": bool(region_stable),
            "region_iterations": len(region_records),
        })

        previous_primal_residual = primal_residual
        status_history.append("continued")

        if verbose:
            print(
                f"AL {al_iter:3d} | "
                f"f={f_original:.10f} | "
                f"phr={phr_augmented_objective:.10f} | "
                f"eq={eq_residual_norm_inf:.3e} | "
                f"ineq={ineq_violation_inf:.3e} | "
                f"region={final_region_indices_at_x.size}/{p} | "
                f"region_iters={len(region_records)} | "
                f"stable={region_stable} | "
                f"primal={primal_residual:.3e} | "
                f"dual={dual_residual:.3e} | "
                f"stat={stationarity_norm_inf:.3e} | "
                f"comp={complementarity_norm_inf:.3e} | "
                f"mu={mu_before:.3e}->{mu:.3e} | "
                f"mu_status={mu_update_status}"
            )

    else:
        status_history.append(
            "stopped: max augmented-Lagrangian iterations reached"
        )

    # ============================================================
    # FINAL OUTPUT
    # ============================================================

    total_runtime = time.time() - start_total
    total_al_iterations = len(eq_residual_history)

    f_final_original = _objective(A, b, c, x)

    if m > 0:
        final_eq_residual = G @ x - r
    else:
        final_eq_residual = np.zeros(0, dtype=float)

    if p > 0:
        final_ineq_residual = H @ x - h
        final_ineq_violation = np.maximum(
            final_ineq_residual,
            0.0
        )

        final_shifted = (
            lambda_ineq + mu * final_ineq_residual
        )

        final_phr_region_mask = (
            final_shifted > phr_region_tol
        )

        final_phr_region_indices = np.where(
            final_phr_region_mask
        )[0]

    else:
        final_ineq_residual = np.zeros(0, dtype=float)
        final_ineq_violation = np.zeros(0, dtype=float)
        final_shifted = np.zeros(0, dtype=float)
        final_phr_region_mask = np.zeros(0, dtype=bool)
        final_phr_region_indices = np.zeros(0, dtype=int)

    final_eq_residual_norm_inf = (
        float(np.linalg.norm(final_eq_residual, ord=np.inf))
        if m > 0 else 0.0
    )

    final_ineq_residual_norm_inf = (
        float(np.linalg.norm(final_ineq_residual, ord=np.inf))
        if p > 0 else 0.0
    )

    final_ineq_violation_inf = (
        float(np.linalg.norm(final_ineq_violation, ord=np.inf))
        if p > 0 else 0.0
    )

    final_residual_norm_inf = max(
        final_eq_residual_norm_inf,
        final_ineq_violation_inf
    )

    final_lambda = np.concatenate([lambda_eq, lambda_ineq])

    final_stationarity = 2.0 * A @ x + b

    if m > 0:
        final_stationarity += G.T @ lambda_eq

    if p > 0:
        final_stationarity += H.T @ lambda_ineq

    final_stationarity_norm_inf = float(
        np.linalg.norm(final_stationarity, ord=np.inf)
    )

    if p > 0:
        final_complementarity_norm_inf = float(
            np.linalg.norm(
                lambda_ineq * final_ineq_residual,
                ord=np.inf
            )
        )
    else:
        final_complementarity_norm_inf = 0.0

    qubo_solver_used = dqup_kwargs.get("qubo_solver", "dvqe")
    classical_time_limit_used = dqup_kwargs.get(
        "classical_time_limit",
        None
    )
    classical_mip_gap_used = dqup_kwargs.get(
        "classical_mip_gap",
        0.0
    )
    classical_threads_used = dqup_kwargs.get(
        "classical_threads",
        None
    )

    info = {
        "x_history": x_history,

        "lambda_history": lambda_history,
        "lambda_eq_history": lambda_eq_history,
        "lambda_ineq_history": lambda_ineq_history,
        "mu_history": mu_history,

        "original_objective_history": original_objective_history,
        "phr_augmented_objective_history": phr_augmented_objective_history,

        # Alias retained for code that used the old name.
        "augmented_objective_history": phr_augmented_objective_history,

        "eq_residual_history": eq_residual_history,
        "ineq_residual_history": ineq_residual_history,
        "ineq_violation_history": ineq_violation_history,

        "eq_residual_norm_inf_history": eq_residual_norm_inf_history,
        "ineq_residual_norm_inf_history": ineq_residual_norm_inf_history,
        "ineq_violation_inf_history": ineq_violation_inf_history,

        "primal_residual_history": primal_residual_history,
        "dual_residual_history": dual_residual_history,
        "stationarity_norm_inf_history": stationarity_norm_inf_history,
        "complementarity_norm_inf_history": complementarity_norm_inf_history,

        "phr_region_mask_history": phr_region_mask_history,
        "phr_region_indices_history": phr_region_indices_history,
        "phr_region_count_history": phr_region_count_history,
        "phr_region_stable_history": phr_region_stable_history,
        "phr_region_iterations_history": phr_region_iterations_history,
        "phr_region_info_history": phr_region_info_history,

        # Compatibility aliases for the prior runner.
        "active_set_history": active_set_history,
        "active_indices_history": active_indices_history,
        "num_active_ineq_history": num_active_ineq_history,

        "dqup_info_history": dqup_info_history,
        "dqup_iterations_history": dqup_iterations_history,
        "dqup_runtime_history": dqup_runtime_history,

        "mu_update_history": mu_update_history,
        "status_history": status_history,

        "final_x": x.copy(),
        "final_original_objective": f_final_original,

        "final_eq_residual": final_eq_residual.copy(),
        "final_ineq_residual": final_ineq_residual.copy(),
        "final_ineq_violation": final_ineq_violation.copy(),

        "final_eq_residual_norm_inf": final_eq_residual_norm_inf,
        "final_ineq_residual_norm_inf": final_ineq_residual_norm_inf,
        "final_ineq_violation_inf": final_ineq_violation_inf,
        "final_residual_norm_inf": final_residual_norm_inf,

        "final_lambda": final_lambda.copy(),
        "final_lambda_eq": lambda_eq.copy(),
        "final_lambda_ineq": lambda_ineq.copy(),
        "final_mu": mu,

        "final_phr_shifted_residual": final_shifted.copy(),
        "final_phr_region_mask": final_phr_region_mask.copy(),
        "final_phr_region_indices": final_phr_region_indices.copy(),

        "final_stationarity_norm_inf": final_stationarity_norm_inf,
        "final_complementarity_norm_inf": (
            final_complementarity_norm_inf
        ),

        "best_feasible_x": best_feasible_x,
        "best_feasible_objective": best_feasible_f,

        "has_eq": has_eq,
        "has_ineq": has_ineq,
        "num_equalities": m,
        "num_inequalities": p,

        "combined_dimension": n,
        "used_slack_variables": False,

        "x_lb": lb.copy(),
        "x_ub": ub.copy(),

        "constraint_tol": constraint_tol,
        "stationarity_tol": stationarity_tol,

        "phr_region_tol": phr_region_tol,
        "max_region_iters": max_region_iters,

        "min_al_iters": min_al_iters,
        "max_al_iters": max_al_iters,

        "update_mu": update_mu,
        "mu_update_rule": mu_update_rule,
        "balance_factor": balance_factor,
        "mu_increase": mu_increase,
        "mu_decrease": mu_decrease,
        "allow_mu_decrease": allow_mu_decrease,
        "primal_stall_ratio": primal_stall_ratio,
        "mu_min": mu_min,
        "mu_max": mu_max,

        "total_al_iterations": total_al_iterations,
        "total_runtime_sec": total_runtime,

        "qubo_solver": qubo_solver_used,
        "classical_method": "gurobi_miqp",
        "classical_time_limit": classical_time_limit_used,
        "classical_mip_gap": classical_mip_gap_used,
        "classical_threads": classical_threads_used,
    }

    if return_info:
        return x, f_final_original, total_al_iterations, info

    return x, f_final_original, total_al_iterations


# ============================================================
# QQP: SCALED DQP WRAPPER FOR BADLY SCALED QPs
# ============================================================

def _normalize_constraint_rows(M, rhs):
    """
    Normalize rows of M y <= rhs or M y = rhs.
    """
    M = np.asarray(M, dtype=float)
    rhs = np.asarray(rhs, dtype=float)

    if M.shape[0] == 0:
        return M, rhs, np.ones(0, dtype=float)

    row_norm = np.maximum(np.linalg.norm(M, axis=1), 1e-12)

    M_scaled = M / row_norm[:, None]
    rhs_scaled = rhs / row_norm

    return M_scaled, rhs_scaled, row_norm


def _add_small_regularization(A, b, c, y_ref=None, eps_reg=1e-8):
    """
    Add eps_reg * ||y - y_ref||^2 to improve weak-cost variables.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    n = A.shape[0]

    if y_ref is None:
        y_ref = np.zeros(n, dtype=float)
    else:
        y_ref = np.asarray(y_ref, dtype=float)

    A_reg = A + eps_reg * np.eye(n)
    b_reg = b - 2.0 * eps_reg * y_ref
    c_reg = float(c + eps_reg * float(y_ref.T @ y_ref))

    return A_reg, b_reg, c_reg


def _recover_original_qp_duals(
    A,
    b,
    G,
    H,
    x_sol,
    lb,
    ub,
    lambda_eq_scaled,
    lambda_ineq_scaled,
    objective_scale,
    G_row_norm,
    H_row_norm,
    bound_active_tol=1e-7
):
    """
    Recover approximate dual variables for the original, unscaled QP.

    Original problem:
        min  x^T A x + b^T x + c
        s.t. Gx = r
             Hx <= h
             lb <= x <= ub

    The multipliers produced by dqp() correspond to the objective-scaled,
    row-normalized problem solved internally by qqp(). For each linear
    constraint row, the original multiplier is recovered as

        lambda_original
            = objective_scale * lambda_scaled / row_norm.

    Bound multipliers are estimated from original-space stationarity using
    the conventions

        lb - x <= 0,  lambda_lb >= 0,
        x - ub <= 0,  lambda_ub >= 0.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    G = np.asarray(G, dtype=float)
    H = np.asarray(H, dtype=float)
    x_sol = np.asarray(x_sol, dtype=float)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    lambda_eq_scaled = np.asarray(lambda_eq_scaled, dtype=float)
    lambda_ineq_scaled = np.asarray(lambda_ineq_scaled, dtype=float)
    G_row_norm = np.asarray(G_row_norm, dtype=float)
    H_row_norm = np.asarray(H_row_norm, dtype=float)

    if lambda_eq_scaled.size > 0:
        lambda_eq = (
            float(objective_scale)
            * lambda_eq_scaled
            / np.maximum(G_row_norm, 1e-12)
        )
    else:
        lambda_eq = np.zeros(0, dtype=float)

    if lambda_ineq_scaled.size > 0:
        lambda_ineq = (
            float(objective_scale)
            * lambda_ineq_scaled
            / np.maximum(H_row_norm, 1e-12)
        )
        lambda_ineq = np.maximum(lambda_ineq, 0.0)
    else:
        lambda_ineq = np.zeros(0, dtype=float)

    # Original-space stationarity before bound multipliers.
    stationarity_without_bounds = 2.0 * A @ x_sol + b

    if G.shape[0] > 0:
        stationarity_without_bounds += G.T @ lambda_eq

    if H.shape[0] > 0:
        stationarity_without_bounds += H.T @ lambda_ineq

    lower_active = x_sol <= lb + bound_active_tol
    upper_active = x_sol >= ub - bound_active_tol

    lambda_lb = np.zeros_like(x_sol)
    lambda_ub = np.zeros_like(x_sol)

    # At a lower bound:
    # grad L_without_bounds - lambda_lb = 0.
    lambda_lb[lower_active] = np.maximum(
        stationarity_without_bounds[lower_active],
        0.0
    )

    # At an upper bound:
    # grad L_without_bounds + lambda_ub = 0.
    lambda_ub[upper_active] = np.maximum(
        -stationarity_without_bounds[upper_active],
        0.0
    )

    stationarity_full = (
        stationarity_without_bounds
        - lambda_lb
        + lambda_ub
    )

    lower_bound_slack = x_sol - lb
    upper_bound_slack = ub - x_sol

    lower_bound_complementarity = lambda_lb * lower_bound_slack
    upper_bound_complementarity = lambda_ub * upper_bound_slack

    return {
        "lambda_eq": lambda_eq.copy(),
        "lambda_ineq": lambda_ineq.copy(),
        "lambda_lb": lambda_lb.copy(),
        "lambda_ub": lambda_ub.copy(),
        "lambda_linear": np.concatenate([lambda_eq, lambda_ineq]),
        "lambda_all": np.concatenate([
            lambda_eq,
            lambda_ineq,
            lambda_lb,
            lambda_ub,
        ]),
        "lambda_eq_scaled": lambda_eq_scaled.copy(),
        "lambda_ineq_scaled": lambda_ineq_scaled.copy(),
        "lower_bound_active": lower_active.copy(),
        "upper_bound_active": upper_active.copy(),
        "stationarity_without_bounds": stationarity_without_bounds.copy(),
        "stationarity_full": stationarity_full.copy(),
        "stationarity_without_bounds_inf": float(
            np.linalg.norm(stationarity_without_bounds, ord=np.inf)
        ),
        "stationarity_full_inf": float(
            np.linalg.norm(stationarity_full, ord=np.inf)
        ),
        "lower_bound_complementarity": (
            lower_bound_complementarity.copy()
        ),
        "upper_bound_complementarity": (
            upper_bound_complementarity.copy()
        ),
        "lower_bound_complementarity_inf": float(
            np.linalg.norm(lower_bound_complementarity, ord=np.inf)
        ),
        "upper_bound_complementarity_inf": float(
            np.linalg.norm(upper_bound_complementarity, ord=np.inf)
        ),
        "bound_active_tol": float(bound_active_tol),
    }


def qqp(
    A,
    b=None,
    c=0.0,

    G=None,
    r=None,

    H=None,
    h=None,

    x0=None,
    lb=None,
    ub=None,

    # Scaling controls
    normalize_constraints=True,
    scale_objective=True,
    regularize_weak_cost=True,
    eps_reg=1e-8,

    # Original-QP dual recovery control
    bound_active_tol=1e-7,

    return_info=True,

    **dqp_kwargs
):
    """
    QQP: scaled wrapper around dqp() with original-QP dual recovery.

    Solves approximately:

        min_x  x^T A x + b^T x + c
        s.t.   Gx = r
               Hx <= h
               lb <= x <= ub

    The internal normalized variables are

        x = x_center + x_scale * y,
        -1 <= y <= 1.

    dqp() returns augmented-Lagrangian multiplier estimates for the scaled,
    row-normalized problem. qqp() maps the equality and inequality
    multipliers back to the original QP scaling and estimates active-bound
    multipliers from original-space stationarity.

    Returns
    -------
    If return_info=True:
        x_sol, f_original, al_iters, info

    Else:
        x_sol, f_original, al_iters

    The original-QP dual estimates and augmented-Lagrangian penalty
    variables are returned automatically inside info. No additional QQP
    output or call argument is required.

    Notes
    -----
    The returned duals are augmented-Lagrangian estimates. Their quality
    should be assessed using primal feasibility, stationarity, and
    complementarity diagnostics. When regularize_weak_cost=True, the solved
    internal objective contains a small regularization term, so the recovered
    duals are approximate multipliers for the unregularized original QP.
    """
    # ------------------------------------------------------------
    # Input processing
    # ------------------------------------------------------------
    A = np.asarray(A, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")

    n = A.shape[0]
    A = _symmetrize(A)

    if b is None:
        b = np.zeros(n, dtype=float)
    else:
        b = np.asarray(b, dtype=float)

    if b.shape != (n,):
        raise ValueError("b must have shape (n,).")

    if lb is None or ub is None:
        raise ValueError("lb and ub must be provided.")

    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    if lb.shape != (n,) or ub.shape != (n,):
        raise ValueError("lb and ub must have shape (n,).")

    if np.any(ub <= lb):
        raise ValueError(
            "Every upper bound must be greater than lower bound."
        )

    if bound_active_tol < 0:
        raise ValueError("bound_active_tol must be nonnegative.")

    # ------------------------------------------------------------
    # Equality constraints
    # ------------------------------------------------------------
    if (G is None) != (r is None):
        raise ValueError(
            "G and r must either both be provided or both be None."
        )

    if G is None:
        G = np.zeros((0, n), dtype=float)
        r = np.zeros(0, dtype=float)
    else:
        G = np.asarray(G, dtype=float)
        r = np.asarray(r, dtype=float)

        if G.ndim != 2 or G.shape[1] != n:
            raise ValueError("G must have shape (m,n).")

        if r.shape != (G.shape[0],):
            raise ValueError("r must have shape (m,).")

    # ------------------------------------------------------------
    # Inequality constraints
    # ------------------------------------------------------------
    if (H is None) != (h is None):
        raise ValueError(
            "H and h must either both be provided or both be None."
        )

    if H is None:
        H = np.zeros((0, n), dtype=float)
        h = np.zeros(0, dtype=float)
    else:
        H = np.asarray(H, dtype=float)
        h = np.asarray(h, dtype=float)

        if H.ndim != 2 or H.shape[1] != n:
            raise ValueError("H must have shape (p,n).")

        if h.shape != (H.shape[0],):
            raise ValueError("h must have shape (p,).")

    if G.shape[0] == 0 and H.shape[0] == 0:
        raise ValueError(
            "At least one equality or inequality constraint must be provided."
        )

    # ------------------------------------------------------------
    # 1) Variable scaling: x = x_center + x_scale * y
    # ------------------------------------------------------------
    x_center = 0.5 * (lb + ub)
    x_scale = 0.5 * (ub - lb)
    x_scale = np.maximum(x_scale, 1e-12)

    y_lb = -np.ones(n, dtype=float)
    y_ub = np.ones(n, dtype=float)

    if x0 is None:
        y0 = np.zeros(n, dtype=float)
    else:
        x0 = np.asarray(x0, dtype=float)

        if x0.shape != (n,):
            raise ValueError("x0 must have shape (n,).")

        x0 = np.clip(x0, lb, ub)
        y0 = (x0 - x_center) / x_scale
        y0 = np.clip(y0, y_lb, y_ub)

    # ------------------------------------------------------------
    # 2) Transform objective into y-space
    # ------------------------------------------------------------
    A_y_unscaled = A * np.outer(x_scale, x_scale)
    b_y_unscaled = x_scale * (2.0 * (A @ x_center) + b)
    c_y_unscaled = _objective(A, b, c, x_center)

    A_y = A_y_unscaled.copy()
    b_y = b_y_unscaled.copy()
    c_y = float(c_y_unscaled)

    # ------------------------------------------------------------
    # 3) Transform constraints into y-space
    # ------------------------------------------------------------
    G_y_unscaled = G * x_scale[None, :]
    r_y_unscaled = r - G @ x_center

    H_y_unscaled = H * x_scale[None, :]
    h_y_unscaled = h - H @ x_center

    G_y = G_y_unscaled.copy()
    r_y = r_y_unscaled.copy()
    H_y = H_y_unscaled.copy()
    h_y = h_y_unscaled.copy()

    # ------------------------------------------------------------
    # 4) Normalize constraint rows
    # ------------------------------------------------------------
    if normalize_constraints:
        G_y, r_y, G_row_norm = _normalize_constraint_rows(G_y, r_y)
        H_y, h_y, H_row_norm = _normalize_constraint_rows(H_y, h_y)
    else:
        G_row_norm = np.ones(G_y.shape[0], dtype=float)
        H_row_norm = np.ones(H_y.shape[0], dtype=float)

    # ------------------------------------------------------------
    # 5) Scale objective coefficients
    # ------------------------------------------------------------
    objective_scale = 1.0

    if scale_objective:
        objective_scale = max(
            float(np.max(np.abs(A_y))) if A_y.size > 0 else 0.0,
            float(np.max(np.abs(b_y))) if b_y.size > 0 else 0.0,
            1.0
        )

        A_y = A_y / objective_scale
        b_y = b_y / objective_scale
        c_y = c_y / objective_scale

    A_y_before_regularization = A_y.copy()
    b_y_before_regularization = b_y.copy()
    c_y_before_regularization = float(c_y)

    # ------------------------------------------------------------
    # 6) Regularize weak-cost / constraint-only variables
    # ------------------------------------------------------------
    if regularize_weak_cost:
        A_y, b_y, c_y = _add_small_regularization(
            A=A_y,
            b=b_y,
            c=c_y,
            y_ref=y0,
            eps_reg=eps_reg
        )

    # ------------------------------------------------------------
    # 7) Solve scaled QP using dqp()
    # ------------------------------------------------------------
    y_sol, f_y, al_iters, info = dqp(
        A=A_y,
        b=b_y,
        c=c_y,
        G=G_y,
        r=r_y,
        H=H_y,
        h=h_y,
        x0=y0,
        lb=y_lb,
        ub=y_ub,
        return_info=True,
        **dqp_kwargs
    )

    # ------------------------------------------------------------
    # 8) Map the primal solution back to original x-space
    # ------------------------------------------------------------
    x_sol = x_center + x_scale * y_sol
    x_sol = np.clip(x_sol, lb, ub)
    f_original = _objective(A, b, c, x_sol)

    if info is None:
        info = {}

    # ------------------------------------------------------------
    # 9) Recover original-QP linear and bound multipliers
    # ------------------------------------------------------------
    lambda_eq_scaled = np.asarray(
        info.get(
            "final_lambda_eq",
            np.zeros(G.shape[0], dtype=float)
        ),
        dtype=float
    )

    lambda_ineq_scaled = np.asarray(
        info.get(
            "final_lambda_ineq",
            np.zeros(H.shape[0], dtype=float)
        ),
        dtype=float
    )

    duals = _recover_original_qp_duals(
        A=A,
        b=b,
        G=G,
        H=H,
        x_sol=x_sol,
        lb=lb,
        ub=ub,
        lambda_eq_scaled=lambda_eq_scaled,
        lambda_ineq_scaled=lambda_ineq_scaled,
        objective_scale=objective_scale,
        G_row_norm=G_row_norm,
        H_row_norm=H_row_norm,
        bound_active_tol=bound_active_tol
    )

    # ------------------------------------------------------------
    # 10) Original-QP residual and complementarity diagnostics
    # ------------------------------------------------------------
    if G.shape[0] > 0:
        original_eq_residual = G @ x_sol - r
    else:
        original_eq_residual = np.zeros(0, dtype=float)

    if H.shape[0] > 0:
        original_ineq_residual = H @ x_sol - h
        original_ineq_violation = np.maximum(
            original_ineq_residual,
            0.0
        )
        original_ineq_complementarity = (
            duals["lambda_ineq"] * original_ineq_residual
        )
    else:
        original_ineq_residual = np.zeros(0, dtype=float)
        original_ineq_violation = np.zeros(0, dtype=float)
        original_ineq_complementarity = np.zeros(0, dtype=float)

    original_eq_residual_inf = (
        float(np.linalg.norm(original_eq_residual, ord=np.inf))
        if original_eq_residual.size > 0 else 0.0
    )

    original_ineq_violation_inf = (
        float(np.linalg.norm(original_ineq_violation, ord=np.inf))
        if original_ineq_violation.size > 0 else 0.0
    )

    original_ineq_complementarity_inf = (
        float(
            np.linalg.norm(
                original_ineq_complementarity,
                ord=np.inf
            )
        )
        if original_ineq_complementarity.size > 0 else 0.0
    )

    duals.update({
        "original_eq_residual": original_eq_residual.copy(),
        "original_ineq_residual": original_ineq_residual.copy(),
        "original_ineq_violation": original_ineq_violation.copy(),
        "original_ineq_complementarity": (
            original_ineq_complementarity.copy()
        ),
        "original_eq_residual_inf": original_eq_residual_inf,
        "original_ineq_violation_inf": original_ineq_violation_inf,
        "original_ineq_complementarity_inf": (
            original_ineq_complementarity_inf
        ),
    })

    # ------------------------------------------------------------
    # 11) Preserve and extend the full info dictionary
    # ------------------------------------------------------------
    info["qqp_used"] = True
    info["scaled_solution_y"] = y_sol.copy()
    info["original_solution_x"] = x_sol.copy()
    info["scaled_objective"] = float(f_y)
    info["original_objective"] = float(f_original)

    # Explicit original-QP dual outputs retained inside info as well.
    info["duals_original_qp"] = duals
    info["final_lambda_eq_original"] = duals["lambda_eq"].copy()
    info["final_lambda_ineq_original"] = duals["lambda_ineq"].copy()
    info["final_lambda_lb_original"] = duals["lambda_lb"].copy()
    info["final_lambda_ub_original"] = duals["lambda_ub"].copy()
    info["final_lambda_linear_original"] = (
        duals["lambda_linear"].copy()
    )
    info["final_lambda_all_original"] = duals["lambda_all"].copy()

    info["original_stationarity_without_bounds"] = (
        duals["stationarity_without_bounds"].copy()
    )
    info["original_stationarity_without_bounds_inf"] = (
        duals["stationarity_without_bounds_inf"]
    )
    info["original_stationarity_full"] = (
        duals["stationarity_full"].copy()
    )
    info["original_stationarity_full_inf"] = (
        duals["stationarity_full_inf"]
    )

    info["original_eq_residual"] = original_eq_residual.copy()
    info["original_ineq_residual"] = original_ineq_residual.copy()
    info["original_ineq_violation"] = original_ineq_violation.copy()
    info["original_eq_residual_inf"] = original_eq_residual_inf
    info["original_ineq_violation_inf"] = (
        original_ineq_violation_inf
    )
    info["original_ineq_complementarity_inf"] = (
        original_ineq_complementarity_inf
    )

    info["scaling_info"] = {
        "x_center": x_center.copy(),
        "x_scale": x_scale.copy(),
        "objective_scale": float(objective_scale),
        "G_row_norm": G_row_norm.copy(),
        "H_row_norm": H_row_norm.copy(),
        "normalize_constraints": normalize_constraints,
        "scale_objective": scale_objective,
        "regularize_weak_cost": regularize_weak_cost,
        "eps_reg": float(eps_reg),
        "dual_recovery_rule": (
            "lambda_original = objective_scale * "
            "lambda_scaled / constraint_row_norm"
        ),
    }

    # Final internal scaled problem actually passed to dqp().
    info["scaled_problem"] = {
        "A_y": A_y.copy(),
        "b_y": b_y.copy(),
        "c_y": float(c_y),
        "G_y": G_y.copy(),
        "r_y": r_y.copy(),
        "H_y": H_y.copy(),
        "h_y": h_y.copy(),
        "y_lb": y_lb.copy(),
        "y_ub": y_ub.copy(),
    }

    # Additional records do not remove any prior QQP/DQP histories.
    info["transformed_problem_before_scaling"] = {
        "A_y_unscaled": A_y_unscaled.copy(),
        "b_y_unscaled": b_y_unscaled.copy(),
        "c_y_unscaled": float(c_y_unscaled),
        "G_y_unscaled": G_y_unscaled.copy(),
        "r_y_unscaled": r_y_unscaled.copy(),
        "H_y_unscaled": H_y_unscaled.copy(),
        "h_y_unscaled": h_y_unscaled.copy(),
    }

    info["scaled_objective_before_regularization"] = {
        "A_y": A_y_before_regularization.copy(),
        "b_y": b_y_before_regularization.copy(),
        "c_y": float(c_y_before_regularization),
    }

    info["dual_recovery_warning"] = (
        "The returned equality and inequality multipliers are "
        "augmented-Lagrangian estimates. When regularize_weak_cost=True, "
        "the internally solved objective includes a small regularization "
        "term, so these are approximate duals for the unregularized QP."
    )

    # ------------------------------------------------------------
    # 12) Store duals and AL penalty variables automatically
    # ------------------------------------------------------------
    # Original-QP dual estimates are always placed in info so that the
    # historical QQP call and four-output unpacking remain unchanged.
    info["duals"] = duals
    info["dual_variables"] = duals

    # Convenience aliases for direct access.
    info["lambda_eq"] = duals["lambda_eq"].copy()
    info["lambda_ineq"] = duals["lambda_ineq"].copy()
    info["lambda_lb"] = duals["lambda_lb"].copy()
    info["lambda_ub"] = duals["lambda_ub"].copy()
    info["lambda_linear"] = duals["lambda_linear"].copy()
    info["lambda_all"] = duals["lambda_all"].copy()

    # The penalty variables used by the PHR augmented-Lagrangian method.
    # These include the final penalty parameter, its history and update
    # records, as well as the scaled internal multiplier estimates.
    penalty_variables = {
        "mu": float(info.get("final_mu", np.nan)),
        "mu_final": float(info.get("final_mu", np.nan)),
        "mu_history": list(info.get("mu_history", [])),
        "mu_update_history": list(info.get("mu_update_history", [])),
        "lambda_eq_scaled": np.asarray(
            info.get("final_lambda_eq", np.zeros(G.shape[0])),
            dtype=float
        ).copy(),
        "lambda_ineq_scaled": np.asarray(
            info.get("final_lambda_ineq", np.zeros(H.shape[0])),
            dtype=float
        ).copy(),
        "lambda_scaled": np.asarray(
            info.get("final_lambda", np.zeros(G.shape[0] + H.shape[0])),
            dtype=float
        ).copy(),
        "phr_region_mask": np.asarray(
            info.get("final_phr_region_mask", np.zeros(H.shape[0], dtype=bool)),
            dtype=bool
        ).copy(),
        "phr_region_indices": np.asarray(
            info.get("final_phr_region_indices", np.zeros(0, dtype=int)),
            dtype=int
        ).copy(),
    }

    info["penalty_variables"] = penalty_variables
    info["penalty_vars"] = penalty_variables
    info["mu"] = penalty_variables["mu"]

    # ------------------------------------------------------------
    # 13) Return using the original QQP interface
    # ------------------------------------------------------------
    if return_info:
        return x_sol, f_original, al_iters, info

    return x_sol, f_original, al_iters


def qqp_runtime_summary(info, print_summary=True):
    """
    Summarize QQP runtime and internal QUBO/MIQP-call diagnostics.

    Use after running:

        x_qqp, f_qqp, al_iters, info = qqp(...)

    Then call:

        summary = qqp_runtime_summary(info)

    This function does not change the optimization result.
    It only reads the info dictionary returned by qqp().
    """

    if info is None:
        raise ValueError("info is None. Run qqp(..., return_info=True).")

    # ------------------------------------------------------------
    # Basic QQP / DQP information
    # ------------------------------------------------------------

    total_qqp_time = float(info.get("total_runtime_sec", 0.0))
    total_al_iterations = int(info.get("total_al_iterations", 0))

    status_history = info.get("status_history", [])
    final_status = status_history[-1] if len(status_history) > 0 else None

    phr_region_iterations = info.get("phr_region_iterations_history", [])
    total_region_iterations = int(np.sum(phr_region_iterations)) if len(phr_region_iterations) > 0 else 0

    # ------------------------------------------------------------
    # DQUP / QUBO information
    # ------------------------------------------------------------

    dqup_info_history = info.get("dqup_info_history", [])

    total_qubo_calls = 0
    total_qubo_time = 0.0

    total_dqup_outer_iterations = 0
    dqup_runtime_total = 0.0

    dqup_call_count = 0
    dqup_qubo_calls_by_call = []
    dqup_qubo_time_by_call = []
    dqup_outer_iters_by_call = []

    for dqup_info in dqup_info_history:

        if dqup_info is None:
            continue

        dqup_call_count += 1

        this_qubo_calls = int(dqup_info.get("total_dvqe_calls", 0))
        this_qubo_times = dqup_info.get("dvqe_runtime_history", [])
        this_qubo_time = float(np.sum(this_qubo_times)) if len(this_qubo_times) > 0 else 0.0

        this_dqup_outer_iters = int(dqup_info.get("total_outer_iterations", 0))
        this_dqup_runtime = float(dqup_info.get("total_runtime_sec", 0.0))

        total_qubo_calls += this_qubo_calls
        total_qubo_time += this_qubo_time

        total_dqup_outer_iterations += this_dqup_outer_iters
        dqup_runtime_total += this_dqup_runtime

        dqup_qubo_calls_by_call.append(this_qubo_calls)
        dqup_qubo_time_by_call.append(this_qubo_time)
        dqup_outer_iters_by_call.append(this_dqup_outer_iters)

    fraction_qubo_time = total_qubo_time / max(total_qqp_time, 1e-12)

    avg_qubo_time = total_qubo_time / max(total_qubo_calls, 1)
    avg_qubo_calls_per_dqup = total_qubo_calls / max(dqup_call_count, 1)
    avg_dqup_time = dqup_runtime_total / max(dqup_call_count, 1)

    # ------------------------------------------------------------
    # Feasibility / final diagnostics
    # ------------------------------------------------------------

    final_eq_residual = float(info.get("final_eq_residual_norm_inf", np.nan))
    final_ineq_violation = float(info.get("final_ineq_violation_inf", np.nan))
    final_total_residual = float(info.get("final_residual_norm_inf", np.nan))
    final_stationarity = float(info.get("final_stationarity_norm_inf", np.nan))
    final_objective = float(info.get("final_original_objective", np.nan))

    qubo_solver = info.get("qubo_solver", None)
    classical_method = info.get("classical_method", None)

    # ------------------------------------------------------------
    # Build summary dictionary
    # ------------------------------------------------------------

    summary = {
        "total_qqp_time_sec": total_qqp_time,
        "total_al_iterations": total_al_iterations,
        "final_status": final_status,

        "total_region_iterations": total_region_iterations,
        "phr_region_iterations_history": phr_region_iterations,

        "dqup_call_count": dqup_call_count,
        "total_dqup_runtime_sec": dqup_runtime_total,
        "avg_dqup_runtime_sec": avg_dqup_time,
        "total_dqup_outer_iterations": total_dqup_outer_iterations,

        "total_qubo_calls": total_qubo_calls,
        "total_qubo_time_sec": total_qubo_time,
        "avg_qubo_time_sec": avg_qubo_time,
        "fraction_time_in_qubo": fraction_qubo_time,
        "avg_qubo_calls_per_dqup": avg_qubo_calls_per_dqup,

        "dqup_qubo_calls_by_call": dqup_qubo_calls_by_call,
        "dqup_qubo_time_by_call": dqup_qubo_time_by_call,
        "dqup_outer_iters_by_call": dqup_outer_iters_by_call,

        "final_objective": final_objective,
        "final_eq_residual_norm_inf": final_eq_residual,
        "final_ineq_violation_inf": final_ineq_violation,
        "final_residual_norm_inf": final_total_residual,
        "final_stationarity_norm_inf": final_stationarity,

        "qubo_solver": qubo_solver,
        "classical_method": classical_method,
    }

    # ------------------------------------------------------------
    # Optional printing
    # ------------------------------------------------------------

    if print_summary:

        print("=" * 80)
        print("QQP Runtime Summary")
        print("=" * 80)

        print("Total QQP time [sec]:", summary["total_qqp_time_sec"])
        print("Total AL iterations:", summary["total_al_iterations"])
        print("Final status:", summary["final_status"])

        print("\nPHR / region information")
        print("-" * 80)
        print("Total region iterations:", summary["total_region_iterations"])
        print("Region iterations history:", summary["phr_region_iterations_history"])

        print("\nDQUP information")
        print("-" * 80)
        print("Number of DQUP calls:", summary["dqup_call_count"])
        print("Total DQUP runtime [sec]:", summary["total_dqup_runtime_sec"])
        print("Average DQUP runtime [sec]:", summary["avg_dqup_runtime_sec"])
        print("Total DQUP outer iterations:", summary["total_dqup_outer_iterations"])

        print("\nQUBO / MIQP information")
        print("-" * 80)
        print("QUBO solver:", summary["qubo_solver"])
        print("Classical method:", summary["classical_method"])
        print("Total local QUBO/MIQP calls:", summary["total_qubo_calls"])
        print("Total local QUBO/MIQP time [sec]:", summary["total_qubo_time_sec"])
        print("Average QUBO/MIQP time [sec]:", summary["avg_qubo_time_sec"])
        print("Fraction of QQP time spent in QUBO/MIQP:", summary["fraction_time_in_qubo"])
        print("Average QUBO calls per DQUP call:", summary["avg_qubo_calls_per_dqup"])

        print("\nFinal diagnostics")
        print("-" * 80)
        print("Final objective:", summary["final_objective"])
        print("Final equality residual inf:", summary["final_eq_residual_norm_inf"])
        print("Final inequality violation inf:", summary["final_ineq_violation_inf"])
        print("Final total residual inf:", summary["final_residual_norm_inf"])
        print("Final stationarity inf:", summary["final_stationarity_norm_inf"])

    return summary