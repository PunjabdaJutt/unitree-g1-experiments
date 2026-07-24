"""Passivity-preserving gain optimizer for PHFingerController (Phase 6).

Tuning strategy
---------------
For the PH closed-loop system with shaped Hamiltonian

    H_d(q,p) = 1/2 p^T M(q)^{-1} p + U_nc(q) + 1/2 (q_c-q*)^T Kp (q_c-q*)

passivity (and hence L2-stability) holds for *any* Kp, Kd >= 0 (Ortega et al.,
"PID passivity-based control of port-Hamiltonian systems", 2021).  Stability is
therefore not a constraint — it is guaranteed by the box kmin>=0.  The optimizer
is purely performance tuning over that box.

Usage
-----
::

    from ph.gain_optimizer import optimize_gains

    def my_run_episode(Kp: np.ndarray, Kd: np.ndarray):
        controller.Kp = Kp
        controller.Kd = Kd
        env.reset()
        J1 = J2 = J3 = 0.0
        for _ in range(T):
            q   = robot.get_joint_positions().cpu().numpy()
            qd  = robot.get_joint_velocities().cpu().numpy()
            tau = controller.compute(q, qd)
            q_c = q[controller.hand12_sim_indices]
            J1 += float(np.sum((q_c - controller.q_des_hand12)**2)) * dt
            J2 += float(np.sum(tau**2)) * dt
            # J3: optional energy overshoot — set 0.0 if not tracked
            robot.set_joint_effort_target(
                controller.scatter_to_full(tau, num_joints))
            sim.step()
        return J1, J2, J3

    Kp_opt, Kd_opt = optimize_gains(
        run_episode=my_run_episode,
        Kp_init=controller.Kp,
        Kd_init=controller.Kd,
        nu=len(controller.ph.control_joint_names),
    )
    controller.Kp = Kp_opt
    controller.Kd = Kd_opt

PHFingerController and PHSystem are never imported or modified here; the
optimizer is a thin layer that only sets Kp/Kd via run_episode.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# run_episode(Kp, Kd) -> (J1, J2, J3)
EpisodeFn = Callable[[np.ndarray, np.ndarray], Tuple[float, float, float]]
# log_callback(iteration, best_cost, Kp, Kd)
LogFn = Callable[[int, float, np.ndarray, np.ndarray], None]


# ═══════════════════════════════════════════════════════════════════ public API


def evaluate_cost(
    theta: np.ndarray,
    run_episode: EpisodeFn,
    alpha1: float = 1.0,
    alpha2: float = 1e-3,
    alpha3: float = 0.0,
) -> float:
    """Return scalar cost J(theta) = alpha1*J1 + alpha2*J2 + alpha3*J3.

    Parameters
    ----------
    theta:
        Array of shape (2*nu,) — [Kp[0:nu], Kd[0:nu]].  Negative entries are
        clipped to 0 before calling run_episode (passivity enforcement).
    run_episode:
        Callable ``(Kp, Kd) -> (J1, J2, J3)`` where
          J1 = integral tracking error  ∫ ||q_c(t) - q*||^2 dt
          J2 = integral effort          ∫ ||tau_hand(t)||^2 dt
          J3 = energy overshoot         max_t H_d(t) - H_d(q*,0)   (0 if unused)
    alpha1, alpha2, alpha3:
        Positive cost weights.  Set alpha3=0 to ignore energy overshoot.
    """
    nu = len(theta) // 2
    Kp = np.clip(theta[:nu], 0.0, None)
    Kd = np.clip(theta[nu:], 0.0, None)
    J1, J2, J3 = run_episode(Kp, Kd)
    return float(alpha1 * J1 + alpha2 * J2 + alpha3 * J3)


def optimize_gains(
    run_episode: EpisodeFn,
    Kp_init: np.ndarray,
    Kd_init: np.ndarray,
    nu: int,
    *,
    kmin: float = 0.0,
    kmax: float = 500.0,
    alpha1: float = 1.0,
    alpha2: float = 1e-3,
    alpha3: float = 0.0,
    max_iters: int = 80,
    num_workers: int = 1,
    log_callback: Optional[LogFn] = None,
    method: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """Derivative-free gain optimizer with passivity box constraints.

    Finds ``(Kp_opt, Kd_opt)`` minimising ``J(theta)`` subject to
    ``kmin <= Kp_i, Kd_i <= kmax``.  Passivity holds for any solution
    because ``kmin >= 0``.

    Parameters
    ----------
    run_episode:
        Episode callback — see ``evaluate_cost``.  Must be thread-safe when
        ``num_workers > 1`` (only used by the DE backend).
    Kp_init, Kd_init:
        Initial diagonal gains, each of shape ``(nu,)``.
    nu:
        Number of controlled finger DOFs (12 for HAND12).
    kmin, kmax:
        Elementwise box bounds.  ``kmin >= 0`` preserves passivity.
    alpha1, alpha2, alpha3:
        Cost weights forwarded to ``evaluate_cost``.
    max_iters:
        Maximum optimizer generations.
    num_workers:
        Parallel episode evaluations (DE backend only).
    log_callback:
        Optional ``log_callback(iter, best_cost, Kp, Kd)`` called after every
        iteration.  Use for live logging or early stopping.
    method:
        ``"cma"``  — CMA-ES (requires ``pip install cma``; recommended).
        ``"de"``   — scipy differential_evolution (always available).
        ``"auto"`` — tries CMA-ES, falls back to DE if ``cma`` not installed.

    Returns
    -------
    Kp_opt, Kd_opt : np.ndarray
        Optimal diagonal gains, shape ``(nu,)`` each, in
        ``ph.control_joint_names`` order.

    Raises
    ------
    RuntimeError
        If ``method="cma"`` and the ``cma`` package is not installed.
    ValueError
        If ``Kp_init`` / ``Kd_init`` size mismatches ``nu``.
    """
    Kp_init = np.asarray(Kp_init, float).reshape(-1)
    Kd_init = np.asarray(Kd_init, float).reshape(-1)
    if Kp_init.size != nu or Kd_init.size != nu:
        raise ValueError(f"Kp_init and Kd_init must each have {nu} elements")
    if kmin < 0:
        raise ValueError("kmin must be >= 0 to preserve passivity")

    theta0 = np.concatenate([Kp_init, Kd_init])
    bounds = np.column_stack([
        np.full(2 * nu, kmin),
        np.full(2 * nu, kmax),
    ])

    cost_fn = lambda th: evaluate_cost(th, run_episode, alpha1, alpha2, alpha3)

    initial_cost = cost_fn(theta0)
    logger.info("optimize_gains: initial cost=%.4f", initial_cost)

    use_cma = _resolve_method(method)

    if use_cma:
        best_theta, best_cost = _run_cma(
            cost_fn, theta0, bounds, max_iters, nu, log_callback
        )
    else:
        best_theta, best_cost = _run_de(
            cost_fn, bounds, max_iters, num_workers, nu, log_callback
        )

    Kp_opt = np.clip(best_theta[:nu], kmin, kmax)
    Kd_opt = np.clip(best_theta[nu:], kmin, kmax)
    logger.info(
        "optimize_gains: final cost=%.4f\n  Kp=%s\n  Kd=%s",
        best_cost, np.round(Kp_opt, 3), np.round(Kd_opt, 3),
    )
    return Kp_opt, Kd_opt


# ══════════════════════════════════════════════════════════════════ backends


def _resolve_method(method: str) -> bool:
    """Return True if CMA-ES should be used, False for DE."""
    if method == "de":
        return False
    try:
        import cma  # noqa: F401
        return True
    except ImportError:
        if method == "cma":
            raise RuntimeError(
                "CMA-ES requested but 'cma' package not found.\n"
                "Install with:  pip install cma"
            )
        logger.info("'cma' not installed; using scipy differential_evolution")
        return False


def _run_cma(
    cost_fn: Callable[[np.ndarray], float],
    theta0: np.ndarray,
    bounds: np.ndarray,
    max_iters: int,
    nu: int,
    log_callback: Optional[LogFn],
) -> Tuple[np.ndarray, float]:
    """CMA-ES optimizer backend (requires ``cma`` package)."""
    import cma

    lo, hi = bounds[:, 0], bounds[:, 1]
    spread = hi - lo
    sigma0 = float(np.mean(spread[spread > 0])) * 0.2
    if sigma0 == 0.0:
        sigma0 = 1.0

    opts = cma.CMAOptions()
    opts["bounds"] = [lo.tolist(), hi.tolist()]
    opts["maxiter"] = max_iters
    opts["verbose"] = -9
    opts["tolx"] = 1e-5
    opts["tolfun"] = 1e-5

    es = cma.CMAEvolutionStrategy(theta0.tolist(), sigma0, opts)

    best_cost = np.inf
    best_theta = theta0.copy()

    for iteration in range(max_iters):
        if es.stop():
            break
        solutions = es.ask()
        fitnesses = [cost_fn(np.asarray(s)) for s in solutions]
        es.tell(solutions, fitnesses)

        gen_best_idx = int(np.argmin(fitnesses))
        gen_best_cost = fitnesses[gen_best_idx]
        if gen_best_cost < best_cost:
            best_cost = gen_best_cost
            best_theta = np.asarray(solutions[gen_best_idx]).copy()

        logger.debug("CMA-ES iter %d | gen_best=%.4f | best=%.4f", iteration, gen_best_cost, best_cost)
        if log_callback is not None:
            log_callback(
                iteration, best_cost,
                np.clip(best_theta[:nu], 0.0, None),
                np.clip(best_theta[nu:], 0.0, None),
            )

    return best_theta, best_cost


def _run_de(
    cost_fn: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    max_iters: int,
    num_workers: int,
    nu: int,
    log_callback: Optional[LogFn],
) -> Tuple[np.ndarray, float]:
    """Differential Evolution backend (scipy, no extra dependencies)."""
    from scipy.optimize import differential_evolution

    best_state: dict = {"cost": np.inf, "theta": None, "iter": 0}

    def _callback(xk: np.ndarray, convergence: float) -> None:
        it = best_state["iter"]
        cost = cost_fn(xk)
        if cost < best_state["cost"]:
            best_state["cost"] = cost
            best_state["theta"] = xk.copy()
        logger.debug("DE iter %d | best=%.4f | convergence=%.4f", it, best_state["cost"], convergence)
        if log_callback is not None:
            log_callback(
                it, best_state["cost"],
                np.clip(xk[:nu], 0.0, None),
                np.clip(xk[nu:], 0.0, None),
            )
        best_state["iter"] += 1

    result = differential_evolution(
        cost_fn,
        bounds=[(lo, hi) for lo, hi in bounds],
        maxiter=max_iters,
        seed=42,
        tol=1e-5,
        mutation=(0.5, 1.0),
        recombination=0.9,
        popsize=12,
        workers=num_workers if num_workers > 1 else 1,
        updating="deferred" if num_workers > 1 else "immediate",
        callback=_callback,
        polish=False,
    )

    return result.x, float(result.fun)
