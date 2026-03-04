"""
solver.py

Main solver for problem P1. Uses a two-stage decomposition:
  Stage 1 - greedy assignment of users to (DT, slice) pairs
  Stage 2 - L-BFGS-B optimization for bandwidth/compute allocation
  Stage 3 - kick out anyone still violating deadlines

This avoids needing Gurobi or any commercial solver which is nice.
"""

from __future__ import annotations

import time
from typing import List, Tuple, Optional

import numpy as np
from scipy.optimize import minimize

from .system_model import (
    NetworkConfig, Solution, DELTA,
    total_latency, total_energy, compute_objective,
    populate_metrics, validate_solution,
)


def solve_p1(cfg: NetworkConfig, verbose: bool = False) -> Solution:
    """
    Solve problem P1 using our two-stage approach.
    Returns a Solution with all metrics filled in.
    """
    t0 = time.perf_counter()

    # stage 1: figure out binary assignments
    y, x = _stage1_greedy_assignment(cfg, verbose=verbose)

    # stage 2: optimize continuous variables (bandwidth, compute)
    b, f = _stage2_continuous_allocation(cfg, y, x, verbose=verbose)

    # stage 3: kick out users who still violate deadlines
    y, x, b, f = _enforce_feasibility(cfg, y, x, b, f, verbose=verbose)

    sol = Solution(y=y, x=x, b=b, f=f, solver_name="proposed_two_stage")
    populate_metrics(cfg, sol)
    sol.solve_time_s = time.perf_counter() - t0

    # check for any remaining violations
    violations = validate_solution(cfg, sol)
    if violations and verbose:
        print(f"  [WARN] {len(violations)} constraint violation(s):")
        for v in violations[:5]:
            print(f"    {v}")

    return sol


# --- Stage 1: Greedy Binary Assignment ---

def _check_feasibility(
    cfg: NetworkConfig,
    u: int, i: int, s: int,
) -> Tuple[bool, float, float, float]:
    """
    Check if user u can be assigned to (DT i, slice s) within deadline.
    Returns (feasible, bw_needed, compute_needed, latency_estimate).
    """
    # reliability check first
    if cfg.p_us[u, s] < (1 - cfg.eps_u[u]):
        return False, 0, 0, np.inf

    d = cfg.d_u[u]
    eta = cfg.eta_us[u, s]
    alpha = cfg.alpha_u[u]
    tau = cfg.tau_u[u]
    h_tr = cfg.h_is[i, s]
    delta = cfg.delta

    # how much time is left after transport?
    avail_time = tau - h_tr
    if avail_time <= 0:
        return False, 0, 0, np.inf

    # split available time 50/50 between tx and compute
    # this is a simple heuristic - could be improved
    t_tx_budget = avail_time * 0.5
    t_cp_budget = avail_time * 0.5

    # minimum bandwidth needed
    bw_need = d / (eta * t_tx_budget + delta)
    bw_need = max(bw_need, 100.0)

    # minimum compute needed
    f_need = alpha * d / (t_cp_budget + delta)
    f_need = max(f_need, 100.0)

    # check capacity
    bw_avail = cfg.beta_s[s] * cfg.B_s[s]
    f_avail = cfg.C_i[i]

    if bw_need > bw_avail:
        return False, bw_need, f_need, np.inf
    if f_need > f_avail:
        return False, bw_need, f_need, np.inf

    # estimate actual latency
    T_tx = d / (eta * bw_need + delta)
    T_cp = alpha * d / (f_need + delta)
    T_total = T_tx + T_cp + h_tr

    return True, float(bw_need), float(f_need), float(T_total)


def _score_assignment(
    cfg: NetworkConfig,
    u: int, i: int, s: int,
) -> Tuple[bool, float]:
    """
    Score a (user, DT, slice) assignment.
    Higher score = better assignment.
    """
    feasible, bw_need, f_need, T_est = _check_feasibility(cfg, u, i, s)
    if not feasible:
        return False, -np.inf

    d = cfg.d_u[u]
    eta = cfg.eta_us[u, s]
    alpha = cfg.alpha_u[u]
    delta = cfg.delta

    # estimate energy for this assignment
    T_tx = d / (eta * bw_need + delta)
    E_tx = cfg.P_tx_us[u, s] * T_tx

    T_cp = alpha * d / (f_need + delta)
    E_cp = cfg.kappa_i[i] * (f_need ** 2) * T_cp

    # combined score - we want low latency and low energy
    score = (1.0
             - cfg.lambda_T * T_est
             - cfg.lambda_E * (E_tx + E_cp)
             + 0.01 * cfg.p_us[u, s])
    return True, score


def _stage1_greedy_assignment(
    cfg: NetworkConfig,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Greedily assign users to their best (DT, slice) pair."""
    U, S_, I_ = cfg.U, cfg.S, cfg.I
    y = np.zeros(U)
    x = np.zeros((U, I_, S_))

    # find best assignment for each user
    user_info: List[Tuple[int, int, int, float, float, float]] = []
    for u in range(U):
        best_score = -np.inf
        best_i, best_s = -1, -1
        best_bw, best_f = 0.0, 0.0
        for i in range(I_):
            for s in range(S_):
                feasible, bw_need, f_need, _ = _check_feasibility(cfg, u, i, s)
                if not feasible:
                    continue
                _, score = _score_assignment(cfg, u, i, s)
                if score > best_score:
                    best_score = score
                    best_i, best_s = i, s
                    best_bw, best_f = bw_need, f_need
        if best_i >= 0:
            user_info.append((u, best_i, best_s, best_score, best_bw, best_f))

    # sort by tightest deadline first, then by score
    user_info.sort(key=lambda t: (cfg.tau_u[t[0]], -t[3]))

    # track remaining capacity
    bw_remaining = (cfg.beta_s * cfg.B_s).copy()
    compute_remaining = cfg.C_i.copy()
    traffic_remaining = cfg.H_s * cfg.T_win

    for u, best_i, best_s, score, bw_need, f_need in user_info:
        # check if we still have enough resources
        if bw_need > bw_remaining[best_s] + 1e-6:
            continue
        if f_need > compute_remaining[best_i] + 1e-6:
            continue
        if cfg.d_u[u] > traffic_remaining[best_s] + 1e-6:
            continue

        # admit this user
        y[u] = 1.0
        x[u, best_i, best_s] = 1.0
        bw_remaining[best_s] -= bw_need
        compute_remaining[best_i] -= f_need
        traffic_remaining[best_s] -= cfg.d_u[u]

    if verbose:
        print(f"  Stage 1: admitted {int(y.sum())}/{U} users")

    return y, x


# --- Stage 2: Continuous Resource Allocation ---

def _stage2_continuous_allocation(
    cfg: NetworkConfig,
    y: np.ndarray,
    x: np.ndarray,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Optimize bandwidth and compute for each admitted user."""
    U, S_, I_ = cfg.U, cfg.S, cfg.I
    b = np.zeros((U, S_))
    f = np.zeros((U, I_))

    admitted = np.where(y > 0.5)[0]
    if len(admitted) == 0:
        return b, f

    # group users by their (DT, slice) assignment
    groups = {}
    for u in admitted:
        i_star, s_star = np.unravel_index(x[u].argmax(), (I_, S_))
        key = (int(i_star), int(s_star))
        groups.setdefault(key, []).append(u)

    # count users per slice/DT for fair sharing
    users_per_slice = np.zeros(S_)
    users_per_dt = np.zeros(I_)
    for (i, s), users in groups.items():
        users_per_slice[s] += len(users)
        users_per_dt[i] += len(users)

    bw_budget = cfg.beta_s * cfg.B_s
    compute_budget = cfg.C_i.copy()

    # optimize for each user
    for (i_star, s_star), users in groups.items():
        n_users = len(users)
        if n_users == 0:
            continue

        bw_per_user = bw_budget[s_star] / max(users_per_slice[s_star], 1)
        f_per_user = compute_budget[i_star] / max(users_per_dt[i_star], 1)

        for u in users:
            b_opt, f_opt = _optimise_single_user(
                cfg, u, i_star, s_star,
                bw_per_user, f_per_user,
            )
            b[u, s_star] = b_opt
            f[u, i_star] = f_opt

    if verbose:
        print(f"  Stage 2: allocated resources for {len(admitted)} users")

    return b, f


def _optimise_single_user(
    cfg: NetworkConfig,
    u: int, i: int, s: int,
    bw_budget: float, f_budget: float,
) -> Tuple[float, float]:
    """
    Optimize (b, f) for a single user.
    Minimizes lambda_T * T + lambda_E * E subject to deadline.
    """
    d = cfg.d_u[u]
    eta = cfg.eta_us[u, s]
    alpha = cfg.alpha_u[u]
    tau = cfg.tau_u[u]
    P_tx = cfg.P_tx_us[u, s]
    kappa = cfg.kappa_i[i]
    h_tr = cfg.h_is[i, s]
    lT = cfg.lambda_T
    lE = cfg.lambda_E
    delta = cfg.delta

    bw_lo = max(bw_budget * 0.001, 100.0)
    bw_hi = bw_budget
    f_lo = max(f_budget * 0.001, 100.0)
    f_hi = f_budget

    def cost(z):
        bv, fv = z
        bv = np.clip(bv, bw_lo, bw_hi)
        fv = np.clip(fv, f_lo, f_hi)
        T_tx = d / (eta * bv + delta)
        T_cp = alpha * d / (fv + delta)
        T_total = T_tx + T_cp + h_tr
        E_tx = P_tx * T_tx
        E_cp = kappa * (fv ** 2) * (alpha * d / (fv + delta))
        E_total = E_tx + E_cp
        obj = lT * T_total + lE * E_total
        # penalize deadline violations heavily
        if T_total > tau:
            obj += 1e6 * (T_total - tau) ** 2
        return obj

    # start with high resource allocation to meet deadline
    z0 = np.array([0.8 * bw_hi, 0.8 * f_hi])
    bounds = [(bw_lo, bw_hi), (f_lo, f_hi)]

    res = minimize(cost, z0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 500, 'ftol': 1e-15})

    b_opt = np.clip(res.x[0], bw_lo, bw_hi)
    f_opt = np.clip(res.x[1], f_lo, f_hi)
    return float(b_opt), float(f_opt)


# --- Stage 3: Feasibility Enforcement ---

def _enforce_feasibility(
    cfg: NetworkConfig,
    y: np.ndarray,
    x: np.ndarray,
    b: np.ndarray,
    f: np.ndarray,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove users that violate deadline constraints.
    This ensures the final solution is always feasible.
    """
    sol_tmp = Solution(y=y.copy(), x=x.copy(), b=b.copy(), f=f.copy())
    T_u = total_latency(cfg, sol_tmp)

    rejected = 0
    for u in range(cfg.U):
        if y[u] < 0.5:
            continue
        if T_u[u] > cfg.tau_u[u] + 1e-9:
            y[u] = 0.0
            x[u, :, :] = 0.0
            b[u, :] = 0.0
            f[u, :] = 0.0
            rejected += 1

    if verbose and rejected:
        print(f"  Stage 3: rejected {rejected} users for deadline violation")

    return y, x, b, f
