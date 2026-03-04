"""
baselines.py

Baseline solvers that we compare against. Implemented 5 of them:
  - random assignment
  - greedy-latency (pick lowest latency option)
  - greedy-energy (pick lowest energy option)
  - latency-only (our solver but with lambda_E=0)
  - energy-only (our solver but with lambda_T=0)

All have the same interface: solve_xxx(cfg, verbose) -> Solution
"""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np

from .system_model import (
    NetworkConfig, Solution, DELTA,
    total_latency, total_energy,
    populate_metrics, validate_solution,
)
from .solver import _stage2_continuous_allocation


# --- 1. Random Assignment ---

def solve_random(cfg: NetworkConfig, verbose: bool = False) -> Solution:
    """Randomly assign users to (DT, slice) pairs, then allocate resources."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.seed if cfg.seed else 0)
    U, S_, I_ = cfg.U, cfg.S, cfg.I

    y = np.zeros(U)
    x = np.zeros((U, I_, S_))

    order = rng.permutation(U)
    bw_remaining = (cfg.beta_s * cfg.B_s).copy()
    compute_remaining = cfg.C_i.copy()

    for u in order:
        i = rng.integers(0, I_)
        s = rng.integers(0, S_)

        # check reliability
        if cfg.p_us[u, s] < (1 - cfg.eps_u[u]):
            continue

        # rough capacity check
        bw_need = bw_remaining[s] / max(U - int(y.sum()), 1)
        f_need = compute_remaining[i] / max(U - int(y.sum()), 1)
        if bw_need < 1e3 or f_need < 1e3:
            continue

        y[u] = 1.0
        x[u, i, s] = 1.0
        bw_remaining[s] -= bw_need
        compute_remaining[i] -= f_need

    b, f = _stage2_continuous_allocation(cfg, y, x, verbose=False)

    sol = Solution(y=y, x=x, b=b, f=f, solver_name="random")
    populate_metrics(cfg, sol)
    sol.solve_time_s = time.perf_counter() - t0
    validate_solution(cfg, sol)
    return sol


# --- 2. Greedy-Latency ---

def solve_greedy_latency(cfg: NetworkConfig, verbose: bool = False) -> Solution:
    """Pick the (DT, slice) pair with lowest estimated latency for each user."""
    t0 = time.perf_counter()
    U, S_, I_ = cfg.U, cfg.S, cfg.I

    y = np.zeros(U)
    x = np.zeros((U, I_, S_))

    # process users with tightest deadlines first
    order = np.argsort(cfg.tau_u)

    bw_remaining = (cfg.beta_s * cfg.B_s).copy()
    compute_remaining = cfg.C_i.copy()
    traffic_remaining = (cfg.H_s * cfg.T_win).copy()

    for u in order:
        best_lat = np.inf
        best_i, best_s = -1, -1
        for i in range(I_):
            for s in range(S_):
                if cfg.p_us[u, s] < (1 - cfg.eps_u[u]):
                    continue
                bw_share = bw_remaining[s] / max(U, 1)
                f_share = compute_remaining[i] / max(U, 1)
                if bw_share < 1e3 or f_share < 1e3:
                    continue

                T_tx = cfg.d_u[u] / (cfg.eta_us[u, s] * bw_share + cfg.delta)
                T_cp = cfg.alpha_u[u] * cfg.d_u[u] / (f_share + cfg.delta)
                T_tr = cfg.h_is[i, s]
                T = T_tx + T_cp + T_tr
                if T < best_lat:
                    best_lat = T
                    best_i, best_s = i, s

        if best_i >= 0:
            bw_need = cfg.d_u[u] / (cfg.eta_us[u, best_s] * cfg.tau_u[u] * 0.4 + cfg.delta)
            bw_need = min(bw_need, bw_remaining[best_s] * 0.5)
            f_need = cfg.alpha_u[u] * cfg.d_u[u] / (cfg.tau_u[u] * 0.4 + cfg.delta)
            f_need = min(f_need, compute_remaining[best_i] * 0.5)

            if (bw_need <= bw_remaining[best_s] + 1e-6 and
                f_need <= compute_remaining[best_i] + 1e-6 and
                cfg.d_u[u] <= traffic_remaining[best_s] + 1e-6):

                y[u] = 1.0
                x[u, best_i, best_s] = 1.0
                bw_remaining[best_s] -= bw_need
                compute_remaining[best_i] -= f_need
                traffic_remaining[best_s] -= cfg.d_u[u]

    b, f = _stage2_continuous_allocation(cfg, y, x, verbose=False)
    sol = Solution(y=y, x=x, b=b, f=f, solver_name="greedy_latency")
    populate_metrics(cfg, sol)
    sol.solve_time_s = time.perf_counter() - t0
    validate_solution(cfg, sol)
    return sol


# --- 3. Greedy-Energy ---

def solve_greedy_energy(cfg: NetworkConfig, verbose: bool = False) -> Solution:
    """Pick the (DT, slice) pair with lowest estimated energy for each user."""
    t0 = time.perf_counter()
    U, S_, I_ = cfg.U, cfg.S, cfg.I

    y = np.zeros(U)
    x = np.zeros((U, I_, S_))

    # estimate energy for each (user, DT, slice) combo
    user_energy = []
    for u in range(U):
        best_e = np.inf
        best_i, best_s = -1, -1
        for i in range(I_):
            for s in range(S_):
                if cfg.p_us[u, s] < (1 - cfg.eps_u[u]):
                    continue
                bw_share = cfg.beta_s[s] * cfg.B_s[s] / max(U, 1)
                f_share = cfg.C_i[i] / max(U, 1)
                T_tx = cfg.d_u[u] / (cfg.eta_us[u, s] * bw_share + cfg.delta)
                T_cp = cfg.alpha_u[u] * cfg.d_u[u] / (f_share + cfg.delta)
                E_tx = cfg.P_tx_us[u, s] * T_tx
                E_cp = cfg.kappa_i[i] * (f_share ** 2) * T_cp
                E = E_tx + E_cp
                if E < best_e:
                    best_e = E
                    best_i, best_s = i, s
        if best_i >= 0:
            user_energy.append((u, best_i, best_s, best_e))

    user_energy.sort(key=lambda t: t[3])

    bw_remaining = (cfg.beta_s * cfg.B_s).copy()
    compute_remaining = cfg.C_i.copy()
    traffic_remaining = (cfg.H_s * cfg.T_win).copy()

    for u, best_i, best_s, _ in user_energy:
        bw_need = cfg.d_u[u] / (cfg.eta_us[u, best_s] * cfg.tau_u[u] * 0.4 + cfg.delta)
        bw_need = min(bw_need, bw_remaining[best_s] * 0.5)
        f_need = cfg.alpha_u[u] * cfg.d_u[u] / (cfg.tau_u[u] * 0.4 + cfg.delta)
        f_need = min(f_need, compute_remaining[best_i] * 0.5)

        if (bw_need <= bw_remaining[best_s] + 1e-6 and
            f_need <= compute_remaining[best_i] + 1e-6 and
            cfg.d_u[u] <= traffic_remaining[best_s] + 1e-6):

            y[u] = 1.0
            x[u, best_i, best_s] = 1.0
            bw_remaining[best_s] -= bw_need
            compute_remaining[best_i] -= f_need
            traffic_remaining[best_s] -= cfg.d_u[u]

    b, f = _stage2_continuous_allocation(cfg, y, x, verbose=False)
    sol = Solution(y=y, x=x, b=b, f=f, solver_name="greedy_energy")
    populate_metrics(cfg, sol)
    sol.solve_time_s = time.perf_counter() - t0
    validate_solution(cfg, sol)
    return sol


# --- 4. Latency-Only (lambda_E = 0) ---

def solve_latency_only(cfg: NetworkConfig, verbose: bool = False) -> Solution:
    """Run our solver but ignoring energy (lambda_E = 0)."""
    from .solver import solve_p1
    import dataclasses
    cfg2 = dataclasses.replace(cfg, lambda_E=0.0)
    sol = solve_p1(cfg2, verbose=verbose)
    sol.solver_name = "latency_only"
    return sol


# --- 5. Energy-Only (lambda_T = 0) ---

def solve_energy_only(cfg: NetworkConfig, verbose: bool = False) -> Solution:
    """Run our solver but ignoring latency (lambda_T = 0)."""
    from .solver import solve_p1
    import dataclasses
    cfg2 = dataclasses.replace(cfg, lambda_T=0.0)
    sol = solve_p1(cfg2, verbose=verbose)
    sol.solver_name = "energy_only"
    return sol


# all baselines in one dict for easy access
BASELINES = {
    "random": solve_random,
    "greedy_latency": solve_greedy_latency,
    "greedy_energy": solve_greedy_energy,
    "latency_only": solve_latency_only,
    "energy_only": solve_energy_only,
}
