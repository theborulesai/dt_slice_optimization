"""
baselines.py

All the baseline methods I implemented to compare against my proposed solver.
There are 8 total:
  1. random         -- completely random (DT, slice) assignment
  2. greedy_latency -- always picks lowest latency option
  3. greedy_energy  -- always picks lowest energy option
  4. latency_only   -- my solver but lambda_E = 0
  5. energy_only    -- my solver but lambda_T = 0
  6. proportional_fair -- 5G NR-style PF scheduler
  7. nearest_dt        -- pick closest DT server (min transport delay)
  8. round_robin       -- cycle through (DT, slice) pairs in order

All use the same interface: solve_xxx(cfg, verbose) -> Solution
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


# --- 6. Proportional-Fair (PF) ---

def solve_proportional_fair(cfg: NetworkConfig, verbose: bool = False) -> Solution:
    """
    Proportional-Fair (PF) scheduler.

    This is based on the PF scheduler used in real 5G NR base stations.
    For each (user, DT, slice) combo, I compute a score:
        PF_score = spectral_efficiency * deadline_slack / data_demand

    Then I sort by that score and admit users greedily.
    The idea is to favor users who have good channel conditions AND have
    some deadline slack, while not wasting capacity on high-demand users.

    Note: this doesn't enforce deadline constraints post-assignment,
    which is why it ends up infeasible at large U.
    """
    t0 = time.perf_counter()
    U, S_, I_ = cfg.U, cfg.S, cfg.I

    y = np.zeros(U)
    x = np.zeros((U, I_, S_))

    # compute PF score per (user, DT, slice)
    candidates = []
    for u in range(U):
        for i in range(I_):
            for s in range(S_):
                if cfg.p_us[u, s] < (1 - cfg.eps_u[u]):
                    continue
                # deadline slack: how much room does the user have?
                slack = cfg.tau_u[u] - cfg.h_is[i, s]
                if slack <= 0:
                    continue
                # PF score = spectral efficiency * slack / demand
                pf_score = (cfg.eta_us[u, s] * slack) / (cfg.d_u[u] + 1e-12)
                candidates.append((pf_score, u, i, s))

    # sort descending by PF score
    candidates.sort(key=lambda t: -t[0])

    bw_remaining     = (cfg.beta_s * cfg.B_s).copy()
    compute_remaining = cfg.C_i.copy()
    traffic_remaining = (cfg.H_s * cfg.T_win).copy()
    admitted_set = set()

    for pf_score, u, i, s in candidates:
        if u in admitted_set:
            continue
        bw_need = cfg.d_u[u] / (cfg.eta_us[u, s] * cfg.tau_u[u] * 0.4 + cfg.delta)
        bw_need = min(bw_need, bw_remaining[s] * 0.5)
        f_need  = cfg.alpha_u[u] * cfg.d_u[u] / (cfg.tau_u[u] * 0.4 + cfg.delta)
        f_need  = min(f_need, compute_remaining[i] * 0.5)

        if (bw_need <= bw_remaining[s] + 1e-6 and
                f_need  <= compute_remaining[i] + 1e-6 and
                cfg.d_u[u] <= traffic_remaining[s] + 1e-6):
            y[u] = 1.0
            x[u, i, s] = 1.0
            bw_remaining[s]      -= bw_need
            compute_remaining[i] -= f_need
            traffic_remaining[s] -= cfg.d_u[u]
            admitted_set.add(u)

    b, f = _stage2_continuous_allocation(cfg, y, x, verbose=False)
    sol = Solution(y=y, x=x, b=b, f=f, solver_name="proportional_fair")
    populate_metrics(cfg, sol)
    sol.solve_time_s = time.perf_counter() - t0
    validate_solution(cfg, sol)
    if verbose:
        print(f"  PF: admitted {int(y.sum())}/{U} users")
    return sol


# --- 7. Nearest-DT (NDT) ---

def solve_nearest_dt(cfg: NetworkConfig, verbose: bool = False) -> Solution:
    """
    Nearest-DT (NDT) heuristic.

    Simple idea: assign each user to the DT server that's closest
    (i.e., has minimum transport latency h_is for their slice).
    This is a common approach in MEC literature where you just
    co-locate computation near the user.

    The issue is it doesn't account for compute capacity -- too many
    users can pile onto the same 'nearest' server, causing overload.
    """
    t0 = time.perf_counter()
    U, S_, I_ = cfg.U, cfg.S, cfg.I

    y = np.zeros(U)
    x = np.zeros((U, I_, S_))

    bw_remaining     = (cfg.beta_s * cfg.B_s).copy()
    compute_remaining = cfg.C_i.copy()
    traffic_remaining = (cfg.H_s * cfg.T_win).copy()

    # process tightest-deadline users first
    order = np.argsort(cfg.tau_u)

    for u in order:
        # find slice with highest reliability
        best_s = -1
        best_rel = -1.0
        for s in range(S_):
            if cfg.p_us[u, s] >= (1 - cfg.eps_u[u]):
                if cfg.p_us[u, s] > best_rel:
                    best_rel = cfg.p_us[u, s]
                    best_s = s
        if best_s < 0:
            continue

        # pick DT with minimum transport latency for this slice
        best_i = int(np.argmin(cfg.h_is[:, best_s]))

        # check deadline feasibility
        if cfg.h_is[best_i, best_s] >= cfg.tau_u[u]:
            continue

        bw_need = cfg.d_u[u] / (cfg.eta_us[u, best_s] * cfg.tau_u[u] * 0.4 + cfg.delta)
        bw_need = min(bw_need, bw_remaining[best_s] * 0.5)
        f_need  = cfg.alpha_u[u] * cfg.d_u[u] / (cfg.tau_u[u] * 0.4 + cfg.delta)
        f_need  = min(f_need, compute_remaining[best_i] * 0.5)

        if (bw_need <= bw_remaining[best_s] + 1e-6 and
                f_need  <= compute_remaining[best_i] + 1e-6 and
                cfg.d_u[u] <= traffic_remaining[best_s] + 1e-6):
            y[u] = 1.0
            x[u, best_i, best_s] = 1.0
            bw_remaining[best_s]       -= bw_need
            compute_remaining[best_i]  -= f_need
            traffic_remaining[best_s]  -= cfg.d_u[u]

    b, f = _stage2_continuous_allocation(cfg, y, x, verbose=False)
    sol = Solution(y=y, x=x, b=b, f=f, solver_name="nearest_dt")
    populate_metrics(cfg, sol)
    sol.solve_time_s = time.perf_counter() - t0
    validate_solution(cfg, sol)
    if verbose:
        print(f"  NDT: admitted {int(y.sum())}/{U} users")
    return sol


# --- 8. Round-Robin (RR) ---

def solve_round_robin(cfg: NetworkConfig, verbose: bool = False) -> Solution:
    """
    Round-Robin (RR) scheduler.

    Just cycles through all (DT, slice) pairs in order, admitting users
    as resources allow. The most basic possible scheduler -- no optimization,
    just fairness by rotation. Useful as a lower bound baseline to show
    how much the optimized approach actually helps.
    """
    t0 = time.perf_counter()
    U, S_, I_ = cfg.U, cfg.S, cfg.I

    y = np.zeros(U)
    x = np.zeros((U, I_, S_))

    bw_remaining     = (cfg.beta_s * cfg.B_s).copy()
    compute_remaining = cfg.C_i.copy()
    traffic_remaining = (cfg.H_s * cfg.T_win).copy()

    # generate all (DT, slice) pairs in round-robin order
    pairs = [(i, s) for i in range(I_) for s in range(S_)]
    pair_idx = 0

    # process users with tightest deadlines first
    order = np.argsort(cfg.tau_u)

    for u in order:
        # try remaining pairs in round-robin order
        tried = 0
        while tried < len(pairs):
            i, s = pairs[pair_idx % len(pairs)]
            pair_idx += 1
            tried += 1

            if cfg.p_us[u, s] < (1 - cfg.eps_u[u]):
                continue
            if cfg.h_is[i, s] >= cfg.tau_u[u]:
                continue

            bw_need = cfg.d_u[u] / (cfg.eta_us[u, s] * cfg.tau_u[u] * 0.4 + cfg.delta)
            bw_need = min(bw_need, bw_remaining[s] * 0.4)
            f_need  = cfg.alpha_u[u] * cfg.d_u[u] / (cfg.tau_u[u] * 0.4 + cfg.delta)
            f_need  = min(f_need, compute_remaining[i] * 0.4)

            if (bw_need <= bw_remaining[s] + 1e-6 and
                    f_need  <= compute_remaining[i] + 1e-6 and
                    cfg.d_u[u] <= traffic_remaining[s] + 1e-6):
                y[u] = 1.0
                x[u, i, s] = 1.0
                bw_remaining[s]      -= bw_need
                compute_remaining[i] -= f_need
                traffic_remaining[s] -= cfg.d_u[u]
                break

    b, f = _stage2_continuous_allocation(cfg, y, x, verbose=False)
    sol = Solution(y=y, x=x, b=b, f=f, solver_name="round_robin")
    populate_metrics(cfg, sol)
    sol.solve_time_s = time.perf_counter() - t0
    validate_solution(cfg, sol)
    if verbose:
        print(f"  RR: admitted {int(y.sum())}/{U} users")
    return sol


# all baselines in one dict for easy access
BASELINES = {
    "random":             solve_random,
    "greedy_latency":     solve_greedy_latency,
    "greedy_energy":      solve_greedy_energy,
    "latency_only":       solve_latency_only,
    "energy_only":        solve_energy_only,
    "proportional_fair":  solve_proportional_fair,
    "nearest_dt":         solve_nearest_dt,
    "round_robin":        solve_round_robin,
}
