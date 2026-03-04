"""
analysis.py

Extra analysis stuff - tracks how the objective changes across solver stages
(convergence), does sensitivity analysis by perturbing parameters, and
benchmarks scalability with different numbers of users.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .system_model import (
    NetworkConfig, Solution, generate_scenario,
    total_latency, total_energy, populate_metrics, validate_solution,
)
from .solver import solve_p1
from .baselines import BASELINES


# --- Convergence Tracking ---

def track_convergence(cfg: NetworkConfig) -> Dict[str, List[float]]:
    """
    Track objective value at each stage of the proposed solver.
    Returns dict mapping solver name to list of per-stage objectives.
    """
    from .solver import (
        _stage1_greedy_assignment, _stage2_continuous_allocation,
        _enforce_feasibility,
    )

    data = {}

    # proposed solver - track each stage
    y, x = _stage1_greedy_assignment(cfg)
    sol1 = Solution(
        y=y.copy(), x=x.copy(),
        b=np.zeros((cfg.U, cfg.S)),
        f=np.zeros((cfg.U, cfg.I)),
    )
    populate_metrics(cfg, sol1)
    obj_s1 = sol1.objective

    b, f = _stage2_continuous_allocation(cfg, y, x)
    sol2 = Solution(y=y.copy(), x=x.copy(), b=b.copy(), f=f.copy())
    populate_metrics(cfg, sol2)
    obj_s2 = sol2.objective

    y3, x3, b3, f3 = _enforce_feasibility(cfg, y.copy(), x.copy(), b.copy(), f.copy())
    sol3 = Solution(y=y3, x=x3, b=b3, f=f3)
    populate_metrics(cfg, sol3)
    obj_s3 = sol3.objective

    data["proposed_two_stage"] = [obj_s1, obj_s2, obj_s3]

    return data


# --- Sensitivity Analysis ---

def sensitivity_analysis(
    base_cfg: NetworkConfig,
    perturbation: float = 0.3,
) -> Dict[str, float]:
    """
    Perturb each key parameter by +/- perturbation and see how the
    objective changes. Returns normalized sensitivity scores in [0, 1].
    """
    sol_base = solve_p1(base_cfg, verbose=False)
    obj_base = sol_base.objective if sol_base.objective != 0 else 1e-6

    params = {
        "Traffic Load ($d_u$)": "d_u",
        "Deadline ($\\tau_u$)": "tau_u",
        "Compute ($\\alpha_u$)": "alpha_u",
        "Bandwidth ($B_s$)": "B_s",
        "DT Compute ($C_i$)": "C_i",
        "Tx Power ($P^{tx}$)": "P_tx_us",
    }

    raw_scores = {}
    for label, attr in params.items():
        # perturb up
        cfg_up = dataclasses.replace(base_cfg)
        arr_up = getattr(cfg_up, attr).copy() * (1 + perturbation)
        object.__setattr__(cfg_up, attr, arr_up)
        sol_up = solve_p1(cfg_up, verbose=False)

        # perturb down
        cfg_dn = dataclasses.replace(base_cfg)
        arr_dn = getattr(cfg_dn, attr).copy() * (1 - perturbation)
        object.__setattr__(cfg_dn, attr, arr_dn)
        sol_dn = solve_p1(cfg_dn, verbose=False)

        delta = abs(sol_up.objective - sol_dn.objective)
        raw_scores[label] = delta / max(abs(obj_base), 1e-6)

    # normalize to [0, 1]
    max_score = max(raw_scores.values()) if raw_scores else 1
    return {k: min(v / max(max_score, 1e-6), 1.0) for k, v in raw_scores.items()}


# --- Scalability Benchmark ---

def scalability_benchmark(
    user_counts: Sequence[int] = (5, 10, 15, 20, 30, 40, 50, 75, 100),
    S: int = 3, I: int = 3,
    seed: int = 42,
    verbose: bool = False,
) -> pd.DataFrame:
    """Measure solve time for different problem sizes."""
    rows = []
    for U in user_counts:
        cfg = generate_scenario(U=U, S=S, I=I, seed=seed)

        # proposed solver
        t0 = time.perf_counter()
        sol = solve_p1(cfg, verbose=False)
        t_prop = time.perf_counter() - t0
        rows.append({"U": U, "solver": "proposed_two_stage",
                      "solve_time_s": t_prop, **sol.summary_dict()})

        # baselines
        for name, fn in BASELINES.items():
            t0 = time.perf_counter()
            sol = fn(cfg, verbose=False)
            t_base = time.perf_counter() - t0
            rows.append({"U": U, "solver": name,
                          "solve_time_s": t_base, **sol.summary_dict()})

        if verbose:
            print(f"    U={U:3d}  done  ({t_prop*1e3:.1f}ms)")

    return pd.DataFrame(rows)


# --- Statistical Summary ---

def statistical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate mean +/- std summary table across seeds."""
    metrics = ["admitted", "objective", "avg_latency_ms", "avg_energy_mJ"]
    present = [m for m in metrics if m in df.columns]

    rows = []
    for solver, grp in df.groupby("solver"):
        row = {"Method": solver}
        for m in present:
            mean = grp[m].mean()
            std = grp[m].std()
            row[m] = f"{mean:.3f} ± {std:.3f}"
        rows.append(row)
    return pd.DataFrame(rows)
