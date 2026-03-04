"""
simulation.py

Handles running all the experiments - single runs, multi-seed comparisons,
user sweeps, and lambda sweeps for Pareto analysis. Basically the glue
between the solver and the plotting code.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .system_model import NetworkConfig, Solution, generate_scenario, populate_metrics
from .solver import solve_p1
from .baselines import BASELINES


def run_single(
    cfg: NetworkConfig,
    solver_fn: Callable,
    verbose: bool = False,
) -> Solution:
    """Run one solver on one instance."""
    return solver_fn(cfg, verbose=verbose)


def run_comparison(
    cfg: NetworkConfig,
    include_proposed: bool = True,
    baseline_names: Optional[List[str]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run proposed solver + baselines on the same config.
    Returns a DataFrame with one row per method.
    """
    if baseline_names is None:
        baseline_names = list(BASELINES.keys())

    results = []

    if include_proposed:
        sol = solve_p1(cfg, verbose=verbose)
        results.append(sol.summary_dict())

    for name in baseline_names:
        fn = BASELINES[name]
        sol = fn(cfg, verbose=verbose)
        results.append(sol.summary_dict())

    return pd.DataFrame(results)


def run_multi_seed(
    base_U: int = 15,
    base_S: int = 3,
    base_I: int = 3,
    lambda_T: float = 1.0,
    lambda_E: float = 1.0,
    seeds: Sequence[int] = (42, 123, 456),
    verbose: bool = False,
) -> pd.DataFrame:
    """Run comparison across multiple seeds."""
    frames = []
    for seed in seeds:
        cfg = generate_scenario(U=base_U, S=base_S, I=base_I,
                                lambda_T=lambda_T, lambda_E=lambda_E, seed=seed)
        df = run_comparison(cfg, verbose=verbose)
        df["seed"] = seed
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# --- Parameter sweeps ---

def run_sweep_users(
    user_counts: Sequence[int] = (5, 10, 15, 20, 25, 30),
    S: int = 3,
    I: int = 3,
    lambda_T: float = 1.0,
    lambda_E: float = 1.0,
    seeds: Sequence[int] = (42, 123, 456),
    verbose: bool = False,
) -> pd.DataFrame:
    """Sweep number of users, run all solvers at each count."""
    frames = []
    for U in user_counts:
        for seed in seeds:
            cfg = generate_scenario(U=U, S=S, I=I,
                                    lambda_T=lambda_T, lambda_E=lambda_E, seed=seed)
            df = run_comparison(cfg, verbose=verbose)
            df["U"] = U
            df["seed"] = seed
            frames.append(df)
        if verbose:
            print(f"  Sweep: U={U} done")
    return pd.concat(frames, ignore_index=True)


def run_sweep_lambda(
    lambda_T_values: Sequence[float] = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    lambda_E_values: Sequence[float] = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    U: int = 15, S: int = 3, I: int = 3,
    seed: int = 42,
    verbose: bool = False,
) -> pd.DataFrame:
    """Sweep lambda_T vs lambda_E for Pareto analysis (proposed solver only)."""
    rows = []
    for lT in lambda_T_values:
        for lE in lambda_E_values:
            cfg = generate_scenario(U=U, S=S, I=I,
                                    lambda_T=lT, lambda_E=lE, seed=seed)
            sol = solve_p1(cfg, verbose=False)
            row = sol.summary_dict()
            row["lambda_T"] = lT
            row["lambda_E"] = lE
            rows.append(row)
    return pd.DataFrame(rows)


def save_results(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")
