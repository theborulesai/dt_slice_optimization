#!/usr/bin/env python3
"""
main.py - DT-Slice Optimization Experiment

Runs everything in one go:
  1. Compare all 8 solvers on the same scenario
  2. Sweep number of users to see how methods scale
  3. Generate all 12 result figures
  4. Compare with reference paper and print report

Usage:
    python3 main.py
    python3 main.py --users 300 --slices 3 --dts 3
    python3 main.py --users 50 --seeds 2   # faster run for testing
    python3 main.py --config configs/default.yaml
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import yaml

from src.system_model import NetworkConfig, generate_scenario
from src.simulation import run_comparison, run_sweep_users, save_results
from src.visualization import generate_simple_plots
from src.paper_comparison import compare_with_paper, print_comparison_report


def parse_args():
    p = argparse.ArgumentParser(
        description="Energy-Latency Aware DT-Slice Co-Optimization")
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config file")
    p.add_argument("--users",    "-U", type=int,   default=300)
    p.add_argument("--slices",   "-S", type=int,   default=3)
    p.add_argument("--dts",      "-I", type=int,   default=3)
    p.add_argument("--lambda-T",       type=float, default=1.0)
    p.add_argument("--lambda-E",       type=float, default=1.0)
    p.add_argument("--seeds",          type=int,   default=5)
    p.add_argument("--out-dir",        default="results")
    p.add_argument("-v", "--verbose",  action="store_true")
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    t_start = time.perf_counter()

    # Load config if provided
    cfg_yaml = None
    if args.config and os.path.exists(args.config):
        cfg_yaml = load_config(args.config)

    # Parameters (CLI overrides YAML)
    U  = args.users
    S  = args.slices
    I  = args.dts
    lT = args.lambda_T
    lE = args.lambda_E
    n_seeds = args.seeds
    out_dir = args.out_dir

    if cfg_yaml:
        net = cfg_yaml.get("network", {})
        U   = net.get("users",      U)
        S   = net.get("slices",     S)
        I   = net.get("dt_servers", I)
        w   = cfg_yaml.get("weights", {})
        lT  = w.get("lambda_T", lT)
        lE  = w.get("lambda_E", lE)
        out_dir = cfg_yaml.get("output", {}).get("results_dir", out_dir)

    seeds   = list(range(42, 42 + n_seeds))
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("\n--- DT-Slice Co-Optimization Experiment ---")
    print(f"  Users:      {U}")
    print(f"  Slices:     {S}  (eMBB / URLLC / mMTC)")
    print(f"  DT Servers: {I}")
    print(f"  lambda_T = {lT}   lambda_E = {lE}")
    print(f"  Seeds:      {seeds}")
    print(f"  Output:     {out_dir}/")
    print()

    # ---- Phase 1: Solver Comparison ----
    print("Phase 1: Run all 8 solvers on the same scenario")
    print("-" * 40)
    cfg     = generate_scenario(U=U, S=S, I=I, lambda_T=lT, lambda_E=lE, seed=seeds[0])
    comp_df = run_comparison(cfg, verbose=args.verbose)
    print()
    print(comp_df.to_string(index=False))
    save_results(comp_df, os.path.join(out_dir, "comparison.csv"))
    print()

    # ---- Phase 2: User Sweep ----
    print("Phase 2: Sweep number of users (scalability test)")
    print("-" * 40)
    user_counts = [5, 10, 15, 20, 25, 30, 50, 100, 150, 200, 250, 300]
    if cfg_yaml:
        user_counts = cfg_yaml.get("sweep_users", {}).get("user_counts", user_counts)

    sweep_df = run_sweep_users(
        user_counts=user_counts, S=S, I=I,
        lambda_T=lT, lambda_E=lE, seeds=seeds, verbose=args.verbose,
    )
    save_results(sweep_df, os.path.join(out_dir, "sweep_users.csv"))
    print()

    # ---- Phase 3: Generate Figures ----
    print("Phase 3: Save all 12 figures")
    print("-" * 40)
    generate_simple_plots(
        sweep_df=sweep_df,
        comparison_df=comp_df,
        out_dir=fig_dir,
        U=U,
    )

    # ---- Phase 4: Paper Comparison ----
    print("Phase 4: Compare results against reference paper")
    print("-" * 40)
    print_comparison_report(comp_df, U_our=U)
    compare_with_paper(comp_df, out_dir=fig_dir, U_ref=50)

    # Done
    elapsed = time.perf_counter() - t_start
    print()
    print("=" * 50)
    print(f"  Experiment complete!  ({elapsed:.2f}s)")
    print(f"  Results in: {out_dir}/")
    print(f"  Figures in: {fig_dir}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
