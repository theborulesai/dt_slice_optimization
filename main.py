#!/usr/bin/env python3
"""
main.py - Entry point for DT-Slice Co-Optimization experiments

Runs the whole pipeline end to end:
  1. Solver comparison (proposed vs 5 baselines)
  2. User sweep (5 to 30 users)
  3. Pareto sweep (lambda_T x lambda_E grid)
  4. Convergence analysis
  5. Sensitivity analysis
  6. Scalability benchmark (up to 100 users)
  7. Plot generation

Usage:
    python3 main.py
    python3 main.py --config configs/default.yaml -v
    python3 main.py --users 20 --slices 3 --dts 4
"""

import argparse
import os
import sys
import time
import yaml

import numpy as np
import pandas as pd

from src.system_model import (
    NetworkConfig, Solution, generate_scenario,
    total_latency, total_energy, validate_solution, populate_metrics,
)
from src.solver import solve_p1
from src.baselines import BASELINES
from src.simulation import (
    run_comparison, run_sweep_users, run_sweep_lambda, save_results,
)
from src.visualization import (
    generate_all_plots, plot_admission_vs_users, plot_latency_vs_users,
    plot_energy_vs_users, plot_pareto_front, plot_resource_heatmap,
    plot_latency_cdf, plot_objective_comparison, plot_summary_table,
    plot_scalability, plot_tradeoff_3d, plot_convergence,
    plot_sensitivity_radar, plot_assignment_map,
)
from src.analysis import (
    track_convergence, sensitivity_analysis, scalability_benchmark,
    statistical_summary,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Energy-Latency Aware DT-Slice Co-Optimization")
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config file")
    p.add_argument("--users", "-U", type=int, default=15)
    p.add_argument("--slices", "-S", type=int, default=3)
    p.add_argument("--dts", "-I", type=int, default=3)
    p.add_argument("--lambda-T", type=float, default=1.0)
    p.add_argument("--lambda-E", type=float, default=1.0)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--out-dir", default="results")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def load_config(path):
    """Load YAML config and return as dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    t_start = time.perf_counter()

    # load config if provided
    cfg_yaml = None
    if args.config and os.path.exists(args.config):
        cfg_yaml = load_config(args.config)

    # get parameters (CLI overrides YAML)
    U = args.users
    S = args.slices
    I = args.dts
    lT = args.lambda_T
    lE = args.lambda_E
    n_seeds = args.seeds
    out_dir = args.out_dir

    if cfg_yaml:
        net = cfg_yaml.get("network", {})
        U = net.get("users", U)
        S = net.get("slices", S)
        I = net.get("dt_servers", I)
        w = cfg_yaml.get("weights", {})
        lT = w.get("lambda_T", lT)
        lE = w.get("lambda_E", lE)
        seeds_list = cfg_yaml.get("seeds", None)
        if seeds_list:
            n_seeds = len(seeds_list)
        out_cfg = cfg_yaml.get("output", {})
        out_dir = out_cfg.get("results_dir", out_dir)

    seeds = list(range(42, 42 + n_seeds))
    if cfg_yaml and "seeds" in cfg_yaml:
        seeds = cfg_yaml["seeds"]

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
    print("Phase 1: Solver Comparison")
    print("-" * 40)

    cfg = generate_scenario(U=U, S=S, I=I, lambda_T=lT, lambda_E=lE, seed=seeds[0])
    comp_df = run_comparison(cfg, verbose=args.verbose)
    print()
    print(comp_df.to_string(index=False))
    save_results(comp_df, os.path.join(out_dir, "comparison.csv"))

    # statistical summary across seeds
    from src.simulation import run_multi_seed
    multi_df = run_multi_seed(base_U=U, base_S=S, base_I=I,
                              lambda_T=lT, lambda_E=lE, seeds=seeds)
    stats_df = statistical_summary(multi_df)
    print("\nStatistical Summary (mean +/- std):")
    print(stats_df.to_string(index=False))
    save_results(stats_df, os.path.join(out_dir, "statistical_summary.csv"))

    # get the proposed solution for later analysis
    sol_proposed = solve_p1(cfg, verbose=False)

    # resource utilization
    bw_util = np.array([sol_proposed.b[:, s].sum() / (cfg.beta_s[s] * cfg.B_s[s])
                        for s in range(cfg.S)])
    cp_util = np.array([sol_proposed.f[:, i].sum() / cfg.C_i[i]
                        for i in range(cfg.I)])

    # collect latency data for CDF plot
    latencies_dict = {}
    for solver_name in ["proposed_two_stage"] + list(BASELINES.keys()):
        if solver_name == "proposed_two_stage":
            s_sol = sol_proposed
        else:
            s_sol = BASELINES[solver_name](cfg, verbose=False)
        admitted = s_sol.y > 0.5
        if admitted.any():
            T_u = total_latency(cfg, s_sol)
            latencies_dict[solver_name] = T_u[admitted] * 1e3

    print()

    # ---- Phase 2: User Sweep ----
    do_user_sweep = True
    user_counts = [5, 10, 15, 20, 25, 30]
    if cfg_yaml:
        sw = cfg_yaml.get("sweep_users", {})
        do_user_sweep = sw.get("enabled", True)
        user_counts = sw.get("user_counts", user_counts)

    sweep_df = None
    if do_user_sweep:
        print("Phase 2: User Sweep")
        print("-" * 40)
        sweep_df = run_sweep_users(
            user_counts=user_counts, S=S, I=I,
            lambda_T=lT, lambda_E=lE, seeds=seeds, verbose=args.verbose,
        )
        save_results(sweep_df, os.path.join(out_dir, "sweep_users.csv"))
        print()

    # ---- Phase 3: Pareto Sweep ----
    do_pareto = True
    lT_vals = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    lE_vals = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    if cfg_yaml:
        sp = cfg_yaml.get("sweep_pareto", {})
        do_pareto = sp.get("enabled", True)
        lT_vals = sp.get("lambda_T_values", lT_vals)
        lE_vals = sp.get("lambda_E_values", lE_vals)

    pareto_df = None
    if do_pareto:
        print("Phase 3: Lambda Pareto Sweep")
        print("-" * 40)
        pareto_df = run_sweep_lambda(
            lambda_T_values=lT_vals, lambda_E_values=lE_vals,
            U=U, S=S, I=I, seed=seeds[0], verbose=args.verbose,
        )
        save_results(pareto_df, os.path.join(out_dir, "sweep_pareto.csv"))
        print()

    # ---- Phase 4: Convergence Analysis ----
    print("Phase 4: Convergence Analysis")
    print("-" * 40)
    convergence_data = track_convergence(cfg)
    for solver, vals in convergence_data.items():
        print(f"  {solver}: {[f'{v:.4f}' for v in vals]}")
    print()

    # ---- Phase 5: Sensitivity Analysis ----
    do_sensitivity = True
    perturb = 0.3
    if cfg_yaml:
        ss = cfg_yaml.get("sensitivity", {})
        do_sensitivity = ss.get("enabled", True)
        perturb = ss.get("perturbation", 0.3)

    sensitivity_metrics = None
    if do_sensitivity:
        print("Phase 5: Sensitivity Analysis")
        print("-" * 40)
        sensitivity_metrics = sensitivity_analysis(cfg, perturbation=perturb)
        for param, score in sensitivity_metrics.items():
            bar = "#" * int(score * 20) + "." * (20 - int(score * 20))
            print(f"  {param:30s} [{bar}] {score:.3f}")
        print()

    # ---- Phase 6: Scalability Benchmark ----
    do_scalability = True
    scale_counts = [5, 10, 15, 20, 30, 40, 50, 75, 100]
    if cfg_yaml:
        sc = cfg_yaml.get("scalability", {})
        do_scalability = sc.get("enabled", True)
        scale_counts = sc.get("user_counts", scale_counts)

    scale_df = None
    if do_scalability:
        print("Phase 6: Scalability Benchmark")
        print("-" * 40)
        scale_df = scalability_benchmark(
            user_counts=scale_counts, S=S, I=I, seed=seeds[0],
            verbose=args.verbose,
        )
        save_results(scale_df, os.path.join(out_dir, "scalability.csv"))
        print()

    # ---- Phase 7: Generate Plots ----
    print("Phase 7: Generating Figures")
    print("-" * 40)

    plot_sweep = sweep_df if sweep_df is not None else scale_df

    if plot_sweep is not None:
        generate_all_plots(
            sweep_df=plot_sweep,
            pareto_df=pareto_df,
            comparison_df=comp_df,
            bw_util=bw_util,
            cp_util=cp_util,
            latencies_dict=latencies_dict,
            convergence_data=convergence_data,
            sensitivity_metrics=sensitivity_metrics,
            x_assignment=sol_proposed.x,
            y_admission=sol_proposed.y,
            S=S, I=I,
            out_dir=fig_dir,
        )

    # done
    elapsed = time.perf_counter() - t_start
    print()
    print("=" * 50)
    print(f"  Experiment complete!  ({elapsed:.2f}s)")
    print(f"  Results in: {out_dir}/")
    print(f"  Figures in: {fig_dir}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
