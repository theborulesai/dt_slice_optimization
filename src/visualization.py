"""
visualization.py

All the plotting functions. Uses a dark theme because it looks way better
in presentations. Generates 13 different plots covering admission rates,
latency/energy comparisons, Pareto fronts, CDF, heatmaps, convergence,
sensitivity radar, assignment maps, scalability, 3D tradeoffs, and
a summary table.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import seaborn as sns

# --- Style config ---
# dark background looks nice for presentations
_BG        = "#0d1117"
_FG        = "#c9d1d9"
_GRID      = "#21262d"
_ACCENT    = "#58a6ff"
_PANEL     = "#161b22"

# colors for each solver
_PALETTE = {
    "proposed_two_stage": "#ff6b6b",
    "milp_solver":        "#ffd93d",
    "random":             "#6bcb77",
    "greedy_latency":     "#4d96ff",
    "greedy_energy":      "#ff922b",
    "latency_only":       "#cc5de8",
    "energy_only":        "#20c997",
}

_MARKERS = {
    "proposed_two_stage": "o",
    "milp_solver":        "H",
    "random":             "s",
    "greedy_latency":     "^",
    "greedy_energy":      "D",
    "latency_only":       "v",
    "energy_only":        "P",
}

_LABELS = {
    "proposed_two_stage": "Proposed (Two-Stage)",
    "milp_solver":        "MILP Solver",
    "random":             "Random",
    "greedy_latency":     "Greedy-Latency",
    "greedy_energy":      "Greedy-Energy",
    "latency_only":       "Latency-Only",
    "energy_only":        "Energy-Only",
}

_LS = {
    "proposed_two_stage": "-",
    "milp_solver":        "-",
    "random":             "--",
    "greedy_latency":     "-.",
    "greedy_energy":      ":",
    "latency_only":       "--",
    "energy_only":        "-.",
}

def _c(solver: str) -> str:
    return _PALETTE.get(solver, "#888888")

def _m(solver: str) -> str:
    return _MARKERS.get(solver, "x")

def _l(solver: str) -> str:
    return _LABELS.get(solver, solver)

def _ls(solver: str) -> str:
    return _LS.get(solver, "-")


def _apply_dark_style():
    """Set up the dark theme for all plots."""
    plt.rcParams.update({
        "figure.facecolor":   _BG,
        "axes.facecolor":     _PANEL,
        "axes.edgecolor":     _GRID,
        "axes.labelcolor":    _FG,
        "axes.titlecolor":    "#ffffff",
        "text.color":         _FG,
        "xtick.color":        _FG,
        "ytick.color":        _FG,
        "grid.color":         _GRID,
        "grid.alpha":         0.5,
        "legend.facecolor":   _PANEL,
        "legend.edgecolor":   _GRID,
        "legend.labelcolor":  _FG,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.facecolor":  _BG,
        "savefig.edgecolor":  _BG,
        "font.family":        "sans-serif",
        "font.size":          11,
        "axes.labelsize":     13,
        "axes.titlesize":     15,
        "axes.titleweight":   "bold",
        "legend.fontsize":    9,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "lines.linewidth":    2.2,
        "lines.markersize":   7,
    })

_apply_dark_style()


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {os.path.basename(path)}")


def _solver_order(df):
    """Get consistent ordering for solvers in plots."""
    order = ["proposed_two_stage", "milp_solver", "random",
             "greedy_latency", "greedy_energy", "latency_only", "energy_only"]
    present = [s for s in order if s in df["solver"].unique()]
    remaining = [s for s in df["solver"].unique() if s not in present]
    return present + remaining


# --- Plot 1: Admission Ratio vs Users ---

def plot_admission_vs_users(df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    solvers = _solver_order(df)

    for solver in solvers:
        grp = df[df["solver"] == solver]
        agg = grp.groupby("U")["admitted"].mean().reset_index()
        agg["ratio"] = agg["admitted"] / agg["U"]
        ax.plot(agg["U"], agg["ratio"], label=_l(solver),
                color=_c(solver), marker=_m(solver), ls=_ls(solver),
                markeredgecolor="white", markeredgewidth=0.5)

    ax.set_xlabel("Number of Users ($U$)")
    ax.set_ylabel("Admission Ratio")
    ax.set_title("User Admission Ratio vs. Network Load")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.08)
    _save(fig, os.path.join(out_dir, "01_admission_vs_users.png"))


# --- Plot 2: Average Latency vs Users ---

def plot_latency_vs_users(df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    solvers = _solver_order(df)
    for solver in solvers:
        grp = df[df["solver"] == solver]
        agg = grp.groupby("U")["avg_latency_ms"].mean().reset_index()
        ax.plot(agg["U"], agg["avg_latency_ms"], label=_l(solver),
                color=_c(solver), marker=_m(solver), ls=_ls(solver),
                markeredgecolor="white", markeredgewidth=0.5)

    ax.set_xlabel("Number of Users ($U$)")
    ax.set_ylabel("Avg. End-to-End Latency [ms]")
    ax.set_title("Average Latency vs. Network Load")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, "02_latency_vs_users.png"))


# --- Plot 3: Average Energy vs Users ---

def plot_energy_vs_users(df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    solvers = _solver_order(df)
    for solver in solvers:
        grp = df[df["solver"] == solver]
        agg = grp.groupby("U")["avg_energy_mJ"].mean().reset_index()
        ax.plot(agg["U"], agg["avg_energy_mJ"], label=_l(solver),
                color=_c(solver), marker=_m(solver), ls=_ls(solver),
                markeredgecolor="white", markeredgewidth=0.5)

    ax.set_xlabel("Number of Users ($U$)")
    ax.set_ylabel("Avg. Energy [mJ]")
    ax.set_title("Average Energy Consumption vs. Network Load")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, "03_energy_vs_users.png"))


# --- Plot 4: Objective Comparison ---

def plot_objective_comparison(df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    agg = df.groupby("solver")["objective"].mean().reset_index()
    solvers = _solver_order(df)
    agg = agg.set_index("solver").loc[solvers].reset_index()

    bars = ax.bar(
        range(len(agg)),
        agg["objective"],
        color=[_c(s) for s in agg["solver"]],
        edgecolor="white", linewidth=0.5, width=0.65,
    )
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels([_l(s) for s in agg["solver"]], rotation=25, ha="right")
    ax.set_ylabel("Objective Value")
    ax.set_title("Objective Comparison Across Methods")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, agg["objective"]):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02 * max(abs(h), 0.1),
                f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                color="white", fontweight="bold")

    _save(fig, os.path.join(out_dir, "04_objective_comparison.png"))


# --- Plot 5: Pareto Front ---

def plot_pareto_front(df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sc = ax.scatter(
        df["avg_latency_ms"], df["avg_energy_mJ"],
        c=df["admitted"], cmap="plasma", s=80,
        edgecolors="white", linewidths=0.6, zorder=3,
    )
    cbar = fig.colorbar(sc, ax=ax, label="Admitted Users", pad=0.02)
    cbar.ax.yaxis.label.set_color(_FG)
    cbar.ax.tick_params(colors=_FG)
    ax.set_xlabel("Avg. Latency [ms]")
    ax.set_ylabel("Avg. Energy [mJ]")
    ax.set_title("Latency-Energy Pareto Front ($\\lambda_T$ / $\\lambda_E$ Sweep)")
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, "05_pareto_front.png"))


# --- Plot 6: Latency CDF ---

def plot_latency_cdf(latencies_dict: Dict[str, np.ndarray], out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    solver_order = ["proposed_two_stage", "milp_solver", "random",
                    "greedy_latency", "greedy_energy", "latency_only", "energy_only"]
    for solver in solver_order:
        if solver not in latencies_dict:
            continue
        lats = np.sort(latencies_dict[solver])
        cdf = np.arange(1, len(lats) + 1) / len(lats)
        ax.step(lats, cdf, where="post", label=_l(solver),
                color=_c(solver), ls=_ls(solver), linewidth=2.5)

    ax.set_xlabel("End-to-End Latency [ms]")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of Per-User Latency (Admitted Users)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.08)
    _save(fig, os.path.join(out_dir, "06_latency_cdf.png"))


# --- Plot 7: Resource Utilization Heatmap ---

def plot_resource_heatmap(bw_util: np.ndarray, cp_util: np.ndarray, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    for idx, (data, title, xlabels) in enumerate([
        (bw_util, "Bandwidth Utilisation", [f"Slice {s+1}" for s in range(len(bw_util))]),
        (cp_util, "Compute Utilisation",   [f"DT {i+1}" for i in range(len(cp_util))]),
    ]):
        ax = axes[idx]
        mat = data.reshape(1, -1)
        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(xlabels)
        ax.set_yticks([0])
        ax.set_yticklabels(["Usage"])
        ax.set_title(title, fontsize=13, fontweight="bold", color="white")
        for j in range(len(data)):
            ax.text(j, 0, f"{data[j]:.1%}", ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if data[j] > 0.5 else "black")

    fig.suptitle("Resource Utilization (Proposed Solver)", y=1.0,
                 fontsize=14, fontweight="bold", color="white")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "07_resource_heatmap.png"))


# --- Plot 8: Convergence ---

def plot_convergence(convergence_data: Dict[str, List[float]], out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for solver, values in convergence_data.items():
        ax.plot(range(1, len(values) + 1), values, label=_l(solver),
                color=_c(solver), linewidth=2.5, marker=_m(solver),
                markeredgecolor="white", markeredgewidth=0.5, markersize=5)

    ax.set_xlabel("Iteration / Stage")
    ax.set_ylabel("Objective Value")
    ax.set_title("Convergence Analysis")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, "08_convergence.png"))


# --- Plot 9: Sensitivity Radar ---

def plot_sensitivity_radar(metrics: Dict, out_dir: str):
    labels = list(metrics.keys())
    values = list(metrics.values())
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_facecolor(_PANEL)
    ax.plot(angles, values, 'o-', linewidth=2.5, color="#ff6b6b",
            markeredgecolor="white", markeredgewidth=0.5)
    ax.fill(angles, values, alpha=0.25, color="#ff6b6b")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, color=_FG)
    ax.set_ylim(0, 1.1)
    ax.set_title("Sensitivity Analysis",
                 fontsize=14, fontweight="bold", color="white", pad=20)
    ax.grid(True, color=_GRID, alpha=0.5)
    ax.tick_params(axis="y", colors=_FG)
    _save(fig, os.path.join(out_dir, "09_sensitivity_radar.png"))


# --- Plot 10: Assignment Map ---

def plot_assignment_map(x: np.ndarray, y: np.ndarray,
                        S: int, I: int, out_dir: str):
    """Shows which (DT, slice) each admitted user got assigned to."""
    fig, ax = plt.subplots(figsize=(10, 5))
    U = len(y)
    admitted = np.where(y > 0.5)[0]
    slice_names = ["eMBB", "URLLC", "mMTC"]
    colors_slice = ["#4d96ff", "#ff6b6b", "#6bcb77"]

    for idx, u in enumerate(admitted):
        i_star, s_star = np.unravel_index(x[u].argmax(), (I, S))
        ax.barh(idx, 1, color=colors_slice[s_star % len(colors_slice)],
                edgecolor="white", linewidth=0.5, height=0.7)
        ax.text(0.5, idx, f"U{u+1} -> DT{i_star+1}, {slice_names[s_star % len(slice_names)]}",
                ha="center", va="center", fontsize=9, fontweight="bold", color="white")

    ax.set_yticks(range(len(admitted)))
    ax.set_yticklabels([f"User {u+1}" for u in admitted])
    ax.set_xlabel("")
    ax.set_title(f"User Assignment Map  |  {len(admitted)}/{U} admitted",
                 fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.grid(False)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors_slice[i], edgecolor="white",
                             label=slice_names[i]) for i in range(min(S, 3))]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)

    _save(fig, os.path.join(out_dir, "10_assignment_map.png"))


# --- Plot 11: Scalability ---

def plot_scalability(df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    solvers = _solver_order(df)
    for solver in solvers:
        grp = df[df["solver"] == solver]
        agg = grp.groupby("U")["solve_time_s"].mean().reset_index()
        ax.plot(agg["U"], agg["solve_time_s"] * 1e3, label=_l(solver),
                color=_c(solver), marker=_m(solver), ls=_ls(solver),
                markeredgecolor="white", markeredgewidth=0.5)

    ax.set_xlabel("Number of Users ($U$)")
    ax.set_ylabel("Solve Time [ms]")
    ax.set_title("Computational Scalability")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, "11_scalability.png"))


# --- Plot 12: 3D Tradeoff Surface ---

def plot_tradeoff_3d(df: pd.DataFrame, out_dir: str):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(_BG)
    fig.patch.set_facecolor(_BG)

    sc = ax.scatter(
        df["lambda_T"], df["lambda_E"], df["objective"],
        c=df["admitted"], cmap="plasma", s=60,
        edgecolors="white", linewidths=0.4, depthshade=True,
    )
    ax.set_xlabel("$\\lambda_T$", fontsize=12, color=_FG, labelpad=10)
    ax.set_ylabel("$\\lambda_E$", fontsize=12, color=_FG, labelpad=10)
    ax.set_zlabel("Objective", fontsize=12, color=_FG, labelpad=10)
    ax.set_title("3D Trade-off Surface ($\\lambda_T$, $\\lambda_E$, Objective)",
                 fontsize=14, fontweight="bold", color="white", pad=15)

    ax.tick_params(axis='x', colors=_FG)
    ax.tick_params(axis='y', colors=_FG)
    ax.tick_params(axis='z', colors=_FG)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, label="Admitted Users", pad=0.1)
    cbar.ax.yaxis.label.set_color(_FG)
    cbar.ax.tick_params(colors=_FG)

    _save(fig, os.path.join(out_dir, "12_tradeoff_3d.png"))


# --- Plot 13: Summary Table ---

def plot_summary_table(df: pd.DataFrame, out_dir: str):
    """Render a comparison table as an image."""
    cols = ["solver", "admitted", "objective", "avg_latency_ms",
            "avg_energy_mJ", "feasible", "solve_time_s"]
    present_cols = [c for c in cols if c in df.columns]
    table_df = df[present_cols].copy()
    table_df["solver"] = table_df["solver"].map(lambda s: _l(s))

    rename = {
        "solver": "Method",
        "admitted": "Admitted",
        "objective": "Objective",
        "avg_latency_ms": "Avg Lat [ms]",
        "avg_energy_mJ": "Avg Eng [mJ]",
        "feasible": "Feasible",
        "solve_time_s": "Time [s]",
    }
    table_df = table_df.rename(columns=rename)

    fig, ax = plt.subplots(figsize=(12, 0.8 + 0.5 * len(table_df)))
    ax.axis("off")
    ax.set_title("Solver Comparison Summary", fontsize=15,
                 fontweight="bold", color="white", pad=15)

    tbl = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(_GRID)
        if row == 0:
            cell.set_facecolor("#30363d")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor(_PANEL)
            cell.set_text_props(color=_FG)
            # highlight our proposed solver
            if table_df.iloc[row - 1, 0] == _l("proposed_two_stage"):
                cell.set_facecolor("#1a2332")
                cell.set_text_props(color="#ff6b6b", fontweight="bold")

    _save(fig, os.path.join(out_dir, "13_summary_table.png"))


# --- Generate all plots at once ---

def generate_all_plots(
    sweep_df: pd.DataFrame,
    pareto_df: Optional[pd.DataFrame],
    comparison_df: pd.DataFrame,
    bw_util: Optional[np.ndarray] = None,
    cp_util: Optional[np.ndarray] = None,
    latencies_dict: Optional[Dict] = None,
    convergence_data: Optional[Dict] = None,
    sensitivity_metrics: Optional[Dict] = None,
    x_assignment: Optional[np.ndarray] = None,
    y_admission: Optional[np.ndarray] = None,
    S: int = 3, I: int = 3,
    out_dir: str = "results/figures",
):
    """Generate all result plots."""
    _apply_dark_style()
    print("\n  Generating plots...")

    # sweep plots
    plot_admission_vs_users(sweep_df, out_dir)
    plot_latency_vs_users(sweep_df, out_dir)
    plot_energy_vs_users(sweep_df, out_dir)
    plot_scalability(sweep_df, out_dir)

    # comparison
    plot_objective_comparison(comparison_df, out_dir)
    plot_summary_table(comparison_df, out_dir)

    # pareto and 3D
    if pareto_df is not None and len(pareto_df) > 0:
        plot_pareto_front(pareto_df, out_dir)
        if "lambda_T" in pareto_df.columns:
            plot_tradeoff_3d(pareto_df, out_dir)

    # resource heatmap
    if bw_util is not None and cp_util is not None:
        plot_resource_heatmap(bw_util, cp_util, out_dir)

    # CDF
    if latencies_dict is not None:
        plot_latency_cdf(latencies_dict, out_dir)

    # convergence
    if convergence_data is not None:
        plot_convergence(convergence_data, out_dir)

    # sensitivity
    if sensitivity_metrics is not None:
        plot_sensitivity_radar(sensitivity_metrics, out_dir)

    # assignment map
    if x_assignment is not None and y_admission is not None:
        plot_assignment_map(x_assignment, y_admission, S, I, out_dir)

    print(f"\n  All plots saved to {out_dir}/")
