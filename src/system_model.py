"""
system_model.py

System model stuff for the DT-Slice co-optimization project.
Has the network config dataclass, solution container, scenario generator,
and all the latency/energy math from the formulation doc.

Reference: see docs/formulation.md for the full problem definition
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

# small constant to avoid division by zero
DELTA = 1e-12


# --- Data classes ---

@dataclass
class NetworkConfig:
    """Holds all parameters for one problem instance."""

    # dimensions
    U: int  # number of users
    S: int  # number of slices
    I: int  # number of DT servers

    # per-user params (shape U)
    d_u: np.ndarray          # data demand [bits]
    tau_u: np.ndarray        # latency deadline [s]
    eps_u: np.ndarray        # reliability target
    alpha_u: np.ndarray      # compute intensity [cycles/bit]

    # per-slice params (shape S)
    B_s: np.ndarray          # bandwidth budget [Hz]
    beta_s: np.ndarray       # usable fraction (0,1)
    H_s: np.ndarray          # backhaul traffic budget [bits/s]

    # per-DT params (shape I)
    C_i: np.ndarray          # compute capacity [cycles/s]
    kappa_i: np.ndarray      # DVFS energy coefficient

    # cross parameters
    eta_us: np.ndarray       # spectral efficiency  (U x S)
    p_us: np.ndarray         # reliability proxy     (U x S)
    h_is: np.ndarray         # transport latency     (I x S)  [s]
    P_tx_us: np.ndarray      # transmit power        (U x S)  [W]

    # scalar params
    T_win: float = 1.0       # scheduling window [s]
    lambda_T: float = 1.0    # latency weight
    lambda_E: float = 1.0    # energy weight
    M: float = 1e6           # big-M for constraint relaxation
    delta: float = DELTA

    seed: Optional[int] = None

    @property
    def users(self):
        return range(self.U)

    @property
    def slices(self):
        return range(self.S)

    @property
    def dts(self):
        return range(self.I)


@dataclass
class Solution:
    """Container for solver output."""

    # decision variables
    y: np.ndarray             # (U,)     admission
    x: np.ndarray             # (U,I,S)  assignment
    b: np.ndarray             # (U,S)    bandwidth allocation
    f: np.ndarray             # (U,I)    compute allocation

    # metrics (filled in after solving)
    objective: float = 0.0
    admitted_count: int = 0
    avg_latency: float = 0.0
    avg_energy: float = 0.0
    total_latency: float = 0.0
    total_energy: float = 0.0
    feasible: bool = True
    solver_name: str = ""
    solve_time_s: float = 0.0

    def summary_dict(self) -> Dict:
        return {
            "solver": self.solver_name,
            "admitted": self.admitted_count,
            "objective": round(self.objective, 6),
            "avg_latency_ms": round(self.avg_latency * 1e3, 4),
            "avg_energy_mJ": round(self.avg_energy * 1e3, 4),
            "total_latency_ms": round(self.total_latency * 1e3, 4),
            "total_energy_mJ": round(self.total_energy * 1e3, 4),
            "feasible": self.feasible,
            "solve_time_s": round(self.solve_time_s, 4),
        }


# --- Scenario generation ---

# slice archetypes: (B_s [Hz], beta, H_s [bps])
_SLICE_PRESETS = {
    "eMBB":  (100e6, 0.8, 500e6),
    "URLLC": (20e6,  0.6, 100e6),
    "mMTC":  (10e6,  0.9, 200e6),
}

# user service types: (data [bits], deadline [s], reliability, compute [cyc/bit])
_USER_PRESETS = {
    "robot_control":     (50e3,   1e-3, 1e-5, 500),
    "xr_telepresence":   (500e3,  5e-3, 1e-3, 200),
    "vehicular_dt_sync": (200e3,  2e-3, 1e-4, 300),
    "smart_factory":     (100e3, 10e-3, 1e-2, 100),
}


def generate_scenario(
    U: int = 15,
    S: int = 3,
    I: int = 3,
    lambda_T: float = 1.0,
    lambda_E: float = 1.0,
    seed: int = 42,
) -> NetworkConfig:
    """Generate a random but realistic problem instance."""
    rng = np.random.default_rng(seed)

    # slices
    slice_names = list(_SLICE_PRESETS.keys())
    chosen_slices = [slice_names[s % len(slice_names)] for s in range(S)]
    B_s = np.array([_SLICE_PRESETS[n][0] for n in chosen_slices])
    beta_s = np.array([_SLICE_PRESETS[n][1] for n in chosen_slices])
    H_s = np.array([_SLICE_PRESETS[n][2] for n in chosen_slices])
    # add some randomness
    B_s *= (1 + rng.uniform(-0.1, 0.1, S))
    H_s *= (1 + rng.uniform(-0.1, 0.1, S))

    # DT servers
    C_i = rng.uniform(5e9, 20e9, I)        # 5-20 GHz compute
    kappa_i = rng.uniform(1e-27, 5e-27, I)  # DVFS coefficient

    # users - cycle through service types
    service_names = list(_USER_PRESETS.keys())
    d_u = np.zeros(U)
    tau_u = np.zeros(U)
    eps_u = np.zeros(U)
    alpha_u = np.zeros(U)
    for u in range(U):
        svc = service_names[u % len(service_names)]
        base = _USER_PRESETS[svc]
        d_u[u] = base[0] * rng.uniform(0.8, 1.2)
        tau_u[u] = base[1] * rng.uniform(0.9, 1.1)
        eps_u[u] = base[2]
        alpha_u[u] = base[3] * rng.uniform(0.9, 1.1)

    # spectral efficiency (2-8 bps/Hz range)
    eta_us = rng.uniform(2.0, 8.0, (U, S))

    # reliability proxy - URLLC slices get higher values
    p_us = np.zeros((U, S))
    for s in range(S):
        base_rel = 0.9999 if "URLLC" in chosen_slices[s] else 0.999
        p_us[:, s] = rng.uniform(base_rel * 0.98, min(base_rel * 1.01, 1.0), U)
    p_us = np.clip(p_us, 0, 1)

    # transport latency between DT servers and slices
    h_is = rng.uniform(0.1e-3, 1.0e-3, (I, S))

    # transmit power
    P_tx_us = rng.uniform(0.1, 0.5, (U, S))

    M = 1e3  # big-M value

    cfg = NetworkConfig(
        U=U, S=S, I=I,
        d_u=d_u, tau_u=tau_u, eps_u=eps_u, alpha_u=alpha_u,
        B_s=B_s, beta_s=beta_s, H_s=H_s,
        C_i=C_i, kappa_i=kappa_i,
        eta_us=eta_us, p_us=p_us, h_is=h_is, P_tx_us=P_tx_us,
        T_win=1.0,
        lambda_T=lambda_T, lambda_E=lambda_E,
        M=M, delta=DELTA,
        seed=seed,
    )
    return cfg


# --- Latency and energy helper functions ---

def tx_latency_per_slice(cfg: NetworkConfig, sol: Solution) -> np.ndarray:
    """Transmission latency per (u, s). Shape (U, S).
    T_tx = d_u / (eta * b + delta)
    """
    denom = cfg.eta_us * sol.b + cfg.delta
    return cfg.d_u[:, None] / denom


def tx_latency(cfg: NetworkConfig, sol: Solution) -> np.ndarray:
    """Total transmission latency per user. Shape (U,)."""
    per_slice = tx_latency_per_slice(cfg, sol)
    # only count slices where user is actually assigned
    assigned = sol.x.sum(axis=1)  # (U, S)
    return (per_slice * assigned).sum(axis=1)


def cp_latency_per_dt(cfg: NetworkConfig, sol: Solution) -> np.ndarray:
    """Compute latency per (u, i). Shape (U, I)."""
    denom = sol.f + cfg.delta
    return (cfg.alpha_u[:, None] * cfg.d_u[:, None]) / denom


def cp_latency(cfg: NetworkConfig, sol: Solution) -> np.ndarray:
    """Total compute latency per user. Shape (U,)."""
    per_dt = cp_latency_per_dt(cfg, sol)
    assigned = sol.x.sum(axis=2)  # (U, I)
    return (per_dt * assigned).sum(axis=1)


def transport_latency(cfg: NetworkConfig, sol: Solution) -> np.ndarray:
    """Transport latency per user. Shape (U,)."""
    return np.einsum("uis,is->u", sol.x, cfg.h_is)


def total_latency(cfg: NetworkConfig, sol: Solution) -> np.ndarray:
    """End-to-end latency per user (sum of tx + compute + transport). Shape (U,)."""
    return tx_latency(cfg, sol) + cp_latency(cfg, sol) + transport_latency(cfg, sol)


def tx_energy_per_slice(cfg: NetworkConfig, sol: Solution) -> np.ndarray:
    """Communication energy per (u,s). Shape (U,S)."""
    T_tx = tx_latency_per_slice(cfg, sol)
    return cfg.P_tx_us * T_tx


def tx_energy(cfg: NetworkConfig, sol: Solution) -> np.ndarray:
    """Total communication energy per user. Shape (U,)."""
    per_slice = tx_energy_per_slice(cfg, sol)
    assigned = sol.x.sum(axis=1)
    return (per_slice * assigned).sum(axis=1)


def cp_energy_per_dt(cfg: NetworkConfig, sol: Solution) -> np.ndarray:
    """Computation energy per (u,i). Shape (U,I). Uses DVFS model."""
    T_cp = cp_latency_per_dt(cfg, sol)
    return cfg.kappa_i[None, :] * (sol.f ** 2) * T_cp


def cp_energy(cfg: NetworkConfig, sol: Solution) -> np.ndarray:
    """Total computation energy per user. Shape (U,)."""
    per_dt = cp_energy_per_dt(cfg, sol)
    assigned = sol.x.sum(axis=2)
    return (per_dt * assigned).sum(axis=1)


def total_energy(cfg: NetworkConfig, sol: Solution) -> np.ndarray:
    """Total energy per user (tx + compute). Shape (U,)."""
    return tx_energy(cfg, sol) + cp_energy(cfg, sol)


def compute_objective(cfg: NetworkConfig, sol: Solution) -> float:
    """Evaluate the P1 objective: sum(y) - lambda_T * sum(y*T) - lambda_E * sum(y*E)."""
    T_u = total_latency(cfg, sol)
    E_u = total_energy(cfg, sol)
    y = sol.y
    obj = y.sum() - cfg.lambda_T * (y * T_u).sum() - cfg.lambda_E * (y * E_u).sum()
    return float(obj)


def populate_metrics(cfg: NetworkConfig, sol: Solution) -> Solution:
    """Fill in the metric fields of a solution (in-place)."""
    T_u = total_latency(cfg, sol)
    E_u = total_energy(cfg, sol)
    admitted = sol.y > 0.5
    sol.admitted_count = int(admitted.sum())
    sol.objective = compute_objective(cfg, sol)
    if sol.admitted_count > 0:
        sol.avg_latency = float(T_u[admitted].mean())
        sol.avg_energy = float(E_u[admitted].mean())
        sol.total_latency = float((sol.y * T_u).sum())
        sol.total_energy = float((sol.y * E_u).sum())
    else:
        sol.avg_latency = 0.0
        sol.avg_energy = 0.0
        sol.total_latency = 0.0
        sol.total_energy = 0.0
    return sol


# --- Constraint checking ---

def validate_solution(cfg: NetworkConfig, sol: Solution, tol: float = 1e-6) -> list:
    """
    Check all constraints C1 through C9.
    Returns list of violation strings. Empty = feasible.
    """
    violations = []
    U, S_, I_ = cfg.U, cfg.S, cfg.I

    # C1: each admitted user has exactly one (DT, slice) assignment
    for u in range(U):
        lhs = sol.x[u].sum()
        if abs(lhs - sol.y[u]) > tol:
            violations.append(f"C1 violated for u={u}: Σx={lhs:.4f}, y={sol.y[u]:.4f}")

    # C2: bandwidth only if assigned to that slice
    for u in range(U):
        for s in range(S_):
            x_sum = sol.x[u, :, s].sum()
            ub = x_sum * cfg.B_s[s]
            if sol.b[u, s] < -tol or sol.b[u, s] > ub + tol:
                violations.append(f"C2 violated u={u}, s={s}: b={sol.b[u, s]:.4f}, UB={ub:.4f}")

    # C3: compute only if assigned to that DT
    for u in range(U):
        for i in range(I_):
            x_sum = sol.x[u, i, :].sum()
            ub = x_sum * cfg.C_i[i]
            if sol.f[u, i] < -tol or sol.f[u, i] > ub + tol:
                violations.append(f"C3 violated u={u}, i={i}: f={sol.f[u, i]:.4f}, UB={ub:.4f}")

    # C4: deadline constraint (with big-M relaxation)
    T_u = total_latency(cfg, sol)
    for u in range(U):
        rhs = cfg.tau_u[u] + cfg.M * (1 - sol.y[u])
        if T_u[u] > rhs + tol:
            violations.append(f"C4 violated u={u}: T={T_u[u]*1e3:.4f}ms > deadline={cfg.tau_u[u]*1e3:.4f}ms")

    # C5: reliability constraint
    for u in range(U):
        lhs = 0.0
        for i in range(I_):
            for s in range(S_):
                lhs += sol.x[u, i, s] * cfg.p_us[u, s]
        rhs = (1 - cfg.eps_u[u]) * sol.y[u]
        if lhs < rhs - tol:
            violations.append(f"C5 violated u={u}: rel={lhs:.6f} < required={rhs:.6f}")

    # C6: bandwidth budget per slice
    for s in range(S_):
        lhs = sol.b[:, s].sum()
        rhs = cfg.beta_s[s] * cfg.B_s[s]
        if lhs > rhs + tol:
            violations.append(f"C6 violated s={s}: total_bw={lhs:.0f} > budget={rhs:.0f}")

    # C7: traffic budget per slice
    for s in range(S_):
        lhs = 0.0
        for u in range(U):
            assigned_to_s = sol.x[u, :, s].sum()
            lhs += assigned_to_s * cfg.d_u[u]
        rhs = cfg.H_s[s] * cfg.T_win
        if lhs > rhs + tol:
            violations.append(f"C7 violated s={s}: traffic={lhs:.0f} > budget={rhs:.0f}")

    # C8: compute budget per DT server
    for i in range(I_):
        lhs = sol.f[:, i].sum()
        if lhs > cfg.C_i[i] + tol:
            violations.append(f"C8 violated i={i}: total_f={lhs:.2e} > C={cfg.C_i[i]:.2e}")

    # C9: binary integrality check
    for u in range(U):
        if not (abs(sol.y[u]) < tol or abs(sol.y[u] - 1) < tol):
            violations.append(f"C9 violated: y[{u}]={sol.y[u]:.4f} not binary")
        for i in range(I_):
            for s in range(S_):
                v = sol.x[u, i, s]
                if not (abs(v) < tol or abs(v - 1) < tol):
                    violations.append(f"C9 violated: x[{u},{i},{s}]={v:.4f} not binary")

    sol.feasible = len(violations) == 0
    return violations
