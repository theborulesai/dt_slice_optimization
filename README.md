# Energy-Latency Aware DT-Slice Co-Optimization in 5G Networks

This is project for the 5G/6G wireless networks course. The goal is to jointly optimize user admission, DT server assignment, and resource allocation across network slices, while keeping latency and energy consumption low.

---

## What is this about?

We have a 5G base station (gNB) serving **U users** across **S network slices** — eMBB, URLLC, and mMTC. There are also **I digital twin (DT) edge servers** that handle the computation offloading. For each user, we need to decide:

1. Should this user be admitted? (binary: yes/no)
2. Which DT server and which slice should handle them?
3. How much bandwidth and compute to allocate?

The objective is to maximize number of admitted users while penalizing high latency and high energy:

```
maximize:  sum(y_u) - lambda_T * sum(y_u * T_u) - lambda_E * sum(y_u * E_u)
```

There are 9 constraints (C1–C9) covering: assignment consistency, bandwidth/compute/traffic budgets, latency deadlines, and reliability requirements. It's a mixed-integer non-linear program (MINLP) which is NP-hard in general, so I used a decomposition approach instead of brute force.

---

## Project Structure

```
dt_slice_optimization/
├── main.py                     # run this to reproduce all results
├── configs/
│   └── default.yaml            # tweak experiment parameters here
├── src/
│   ├── system_model.py         # network config, latency/energy math
│   ├── solver.py               # my two-stage solver (the main contribution)
│   ├── baselines.py            # 8 comparison methods
│   ├── simulation.py           # runs experiments + sweeps
│   ├── visualization.py        # generates all 12 figures
│   └── paper_comparison.py     # compares results against a reference paper
├── tests/
│   ├── test_model.py
│   └── test_solver.py
├── results/
│   ├── comparison.csv          # per-method results at U=300
│   ├── sweep_users.csv         # scalability sweep data
│   ├── paper_comparison.csv    # vs reference paper numbers
│   └── figures/                # all 13 output figures
├── paper_comparison_report.md  # written comparison vs reference paper
└── requirements.txt
```

---

## How to Run

```bash
# install required packages
pip install -r requirements.txt

# run the full experiment (takes a few minutes due to user sweep)
python3 main.py

# smaller/faster run for testing
python3 main.py --users 50 --seeds 2

# use config file
python3 main.py --config configs/default.yaml -v

# run tests
python3 -m pytest tests/ -v
```

The script runs 4 phases:
1. **Solver comparison** — runs all 8 methods on the same scenario at U=300
2. **User sweep** — varies U from 5 to 300, runs all methods at each count
3. **Figure generation** — saves all 12 result plots to `results/figures/`
4. **Paper comparison** — prints a terminal report + saves comparison figure

---

## My Solver (Two-Stage Decomposition)

I couldn't use Gurobi (no license), so I built my own solver using a 3-stage decomposition:

**Stage 1 – Greedy Assignment:**  
For each user, score all possible (DT server, slice) pairs using a combined latency + energy score. Then admit users in order of tightest deadline first, checking capacity constraints before committing.

**Stage 2 – Continuous Resource Allocation:**  
With assignments fixed, optimize each user's bandwidth and compute using L-BFGS-B (scipy). This minimizes `lambda_T * T + lambda_E * E` subject to deadline and budget constraints.

**Stage 3 – Feasibility Enforcement:**  
After optimization, kick out any user whose total latency still exceeds their deadline. This guarantees the output is always feasible.

---

## Baselines (8 total)

I implemented 8 comparison methods — 5 from the original formulation and 3 new ones I added:

| # | Method | How it works |
|---|--------|-------------|
| 1 | Random | Random (DT, slice) pair assignment |
| 2 | Greedy-Latency | Always picks the lowest latency option |
| 3 | Greedy-Energy | Always picks the lowest energy option |
| 4 | Latency-Only | My solver but with lambda_E = 0 (ignores energy) |
| 5 | Energy-Only | My solver but with lambda_T = 0 (ignores latency) |
| 6 | **Proportional-Fair** *(new)* | 5G NR-style PF scheduler: scores by η × slack / demand |
| 7 | **Nearest-DT** *(new)* | Assigns user to closest DT server (min transport latency) |
| 8 | **Round-Robin** *(new)* | Cycles through (DT, slice) pairs in rotation |

The three new baselines help show how standard telecom scheduling approaches compare against the optimized approach.

---

## Results (at U=300, S=3, I=3)

| Method | Admitted | Avg Latency (ms) | Avg Energy (mJ) | Feasible? |
|--------|----------|-----------------|-----------------|-----------|
| **Proposed** | **9** | **7.69** | **0.81** | **Yes** |
| Random | 37 | 35.35 | 4.98 | No |
| Greedy-Latency | 21 | 60.24 | 8.10 | No |
| Greedy-Energy | 76 | 301.95 | 11.28 | No |
| Latency-Only | 4 | 6.70 | 0.42 | Yes |
| Energy-Only | 9 | 7.69 | 0.81 | Yes |
| Proportional-Fair | 76 | 184.30 | 17.11 | No |
| Nearest-DT | 76 | 204.61 | 16.95 | No |
| Round-Robin | 76 | 110.76 | 18.13 | No |

The proposed method is the only one (along with the single-objective variants) that satisfies all deadline and reliability constraints. Methods like Greedy-Energy and Round-Robin admit more users but violate constraints, which means they're not actually valid solutions.

---

## Generated Figures (12 total)

| Figure | What it shows |
|--------|--------------|
| fig1_admission_ratio.png | Admission % per method (bar chart) |
| fig2_avg_latency.png | Average latency per method (bar chart) |
| fig3_avg_energy.png | Average energy per method (bar chart) |
| fig4_objective.png | Objective value per method (bar chart) |
| fig5_scalability.png | Solve time vs number of users |
| fig6_summary_table.png | Summary table with all metrics |
| fig7_latency_cdf.png | CDF of per-user latency per method |
| fig8_pareto_tradeoff.png | Latency vs Energy scatter (Pareto front) |
| fig9_admission_vs_users.png | Admission % as users increase |
| fig10_energy_vs_users.png | Energy as users increase |
| fig11_latency_vs_users.png | Latency as users increase |
| fig12_slice_heatmap.png | How users distribute across slices per method |

Plus **figPC_paper_comparison.png** which compares this project against the reference paper.

---

## Paper Comparison

I compared my results against:

> **"Energy-Efficient Resource Allocation for Digital Twin-Assisted Network Slicing in 5G MEC Systems"**  
> Lin et al., IEEE Transactions on Vehicular Technology, 2023

Their approach uses SCA-MINLP (Successive Convex Approximation) with a commercial solver (CVX/MOSEK). My solver runs ~40× faster and doesn't need any commercial tools. Full comparison is in `paper_comparison_report.md`.

---

## Network Slice Parameters

| Slice | Bandwidth (MHz) | Usable Fraction | Backhaul Budget |
|-------|----------------|----------------|----------------|
| eMBB | ~100 | 80% | 500 Mbps |
| URLLC | ~20 | 60% | 100 Mbps |
| mMTC | ~10 | 90% | 200 Mbps |

## User Service Types

| Service | Data | Deadline | Reliability | Compute |
|---------|------|---------|------------|---------|
| Robot Control | 50 KB | 1 ms | 99.999% | 500 cyc/bit |
| XR Telepresence | 500 KB | 5 ms | 99.9% | 200 cyc/bit |
| Vehicular DT Sync | 200 KB | 2 ms | 99.99% | 300 cyc/bit |
| Smart Factory | 100 KB | 10 ms | 99% | 100 cyc/bit |

---

## References

Papers I read while working on this:

1. Foukas et al., "Network Slicing in 5G: Survey and Challenges," IEEE Commun. Mag., 2017. https://ieeexplore.ieee.org/document/7926919

2. Masood et al., "Digital Twin for 6G: Taxonomy, Research Challenges, and the Road Ahead," IEEE OJ-COMS, 2022. https://ieeexplore.ieee.org/document/9963577

3. Popovski et al., "5G Wireless Network Slicing for eMBB, URLLC, and mMTC," IEEE Access, 2018. https://ieeexplore.ieee.org/document/8490674

4. Letaief et al., "The Roadmap to 6G: AI Empowered Wireless Networks," IEEE Commun. Mag., 2019. https://ieeexplore.ieee.org/document/8808168

5. Zhou et al., "Edge Intelligence: Paving the Last Mile of AI with Edge Computing," Proc. IEEE, 2019. https://ieeexplore.ieee.org/document/8736011

6. Sun et al., "Digital Twin Assisted Resource Allocation for Network Slicing in 5G," IEEE TVT, 2022. https://ieeexplore.ieee.org/document/9756298

7. Saad et al., "Joint Admission Control and Resource Allocation in Wireless Networks," IEEE Commun. Surv. Tut., 2015. https://ieeexplore.ieee.org/document/7005534

8. Shen et al., "Holistic Network Virtualization and Pervasive Network Intelligence for 6G," IEEE Commun. Surv. Tut., 2022. https://ieeexplore.ieee.org/document/9627832

9. Mao et al., "Energy-Efficient Resource Allocation in Mobile Edge Computing Networks," IEEE Trans. Wireless Commun., 2017. https://ieeexplore.ieee.org/document/7762913

10. Lin et al., "Energy-Efficient Resource Allocation for Digital Twin-Assisted Network Slicing in 5G MEC Systems," IEEE TVT, 2023. (reference paper for comparison) https://ieeexplore.ieee.org/document/10057735
