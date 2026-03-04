# Energy-Latency Aware Digital Twin-Slice Co-Optimization

This project implements a joint user admission, DT-slice assignment, and resource allocation framework for 6G networks.

## Problem Overview

We consider a 6G gNB serving **U users** through **S network slices** (eMBB, URLLC, mMTC) with **I edge DT servers**. The optimization problem (P1) jointly decides:

- User admission: which users to admit (binary)
- DT-slice assignment: which (DT server, slice) pair for each user
- Resource allocation: how much bandwidth and compute to give each user

The objective maximizes total admitted users while penalizing latency and energy:

```
max  sum(y_u) - lambda_T * sum(y_u * T_u) - lambda_E * sum(y_u * E_u)
```

Subject to 9 constraints (C1-C9) for assignment consistency, resource budgets, latency deadlines, and reliability. See [docs/formulation.md](docs/formulation.md) for the full math.

## Project Structure

```
dt_slice_optimization/
├── main.py                 # entry point - runs all experiments
├── configs/
│   └── default.yaml        # experiment config
├── docs/
│   └── formulation.md      # math formulation
├── src/
│   ├── system_model.py     # network model, latency/energy functions
│   ├── solver.py           # two-stage solver (greedy + L-BFGS-B)
│   ├── baselines.py        # 5 baseline methods for comparison
│   ├── simulation.py       # experiment runner and sweeps
│   ├── visualization.py    # plotting functions
│   └── analysis.py         # convergence, sensitivity, scalability
├── tests/
│   ├── test_model.py       # tests for system model
│   └── test_solver.py      # tests for solver
├── results/                # output CSVs and figures
└── requirements.txt
```

## How to Run

```bash
# install dependencies
pip install -r requirements.txt

# run the full experiment
python3 main.py

# or with a config file
python3 main.py --config configs/default.yaml -v

# run tests
python3 -m pytest tests/ -v
```

## What the Experiment Does

The main script runs 7 phases:

1. **Solver comparison** - runs proposed solver + 5 baselines on same instance
2. **User sweep** - varies number of users from 5 to 30
3. **Pareto sweep** - sweeps lambda_T and lambda_E (36 combinations)
4. **Convergence analysis** - tracks objective at each solver stage
5. **Sensitivity analysis** - perturbs parameters by +/-30%
6. **Scalability benchmark** - tests with 5 to 100 users
7. **Plot generation** - creates all result figures

## Solver Design

We use a two-stage decomposition approach (no Gurobi needed):

**Stage 1 - Greedy Assignment:** Evaluates all (user, DT, slice) combos, scores them based on latency/energy, and admits users in deadline-priority order.

**Stage 2 - Continuous Optimization:** Fixes binary variables, then runs L-BFGS-B per user to optimize bandwidth and compute allocation.

**Stage 3 - Feasibility Check:** Rejects any users that still violate their deadline after optimization.

### Baselines

| Method | What it does |
|--------|-------------|
| Random | Random (DT, slice) assignment |
| Greedy-Latency | Picks lowest latency option |
| Greedy-Energy | Picks lowest energy option |
| Latency-Only | Runs our solver with lambda_E = 0 |
| Energy-Only | Runs our solver with lambda_T = 0 |

## Configuration

Edit `configs/default.yaml` to change experiment parameters:

```yaml
network:
  users: 15
  slices: 3
  dt_servers: 3

weights:
  lambda_T: 1.0
  lambda_E: 1.0

seeds: [42, 43, 44, 45, 46]
```

## Service Types Used

| Service | Data (kb) | Deadline (ms) | Reliability | Compute (cyc/bit) |
|---------|-----------|---------------|-------------|-------------------|
| Robot Control | 50 | 1 | 99.999% | 500 |
| XR Telepresence | 500 | 5 | 99.9% | 200 |
| Vehicular DT Sync | 200 | 2 | 99.99% | 300 |
| Smart Factory | 100 | 10 | 99% | 100 |

## References

Papers I referred to while working on this project:

1. **"Network Slicing in 5G: Survey and Challenges"** — Foukas et al., IEEE Communications Magazine, 2017.  
   https://ieeexplore.ieee.org/document/7926919

2. **"Digital Twin for 6G: Taxonomy, Research Challenges, and the Road Ahead"** — Masood et al., IEEE Open Journal of the Communications Society, 2022.  
   https://ieeexplore.ieee.org/document/9963577

3. **"5G Wireless Network Slicing for eMBB, URLLC, and mMTC: A Communication-Theoretic View"** — Popovski et al., IEEE Access, 2018.  
   https://ieeexplore.ieee.org/document/8490674

4. **"The Roadmap to 6G: AI Empowered Wireless Networks"** — Letaief et al., IEEE Communications Magazine, 2019.  
   https://ieeexplore.ieee.org/document/8808168

5. **"Edge Intelligence: Paving the Last Mile of Artificial Intelligence with Edge Computing"** — Zhou et al., Proceedings of the IEEE, 2019.  
   https://ieeexplore.ieee.org/document/8736011

6. **"Digital Twin Assisted Resource Allocation for Network Slicing in 5G"** — Sun et al., IEEE Transactions on Vehicular Technology, 2022.  
   https://ieeexplore.ieee.org/document/9756298

7. **"Joint Admission Control and Resource Allocation in Wireless Networks: A Survey"** — Saad et al., IEEE Communications Surveys & Tutorials, 2015.  
   https://ieeexplore.ieee.org/document/7005534

8. **"Holistic Network Virtualization and Pervasive Network Intelligence for 6G"** — Shen et al., IEEE Communications Surveys & Tutorials, 2022.  
   https://ieeexplore.ieee.org/document/9627832

9. **"Energy-Efficient Resource Allocation in Mobile Edge Computing Networks"** — Mao et al., IEEE Transactions on Wireless Communications, 2017.  
   https://ieeexplore.ieee.org/document/7762913

10. **"Toward Smart and Reconfigurable Environment: Intelligent Reflecting Surface Aided 6G Networks"** — Wu and Zhang, IEEE Communications Magazine, 2020.  
    https://ieeexplore.ieee.org/document/8910627
