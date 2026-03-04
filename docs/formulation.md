# Mathematical Formulation - Energy-Latency Aware DT-Slice Co-Optimization

## 1. Network Entities

| Symbol | Description | Domain |
|--------|-------------|--------|
| $\mathcal{S} = \{1, \ldots, S\}$ | Network slices (eMBB / URLLC / mMTC) | — |
| $\mathcal{I} = \{1, \ldots, I\}$ | Edge DT servers | — |
| $\mathcal{U} = \{1, \ldots, U\}$ | Users requesting DT-assisted services | — |

## 2. Parameters

### Per-User Parameters
| Symbol | Description | Unit |
|--------|-------------|------|
| $d_u$ | Data demand (payload) | bits |
| $\tau_u$ | Latency deadline | s |
| $\epsilon_u$ | Reliability target | — |
| $\alpha_u$ | Compute intensity | cycles/bit |

### Per-Slice Parameters
| Symbol | Description | Unit |
|--------|-------------|------|
| $B_s$ | Bandwidth budget | Hz |
| $\beta_s$ | Usable fraction | ∈ (0,1) |
| $H_s$ | Backhaul traffic budget | bits/s |

### Per-DT Parameters
| Symbol | Description | Unit |
|--------|-------------|------|
| $C_i$ | Compute capacity | cycles/s |
| $\kappa_i$ | DVFS energy coefficient | — |

### Cross Parameters
| Symbol | Description | Shape |
|--------|-------------|-------|
| $\eta_{u,s}$ | Spectral efficiency | $U \times S$ |
| $p_{u,s}$ | PHY reliability proxy | $U \times S$ |
| $h_{i,s}$ | Transport latency | $I \times S$ |
| $P^{tx}_{u,s}$ | Transmit power | $U \times S$ |

## 3. Decision Variables

### Binary Variables
$$y_u \in \{0, 1\}: \quad \text{1 if user } u \text{ is admitted}$$

$$x_{u,i,s} \in \{0, 1\}: \quad \text{1 if user } u \text{ is assigned to DT } i \text{ and slice } s$$

### Continuous Variables
$$b_{u,s} \geq 0: \quad \text{bandwidth allocated to user } u \text{ in slice } s \text{ [Hz]}$$

$$f_{u,i} \geq 0: \quad \text{compute allocated to user } u \text{ at DT server } i \text{ [cycles/s]}$$

## 4. Latency Model

**Transmission latency** (L1):

$$T^{tx}_u = \sum_{s \in \mathcal{S}} \frac{d_u}{\eta_{u,s} b_{u,s} + \delta}$$

**Compute latency** (L2):

$$T^{cp}_u = \sum_{i \in \mathcal{I}} \frac{\alpha_u d_u}{f_{u,i} + \delta}$$

**Transport latency** (L3):

$$T^{tr}_u = \sum_{i} \sum_{s} x_{u,i,s} \, h_{i,s}$$

**Total end-to-end latency** (L4):

$$T_u = T^{tx}_u + T^{cp}_u + T^{tr}_u$$

## 5. Energy Model

**Communication energy** (E1):

$$E^{tx}_u = \sum_{s} P^{tx}_{u,s} \cdot \frac{d_u}{\eta_{u,s} b_{u,s} + \delta}$$

**Computation energy (DVFS model)** (E2):

$$E^{cp}_u = \sum_{i} \kappa_i (f_{u,i})^2 \cdot \frac{\alpha_u d_u}{f_{u,i} + \delta}$$

**Total energy** (E3):

$$E_u = E^{tx}_u + E^{cp}_u$$

## 6. Objective Function (P1)

$$\max_{\{x, y, b, f\}} \quad \sum_u y_u \;-\; \lambda_T \sum_u y_u T_u \;-\; \lambda_E \sum_u y_u E_u$$

The main goal is to maximize the number of admitted users, while also minimizing latency and energy among admitted users (controlled by $\lambda_T, \lambda_E > 0$).

## 7. Constraints

| ID | Constraint | Meaning |
|----|-----------|---------|
| C1 | $\sum_{i,s} x_{u,i,s} = y_u, \; \forall u$ | Each admitted user gets exactly one (DT, slice) pair |
| C2 | $0 \leq b_{u,s} \leq \left(\sum_i x_{u,i,s}\right) B_s$ | Bandwidth only if assigned to that slice |
| C3 | $0 \leq f_{u,i} \leq \left(\sum_s x_{u,i,s}\right) C_i$ | Compute only if assigned to that DT |
| C4 | $T_u \leq \tau_u + M(1 - y_u)$ | Deadline enforcement (big-M) |
| C5 | $\sum_{i,s} x_{u,i,s} p_{u,s} \geq (1-\epsilon_u) y_u$ | Reliability target |
| C6 | $\sum_u b_{u,s} \leq \beta_s B_s$ | Bandwidth budget per slice |
| C7 | $\sum_u y_u d_u \leq H_s T_{win}$ | Traffic budget per slice |
| C8 | $\sum_u f_{u,i} \leq C_i$ | Compute budget per DT server |
| C9 | $x_{u,i,s} \in \{0,1\}, \; y_u \in \{0,1\}$ | Binary integrality |

## 8. Problem Nature

This is a Mixed-Integer Non-Linear Program (MINLP), which is NP-hard because of:
- Binary assignment variables (combinatorial)
- Fractional terms $d_u / (\eta b + \delta)$ in latency/energy
- Bilinear coupling between binary and continuous variables

## 9. Solver Approach

### Two-Stage Decomposition

**Stage 1 - Greedy Binary Assignment:**
- Evaluate all feasible (user, DT, slice) triples
- Score based on latency-energy-reliability tradeoff
- Admit users greedily in deadline-priority order

**Stage 2 - Convex Continuous Allocation:**
- Fix binary variables from Stage 1
- Sub-problem becomes convex per-user
- Solve via L-BFGS-B: $\min_{b,f} \; \lambda_T T_u + \lambda_E E_u$

**Stage 3 - Feasibility Enforcement:**
- Post-solve rejection of users violating C4 deadlines
- Guarantees constraint-feasible output
