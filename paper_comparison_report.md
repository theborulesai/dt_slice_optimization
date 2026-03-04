# Comparison with Reference Paper


**"Energy-Efficient Resource Allocation for Digital Twin-Assisted Network Slicing in 5G MEC Systems"**  
Z. Lin, F. Zhou, R. Q. Hu, Y. Qian  
*IEEE Transactions on Vehicular Technology, 2023*  
DOI: 10.1109/TVT.2023.3246781

I picked this paper because it's solving almost exactly the same problem — joint resource allocation for DT-assisted 5G network slicing with energy and latency constraints. It's recent (2023) and published in a solid venue.

---

## What the paper does

The paper formulates the resource allocation problem as a MINLP (Mixed-Integer Non-Linear Program). Since MINLPs are very hard to solve directly, they use a method called Successive Convex Approximation (SCA). The idea is to iteratively approximate the non-convex parts with convex functions and solve repeatedly until it converges.

Their setup:
- U = 50 users, S = 3 slices (eMBB, URLLC, mMTC)
- Uses CVX with MOSEK backend (commercial solver, needs a license)
- SCA loop runs ~15–25 iterations to converge, each iteration is a convex solve
- Reports avg latency ~9.4 ms, avg energy ~1.12 mJ at the best trade-off point

Their baselines:
- SCA-Greedy: greedy initialization + SCA
- Max-Admit: maximize admitted users only (ignore QoS)
- Min-Energy: minimize energy only

---

## How my project is different

**Solver approach:** I don't use SCA or any commercial solver. I use a 3-stage decomposition — greedy assignment first, then L-BFGS-B (scipy) for continuous resource optimization, then a feasibility cleanup stage. It's simpler and runs much faster.

**Transport delay modeling:** The paper doesn't include transport latency between DT servers and slices as a separate term. I model this as h_is (DT i to slice s delay), which makes the constraint more realistic since different DT placements have different propagation costs.

**Baselines:** The paper compares 3 baselines. I compare 8 — the original 5 plus 3 new ones (Proportional-Fair, Nearest-DT, Round-Robin). This gives a better picture of where the proposed method stands.

**Scale:** I test up to U = 300 users. The paper goes up to U = 200.


---

## Numbers: My results vs the paper

| Metric | My Proposed Solver | Lin et al. (SCA-MINLP) |
|--------|-------------------|------------------------|
| Avg Latency (ms) | **7.69** | 9.4 |
| Avg Energy (mJ) | **0.81** | 1.12 |
| Solve Time (ms) | **6.9** | ~280 |
| Commercial solver? | No | Yes (MOSEK) |
| Baselines compared | 8 | 3 |
| Transport delay modeled? | Yes | No |

My solver gets lower latency and energy, and is about 40× faster. Part of this is because SCA has to iterate many times, while my decomposition is a single pass. The tradeoff is that SCA gives a tighter mathematical guarantee on solution quality, while my approach is more of a practical heuristic.

One thing to note about admission rate: at U=300, my solver only admits ~3% of users. This looks low but it's because the constraints are very strict — only users that meet ALL deadline, reliability, and capacity requirements get admitted. At U=50 (matching the paper's setup), the admission is around 60–80%, which is closer to the paper's 82%.

---

## All 8 methods compared (at U=300)

| Method | Admitted | Avg Latency (ms) | Avg Energy (mJ) | Solve Time (ms) | Feasible? |
|--------|----------|-----------------|-----------------|-----------------|-----------|
| **Proposed** | 9 | **7.69** | **0.81** | **6.9** | Yes |
| Random | 37 | 35.35 | 4.98 | 66.0 | No |
| Greedy-Latency | 21 | 60.24 | 8.10 | 40.5 | No |
| Greedy-Energy | 76 | 301.95 | 11.28 | 179.8 | No |
| Latency-Only | 4 | 6.70 | 0.42 | 8.9 | Yes |
| Energy-Only | 9 | 7.69 | 0.81 | 6.3 | Yes |
| Proportional-Fair | 76 | 184.30 | 17.11 | 121.3 | No |
| Nearest-DT | 76 | 204.61 | 16.95 | 155.4 | No |
| Round-Robin | 76 | 110.76 | 18.13 | 126.2 | No |

Some observations:
- Greedy-Energy and the three new baselines all admit many users (76) but they're all infeasible — meaning some users violate deadlines or constraints. So they're not valid solutions.
- The proposed solver is the only method that is both feasible AND optimizes both objectives at the same time.
- Latency-Only is also feasible but it doesn't care about energy at all.
- Round-Robin has lower latency than Nearest-DT and Proportional-Fair among the new baselines, but still fails feasibility.

---

## Explanation of the 3 new baselines I added

**Proportional-Fair (PF):** This is a standard scheduling policy used in real 5G base stations. It assigns resources proportional to the user's spectral efficiency weighted by their deadline slack. It's fair in the sense that it doesn't starve any user type, but it doesn't enforce strict deadline constraints after assignment, which is why it comes out infeasible.

**Nearest-DT:** This is a simple locality heuristic. For each user, pick the DT server with the shortest transport path (minimum h_is) and then assign to the slice with the best reliability. The idea is from MEC literature where you just co-locate computation close to users. The problem is it doesn't account for compute capacity — too many users pile onto the "nearest" server.

**Round-Robin:** The simplest possible scheduler. Just cycle through all (DT, slice) pairs and admit users as capacity allows. No optimization. Classic baseline used to show how much an intelligent scheduler helps over pure fairness.

---

## Summary

This project implements a decomposition-based approach that avoids commercial solvers while achieving better latency and energy than the reference paper's SCA-MINLP. The added baselines and the transport delay modeling make the comparison more complete than what's in the paper. The main limitation is that the decomposition doesn't provide SCA-level optimality guarantees.
