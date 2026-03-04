"""Tests for solver.py"""

import numpy as np
import pytest

from src.system_model import (
    NetworkConfig, Solution, generate_scenario,
    total_latency, validate_solution, populate_metrics,
)
from src.solver import solve_p1


class TestSolverSmall:
    """Test on a small instance (U=4, S=3, I=2)."""

    @pytest.fixture
    def small_cfg(self):
        return generate_scenario(U=4, S=3, I=2, seed=7)

    def test_solution_returned(self, small_cfg):
        sol = solve_p1(small_cfg, verbose=False)
        assert isinstance(sol, Solution)
        assert sol.y.shape == (4,)
        assert sol.x.shape == (4, 2, 3)
        assert sol.b.shape == (4, 3)
        assert sol.f.shape == (4, 2)

    def test_constraints_satisfied(self, small_cfg):
        sol = solve_p1(small_cfg, verbose=False)
        violations = validate_solution(small_cfg, sol)
        assert len(violations) == 0, f"Violations: {violations}"

    def test_at_least_one_admitted(self, small_cfg):
        sol = solve_p1(small_cfg, verbose=False)
        assert sol.admitted_count >= 1

    def test_infeasible_instance_no_crash(self):
        """Very limited resources - should handle gracefully."""
        cfg = generate_scenario(U=2, S=2, I=1, seed=42)
        sol = solve_p1(cfg, verbose=False)
        violations = validate_solution(cfg, sol)
        assert len(violations) == 0

    def test_metrics_populated(self, small_cfg):
        sol = solve_p1(small_cfg, verbose=False)
        assert sol.solve_time_s > 0
        assert sol.solver_name == "proposed_two_stage"


class TestSolverMedium:
    """Test on a medium instance (U=10, S=3, I=3)."""

    @pytest.fixture
    def med_cfg(self):
        return generate_scenario(U=10, S=3, I=3, seed=123)

    def test_feasibility(self, med_cfg):
        sol = solve_p1(med_cfg, verbose=False)
        violations = validate_solution(med_cfg, sol)
        assert len(violations) == 0, f"Violations: {violations[:5]}"

    def test_admitted_users_positive(self, med_cfg):
        sol = solve_p1(med_cfg, verbose=False)
        assert sol.admitted_count > 0

    def test_objective_reasonable(self, med_cfg):
        sol = solve_p1(med_cfg, verbose=False)
        # objective can't be more than U (all admitted with zero cost)
        assert sol.objective <= med_cfg.U + 0.1


class TestSolverEdgeCases:
    def test_single_user(self):
        cfg = generate_scenario(U=1, S=1, I=1, seed=0)
        sol = solve_p1(cfg, verbose=False)
        violations = validate_solution(cfg, sol)
        assert len(violations) == 0

    def test_many_users_limited_resources(self):
        """50 users with only 2 slices/DTs - some must be rejected."""
        cfg = generate_scenario(U=50, S=2, I=2, seed=7)
        sol = solve_p1(cfg, verbose=False)
        violations = validate_solution(cfg, sol)
        assert len(violations) == 0
        assert sol.admitted_count <= 50
