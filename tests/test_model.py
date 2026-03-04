"""Tests for system_model.py"""

import numpy as np
import pytest

from src.system_model import (
    NetworkConfig, Solution, generate_scenario,
    tx_latency, cp_latency, transport_latency,
    total_latency, total_energy,
    tx_energy, cp_energy,
    compute_objective, populate_metrics, validate_solution,
)


class TestScenarioGeneration:
    def test_dimensions(self):
        cfg = generate_scenario(U=10, S=3, I=2, seed=0)
        assert cfg.U == 10
        assert cfg.S == 3
        assert cfg.I == 2
        assert cfg.d_u.shape == (10,)
        assert cfg.tau_u.shape == (10,)
        assert cfg.eta_us.shape == (10, 3)
        assert cfg.p_us.shape == (10, 3)
        assert cfg.h_is.shape == (2, 3)
        assert cfg.P_tx_us.shape == (10, 3)
        assert cfg.C_i.shape == (2,)

    def test_positive_values(self):
        cfg = generate_scenario(U=20, S=4, I=3, seed=42)
        assert np.all(cfg.d_u > 0)
        assert np.all(cfg.tau_u > 0)
        assert np.all(cfg.B_s > 0)
        assert np.all(cfg.C_i > 0)
        assert np.all(cfg.eta_us > 0)
        assert np.all(cfg.h_is >= 0)
        assert np.all(cfg.P_tx_us > 0)
        assert np.all(cfg.p_us >= 0) and np.all(cfg.p_us <= 1)
        assert np.all(cfg.beta_s > 0) and np.all(cfg.beta_s < 1)

    def test_reproducibility(self):
        c1 = generate_scenario(U=5, S=2, I=2, seed=123)
        c2 = generate_scenario(U=5, S=2, I=2, seed=123)
        np.testing.assert_array_equal(c1.d_u, c2.d_u)
        np.testing.assert_array_equal(c1.eta_us, c2.eta_us)


class TestLatencyComputation:
    @pytest.fixture
    def small_instance(self):
        """Small 2-user, 2-slice, 1-DT instance for testing."""
        cfg = NetworkConfig(
            U=2, S=2, I=1,
            d_u=np.array([1000.0, 2000.0]),
            tau_u=np.array([0.01, 0.02]),
            eps_u=np.array([1e-3, 1e-3]),
            alpha_u=np.array([100.0, 200.0]),
            B_s=np.array([1e6, 2e6]),
            beta_s=np.array([0.8, 0.9]),
            H_s=np.array([1e9, 1e9]),
            C_i=np.array([1e9]),
            kappa_i=np.array([1e-27]),
            eta_us=np.array([[4.0, 6.0],
                             [5.0, 3.0]]),
            p_us=np.array([[0.999, 0.9999],
                           [0.999, 0.9999]]),
            h_is=np.array([[0.0005, 0.001]]),
            P_tx_us=np.array([[0.2, 0.3],
                              [0.25, 0.15]]),
            lambda_T=1.0, lambda_E=1.0, M=1e3,
        )
        # user 0 -> (DT 0, slice 0), user 1 -> (DT 0, slice 1)
        y = np.array([1.0, 1.0])
        x = np.zeros((2, 1, 2))
        x[0, 0, 0] = 1.0
        x[1, 0, 1] = 1.0
        b = np.zeros((2, 2))
        b[0, 0] = 100000.0   # 100 kHz
        b[1, 1] = 200000.0   # 200 kHz
        f = np.zeros((2, 1))
        f[0, 0] = 5e8        # 0.5 GHz
        f[1, 0] = 3e8        # 0.3 GHz
        sol = Solution(y=y, x=x, b=b, f=f)
        return cfg, sol

    def test_tx_latency_formula(self, small_instance):
        cfg, sol = small_instance
        T_tx = tx_latency(cfg, sol)
        # user 0: d=1000, eta=4, b=100000 -> 1000/(4*100000) = 0.0025 s
        expected_0 = 1000.0 / (4.0 * 100000.0 + cfg.delta)
        np.testing.assert_allclose(T_tx[0], expected_0, rtol=1e-6)

    def test_cp_latency_formula(self, small_instance):
        cfg, sol = small_instance
        T_cp = cp_latency(cfg, sol)
        # user 0: alpha=100, d=1000, f=5e8 -> 100*1000/5e8 = 2e-4 s
        expected_0 = 100.0 * 1000.0 / (5e8 + cfg.delta)
        np.testing.assert_allclose(T_cp[0], expected_0, rtol=1e-6)

    def test_transport_latency(self, small_instance):
        cfg, sol = small_instance
        T_tr = transport_latency(cfg, sol)
        np.testing.assert_allclose(T_tr[0], 0.0005, rtol=1e-6)
        np.testing.assert_allclose(T_tr[1], 0.001, rtol=1e-6)

    def test_total_is_sum(self, small_instance):
        cfg, sol = small_instance
        T_total = total_latency(cfg, sol)
        T_tx = tx_latency(cfg, sol)
        T_cp = cp_latency(cfg, sol)
        T_tr = transport_latency(cfg, sol)
        np.testing.assert_allclose(T_total, T_tx + T_cp + T_tr, rtol=1e-10)


class TestEnergyComputation:
    def test_energy_positive(self):
        cfg = generate_scenario(U=5, S=2, I=2, seed=0)
        y = np.ones(5)
        x = np.zeros((5, 2, 2))
        for u in range(5):
            x[u, u % 2, u % 2] = 1.0
        b = np.ones((5, 2)) * 1e5
        f = np.ones((5, 2)) * 1e8
        sol = Solution(y=y, x=x, b=b, f=f)
        E_tx = tx_energy(cfg, sol)
        E_cp = cp_energy(cfg, sol)
        assert np.all(E_tx >= 0)
        assert np.all(E_cp >= 0)


class TestObjective:
    def test_all_rejected(self):
        cfg = generate_scenario(U=5, S=2, I=2, seed=0)
        y = np.zeros(5)
        x = np.zeros((5, 2, 2))
        b = np.zeros((5, 2))
        f = np.zeros((5, 2))
        sol = Solution(y=y, x=x, b=b, f=f)
        obj = compute_objective(cfg, sol)
        assert obj == 0.0


class TestValidation:
    def test_trivial_feasible(self):
        """All users rejected should be trivially feasible."""
        cfg = generate_scenario(U=3, S=2, I=1, seed=0)
        y = np.zeros(3)
        x = np.zeros((3, 1, 2))
        b = np.zeros((3, 2))
        f = np.zeros((3, 1))
        sol = Solution(y=y, x=x, b=b, f=f)
        violations = validate_solution(cfg, sol)
        assert len(violations) == 0

    def test_c1_violation(self):
        """If x doesn't match y, should catch C1 violation."""
        cfg = generate_scenario(U=2, S=2, I=1, seed=0)
        y = np.array([1.0, 0.0])
        x = np.zeros((2, 1, 2))  # user 0 admitted but no assignment
        sol = Solution(y=y, x=x, b=np.zeros((2, 2)), f=np.zeros((2, 1)))
        violations = validate_solution(cfg, sol)
        assert any("C1" in v for v in violations)
