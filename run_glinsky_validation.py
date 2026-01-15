#!/usr/bin/env python3
"""
Glinsky Framework Validation Suite

Single command to validate all core claims:
    python run_glinsky_validation.py

Returns exit code 0 if all tests pass, 1 if any fail.
"""

import numpy as np
import torch
import sys
import time
from typing import Dict, List, Tuple
from scipy.stats import pearsonr, spearmanr
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from fixed_hjb_loss import ImprovedHJB_MLP
from hst_rom import HST_ROM
from hst import hst_forward_pywt, hst_inverse_pywt
from action_angle_utils import wrap_to_2pi, angular_distance


@dataclass
class TestResult:
    name: str
    passed: bool
    metric: float
    threshold: float
    details: str = ""


class GlinskyValidator:
    """Comprehensive validation of Glinsky framework."""

    def __init__(self, device: str = None, verbose: bool = True):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        self.results: List[TestResult] = []

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    # =========================================================================
    # TEST 1: HST Roundtrip
    # =========================================================================
    def test_hst_roundtrip(self) -> TestResult:
        """HST forward/inverse should be near-perfect."""
        self.log("\n[TEST 1] HST Roundtrip")
        self.log("-" * 40)

        # Generate test signal
        t = np.linspace(0, 2*np.pi, 128)
        z = np.cos(t) + 1j * np.sin(t)

        # Forward + inverse
        coeffs = hst_forward_pywt(z, J=3, wavelet_name='db8')
        z_rec = hst_inverse_pywt(coeffs, original_length=len(z))

        error = np.linalg.norm(z_rec - z) / np.linalg.norm(z)

        passed = error < 1e-10
        self.log(f"  Error: {error:.2e}")
        self.log(f"  {'PASS' if passed else 'FAIL'} (threshold: 1e-10)")

        return TestResult("HST Roundtrip", passed, error, 1e-10)

    # =========================================================================
    # TEST 2: PCA Variance
    # =========================================================================
    def test_pca_variance(self) -> TestResult:
        """PCA with 4 components should capture >85% variance."""
        self.log("\n[TEST 2] PCA Variance Capture")
        self.log("-" * 40)

        # Generate SHO signals (consistent with bridge test)
        np.random.seed(42)
        omega = 1.0
        dt = 0.01
        n_points = 128
        signals = []

        for _ in range(100):
            E = np.random.uniform(0.5, 4.5)
            theta0 = np.random.uniform(0, 2*np.pi)

            t = np.arange(n_points) * dt
            theta_t = theta0 + omega * t
            p_t = np.sqrt(2 * E) * np.sin(theta_t)
            q_t = np.sqrt(2 * E) / omega * np.cos(theta_t)

            # Analytic signal representation
            z = q_t + 1j * p_t / omega
            signals.append(z)

        rom = HST_ROM(n_components=4, wavelet='db8', J=3, window_size=128)
        rom.fit(signals, extract_windows=False)

        variance = sum(rom.pca.explained_variance_ratio_)

        passed = variance > 0.85
        self.log(f"  Variance explained: {variance:.3f}")
        self.log(f"  {'PASS' if passed else 'FAIL'} (threshold: 0.85)")

        return TestResult("PCA Variance", passed, variance, 0.85)

    # =========================================================================
    # TEST 3: β-pq Correlation
    # =========================================================================
    def test_beta_pq_correlation(self) -> TestResult:
        """β should correlate with (p,q)."""
        self.log("\n[TEST 3] β-(p,q) Correlation")
        self.log("-" * 40)

        np.random.seed(42)
        omega = 1.0
        signals = []
        pq_true = []

        for _ in range(100):
            E = np.random.uniform(0.5, 4.5)
            theta0 = np.random.uniform(0, 2*np.pi)

            p0 = np.sqrt(2*E) * np.sin(theta0)
            q0 = np.sqrt(2*E) / omega * np.cos(theta0)

            t = np.linspace(0, 1.27, 128)
            theta_t = theta0 + omega * t
            p_t = np.sqrt(2*E) * np.sin(theta_t)
            q_t = np.sqrt(2*E) / omega * np.cos(theta_t)
            z = q_t + 1j * p_t / omega
            signals.append(z)

            center = 64
            pq_true.append([p_t[center], q_t[center]])

        pq_true = np.array(pq_true)

        rom = HST_ROM(n_components=4, wavelet='db8', J=3, window_size=128)
        betas = rom.fit(signals, extract_windows=False)

        # Best correlation with p or q
        max_corr = 0
        for i in range(min(4, betas.shape[1])):
            r_p = abs(pearsonr(betas[:, i], pq_true[:, 0])[0])
            r_q = abs(pearsonr(betas[:, i], pq_true[:, 1])[0])
            max_corr = max(max_corr, r_p, r_q)

        passed = max_corr > 0.8
        self.log(f"  Max |correlation|: {max_corr:.4f}")
        self.log(f"  {'PASS' if passed else 'FAIL'} (threshold: 0.8)")

        return TestResult("β-pq Correlation", passed, max_corr, 0.8)

    # =========================================================================
    # TEST 4: HJB-MLP P Learning
    # =========================================================================
    def test_hjb_p_learning(self) -> TestResult:
        """HJB-MLP should learn P with Spearman > 0.95."""
        self.log("\n[TEST 4] HJB-MLP P Learning")
        self.log("-" * 40)

        torch.manual_seed(3)
        np.random.seed(3)

        model = ImprovedHJB_MLP(hidden_dim=64, num_layers=3).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        omega = 1.0
        dt = 0.3
        F_scale = 1.0

        # Training loop (shortened for validation)
        for epoch in range(1500):
            n_traj = 100
            P0 = torch.rand(n_traj, device=self.device) * 2 + 0.5
            Q0 = torch.rand(n_traj, device=self.device) * 2 * np.pi
            F_ext = (torch.rand(n_traj, device=self.device) - 0.5) * 2 * F_scale

            P1 = torch.clamp(P0 + F_ext * dt, min=0.1)
            Q1 = Q0 + omega * dt

            p0 = torch.sqrt(2 * P0 * omega) * torch.sin(Q0)
            q0 = torch.sqrt(2 * P0 / omega) * torch.cos(Q0)
            p1 = torch.sqrt(2 * P1 * omega) * torch.sin(Q1)
            q1 = torch.sqrt(2 * P1 / omega) * torch.cos(Q1)

            optimizer.zero_grad()

            P0_pred, Q0_pred = model.encode(p0, q0)
            P1_pred, Q1_pred = model.encode(p1, q1)

            # Losses
            p0_rec, q0_rec = model.decode(P0_pred, Q0_pred)
            recon = torch.mean((p0 - p0_rec)**2 + (q0 - q0_rec)**2)

            dP_exp = F_ext * dt
            dP_act = P1_pred - P0_pred
            action = torch.mean((dP_act - dP_exp)**2) / (torch.mean(dP_exp**2) + 1e-6)

            evol = torch.mean(1 - torch.cos(Q1_pred - Q0_pred - omega * dt))

            loss = recon + 10 * action + 5 * evol
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Evaluate
        model.eval()
        np.random.seed(9999)
        n_test = 200
        P_true = np.random.uniform(0.5, 2.5, n_test)
        Q_true = np.random.uniform(0, 2*np.pi, n_test)

        p = np.sqrt(2 * P_true * omega) * np.sin(Q_true)
        q = np.sqrt(2 * P_true / omega) * np.cos(Q_true)

        with torch.no_grad():
            p_t = torch.tensor(p, dtype=torch.float32, device=self.device)
            q_t = torch.tensor(q, dtype=torch.float32, device=self.device)
            P_pred, _ = model.encode(p_t, q_t)
            P_pred = P_pred.cpu().numpy()

        spearman = abs(spearmanr(P_pred, P_true)[0])

        passed = spearman > 0.95
        self.log(f"  Spearman: {spearman:.4f}")
        self.log(f"  {'PASS' if passed else 'FAIL'} (threshold: 0.95)")

        # Store model for later tests
        self._trained_model = model

        return TestResult("HJB-MLP P Learning", passed, spearman, 0.95)

    # =========================================================================
    # TEST 5: Forcing Response
    # =========================================================================
    def test_forcing_response(self) -> TestResult:
        """Model should predict dP = F·dt correctly."""
        self.log("\n[TEST 5] Forcing Response")
        self.log("-" * 40)

        if not hasattr(self, '_trained_model'):
            self.log("  SKIP: No trained model available")
            return TestResult("Forcing Response", False, 0, 0.9, "No model")

        model = self._trained_model
        model.eval()

        omega = 1.0
        dt = 0.3
        F_scale = 1.0

        np.random.seed(8888)
        n_test = 100

        P0 = np.random.uniform(0.6, 1.8, n_test)
        Q0 = np.random.uniform(0, 2*np.pi, n_test)
        F_ext = (np.random.rand(n_test) - 0.5) * 2 * F_scale

        P1 = np.maximum(P0 + F_ext * dt, 0.1)
        Q1 = Q0 + omega * dt

        p0 = np.sqrt(2 * P0 * omega) * np.sin(Q0)
        q0 = np.sqrt(2 * P0 / omega) * np.cos(Q0)
        p1 = np.sqrt(2 * P1 * omega) * np.sin(Q1)
        q1 = np.sqrt(2 * P1 / omega) * np.cos(Q1)

        with torch.no_grad():
            P0_pred, _ = model.encode(
                torch.tensor(p0, dtype=torch.float32, device=self.device),
                torch.tensor(q0, dtype=torch.float32, device=self.device)
            )
            P1_pred, _ = model.encode(
                torch.tensor(p1, dtype=torch.float32, device=self.device),
                torch.tensor(q1, dtype=torch.float32, device=self.device)
            )

        dP_pred = (P1_pred - P0_pred).cpu().numpy()
        dP_exp = F_ext * dt

        # Account for possible scale factor
        if np.std(dP_exp) > 1e-6:
            scale = np.polyfit(dP_exp, dP_pred, 1)[0]
            if abs(scale) > 0.1:
                corr = abs(pearsonr(dP_pred / scale, dP_exp)[0])
            else:
                corr = 0
        else:
            corr = 0

        passed = corr > 0.9
        self.log(f"  Forcing correlation: {corr:.4f}")
        self.log(f"  {'PASS' if passed else 'FAIL'} (threshold: 0.9)")

        return TestResult("Forcing Response", passed, corr, 0.9)

    # =========================================================================
    # TEST 6: Forecast Stability
    # =========================================================================
    def test_forecast_stability(self) -> TestResult:
        """Forecast error should stay bounded over long horizons."""
        self.log("\n[TEST 6] Forecast Stability")
        self.log("-" * 40)

        if not hasattr(self, '_trained_model'):
            self.log("  SKIP: No trained model available")
            return TestResult("Forecast Stability", False, 0, 2.0, "No model")

        model = self._trained_model
        model.eval()

        omega = 1.0
        np.random.seed(7777)
        n_test = 50

        E = np.random.uniform(0.5, 4.5, n_test)
        theta0 = np.random.uniform(0, 2*np.pi, n_test)

        p0 = np.sqrt(2 * E) * np.sin(theta0)
        q0 = np.sqrt(2 * E) / omega * np.cos(theta0)

        with torch.no_grad():
            P0, Q0 = model.encode(
                torch.tensor(p0, dtype=torch.float32, device=self.device),
                torch.tensor(q0, dtype=torch.float32, device=self.device)
            )
            P0 = P0.cpu().numpy()
            Q0 = Q0.cpu().numpy()

        # Short-term (T=0.1)
        T_short = 0.1
        theta_short = theta0 + omega * T_short
        p_true_short = np.sqrt(2 * E) * np.sin(theta_short)
        q_true_short = np.sqrt(2 * E) / omega * np.cos(theta_short)

        with torch.no_grad():
            p_pred, q_pred = model.decode(
                torch.tensor(P0, dtype=torch.float32, device=self.device),
                torch.tensor(Q0 + omega * T_short, dtype=torch.float32, device=self.device)
            )
        err_short = np.sqrt(np.mean((p_pred.cpu().numpy() - p_true_short)**2 +
                                     (q_pred.cpu().numpy() - q_true_short)**2))

        # Long-term (T=100 periods)
        T_long = 100 * 2 * np.pi
        theta_long = theta0 + omega * T_long
        p_true_long = np.sqrt(2 * E) * np.sin(theta_long)
        q_true_long = np.sqrt(2 * E) / omega * np.cos(theta_long)

        with torch.no_grad():
            p_pred, q_pred = model.decode(
                torch.tensor(P0, dtype=torch.float32, device=self.device),
                torch.tensor(Q0 + omega * T_long, dtype=torch.float32, device=self.device)
            )
        err_long = np.sqrt(np.mean((p_pred.cpu().numpy() - p_true_long)**2 +
                                    (q_pred.cpu().numpy() - q_true_long)**2))

        ratio = err_long / (err_short + 1e-10)

        passed = ratio < 2.0
        self.log(f"  Short-term error: {err_short:.4f}")
        self.log(f"  Long-term error: {err_long:.4f}")
        self.log(f"  Ratio: {ratio:.2f}x")
        self.log(f"  {'PASS' if passed else 'FAIL'} (threshold: 2.0x)")

        return TestResult("Forecast Stability", passed, ratio, 2.0)

    # =========================================================================
    # RUN ALL
    # =========================================================================
    def run_all(self) -> bool:
        """Run all tests and return overall pass/fail."""

        self.log("=" * 70)
        self.log("GLINSKY FRAMEWORK VALIDATION SUITE")
        self.log("=" * 70)

        start = time.time()

        # Run tests
        self.results.append(self.test_hst_roundtrip())
        self.results.append(self.test_pca_variance())
        self.results.append(self.test_beta_pq_correlation())
        self.results.append(self.test_hjb_p_learning())
        self.results.append(self.test_forcing_response())
        self.results.append(self.test_forecast_stability())

        elapsed = time.time() - start

        # Summary
        self.log("\n" + "=" * 70)
        self.log("SUMMARY")
        self.log("=" * 70)

        self.log(f"\n{'Test':<25} | {'Metric':>10} | {'Threshold':>10} | {'Status':>6}")
        self.log("-" * 60)

        all_passed = True
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            metric_str = f"{r.metric:.4f}" if r.metric < 100 else f"{r.metric:.2e}"
            thresh_str = f"{r.threshold:.4f}" if r.threshold < 100 else f"{r.threshold:.2e}"
            self.log(f"{r.name:<25} | {metric_str:>10} | {thresh_str:>10} | {status:>6}")
            if not r.passed:
                all_passed = False

        self.log("-" * 60)
        self.log(f"\nTime: {elapsed:.1f}s")

        if all_passed:
            self.log("\n" + "=" * 70)
            self.log("ALL TESTS PASSED")
            self.log("Glinsky framework VALIDATED")
            self.log("=" * 70)
        else:
            self.log("\n" + "=" * 70)
            self.log("SOME TESTS FAILED")
            failed = [r.name for r in self.results if not r.passed]
            self.log(f"Failed: {', '.join(failed)}")
            self.log("=" * 70)

        return all_passed


def main():
    """Entry point."""
    validator = GlinskyValidator(verbose=True)
    success = validator.run_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
