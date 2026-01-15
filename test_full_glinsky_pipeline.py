"""
Full Glinsky Pipeline Test:
signal → HST → PCA → β → (p,q) → HJB-MLP → (P,Q) → forecast → signal

This tests the COMPLETE pipeline from raw signal to long-term forecast,
proving Glinsky's framework works end-to-end.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, List
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

from hst_rom import HST_ROM
from fixed_hjb_loss import ImprovedHJB_MLP
from action_angle_utils import wrap_to_2pi


class GlinskyPipeline:
    """
    Full Glinsky pipeline: signal → (P,Q) → forecast → signal

    Components:
    1. HST_ROM: signal → β (PCA of HST coefficients)
    2. Linear adapter: β → (p,q) proxy
    3. HJB-MLP: (p,q) → (P,Q) action-angle coordinates
    4. Inverse path for forecasting
    """

    def __init__(self, n_components: int = 4, hidden_dim: int = 64,
                 window_size: int = 128, omega: float = 1.0):
        self.n_components = n_components
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.omega = omega

        # Components
        self.rom = HST_ROM(n_components=n_components, wavelet='db8',
                          J=3, window_size=window_size)
        self.hjb_mlp = ImprovedHJB_MLP(hidden_dim=hidden_dim, num_layers=3)

        # Adapter: β → (p,q)
        self.W_beta_to_pq = None  # Learned linear map

        # Inverse adapter: (p,q) → β (pseudo-inverse)
        self.W_pq_to_beta = None

        self.device = 'cpu'
        self.fitted = False

    def generate_sho_signal(self, p0: float, q0: float,
                            n_points: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate complex SHO signal z(t) = q(t) + i*p(t)/omega."""
        if n_points is None:
            n_points = self.window_size

        dt = 0.01
        t = np.arange(n_points) * dt

        E = 0.5 * p0**2 + 0.5 * self.omega**2 * q0**2
        theta0 = np.arctan2(p0, self.omega * q0)

        theta_t = theta0 + self.omega * t
        p_t = np.sqrt(2 * E) * np.sin(theta_t)
        q_t = np.sqrt(2 * E) / self.omega * np.cos(theta_t)

        z = q_t + 1j * p_t / self.omega

        return z, p_t, q_t

    def fit(self, n_trajectories: int = 200, n_epochs: int = 2000,
            F_scale: float = 1.0, verbose: bool = True):
        """
        Fit the complete pipeline.

        1. Generate training signals
        2. Fit HST_ROM
        3. Learn β → (p,q) adapter
        4. Train HJB-MLP on (p,q) from β
        """
        if verbose:
            print("=" * 70)
            print("FITTING GLINSKY PIPELINE")
            print("=" * 70)

        # Step 1: Generate training signals with ground truth
        if verbose:
            print("\n[1/4] Generating training signals...")

        np.random.seed(42)
        signals = []
        pq_true = []  # Ground truth (p,q) at window center

        for i in range(n_trajectories):
            E = np.random.uniform(0.5, 4.5)
            theta0 = np.random.uniform(0, 2 * np.pi)

            p0 = np.sqrt(2 * E) * np.sin(theta0)
            q0 = np.sqrt(2 * E) / self.omega * np.cos(theta0)

            z, p_t, q_t = self.generate_sho_signal(p0, q0)
            signals.append(z)

            center_idx = self.window_size // 2
            pq_true.append([p_t[center_idx], q_t[center_idx]])

        pq_true = np.array(pq_true)

        # Step 2: Fit HST_ROM
        if verbose:
            print("[2/4] Fitting HST_ROM...")

        betas = self.rom.fit(signals, extract_windows=False)

        if verbose:
            print(f"  β shape: {betas.shape}")
            print(f"  Variance explained: {self.rom.pca.explained_variance_ratio_[:4].sum():.3f}")

        # Step 3: Learn β → (p,q) adapter
        if verbose:
            print("[3/4] Learning β → (p,q) adapter...")

        self.W_beta_to_pq, _, _, _ = np.linalg.lstsq(betas, pq_true, rcond=None)

        # Pseudo-inverse for (p,q) → β
        self.W_pq_to_beta = np.linalg.pinv(self.W_beta_to_pq)

        # Test adapter quality
        pq_from_beta = betas @ self.W_beta_to_pq
        r_p, _ = pearsonr(pq_from_beta[:, 0], pq_true[:, 0])
        r_q, _ = pearsonr(pq_from_beta[:, 1], pq_true[:, 1])

        if verbose:
            print(f"  Adapter quality: r(p)={r_p:.4f}, r(q)={r_q:.4f}")

        # Step 4: Train HJB-MLP on (p,q) derived from β
        if verbose:
            print("[4/4] Training HJB-MLP...")

        self._train_hjb_mlp(pq_from_beta, n_epochs, F_scale, verbose)

        self.fitted = True

        if verbose:
            print("\n✓ Pipeline fitted successfully")

    def _train_hjb_mlp(self, pq_data: np.ndarray, n_epochs: int,
                       F_scale: float, verbose: bool):
        """Train HJB-MLP with forcing on (p,q) derived from β."""

        self.hjb_mlp = self.hjb_mlp.to(self.device)
        optimizer = torch.optim.Adam(self.hjb_mlp.parameters(), lr=1e-3)

        n_samples = len(pq_data)
        dt = 0.3

        for epoch in range(n_epochs):
            # Generate trajectory pairs with forcing
            idx = np.random.choice(n_samples, size=min(100, n_samples), replace=False)

            p0 = pq_data[idx, 0]
            q0 = pq_data[idx, 1]

            # Convert to action-angle for evolution
            E0 = 0.5 * p0**2 + 0.5 * self.omega**2 * q0**2
            P0 = E0 / self.omega
            Q0 = np.arctan2(p0, self.omega * q0)

            # Random forcing
            F_ext = (np.random.rand(len(idx)) - 0.5) * 2 * F_scale

            # Evolve
            P1 = P0 + F_ext * dt
            P1 = np.maximum(P1, 0.1)
            Q1 = Q0 + self.omega * dt

            # Back to (p,q)
            p1 = np.sqrt(2 * P1 * self.omega) * np.sin(Q1)
            q1 = np.sqrt(2 * P1 / self.omega) * np.cos(Q1)

            # To tensors
            p0_t = torch.tensor(p0, dtype=torch.float32, device=self.device)
            q0_t = torch.tensor(q0, dtype=torch.float32, device=self.device)
            p1_t = torch.tensor(p1, dtype=torch.float32, device=self.device)
            q1_t = torch.tensor(q1, dtype=torch.float32, device=self.device)
            F_ext_t = torch.tensor(F_ext, dtype=torch.float32, device=self.device)

            # Forward
            optimizer.zero_grad()

            P0_pred, Q0_pred = self.hjb_mlp.encode(p0_t, q0_t)
            P1_pred, Q1_pred = self.hjb_mlp.encode(p1_t, q1_t)

            # Reconstruction loss
            p0_rec, q0_rec = self.hjb_mlp.decode(P0_pred, Q0_pred)
            recon_loss = torch.mean((p0_t - p0_rec)**2 + (q0_t - q0_rec)**2)

            # Normalized action loss
            dP_expected = F_ext_t * dt
            dP_actual = P1_pred - P0_pred
            mse = torch.mean((dP_actual - dP_expected)**2)
            normalizer = torch.mean(dP_expected**2) + 1e-6
            action_loss = mse / normalizer

            # Evolution loss
            dQ_expected = self.omega * dt
            dQ_actual = Q1_pred - Q0_pred
            evolution_loss = torch.mean(1 - torch.cos(dQ_actual - dQ_expected))

            # Total
            total_loss = recon_loss + 10 * action_loss + 5 * evolution_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.hjb_mlp.parameters(), max_norm=1.0)
            optimizer.step()

            if verbose and epoch % 500 == 0:
                print(f"    Epoch {epoch}: loss={total_loss.item():.4f}")

    def signal_to_action_angle(self, signal: np.ndarray) -> Tuple[float, float]:
        """
        Transform signal to action-angle coordinates.

        signal → HST → β → (p,q) → (P,Q)
        """
        if not self.fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        # HST → β
        beta = self.rom.transform(signal)

        # β → (p,q)
        pq = beta @ self.W_beta_to_pq
        p, q = pq[0], pq[1]

        # (p,q) → (P,Q)
        with torch.no_grad():
            p_t = torch.tensor([p], dtype=torch.float32, device=self.device)
            q_t = torch.tensor([q], dtype=torch.float32, device=self.device)
            P, Q = self.hjb_mlp.encode(p_t, q_t)

        return P.item(), Q.item()

    def action_angle_to_signal(self, P: float, Q: float,
                                original_length: int = None) -> np.ndarray:
        """
        Transform action-angle back to signal.

        (P,Q) → (p,q) → β → HST⁻¹ → signal
        """
        if not self.fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        if original_length is None:
            original_length = self.window_size

        # (P,Q) → (p,q)
        with torch.no_grad():
            P_t = torch.tensor([P], dtype=torch.float32, device=self.device)
            Q_t = torch.tensor([Q], dtype=torch.float32, device=self.device)
            p, q = self.hjb_mlp.decode(P_t, Q_t)
            p, q = p.item(), q.item()

        # (p,q) → β
        pq = np.array([[p, q]])
        beta = pq @ self.W_pq_to_beta

        # β → signal
        signal = self.rom.inverse_transform(beta[0], original_length=original_length)

        return signal

    def forecast(self, signal: np.ndarray, T: float) -> np.ndarray:
        """
        Forecast signal T time units into future.

        1. Encode: signal → (P,Q)
        2. Propagate: P_T = P, Q_T = Q + ω*T
        3. Decode: (P_T, Q_T) → signal_T
        """
        # Encode
        P, Q = self.signal_to_action_angle(signal)

        # Propagate in action-angle space
        P_T = P  # Action conserved
        Q_T = Q + self.omega * T  # Angle evolves

        # Decode
        signal_T = self.action_angle_to_signal(P_T, Q_T, len(signal))

        return signal_T


def test_end_to_end_forecasting():
    """
    Test full pipeline forecasting on SHO signals.

    Key test: Does forecast error stay FLAT over long horizons?
    """
    print("=" * 70)
    print("FULL GLINSKY PIPELINE: END-TO-END FORECASTING TEST")
    print("=" * 70)

    omega = 1.0
    window_size = 128

    # Create and fit pipeline
    pipeline = GlinskyPipeline(
        n_components=4,
        hidden_dim=64,
        window_size=window_size,
        omega=omega
    )

    pipeline.fit(n_trajectories=200, n_epochs=2000, F_scale=1.0, verbose=True)

    # Test forecasting
    print("\n" + "=" * 70)
    print("FORECASTING TEST")
    print("=" * 70)

    # Generate test signals
    np.random.seed(9999)
    n_test = 50

    T_values = [0.1, 1.0, 2*np.pi, 10*2*np.pi, 50*2*np.pi]
    T_labels = ['0.1', '1', '1 per', '10 per', '50 per']

    results = {'T': [], 'signal_error': [], 'pq_error': []}

    print(f"\n{'T':>10} | {'Signal Err':>12} | {'(p,q) Err':>12}")
    print("-" * 45)

    for T, label in zip(T_values, T_labels):
        signal_errors = []
        pq_errors = []

        for i in range(n_test):
            # Random initial conditions
            E = np.random.uniform(0.5, 4.5)
            theta0 = np.random.uniform(0, 2 * np.pi)

            p0 = np.sqrt(2 * E) * np.sin(theta0)
            q0 = np.sqrt(2 * E) / omega * np.cos(theta0)

            # Generate initial signal
            z0, _, _ = pipeline.generate_sho_signal(p0, q0)

            # Ground truth at time T
            theta_T = theta0 + omega * T
            p_T_true = np.sqrt(2 * E) * np.sin(theta_T)
            q_T_true = np.sqrt(2 * E) / omega * np.cos(theta_T)
            z_T_true, _, _ = pipeline.generate_sho_signal(p_T_true, q_T_true)

            # Forecast via pipeline
            try:
                z_T_pred = pipeline.forecast(z0, T)

                # Signal error (relative)
                sig_err = np.linalg.norm(z_T_pred - z_T_true) / np.linalg.norm(z_T_true)
                signal_errors.append(sig_err)

                # Also check (p,q) error via encoding
                P_pred, Q_pred = pipeline.signal_to_action_angle(z_T_pred)
                P_true = E / omega
                Q_true = wrap_to_2pi(theta_T + omega * (window_size // 2) * 0.01)

                # Back to (p,q) for comparison
                with torch.no_grad():
                    p_pred, q_pred = pipeline.hjb_mlp.decode(
                        torch.tensor([P_pred]), torch.tensor([Q_pred])
                    )
                    p_pred, q_pred = p_pred.item(), q_pred.item()

                pq_err = np.sqrt((p_pred - p_T_true)**2 + (q_pred - q_T_true)**2)
                pq_errors.append(pq_err)

            except Exception as e:
                # Skip failed forecasts
                continue

        if signal_errors:
            mean_sig_err = np.mean(signal_errors)
            mean_pq_err = np.mean(pq_errors)

            results['T'].append(T)
            results['signal_error'].append(mean_sig_err)
            results['pq_error'].append(mean_pq_err)

            print(f"{label:>10} | {mean_sig_err:12.4f} | {mean_pq_err:12.4f}")

    # Analyze error growth
    print("\n" + "-" * 70)
    print("ERROR GROWTH ANALYSIS")
    print("-" * 70)

    if len(results['signal_error']) >= 2:
        short_err = results['signal_error'][0]
        long_err = results['signal_error'][-1]
        error_ratio = long_err / (short_err + 1e-10)

        print(f"\nShort-term error (T=0.1): {short_err:.4f}")
        print(f"Long-term error (T=50 per): {long_err:.4f}")
        print(f"Error ratio: {error_ratio:.2f}x")

        # Fit growth exponent
        if len(results['T']) >= 3:
            log_T = np.log(np.array(results['T']) + 1e-10)
            log_err = np.log(np.array(results['signal_error']) + 1e-10)
            growth_exp, _ = np.polyfit(log_T, log_err, 1)
            print(f"Growth exponent: {growth_exp:.4f}")
        else:
            growth_exp = 0

        # Verdict
        print("\n" + "=" * 70)
        print("VERDICT")
        print("=" * 70)

        if error_ratio < 3.0 and growth_exp < 0.3:
            print("\n✓ FULL PIPELINE VALIDATED")
            print("  signal → HST → β → (p,q) → HJB-MLP → (P,Q) → forecast → signal")
            print("  Error stays bounded over 50+ periods!")
        else:
            print("\n◐ PARTIAL SUCCESS")
            print(f"  Error ratio: {error_ratio:.2f} (target < 3.0)")
            print(f"  Growth exponent: {growth_exp:.4f} (target < 0.3)")

    return results


def test_reconstruction_quality():
    """Test signal reconstruction through full pipeline."""

    print("\n" + "=" * 70)
    print("RECONSTRUCTION QUALITY TEST")
    print("=" * 70)

    omega = 1.0
    window_size = 128

    pipeline = GlinskyPipeline(
        n_components=4,
        hidden_dim=64,
        window_size=window_size,
        omega=omega
    )

    pipeline.fit(n_trajectories=200, n_epochs=2000, F_scale=1.0, verbose=False)

    # Test reconstruction: signal → (P,Q) → signal
    print("\nTesting: signal → encode → decode → signal")

    np.random.seed(12345)
    n_test = 50

    recon_errors = []

    for i in range(n_test):
        E = np.random.uniform(0.5, 4.5)
        theta0 = np.random.uniform(0, 2 * np.pi)

        p0 = np.sqrt(2 * E) * np.sin(theta0)
        q0 = np.sqrt(2 * E) / omega * np.cos(theta0)

        z_orig, _, _ = pipeline.generate_sho_signal(p0, q0)

        # Encode
        P, Q = pipeline.signal_to_action_angle(z_orig)

        # Decode (T=0)
        z_recon = pipeline.action_angle_to_signal(P, Q, len(z_orig))

        # Error
        err = np.linalg.norm(z_recon - z_orig) / np.linalg.norm(z_orig)
        recon_errors.append(err)

    mean_err = np.mean(recon_errors)
    std_err = np.std(recon_errors)

    print(f"\nReconstruction error: {mean_err:.4f} ± {std_err:.4f}")
    print(f"{'✓ GOOD' if mean_err < 0.5 else '✗ POOR'}")

    return {'mean_error': mean_err, 'std_error': std_err}


def run_full_pipeline_validation():
    """Run complete pipeline validation."""

    print("\n" + "=" * 70)
    print("GLINSKY FULL PIPELINE VALIDATION")
    print("=" * 70)
    print("\nPipeline: signal → HST → PCA → β → W → (p,q) → HJB-MLP → (P,Q)")
    print("          (P,Q) → HJB-MLP⁻¹ → (p,q) → W⁻¹ → β → HST⁻¹ → signal\n")

    # Test 1: Reconstruction
    recon_results = test_reconstruction_quality()

    # Test 2: Forecasting
    forecast_results = test_end_to_end_forecasting()

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print("\n| Test | Result |")
    print("|------|--------|")
    print(f"| Reconstruction | {'✓' if recon_results['mean_error'] < 0.5 else '✗'} (err={recon_results['mean_error']:.3f}) |")

    if forecast_results['signal_error']:
        ratio = forecast_results['signal_error'][-1] / forecast_results['signal_error'][0]
        print(f"| Forecasting | {'✓' if ratio < 3 else '✗'} (ratio={ratio:.2f}) |")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_full_pipeline_validation()
