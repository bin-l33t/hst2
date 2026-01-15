"""
Ablation study: Which features are necessary for the nullspace decoder?

Tests 5 configurations to determine what β_⊥ actually depends on:
- A: Full (8 features) - baseline
- B: Action-angle only (3 features)
- C: Phase space only (2 features)
- D: Phase space + quadratic (5 features)
- E: Action only (1 feature) - expected to fail
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from hst_rom import HST_ROM
from fixed_hjb_loss import ImprovedHJB_MLP, train_on_sho_fixed


class AblatedNullspaceDecoder(nn.Module):
    """
    Nullspace decoder with configurable input features.

    Allows testing which features g_θ actually needs.
    """

    CONFIGS = {
        'full': 8,           # p, q, p², q², pq, P, sin(Q), cos(Q)
        'action_angle': 3,   # P, sin(Q), cos(Q)
        'phase_space': 2,    # p, q
        'phase_quad': 5,     # p, q, p², q², pq
        'action_only': 1,    # P
    }

    def __init__(self, W_beta_to_pq, hjb_encoder, feature_config='full', hidden_dim=64):
        super().__init__()

        self.feature_config = feature_config
        n_features = self.CONFIGS[feature_config]
        n_beta = W_beta_to_pq.shape[0]

        # Register buffers
        self.register_buffer('W', torch.tensor(W_beta_to_pq, dtype=torch.float32))
        W_pinv = np.linalg.pinv(W_beta_to_pq)
        self.register_buffer('W_pinv', torch.tensor(W_pinv, dtype=torch.float32))

        # Nullspace projector
        P_parallel = W_beta_to_pq @ W_pinv
        P_perp = np.eye(n_beta) - P_parallel
        self.register_buffer('P_perp', torch.tensor(P_perp, dtype=torch.float32))

        # Frozen HJB encoder
        self.hjb_encoder = hjb_encoder
        for param in self.hjb_encoder.parameters():
            param.requires_grad = False

        self.n_beta = n_beta

        # Correction MLP with configurable input size
        self.g = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_beta)
        )

        # Small initialization
        for m in self.g.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def _get_features(self, p, q, P, Q):
        """Get features based on configuration."""
        if self.feature_config == 'full':
            return torch.stack([
                p, q, p**2, q**2, p*q,
                P, torch.sin(Q), torch.cos(Q)
            ], dim=-1)
        elif self.feature_config == 'action_angle':
            return torch.stack([P, torch.sin(Q), torch.cos(Q)], dim=-1)
        elif self.feature_config == 'phase_space':
            return torch.stack([p, q], dim=-1)
        elif self.feature_config == 'phase_quad':
            return torch.stack([p, q, p**2, q**2, p*q], dim=-1)
        elif self.feature_config == 'action_only':
            return P.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown config: {self.feature_config}")

    def forward(self, p, q):
        if p.dim() == 0:
            p = p.unsqueeze(0)
            q = q.unsqueeze(0)

        pq = torch.stack([p, q], dim=-1)
        beta_linear = pq @ self.W_pinv

        with torch.no_grad():
            P, Q = self.hjb_encoder.encode(p, q)

        features = self._get_features(p, q, P, Q)
        g_output = self.g(features)
        beta_correction = g_output @ self.P_perp.T

        return beta_linear + beta_correction


def train_ablated_decoder(decoder, p_train, q_train, beta_train, epochs=1000, lr=1e-3):
    """Train decoder and return final loss."""
    p_train = torch.tensor(p_train, dtype=torch.float32)
    q_train = torch.tensor(q_train, dtype=torch.float32)
    beta_train = torch.tensor(beta_train, dtype=torch.float32)

    optimizer = torch.optim.Adam(decoder.g.parameters(), lr=lr)

    final_loss = None
    for epoch in range(epochs):
        decoder.train()
        beta_pred = decoder(p_train, q_train)
        loss = F.mse_loss(beta_pred, beta_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        final_loss = loss.item()

    return final_loss


def generate_sho_signal(p0, q0, omega=1.0, dt=0.01, n_points=128):
    """Generate SHO signal."""
    t = np.arange(n_points) * dt
    E = 0.5 * p0**2 + 0.5 * omega**2 * q0**2
    theta0 = np.arctan2(p0, omega * q0)
    theta_t = theta0 + omega * t
    p_t = np.sqrt(2 * E) * np.sin(theta_t)
    q_t = np.sqrt(2 * E) / omega * np.cos(theta_t)
    z = q_t + 1j * p_t / omega
    return z, p_t, q_t


def run_ablation():
    """Run full ablation study."""
    print("=" * 70)
    print("NULLSPACE DECODER ABLATION STUDY")
    print("=" * 70)

    np.random.seed(42)
    torch.manual_seed(42)

    # Parameters
    omega = 1.0
    n_samples = 300
    window_size = 128
    n_pca = 8
    dt = 0.01

    # Train/test split
    n_train = int(0.8 * n_samples)
    all_idx = np.arange(n_samples)
    np.random.shuffle(all_idx)
    train_idx = all_idx[:n_train]
    test_idx = all_idx[n_train:]

    print(f"\nData split: {n_train} train, {len(test_idx)} test")

    # Generate data
    print("Generating data...")
    E_samples = np.random.uniform(0.5, 2.0, n_samples)
    phase_samples = np.random.uniform(0, 2 * np.pi, n_samples)

    p0_arr = np.sqrt(2 * E_samples) * np.sin(phase_samples)
    q0_arr = np.sqrt(2 * E_samples) / omega * np.cos(phase_samples)

    signals = []
    p_true = []
    q_true = []
    center_idx = window_size // 2

    for i in range(n_samples):
        z, p_t, q_t = generate_sho_signal(p0_arr[i], q0_arr[i], omega=omega,
                                          dt=dt, n_points=window_size)
        signals.append(z)
        p_true.append(p_t[center_idx])
        q_true.append(q_t[center_idx])

    signals = np.array(signals)
    p_true = np.array(p_true)
    q_true = np.array(q_true)

    # Fit HST_ROM on train
    print("Fitting HST_ROM...")
    hst_rom = HST_ROM(n_components=n_pca, J=3, window_size=window_size)
    train_signals = [signals[i] for i in train_idx]
    beta_train = hst_rom.fit(train_signals, extract_windows=False)
    beta_test = np.array([hst_rom.transform(signals[i]) for i in test_idx])

    # Baseline error
    signals_baseline = []
    for i, idx in enumerate(test_idx):
        z_rec = hst_rom.inverse_transform(beta_test[i])
        signals_baseline.append(z_rec)
    signals_baseline = np.array(signals_baseline)
    baseline_error = np.mean(np.abs(signals[test_idx] - signals_baseline))
    print(f"Baseline error (TEST): {baseline_error:.4f}")

    # Learn W on train
    pq_train = np.stack([p_true[train_idx], q_true[train_idx]], axis=1)
    W_beta_to_pq, _, _, _ = np.linalg.lstsq(beta_train, pq_train, rcond=None)

    # Train HJB encoder
    print("Training HJB encoder...")
    hjb_encoder = ImprovedHJB_MLP(hidden_dim=64, num_layers=3)
    train_on_sho_fixed(hjb_encoder, n_epochs=500, omega=omega, dt=0.5, lr=1e-3, device='cpu')

    # Test each configuration
    configs = ['full', 'action_angle', 'phase_space', 'phase_quad', 'action_only']
    results = {}

    print("\n" + "=" * 70)
    print("RUNNING ABLATION EXPERIMENTS")
    print("=" * 70)

    for config in configs:
        print(f"\n--- Config: {config} ({AblatedNullspaceDecoder.CONFIGS[config]} features) ---")

        # Create and train decoder
        decoder = AblatedNullspaceDecoder(W_beta_to_pq, hjb_encoder,
                                          feature_config=config, hidden_dim=64)

        final_loss = train_ablated_decoder(
            decoder, p_true[train_idx], q_true[train_idx], beta_train,
            epochs=1000, lr=1e-3
        )
        print(f"  Final loss: {final_loss:.4f}")

        # Evaluate on test
        decoder.eval()
        p_test_t = torch.tensor(p_true[test_idx], dtype=torch.float32)
        q_test_t = torch.tensor(q_true[test_idx], dtype=torch.float32)

        with torch.no_grad():
            beta_pred = decoder(p_test_t, q_test_t).numpy()

        # Roundtrip error
        signals_decoded = []
        for i in range(len(beta_pred)):
            z_rec = hst_rom.inverse_transform(beta_pred[i])
            signals_decoded.append(z_rec)
        signals_decoded = np.array(signals_decoded)
        roundtrip_error = np.mean(np.abs(signals[test_idx] - signals_decoded))
        roundtrip_ratio = roundtrip_error / baseline_error
        print(f"  Roundtrip/baseline: {roundtrip_ratio:.2f}x")

        # Forecast test
        forecast_errors = {}
        for T_periods in [0.1, 100.0]:
            T = T_periods * 2 * np.pi / omega
            errors = []

            for idx in test_idx:
                p0, q0 = p_true[idx], q_true[idx]

                with torch.no_grad():
                    p0_t = torch.tensor([p0], dtype=torch.float32)
                    q0_t = torch.tensor([q0], dtype=torch.float32)
                    P0, Q0 = hjb_encoder.encode(p0_t, q0_t)

                    Q_T = Q0 + omega * T
                    p_T, q_T = hjb_encoder.decode(P0, Q_T)
                    beta_T = decoder(p_T, q_T).numpy()[0]

                signal_pred = hst_rom.inverse_transform(beta_T)

                E = 0.5 * p0**2 + 0.5 * omega**2 * q0**2
                theta0 = np.arctan2(p0, omega * q0)
                theta_T = theta0 + omega * T
                p_T_true = np.sqrt(2 * E) * np.sin(theta_T)
                q_T_true = np.sqrt(2 * E) / omega * np.cos(theta_T)

                signal_true, _, _ = generate_sho_signal(p_T_true, q_T_true, omega=omega,
                                                        dt=dt, n_points=window_size)

                errors.append(np.mean(np.abs(signal_true - signal_pred)))

            forecast_errors[T_periods] = np.mean(errors)

        forecast_ratio = forecast_errors[100.0] / forecast_errors[0.1]
        print(f"  Forecast ratio: {forecast_ratio:.2f}x")

        results[config] = {
            'n_features': AblatedNullspaceDecoder.CONFIGS[config],
            'final_loss': final_loss,
            'roundtrip_ratio': roundtrip_ratio,
            'forecast_ratio': forecast_ratio,
            'passed': forecast_ratio < 2.0 and roundtrip_ratio < 3.0
        }

    # Summary table
    print("\n" + "=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)

    print(f"\n{'Config':<15} | {'Features':<8} | {'Loss':<8} | {'RT/Base':<10} | {'Forecast':<10} | {'Status':<8}")
    print("-" * 75)

    for config, r in results.items():
        status = "PASS" if r['passed'] else "FAIL"
        print(f"{config:<15} | {r['n_features']:<8} | {r['final_loss']:<8.4f} | "
              f"{r['roundtrip_ratio']:<10.2f}x | {r['forecast_ratio']:<10.2f}x | {status:<8}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Find minimal sufficient config
    passing_configs = [c for c, r in results.items() if r['passed']]
    if passing_configs:
        min_config = min(passing_configs, key=lambda c: results[c]['n_features'])
        print(f"\nMinimal sufficient config: {min_config} ({results[min_config]['n_features']} features)")

    # Check action_only failure
    if not results['action_only']['passed']:
        print("\naction_only FAILED as expected → β_⊥ depends on Q (phase), not just P")
    else:
        print("\nSURPRISE: action_only passed! β_⊥ might only depend on action P")

    # Check action_angle vs phase_space
    if results['action_angle']['passed'] and results['phase_space']['passed']:
        print("\nBoth action_angle and phase_space work → redundant representations")
        if results['action_angle']['forecast_ratio'] < results['phase_space']['forecast_ratio']:
            print("  action_angle is slightly better → cleaner physical interpretation")
        else:
            print("  phase_space is slightly better → direct coordinates sufficient")
    elif results['action_angle']['passed']:
        print("\nOnly action_angle works → β_⊥ = f(P, Q) is the natural representation")
    elif results['phase_space']['passed']:
        print("\nOnly phase_space works → β_⊥ = f(p, q) without needing action-angle")

    # Overall conclusion
    print("\n" + "=" * 70)
    n_passing = len(passing_configs)
    if n_passing > 0:
        print(f"CONCLUSION: {n_passing}/{len(configs)} configurations work")
        print(f"Minimal feature set: {min_config}")
    else:
        print("CONCLUSION: Only full feature set works")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_ablation()

    # Exit with success if at least action_only fails (proves Q-dependence)
    # and at least one other config passes
    passing = [c for c, r in results.items() if r['passed']]
    action_only_failed = not results['action_only']['passed']

    success = action_only_failed and len(passing) > 0
    sys.exit(0 if success else 1)
