"""
Critical Test: Can HJB-MLP Learn True Action?

The key question:
- Raw HST coefficients do NOT capture action (r ≈ 0 in rigorous_tests.py)
- HST DOES capture angle (r = 0.9999)
- Can the HJB-MLP stage LEARN action from HST features?

This test uses SHO where true action I = E/ω₀ is analytically known.

Success criterion: r(P_learned, I_true) > 0.95
"""

import numpy as np
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim

from hamiltonian_systems import (
    SimpleHarmonicOscillator, AnharmonicOscillator, PendulumOscillator,
    simulate_hamiltonian
)
from hst import hst_forward_pywt


class HJB_ActionMLP(nn.Module):
    """
    MLP that learns to extract action from HST coefficients.

    Input: Flattened HST coefficients
    Output: (P, omega) where P should correspond to true action I
    """

    def __init__(self, input_dim, hidden_dims=[128, 64, 32]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        # Output: P (action) and omega (frequency)
        layers.append(nn.Linear(prev_dim, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = self.network(x)
        P = out[:, 0:1]      # Action (should be constant per trajectory)
        omega = out[:, 1:2]  # Frequency (should depend on P for non-degenerate systems)
        return P, omega


def extract_hst_features(z, J=3, wavelet='db8'):
    """Extract flattened HST coefficients as feature vector."""
    coeffs = hst_forward_pywt(z.real, J=J, wavelet_name=wavelet)

    # Flatten all coefficients into a single feature vector
    features = []
    for c in coeffs['cD']:
        # Use statistics of each level (mean, std, energy)
        features.extend([
            np.mean(np.abs(c)),
            np.std(np.abs(c)),
            np.mean(np.abs(c)**2),  # Energy
        ])

    # Final approximation
    ca = coeffs['cA_final']
    features.extend([
        np.mean(np.abs(ca)),
        np.std(np.abs(ca)),
        np.mean(np.abs(ca)**2),
    ])

    return np.array(features)


def generate_sho_dataset(n_trajectories=50, T=50, dt=0.01):
    """
    Generate SHO trajectories at different energies.

    Returns HST features, true action I, and true frequency omega.
    """
    sho = SimpleHarmonicOscillator(omega0=1.0)

    energies = np.linspace(0.5, 5.0, n_trajectories)

    features_list = []
    I_true_list = []
    omega_true_list = []

    for E in energies:
        q0, p0 = sho.initial_condition_for_energy(E)
        t, q, p, z, E_actual = simulate_hamiltonian(sho, q0, p0, T=T, dt=dt)

        # True action for SHO: I = E/ω₀
        I_true = E_actual / sho.omega0
        omega_true = sho.omega0  # Constant for SHO (degenerate)

        # Extract HST features
        features = extract_hst_features(z)

        features_list.append(features)
        I_true_list.append(I_true)
        omega_true_list.append(omega_true)

    return (np.array(features_list),
            np.array(I_true_list),
            np.array(omega_true_list))


def generate_duffing_dataset(n_trajectories=50, T=50, dt=0.01):
    """
    Generate Duffing trajectories at different energies.

    Duffing is NON-degenerate: ω depends on E.
    """
    duffing = AnharmonicOscillator(epsilon=0.3)

    energies = np.linspace(0.5, 3.0, n_trajectories)

    features_list = []
    E_list = []
    omega_list = []

    for E in energies:
        q0, p0 = duffing.initial_condition_for_energy(E)
        t, q, p, z, E_actual = simulate_hamiltonian(duffing, q0, p0, T=T, dt=dt)

        # Measure omega from zero crossings
        zero_crossings = np.where((q[:-1] < 0) & (q[1:] >= 0))[0]
        if len(zero_crossings) >= 2:
            periods = np.diff(t[zero_crossings])
            omega = 2 * np.pi / np.mean(periods)
        else:
            omega = 1.0  # Fallback

        features = extract_hst_features(z)

        features_list.append(features)
        E_list.append(E_actual)
        omega_list.append(omega)

    return (np.array(features_list),
            np.array(E_list),
            np.array(omega_list))


def train_action_mlp(features, I_true, omega_true, n_epochs=1000, lr=1e-3, verbose=True):
    """
    Train MLP to predict action from HST features.

    Loss: MSE on action prediction (I_true) + MSE on omega prediction
    """
    input_dim = features.shape[1]
    model = HJB_ActionMLP(input_dim)

    # Convert to tensors
    X = torch.tensor(features, dtype=torch.float32)
    y_I = torch.tensor(I_true.reshape(-1, 1), dtype=torch.float32)
    y_omega = torch.tensor(omega_true.reshape(-1, 1), dtype=torch.float32)

    # Normalize targets
    I_mean, I_std = y_I.mean(), y_I.std()
    omega_mean, omega_std = y_omega.mean(), y_omega.std()

    y_I_norm = (y_I - I_mean) / (I_std + 1e-8)
    y_omega_norm = (y_omega - omega_mean) / (omega_std + 1e-8)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100)

    best_loss = float('inf')

    for epoch in range(n_epochs):
        model.train()

        P_pred, omega_pred = model(X)

        # Loss: predict normalized action and frequency
        loss_P = nn.MSELoss()(P_pred, y_I_norm)
        loss_omega = nn.MSELoss()(omega_pred, y_omega_norm)
        loss = loss_P + 0.5 * loss_omega

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if loss.item() < best_loss:
            best_loss = loss.item()

        if verbose and epoch % 200 == 0:
            print(f"Epoch {epoch:4d}: loss = {loss.item():.6f} (P: {loss_P.item():.6f}, ω: {loss_omega.item():.6f})")

    # Get final predictions (denormalized)
    model.eval()
    with torch.no_grad():
        P_pred, omega_pred = model(X)
        P_final = P_pred * I_std + I_mean
        omega_final = omega_pred * omega_std + omega_mean

    return model, P_final.numpy().flatten(), omega_final.numpy().flatten()


def test_sho_action_recovery():
    """
    Critical test: Can MLP learn true action for SHO?

    Pass: r(P_learned, I_true) > 0.95
    """
    print("\n" + "="*60)
    print("TEST: SHO Action Recovery via HJB-MLP")
    print("="*60)

    # Generate data
    print("Generating SHO trajectories...")
    features, I_true, omega_true = generate_sho_dataset(n_trajectories=50, T=50)

    print(f"Feature dimension: {features.shape[1]}")
    print(f"I_true range: [{I_true.min():.2f}, {I_true.max():.2f}]")

    # Split train/test
    n_train = 40
    X_train, X_test = features[:n_train], features[n_train:]
    I_train, I_test = I_true[:n_train], I_true[n_train:]
    omega_train, omega_test = omega_true[:n_train], omega_true[n_train:]

    # Train MLP
    print("\nTraining HJB-MLP...")
    model, P_pred_train, omega_pred_train = train_action_mlp(
        X_train, I_train, omega_train, n_epochs=2000, verbose=True
    )

    # Evaluate on training set
    r_train, _ = pearsonr(P_pred_train, I_train)
    print(f"\nTraining set: r(P, I) = {r_train:.4f}")

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        P_pred_test, omega_pred_test = model(X_test_t)

        # Denormalize (use training stats)
        I_mean, I_std = I_train.mean(), I_train.std()
        P_pred_test = (P_pred_test.numpy().flatten() * I_std + I_mean)

    r_test, p_val = pearsonr(P_pred_test, I_test)

    print(f"\nTest set results:")
    print(f"  r(P_learned, I_true) = {r_test:.4f}")
    print(f"  p-value = {p_val:.2e}")
    print(f"  I_true range: [{I_test.min():.2f}, {I_test.max():.2f}]")
    print(f"  P_pred range: [{P_pred_test.min():.2f}, {P_pred_test.max():.2f}]")

    # Verdict
    if r_test > 0.95:
        print(f"\nStatus: \033[92mPASS\033[0m (r > 0.95)")
        status = 'PASS'
    elif r_test > 0.80:
        print(f"\nStatus: \033[93mMARGINAL\033[0m (0.80 < r < 0.95)")
        status = 'MARGINAL'
    else:
        print(f"\nStatus: \033[91mFAIL\033[0m (r < 0.80)")
        status = 'FAIL'

    return status, r_test


def test_duffing_action_recovery():
    """
    Test on Duffing: Non-degenerate system where ω depends on E.
    """
    print("\n" + "="*60)
    print("TEST: Duffing Action Recovery via HJB-MLP")
    print("="*60)

    # Generate data
    print("Generating Duffing trajectories...")
    features, E_true, omega_true = generate_duffing_dataset(n_trajectories=50, T=100)

    print(f"Feature dimension: {features.shape[1]}")
    print(f"E range: [{E_true.min():.2f}, {E_true.max():.2f}]")
    print(f"ω range: [{omega_true.min():.3f}, {omega_true.max():.3f}]")

    # Split train/test
    n_train = 40
    X_train, X_test = features[:n_train], features[n_train:]
    E_train, E_test = E_true[:n_train], E_true[n_train:]
    omega_train, omega_test = omega_true[:n_train], omega_true[n_train:]

    # Train MLP
    print("\nTraining HJB-MLP...")
    model, P_pred_train, omega_pred_train = train_action_mlp(
        X_train, E_train, omega_train, n_epochs=2000, verbose=True
    )

    # Evaluate
    r_P_train, _ = pearsonr(P_pred_train, E_train)
    r_omega_train, _ = pearsonr(omega_pred_train, omega_train)
    print(f"\nTraining: r(P, E) = {r_P_train:.4f}, r(ω_pred, ω_true) = {r_omega_train:.4f}")

    # Test set
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        P_pred_test, omega_pred_test = model(X_test_t)

        E_mean, E_std = E_train.mean(), E_train.std()
        omega_mean, omega_std = omega_train.mean(), omega_train.std()

        P_pred_test = P_pred_test.numpy().flatten() * E_std + E_mean
        omega_pred_test = omega_pred_test.numpy().flatten() * omega_std + omega_mean

    r_P_test, _ = pearsonr(P_pred_test, E_test)
    r_omega_test, _ = pearsonr(omega_pred_test, omega_test)

    print(f"\nTest set results:")
    print(f"  r(P_learned, E_true) = {r_P_test:.4f}")
    print(f"  r(ω_learned, ω_true) = {r_omega_test:.4f}")

    # Verdict
    if r_P_test > 0.95 and r_omega_test > 0.90:
        print(f"\nStatus: \033[92mPASS\033[0m")
        status = 'PASS'
    elif r_P_test > 0.80 and r_omega_test > 0.70:
        print(f"\nStatus: \033[93mMARGINAL\033[0m")
        status = 'MARGINAL'
    else:
        print(f"\nStatus: \033[91mFAIL\033[0m")
        status = 'FAIL'

    return status, r_P_test, r_omega_test


def test_p_conservation():
    """
    Test: Is P conserved along each trajectory?

    For each SHO trajectory (constant E), compute P at multiple windows.
    Check coefficient of variation CV(P) < 0.05.
    """
    print("\n" + "="*60)
    print("TEST: P Conservation Along Trajectories")
    print("="*60)

    # First, train the model
    features_all, I_all, omega_all = generate_sho_dataset(n_trajectories=30, T=100)
    model, _, _ = train_action_mlp(features_all, I_all, omega_all, n_epochs=1000, verbose=False)

    # Now test conservation on individual trajectories
    sho = SimpleHarmonicOscillator(omega0=1.0)
    cv_values = []

    for E in [1.0, 2.0, 3.0, 4.0]:
        q0, p0 = sho.initial_condition_for_energy(E)
        t, q, p, z, E_actual = simulate_hamiltonian(sho, q0, p0, T=100, dt=0.01)

        # Compute P at multiple windows along trajectory
        window_size = 512
        stride = 128
        P_values = []

        for start in range(0, len(z) - window_size, stride):
            window = z[start:start+window_size]
            features = extract_hst_features(window)

            with torch.no_grad():
                X = torch.tensor(features.reshape(1, -1), dtype=torch.float32)
                P_pred, _ = model(X)
                P_values.append(P_pred.item())

        if len(P_values) > 2:
            P_arr = np.array(P_values)
            cv = np.std(P_arr) / (np.abs(np.mean(P_arr)) + 1e-10)
            cv_values.append(cv)
            print(f"  E={E:.1f}: CV(P) = {cv:.4f} (mean P = {np.mean(P_arr):.3f})")

    mean_cv = np.mean(cv_values)
    fraction_good = np.mean(np.array(cv_values) < 0.10)

    print(f"\nMean CV: {mean_cv:.4f}")
    print(f"Fraction with CV < 0.10: {fraction_good:.1%}")

    if mean_cv < 0.05:
        print(f"\nStatus: \033[92mPASS\033[0m (CV < 0.05)")
        return 'PASS'
    elif mean_cv < 0.15:
        print(f"\nStatus: \033[93mMARGINAL\033[0m (0.05 < CV < 0.15)")
        return 'MARGINAL'
    else:
        print(f"\nStatus: \033[91mFAIL\033[0m (CV > 0.15)")
        return 'FAIL'


if __name__ == "__main__":
    print("="*60)
    print("CRITICAL TEST: HJB-MLP Action Learning")
    print("Can MLP learn true action from HST features?")
    print("="*60)

    results = {}

    # Test 1: SHO action recovery
    status, r = test_sho_action_recovery()
    results['sho_action'] = (status, r)

    # Test 2: Duffing action and frequency recovery
    status, r_P, r_omega = test_duffing_action_recovery()
    results['duffing'] = (status, r_P, r_omega)

    # Test 3: P conservation
    status = test_p_conservation()
    results['conservation'] = status

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"SHO Action Recovery: {results['sho_action'][0]} (r = {results['sho_action'][1]:.4f})")
    print(f"Duffing Recovery: {results['duffing'][0]} (r_P = {results['duffing'][1]:.4f}, r_ω = {results['duffing'][2]:.4f})")
    print(f"P Conservation: {results['conservation']}")
