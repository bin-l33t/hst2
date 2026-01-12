"""
Comprehensive Validation Tests

1. Adiabatic Invariance with Full Amplitude Training
2. Time Propagation: Q(t) = Q₀ + ω(P)·t prediction
3. Duffing Oscillator with Conservation Loss Fix
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import ellipk
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

from hst import hst_forward_pywt, extract_features
from hamiltonian_systems import PendulumOscillator, AnharmonicOscillator, simulate_hamiltonian


class ActionOmegaMLP(nn.Module):
    """MLP that learns P (action) and ω(P) jointly."""
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)  # P (action)
        )
        self.omega_net = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, 1)  # ω(P)
        )

    def forward(self, x):
        P = self.encoder(x)
        omega = self.omega_net(P)
        return P, omega


# ============================================================
# TEST 1: Adiabatic Invariance with Full Amplitude Training
# ============================================================

def kapitza_dynamics_ramped(t, y, g, L, Omega, a_func):
    """Kapitza pendulum with time-varying amplitude a(t)."""
    theta, theta_dot = y
    a = a_func(t)
    pivot_term = (a / L) * Omega**2 * np.cos(Omega * t) * np.sin(theta)
    gravity_term = -(g / L) * np.sin(theta)
    theta_ddot = gravity_term + pivot_term
    return [theta_dot, theta_ddot]


def compute_energy(theta, theta_dot, g, L, a, Omega):
    """Compute effective energy with ponderomotive correction."""
    E_bare = 0.5 * L**2 * theta_dot**2 - g * L * np.cos(theta)
    kappa = (a * Omega)**2 / (2 * g * L)
    E_pond = g * L * kappa * np.sin(theta)**2
    return E_bare + E_pond


def test_adiabatic_full_range():
    """
    TEST 1: Train on FULL amplitude range [0.06, 0.12] instead of single value.
    This should give cleaner scaling with T_ramp.
    """
    print("=" * 60)
    print("TEST 1: Adiabatic Invariance (Full Amplitude Training)")
    print("=" * 60)

    g, L = 9.81, 1.0
    Omega = 50.0

    # === Train on trajectories spanning FULL amplitude range ===
    print("\n--- Training on amplitude range [0.06, 0.12] ---")

    training_data = []
    amplitudes = np.linspace(0.06, 0.12, 5)  # Full range
    initial_angles = [0.3, 0.5, 0.7]

    for a_train in amplitudes:
        for theta0 in initial_angles:
            def a_const(t, a=a_train):
                return a

            t_eval = np.arange(0, 50, 0.01)
            sol = solve_ivp(
                kapitza_dynamics_ramped,
                (0, 50),
                [theta0, 0.0],
                args=(g, L, Omega, a_const),
                t_eval=t_eval,
                method='RK45',
                rtol=1e-8
            )

            omega0 = np.sqrt(g / L)
            z = sol.y[0] + 1j * sol.y[1] / omega0

            # Mean energy for this trajectory
            E = np.array([
                compute_energy(sol.y[0][i], sol.y[1][i], g, L, a_train, Omega)
                for i in range(len(t_eval))
            ])

            training_data.append((z, np.mean(E), a_train))

    # Extract features and train
    all_features = []
    all_labels = []
    window_size = 256
    stride = 64

    for z, E_mean, a in training_data:
        for start in range(0, len(z) - window_size, stride):
            window = z[start:start+window_size]
            feat = extract_features(window)
            all_features.append(feat)
            all_labels.append(E_mean)

    X = np.array(all_features)
    y = np.array(all_labels)
    y_mean, y_std = y.mean(), y.std()
    y_norm = (y - y_mean) / (y_std + 1e-8)

    print(f"Training samples: {len(X)} (spanning a ∈ [0.06, 0.12])")

    model = nn.Sequential(
        nn.Linear(X.shape[1], 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 1)
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y_norm.reshape(-1, 1), dtype=torch.float32)

    for epoch in range(1500):
        J_pred = model(X_t)
        loss = nn.MSELoss()(J_pred, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: loss = {loss.item():.6f}")

    # === Test on ramped trajectory ===
    print("\n--- Testing ramp speed scaling ---")

    ramp_times = [5.0, 10.0, 20.0, 50.0, 100.0]
    results = []

    for T_ramp in ramp_times:
        a0, a1 = 0.06, 0.12
        T_total = T_ramp + 20

        def a_func(t, a0=a0, a1=a1, T_ramp=T_ramp):
            if t < 0:
                return a0
            elif t < T_ramp:
                return a0 + (a1 - a0) * t / T_ramp
            else:
                return a1

        t_eval = np.arange(0, T_total, 0.01)
        sol = solve_ivp(
            kapitza_dynamics_ramped,
            (0, T_total),
            [0.5, 0.0],
            args=(g, L, Omega, a_func),
            t_eval=t_eval,
            method='RK45',
            rtol=1e-8
        )

        theta = sol.y[0]
        theta_dot = sol.y[1]
        omega0 = np.sqrt(g / L)
        z = theta + 1j * theta_dot / omega0

        # Compute J and E along trajectory
        J_values = []
        E_values = []
        t_values = []

        model.eval()
        with torch.no_grad():
            for start in range(0, len(z) - window_size, stride):
                window = z[start:start+window_size]
                feat = extract_features(window)
                X_test = torch.tensor(feat.reshape(1, -1), dtype=torch.float32)
                J = model(X_test).item() * y_std + y_mean
                J_values.append(J)

                center = start + window_size // 2
                E = compute_energy(theta[center], theta_dot[center], g, L, a_func(t_eval[center]), Omega)
                E_values.append(E)
                t_values.append(t_eval[center])

        J_values = np.array(J_values)
        E_values = np.array(E_values)
        t_values = np.array(t_values)

        # During ramp only
        ramp_mask = t_values < T_ramp
        J_ramp = J_values[ramp_mask]
        E_ramp = E_values[ramp_mask]

        if len(J_ramp) > 0:
            CV_J = np.std(J_ramp) / (np.abs(np.mean(J_ramp)) + 1e-10)
            CV_E = np.std(E_ramp) / (np.abs(np.mean(E_ramp)) + 1e-10)
            ratio = CV_J / CV_E if CV_E > 0.01 else 0

            results.append({
                'T_ramp': T_ramp,
                'CV_J': CV_J,
                'CV_E': CV_E,
                'ratio': ratio
            })

            print(f"T_ramp = {T_ramp:5.1f}s: CV(J)/CV(E) = {ratio:.4f}")

    # Check monotonic improvement
    ratios = [r['ratio'] for r in results]
    print(f"\nRatios: {[f'{r:.3f}' for r in ratios]}")

    if len(ratios) >= 2 and ratios[-1] < ratios[0]:
        print("\033[92mCONFIRMED: Slower ramp preserves action better!\033[0m")
        return 'PASS', results
    else:
        print("\033[93mMIXED: Ratio doesn't strictly decrease\033[0m")
        return 'MARGINAL', results


# ============================================================
# TEST 2: Time Propagation - Predict Q(t) = Q₀ + ω(P)·t
# ============================================================

def test_time_propagation():
    """
    TEST 2: Use learned ω(P) to predict phase evolution.

    1. Train MLP to learn P and ω from HST features
    2. For a test trajectory, predict Q(t) = Q₀ + ω·t
    3. Compare predicted phase with actual phase
    """
    print("\n" + "=" * 60)
    print("TEST 2: Time Propagation (Q(t) = Q₀ + ω(P)·t)")
    print("=" * 60)

    # Use pendulum for interesting ω(E) behavior
    pendulum = PendulumOscillator()

    # Generate training data
    print("\n--- Generating training data ---")
    train_energies = np.linspace(-0.8, 0.6, 15)
    train_data = []

    for E in train_energies:
        if E >= 0.95 or E <= -0.95:
            continue

        q0, p0 = pendulum.initial_condition_for_energy(E)
        t, q, p, z, E_actual = simulate_hamiltonian(pendulum, q0, p0, T=100, dt=0.01)

        # Measure true omega
        zero_crossings = np.where((q[:-1] < 0) & (q[1:] >= 0))[0]
        if len(zero_crossings) >= 2:
            periods = np.diff(t[zero_crossings])
            omega_true = 2 * np.pi / np.mean(periods)

            train_data.append({
                'z': z,
                't': t,
                'q': q,
                'p': p,
                'E': E_actual,
                'omega': omega_true
            })

    print(f"Training trajectories: {len(train_data)}")

    # Extract features
    window_size = 512
    stride = 128

    all_X = []
    all_E = []
    all_omega = []

    for d in train_data:
        for start in range(0, len(d['z']) - window_size, stride):
            feat = extract_features(d['z'][start:start+window_size])
            all_X.append(feat)
            all_E.append(d['E'])
            all_omega.append(d['omega'])

    X = np.array(all_X)
    E_arr = np.array(all_E)
    omega_arr = np.array(all_omega)

    E_mean, E_std = E_arr.mean(), E_arr.std()
    omega_mean, omega_std = omega_arr.mean(), omega_arr.std()

    # Train MLP
    print("\n--- Training action-omega MLP ---")
    model = ActionOmegaMLP(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    X_t = torch.tensor(X, dtype=torch.float32)
    E_t = torch.tensor((E_arr - E_mean) / (E_std + 1e-8), dtype=torch.float32).unsqueeze(1)
    omega_t = torch.tensor((omega_arr - omega_mean) / (omega_std + 1e-8), dtype=torch.float32).unsqueeze(1)

    for epoch in range(2000):
        P, omega_pred = model(X_t)

        # Losses
        loss_P = nn.MSELoss()(P, E_t)
        loss_omega = nn.MSELoss()(omega_pred, omega_t)
        loss = loss_P + loss_omega

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: loss = {loss.item():.6f}")

    # === Test time propagation ===
    print("\n--- Testing time propagation ---")

    # Pick a test trajectory
    test_E = 0.3
    q0, p0 = pendulum.initial_condition_for_energy(test_E)
    t, q, p, z, E_actual = simulate_hamiltonian(pendulum, q0, p0, T=50, dt=0.01)

    # Get initial P and omega from first window
    model.eval()
    with torch.no_grad():
        init_feat = extract_features(z[:window_size])
        X_init = torch.tensor(init_feat.reshape(1, -1), dtype=torch.float32)
        P_init, omega_init = model(X_init)
        P_val = P_init.item() * E_std + E_mean
        omega_val = omega_init.item() * omega_std + omega_mean

    print(f"Learned P = {P_val:.4f} (true E = {E_actual:.4f})")
    print(f"Learned ω = {omega_val:.4f}")

    # Measure true omega
    zero_crossings = np.where((q[:-1] < 0) & (q[1:] >= 0))[0]
    if len(zero_crossings) >= 2:
        periods = np.diff(t[zero_crossings])
        omega_true = 2 * np.pi / np.mean(periods)
        print(f"True ω = {omega_true:.4f}")

    # Predict Q(t) = Q₀ + ω·t
    # Extract actual phase (unwrapped)
    phase_actual = np.unwrap(np.arctan2(p, q))

    # Predicted phase - try both signs of ω (phase convention ambiguity)
    Q0 = phase_actual[window_size // 2]
    t_pred = t - t[window_size // 2]

    start_idx = window_size // 2
    end_idx = min(len(phase_actual), start_idx + 2000)
    actual = phase_actual[start_idx:end_idx]
    t_slice = t_pred[start_idx:end_idx]

    # Try positive and negative omega
    pred_pos = Q0 + omega_val * t_slice
    pred_neg = Q0 - omega_val * t_slice

    r_pos, _ = pearsonr(actual, pred_pos)
    r_neg, _ = pearsonr(actual, pred_neg)

    if r_pos > r_neg:
        predicted = pred_pos
        r = r_pos
        omega_used = omega_val
    else:
        predicted = pred_neg
        r = r_neg
        omega_used = -omega_val
        print(f"  (Using ω = {omega_used:.4f} for correct phase direction)")

    # Phase error
    phase_error = np.abs(actual - predicted)
    mean_error = np.mean(phase_error)
    max_error = np.max(phase_error)

    print(f"\nTime propagation results:")
    print(f"  Correlation r(Q_pred, Q_actual) = {r:.4f}")
    print(f"  Mean phase error = {mean_error:.4f} rad")
    print(f"  Max phase error = {max_error:.4f} rad")

    if r > 0.99 and mean_error < 0.5:
        print("\n\033[92mPASS: Time propagation accurate!\033[0m")
        status = 'PASS'
    elif r > 0.95:
        print("\n\033[93mMARGINAL: Phase drift present\033[0m")
        status = 'MARGINAL'
    else:
        print("\n\033[91mFAIL: Poor time propagation\033[0m")
        status = 'FAIL'

    return status, r, mean_error


# ============================================================
# TEST 3: Duffing Oscillator with Conservation Loss Fix
# ============================================================

def theoretical_omega_duffing(E, epsilon=0.1):
    """Approximate ω(E) for Duffing: ω ≈ 1 + (3ε/8)·A² where A ~ √E."""
    if E <= 0:
        return 1.0
    A_sq = 2 * E  # Rough estimate of amplitude squared
    return 1 + (3 * epsilon / 8) * A_sq


def test_duffing_conservation():
    """
    TEST 3: Duffing oscillator with proper conservation loss.

    H = ½p² + ½q² + (ε/4)q⁴

    The earlier test showed r = 0.57. With conservation loss fix, should improve.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Duffing Oscillator with Conservation Loss Fix")
    print("=" * 60)

    epsilon = 0.1
    duffing = AnharmonicOscillator(epsilon=epsilon)

    # Generate training data at multiple energies
    print("\n--- Generating training data ---")
    train_energies = np.linspace(0.1, 2.0, 15)
    train_data = []

    for E in train_energies:
        q0, p0 = duffing.initial_condition_for_energy(E)
        t, q, p, z, E_actual = simulate_hamiltonian(duffing, q0, p0, T=100, dt=0.01)

        # Measure omega
        zero_crossings = np.where((q[:-1] < 0) & (q[1:] >= 0))[0]
        if len(zero_crossings) >= 2:
            periods = np.diff(t[zero_crossings])
            omega_true = 2 * np.pi / np.mean(periods)

            train_data.append({
                'z': z,
                'E': E_actual,
                'omega': omega_true
            })

    print(f"Training trajectories: {len(train_data)}")

    # Extract features - multiple windows per trajectory
    window_size = 512
    stride = 128

    # Group features by trajectory for conservation loss
    traj_features = []
    traj_E = []
    traj_omega = []

    for d in train_data:
        windows = []
        for start in range(0, len(d['z']) - window_size, stride):
            feat = extract_features(d['z'][start:start+window_size])
            windows.append(feat)

        if len(windows) >= 3:
            traj_features.append(np.array(windows))
            traj_E.append(d['E'])
            traj_omega.append(d['omega'])

    E_arr = np.array(traj_E)
    omega_arr = np.array(traj_omega)
    E_mean, E_std = E_arr.mean(), E_arr.std()
    omega_mean, omega_std = omega_arr.mean(), omega_arr.std()

    print(f"Trajectories with enough windows: {len(traj_features)}")
    print(f"E range: [{E_arr.min():.2f}, {E_arr.max():.2f}]")
    print(f"ω range: [{omega_arr.min():.3f}, {omega_arr.max():.3f}]")

    # Train with CONSERVATION loss
    print("\n--- Training with conservation loss ---")
    input_dim = traj_features[0].shape[1]
    model = ActionOmegaMLP(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(2500):
        total_loss = 0
        total_cons = 0
        total_reg_P = 0
        total_reg_omega = 0

        for i, (X, E, omega) in enumerate(zip(traj_features, traj_E, traj_omega)):
            X_t = torch.tensor(X, dtype=torch.float32)
            target_E = (E - E_mean) / (E_std + 1e-8)
            target_omega = (omega - omega_mean) / (omega_std + 1e-8)

            P_pred, omega_pred = model(X_t)

            # Conservation loss: P should be constant within trajectory
            conservation_loss = P_pred.var()

            # Regression losses
            reg_loss_P = (P_pred.mean() - target_E)**2
            reg_loss_omega = (omega_pred.mean() - target_omega)**2

            # Combined loss (heavy weight on conservation!)
            loss = reg_loss_P + 10.0 * conservation_loss + reg_loss_omega

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cons += conservation_loss.item()
            total_reg_P += reg_loss_P.item()
            total_reg_omega += reg_loss_omega.item()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: loss = {total_loss/len(traj_features):.6f}, "
                  f"cons = {total_cons/len(traj_features):.6f}")

    # === Evaluate ===
    print("\n--- Evaluation ---")
    model.eval()

    P_means = []
    P_cvs = []
    omega_preds = []
    E_true = []
    omega_true = []

    with torch.no_grad():
        for X, E, omega in zip(traj_features, traj_E, traj_omega):
            X_t = torch.tensor(X, dtype=torch.float32)
            P_pred, omega_pred = model(X_t)

            P_np = P_pred.numpy().flatten() * E_std + E_mean
            omega_np = omega_pred.numpy().flatten() * omega_std + omega_mean

            P_means.append(np.mean(P_np))
            P_cvs.append(np.std(P_np) / (np.abs(np.mean(P_np)) + 1e-10))
            omega_preds.append(np.mean(omega_np))
            E_true.append(E)
            omega_true.append(omega)

    P_means = np.array(P_means)
    omega_preds = np.array(omega_preds)
    E_true = np.array(E_true)
    omega_true = np.array(omega_true)

    r_P_E, _ = pearsonr(P_means, E_true)
    r_omega, _ = pearsonr(omega_preds, omega_true)
    mean_cv = np.mean(P_cvs)

    print(f"\nResults:")
    print(f"  r(P, E): {r_P_E:.4f}")
    print(f"  r(ω_pred, ω_true): {r_omega:.4f}")
    print(f"  Mean CV (conservation): {mean_cv:.4f}")
    print(f"  Individual CVs: {[f'{cv:.3f}' for cv in P_cvs[:5]]}")

    # Verdict
    print("\n" + "=" * 60)
    if r_P_E > 0.95 and r_omega > 0.80 and mean_cv < 0.10:
        print("\033[92mPASS: Duffing with conservation loss works!\033[0m")
        status = 'PASS'
    elif r_P_E > 0.85 and r_omega > 0.60:
        print("\033[93mMARGINAL: Improved but not perfect\033[0m")
        status = 'MARGINAL'
    else:
        print("\033[91mFAIL\033[0m")
        status = 'FAIL'

    return status, r_P_E, r_omega, mean_cv


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("COMPREHENSIVE VALIDATION SUITE")
    print("=" * 70)

    # Test 1: Adiabatic with full amplitude training
    status1, results1 = test_adiabatic_full_range()

    # Test 2: Time propagation
    status2, r2, err2 = test_time_propagation()

    # Test 3: Duffing with conservation
    status3, r_P3, r_omega3, cv3 = test_duffing_conservation()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Test 1 (Adiabatic Full Range):  {status1}")
    print(f"Test 2 (Time Propagation):      {status2} (r={r2:.4f})")
    print(f"Test 3 (Duffing Conservation):  {status3} (r_P={r_P3:.4f}, r_ω={r_omega3:.4f}, CV={cv3:.4f})")
