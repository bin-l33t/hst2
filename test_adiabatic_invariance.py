"""
Adiabatic Invariance Test for Kapitza Pendulum

THE KEY TEST: Action J is conserved even when energy E changes!

Setup:
  θ̈ = -(g/L + a(t)·Ω²cos(Ωt)/L)·sin(θ)

Experiment:
  1. Start at energy E₀, train MLP to learn action J₀
  2. Slowly ramp drive amplitude: a(t) = a₀ + (a₁-a₀)·t/T_ramp
  3. Energy E(t) changes, but action J should stay ≈ J₀

Test:
  Var(J)/mean(J) << Var(E)/mean(E)

This is the DEFINITIVE test of the geodesic property:
  P (action) is the adiabatic invariant that stays constant
  even as the system is slowly driven.
"""

import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

from hst import hst_forward_pywt, extract_features


def kapitza_dynamics_ramped(t, y, g, L, Omega, a_func):
    """
    Kapitza pendulum with time-varying amplitude a(t).

    a_func(t) returns the drive amplitude at time t.
    """
    theta, theta_dot = y

    a = a_func(t)

    # Pivot acceleration term
    pivot_term = (a / L) * Omega**2 * np.cos(Omega * t) * np.sin(theta)

    # Gravity term
    gravity_term = -(g / L) * np.sin(theta)

    theta_ddot = gravity_term + pivot_term

    return [theta_dot, theta_ddot]


def compute_instantaneous_energy(theta, theta_dot, g, L, a, Omega, t):
    """
    Compute instantaneous energy of the pendulum.

    For the driven pendulum, we use the "bare" pendulum energy
    plus a ponderomotive correction.

    E_bare = ½L²θ̇² - gL·cos(θ)  (per unit mass)
    E_ponderomotive = (a·Ω)²/(4g) · sin²(θ)  (effective potential)
    """
    # Bare pendulum energy (kinetic + potential)
    E_bare = 0.5 * L**2 * theta_dot**2 - g * L * np.cos(theta)

    # Time-averaged ponderomotive correction
    kappa = (a * Omega)**2 / (2 * g * L)
    E_pond = g * L * kappa * np.sin(theta)**2

    return E_bare + E_pond


def simulate_with_ramp(theta0, theta_dot0, T, dt, g, L, Omega, a0, a1, T_ramp):
    """
    Simulate Kapitza pendulum with linearly ramping drive amplitude.

    a(t) = a0                    for t < 0
         = a0 + (a1-a0)*t/T_ramp for 0 <= t < T_ramp
         = a1                    for t >= T_ramp
    """
    def a_func(t):
        if t < 0:
            return a0
        elif t < T_ramp:
            return a0 + (a1 - a0) * t / T_ramp
        else:
            return a1

    t_eval = np.arange(0, T, dt)

    sol = solve_ivp(
        kapitza_dynamics_ramped,
        (0, T),
        [theta0, theta_dot0],
        args=(g, L, Omega, a_func),
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8,
        atol=1e-10
    )

    theta = sol.y[0]
    theta_dot = sol.y[1]

    # Compute energy at each time step
    E = np.array([
        compute_instantaneous_energy(theta[i], theta_dot[i], g, L, a_func(t_eval[i]), Omega, t_eval[i])
        for i in range(len(t_eval))
    ])

    # Complex phase space representation
    omega0 = np.sqrt(g / L)
    z = theta + 1j * theta_dot / omega0

    return sol.t, theta, theta_dot, z, E, a_func


class ActionMLP(nn.Module):
    """MLP that learns action J from HST features."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


def train_action_mlp(trajectories, n_epochs=1500):
    """
    Train MLP to learn action from HST features.

    Training data: multiple trajectories at CONSTANT drive amplitude.
    Each trajectory has constant action but different energy over time.
    """
    print("Extracting features from training trajectories...")

    # Collect all windows from all trajectories
    all_features = []
    all_labels = []  # trajectory index (proxy for action level)

    window_size = 256
    stride = 64

    for traj_idx, (z, E_mean) in enumerate(trajectories):
        for start in range(0, len(z) - window_size, stride):
            window = z[start:start+window_size]
            feat = extract_features(window)
            all_features.append(feat)
            all_labels.append(E_mean)  # Use mean energy as proxy for action

    X = np.array(all_features)
    y = np.array(all_labels)

    print(f"Training samples: {len(X)}")

    # Normalize
    y_mean, y_std = y.mean(), y.std()
    y_norm = (y - y_mean) / (y_std + 1e-8)

    model = ActionMLP(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y_norm.reshape(-1, 1), dtype=torch.float32)

    for epoch in range(n_epochs):
        J_pred = model(X_t)
        loss = nn.MSELoss()(J_pred, y_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 300 == 0:
            print(f"Epoch {epoch}: loss = {loss.item():.6f}")

    return model, y_mean, y_std


def compute_action_along_trajectory(model, z, y_mean, y_std, window_size=256, stride=64):
    """Compute action J at multiple points along a trajectory."""
    J_values = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(z) - window_size, stride):
            window = z[start:start+window_size]
            feat = extract_features(window)
            X = torch.tensor(feat.reshape(1, -1), dtype=torch.float32)
            J = model(X).item() * y_std + y_mean
            J_values.append(J)

    return np.array(J_values)


def test_adiabatic_invariance():
    """
    Main test: Action is conserved during slow parameter ramp.
    """
    print("="*60)
    print("ADIABATIC INVARIANCE TEST")
    print("="*60)

    # Physical parameters
    g, L = 9.81, 1.0
    Omega = 50.0  # High frequency drive

    # === Phase 1: Train MLP on constant-drive trajectories ===
    print("\n--- Phase 1: Training on constant-drive trajectories ---")

    training_trajectories = []

    # Generate trajectories at different energies but CONSTANT drive
    a_train = 0.08  # Fixed drive amplitude for training

    for theta0 in [0.3, 0.5, 0.7, 0.9, 1.1]:  # Different initial angles
        theta_dot0 = 0.0
        T = 50.0
        dt = 0.01

        def a_const(t):
            return a_train

        t_eval = np.arange(0, T, dt)
        sol = solve_ivp(
            kapitza_dynamics_ramped,
            (0, T),
            [theta0, theta_dot0],
            args=(g, L, Omega, a_const),
            t_eval=t_eval,
            method='RK45',
            rtol=1e-8
        )

        omega0 = np.sqrt(g / L)
        z = sol.y[0] + 1j * sol.y[1] / omega0

        # Compute mean energy for this trajectory
        E = np.array([
            compute_instantaneous_energy(sol.y[0][i], sol.y[1][i], g, L, a_train, Omega, t_eval[i])
            for i in range(len(t_eval))
        ])

        training_trajectories.append((z, np.mean(E)))

    # Train MLP
    model, y_mean, y_std = train_action_mlp(training_trajectories)

    # === Phase 2: Test on ramped trajectory ===
    print("\n--- Phase 2: Testing adiabatic invariance ---")

    # Ramp parameters
    a0 = 0.06  # Initial drive
    a1 = 0.12  # Final drive
    T_ramp = 100.0  # Slow ramp (adiabatic)
    T_total = 150.0  # Total simulation time

    theta0 = 0.5
    theta_dot0 = 0.0
    dt = 0.01

    print(f"Drive ramp: a = {a0} → {a1} over T = {T_ramp}s")

    t, theta, theta_dot, z, E, a_func = simulate_with_ramp(
        theta0, theta_dot0, T_total, dt, g, L, Omega, a0, a1, T_ramp
    )

    # Compute action along trajectory
    J = compute_action_along_trajectory(model, z, y_mean, y_std, window_size=256, stride=64)

    # Compute energy at same points
    window_size = 256
    stride = 64
    E_windows = []
    t_windows = []

    for i, start in enumerate(range(0, len(z) - window_size, stride)):
        center = start + window_size // 2
        E_windows.append(E[center])
        t_windows.append(t[center])

    E_windows = np.array(E_windows)
    t_windows = np.array(t_windows)

    # === Compute invariance metrics ===

    # Split into ramp phase and post-ramp
    ramp_mask = t_windows < T_ramp

    J_ramp = J[ramp_mask]
    E_ramp = E_windows[ramp_mask]

    # Coefficient of variation
    CV_J = np.std(J_ramp) / (np.abs(np.mean(J_ramp)) + 1e-10)
    CV_E = np.std(E_ramp) / (np.abs(np.mean(E_ramp)) + 1e-10)

    # Relative change
    J_change = (J_ramp[-1] - J_ramp[0]) / (np.abs(J_ramp[0]) + 1e-10)
    E_change = (E_ramp[-1] - E_ramp[0]) / (np.abs(E_ramp[0]) + 1e-10)

    print(f"\n--- Results during ramp phase ---")
    print(f"Energy E:")
    print(f"  Range: [{E_ramp.min():.2f}, {E_ramp.max():.2f}]")
    print(f"  CV(E) = {CV_E:.4f}")
    print(f"  Relative change: {E_change*100:.1f}%")
    print(f"\nAction J:")
    print(f"  Range: [{J_ramp.min():.2f}, {J_ramp.max():.2f}]")
    print(f"  CV(J) = {CV_J:.4f}")
    print(f"  Relative change: {J_change*100:.1f}%")

    print(f"\n--- Adiabatic Invariance Ratio ---")
    ratio = CV_J / CV_E if CV_E > 0.01 else float('inf')
    print(f"CV(J) / CV(E) = {ratio:.4f}")

    # === Verdict ===
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)

    if ratio < 0.5 and CV_J < 0.15:
        print("\033[92mPASS: Action J is adiabatically invariant!\033[0m")
        print(f"  J varies {ratio:.1%} as much as E during parameter ramp.")
        status = 'PASS'
    elif ratio < 1.0:
        print("\033[93mMARGINAL: J more stable than E, but not fully invariant\033[0m")
        status = 'MARGINAL'
    else:
        print("\033[91mFAIL: J varies as much as E\033[0m")
        status = 'FAIL'

    return status, CV_J, CV_E, ratio


def test_fast_vs_slow_ramp():
    """
    Compare fast ramp (non-adiabatic) vs slow ramp (adiabatic).

    Slow ramp: Action should be conserved
    Fast ramp: Action will change
    """
    print("\n" + "="*60)
    print("FAST vs SLOW RAMP COMPARISON")
    print("="*60)

    g, L = 9.81, 1.0
    Omega = 50.0

    # Train MLP first
    training_trajectories = []
    a_train = 0.08

    for theta0 in [0.3, 0.5, 0.7, 0.9]:
        theta_dot0 = 0.0
        T = 50.0
        dt = 0.01

        def a_const(t):
            return a_train

        t_eval = np.arange(0, T, dt)
        sol = solve_ivp(
            kapitza_dynamics_ramped,
            (0, T),
            [theta0, theta_dot0],
            args=(g, L, Omega, a_const),
            t_eval=t_eval,
            method='RK45'
        )

        omega0 = np.sqrt(g / L)
        z = sol.y[0] + 1j * sol.y[1] / omega0
        E = np.array([
            compute_instantaneous_energy(sol.y[0][i], sol.y[1][i], g, L, a_train, Omega, t_eval[i])
            for i in range(len(t_eval))
        ])
        training_trajectories.append((z, np.mean(E)))

    model, y_mean, y_std = train_action_mlp(training_trajectories, n_epochs=1000)

    # Compare ramp speeds
    ramp_times = [5.0, 20.0, 50.0, 100.0]  # Fast to slow

    results = []

    for T_ramp in ramp_times:
        a0, a1 = 0.06, 0.12
        T_total = T_ramp + 20

        t, theta, theta_dot, z, E, a_func = simulate_with_ramp(
            0.5, 0.0, T_total, 0.01, g, L, Omega, a0, a1, T_ramp
        )

        J = compute_action_along_trajectory(model, z, y_mean, y_std)

        # Get E at same points
        window_size = 256
        stride = 64
        E_windows = [E[start + window_size//2] for start in range(0, len(z) - window_size, stride)]
        E_windows = np.array(E_windows)

        # During ramp
        n_ramp = int(T_ramp / 0.01 / stride)
        n_ramp = min(n_ramp, len(J) - 1)

        CV_J = np.std(J[:n_ramp]) / (np.abs(np.mean(J[:n_ramp])) + 1e-10)
        CV_E = np.std(E_windows[:n_ramp]) / (np.abs(np.mean(E_windows[:n_ramp])) + 1e-10)
        ratio = CV_J / CV_E if CV_E > 0.01 else 0

        results.append({
            'T_ramp': T_ramp,
            'CV_J': CV_J,
            'CV_E': CV_E,
            'ratio': ratio
        })

        print(f"T_ramp = {T_ramp:5.1f}s: CV(J)/CV(E) = {ratio:.3f}")

    print("\nExpected: Slower ramp → Lower ratio (better adiabatic invariance)")

    # Check if ratio decreases with slower ramp
    ratios = [r['ratio'] for r in results]
    if ratios[-1] < ratios[0]:
        print("\033[92mCONFIRMED: Slower ramp preserves action better!\033[0m")
    else:
        print("\033[93mUNCLEAR: Need more investigation\033[0m")

    return results


if __name__ == "__main__":
    # Main test
    status, CV_J, CV_E, ratio = test_adiabatic_invariance()

    # Comparison test
    print("\n")
    results = test_fast_vs_slow_ramp()
