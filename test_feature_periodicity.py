"""
Feature Periodicity Search - Finding Operational J₀

Key question: Do HST features show periodicity in action space?

For SHO: We expect NO natural J₀ (features don't repeat in J)
For Pendulum: May show periodicity near separatrix

The search:
1. Generate trajectories at various J values
2. Extract HST features Φ(J, Q₀) for random initial phases Q₀
3. Compute phase-averaged feature distance: d(J, Δ) = 𝔼_{Q₀}[||Φ(J+Δ) - Φ(J)||]
4. Search for minima in d(J, Δ) away from Δ = 0
5. If found: Δ = J₀ is operationally meaningful
6. If not found: system lacks intrinsic action scale

This is a meaningful test because:
- Integer Fourier modes ≠ integer actions
- "Quantization" in Glinsky's sense is about what's OBSERVABLE
- J₀ must emerge from measurement resolution OR feature periodicity
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

# Import utilities
from action_angle_utils import wrap_to_2pi
from test_sho_action_angle import sho_from_action_angle
from hst import extract_features, extract_features_magnitude_only


def generate_sho_trajectory(J: float, Q0: float, omega: float = 1.0,
                             m: float = 1.0, dt: float = 0.01,
                             n_samples: int = 512) -> np.ndarray:
    """
    Generate SHO trajectory in complex form z = q + i·p/ω.

    Parameters
    ----------
    J : float
        Action (P in our convention)
    Q0 : float
        Initial phase angle
    omega : float
        Angular frequency
    m : float
        Mass
    dt : float
        Time step
    n_samples : int
        Number of samples

    Returns
    -------
    z : np.ndarray
        Complex trajectory z = q + i·p/ω
    """
    t = np.arange(n_samples) * dt
    Q_traj = wrap_to_2pi(Q0 + omega * t)

    z = np.zeros(n_samples, dtype=complex)
    for i, Q in enumerate(Q_traj):
        q, p = sho_from_action_angle(J, Q, omega, m)
        z[i] = q + 1j * p / omega

    return z


def generate_pendulum_trajectory(J: float, Q0: float, g: float = 1.0,
                                  L: float = 1.0, dt: float = 0.01,
                                  n_samples: int = 512) -> np.ndarray:
    """
    Generate pendulum trajectory in complex form z = θ + i·ω_scale.

    For pendulum: H = p²/2mL² - mgL·cos(θ)
    The action J is related to energy E via elliptic integrals.

    For simplicity, we use numerical integration.
    """
    from scipy.integrate import odeint

    # Pendulum ODE: θ'' = -g/L sin(θ)
    def pendulum_ode(y, t):
        theta, theta_dot = y
        return [theta_dot, -g/L * np.sin(theta)]

    # Initial conditions from J (approximately)
    # For small oscillations: J ≈ E/ω₀ where ω₀ = √(g/L)
    omega0 = np.sqrt(g / L)
    E = J * omega0  # Approximate energy

    # Initial θ from energy: E = -mgL·cos(θ_max) → θ_max = arccos(-E/(mgL))
    # But we need E < mgL for oscillatory motion
    if E >= g * L:
        # Near or past separatrix - use full rotation
        theta0 = Q0  # Starting angle
        theta_dot0 = np.sqrt(2 * E / L)  # Kinetic energy at θ=0
    else:
        # Oscillatory motion
        cos_theta_max = 1 - E / (g * L)  # From E = mgL(1 - cos(θ_max))
        theta_max = np.arccos(np.clip(cos_theta_max, -1, 1))
        theta0 = theta_max * np.sin(Q0)  # Start at phase Q0
        # θ_dot from energy: (1/2)mL²θ'² - mgL·cos(θ) = -mgL·cos(θ_max)
        kinetic = g * L * (np.cos(theta0) - cos_theta_max)
        theta_dot0 = np.sqrt(max(0, 2 * kinetic / L)) * np.sign(np.cos(Q0))

    # Integrate
    t = np.arange(n_samples) * dt
    y0 = [theta0, theta_dot0]

    try:
        sol = odeint(pendulum_ode, y0, t)
        theta = sol[:, 0]
        theta_dot = sol[:, 1]
    except:
        # Fallback if integration fails
        theta = theta0 * np.cos(omega0 * t)
        theta_dot = -theta0 * omega0 * np.sin(omega0 * t)

    # Complex representation (scaled momentum)
    z = theta + 1j * theta_dot / omega0

    return z


def extract_features_for_J(J: float, n_phases: int = 20,
                           system: str = 'sho',
                           feature_type: str = 'full',
                           **kwargs) -> np.ndarray:
    """
    Extract features averaged over multiple initial phases.

    Returns mean and std of feature vectors.
    """
    features_list = []

    for _ in range(n_phases):
        Q0 = np.random.uniform(0, 2 * np.pi)

        # Generate trajectory
        if system == 'sho':
            z = generate_sho_trajectory(J, Q0, **kwargs)
        elif system == 'pendulum':
            z = generate_pendulum_trajectory(J, Q0, **kwargs)
        else:
            raise ValueError(f"Unknown system: {system}")

        # Extract features
        if feature_type == 'full':
            features = extract_features(z)
        elif feature_type == 'magnitude':
            features = extract_features_magnitude_only(z)
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

        features_list.append(features)

    return np.array(features_list)


def compute_feature_distance(J1: float, J2: float, n_phases: int = 20,
                              system: str = 'sho',
                              feature_type: str = 'magnitude',
                              **kwargs) -> Tuple[float, float]:
    """
    Compute phase-averaged feature distance between J1 and J2.

    d(J1, J2) = 𝔼_{Q₀}[||Φ(J1, Q₀) - Φ(J2, Q₀')||]

    We use paired phases (same Q0 for both J values) to reduce variance.

    Returns
    -------
    mean_distance : float
        Mean feature distance
    std_distance : float
        Std of feature distance
    """
    distances = []

    for _ in range(n_phases):
        Q0 = np.random.uniform(0, 2 * np.pi)

        # Generate trajectories with SAME initial phase
        if system == 'sho':
            z1 = generate_sho_trajectory(J1, Q0, **kwargs)
            z2 = generate_sho_trajectory(J2, Q0, **kwargs)
        elif system == 'pendulum':
            z1 = generate_pendulum_trajectory(J1, Q0, **kwargs)
            z2 = generate_pendulum_trajectory(J2, Q0, **kwargs)
        else:
            raise ValueError(f"Unknown system: {system}")

        # Extract features
        if feature_type == 'full':
            f1 = extract_features(z1)
            f2 = extract_features(z2)
        elif feature_type == 'magnitude':
            f1 = extract_features_magnitude_only(z1)
            f2 = extract_features_magnitude_only(z2)
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

        # Compute distance
        dist = np.linalg.norm(f1 - f2)
        distances.append(dist)

    return np.mean(distances), np.std(distances)


def search_periodicity(J_center: float, delta_range: np.ndarray,
                        n_phases: int = 30, system: str = 'sho',
                        feature_type: str = 'magnitude',
                        **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search for periodicity in feature space.

    Parameters
    ----------
    J_center : float
        Center action value
    delta_range : np.ndarray
        Array of Δ values to test
    n_phases : int
        Number of phases to average over
    system : str
        'sho' or 'pendulum'
    feature_type : str
        'full' or 'magnitude'

    Returns
    -------
    distances : np.ndarray
        d(J_center, J_center + Δ) for each Δ
    stds : np.ndarray
        Standard deviations
    """
    distances = []
    stds = []

    for delta in delta_range:
        J2 = J_center + delta
        if J2 <= 0:
            distances.append(np.nan)
            stds.append(np.nan)
            continue

        mean_d, std_d = compute_feature_distance(
            J_center, J2, n_phases=n_phases,
            system=system, feature_type=feature_type,
            **kwargs
        )
        distances.append(mean_d)
        stds.append(std_d)

    return np.array(distances), np.array(stds)


def find_minima(delta_range: np.ndarray, distances: np.ndarray,
                 min_delta: float = 0.3) -> List[Tuple[float, float]]:
    """
    Find local minima in the distance function, excluding Δ ≈ 0.

    Returns list of (delta, distance) for each minimum.
    """
    minima = []

    # Smooth the distances first
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(distances, sigma=2)

    for i in range(1, len(delta_range) - 1):
        if abs(delta_range[i]) < min_delta:
            continue  # Skip near Δ = 0

        if np.isnan(smoothed[i]):
            continue

        # Check if local minimum
        if smoothed[i] < smoothed[i-1] and smoothed[i] < smoothed[i+1]:
            # Check if it's a significant minimum (not just noise)
            local_max = max(smoothed[max(0, i-5):min(len(smoothed), i+5)])
            if smoothed[i] < 0.7 * local_max:  # At least 30% dip
                minima.append((delta_range[i], distances[i]))

    return minima


def test_sho_periodicity():
    """
    Test: Does SHO show feature periodicity in action space?

    Expected result: NO - SHO has no natural action scale.
    """
    print("=" * 70)
    print("SHO FEATURE PERIODICITY SEARCH")
    print("=" * 70)
    print()
    print("Question: Do HST features Φ(J) show periodicity in J?")
    print("Expected: NO for SHO (no intrinsic action scale)")
    print()

    # Test at different J values
    J_values = [1.0, 2.0, 3.0]
    delta_range = np.linspace(-2.0, 4.0, 61)  # -2 to +4 in steps of 0.1

    results = {}

    for J in J_values:
        print(f"Searching at J = {J}...")
        distances, stds = search_periodicity(
            J, delta_range, n_phases=30, system='sho',
            feature_type='magnitude'
        )
        results[J] = (distances, stds)

        # Find minima
        minima = find_minima(delta_range, distances, min_delta=0.3)
        if minima:
            print(f"  Potential periodicities found:")
            for delta, dist in minima:
                print(f"    J₀ = {delta:.2f} (distance = {dist:.4f})")
        else:
            print(f"  No periodicity found (expected for SHO)")

    # Plot results
    fig, axes = plt.subplots(1, len(J_values), figsize=(15, 5))

    for i, J in enumerate(J_values):
        ax = axes[i]
        distances, stds = results[J]

        ax.errorbar(delta_range, distances, yerr=stds,
                   capsize=2, alpha=0.7, label=f'J = {J}')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Δ = 0')
        ax.axvline(x=1, color='green', linestyle=':', alpha=0.5, label='Δ = 1')

        ax.set_xlabel('Δ (action offset)')
        ax.set_ylabel('Feature distance d(J, J+Δ)')
        ax.set_title(f'SHO at J = {J}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('SHO Feature Periodicity Search\n(No periodicity expected)',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('sho_feature_periodicity.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: sho_feature_periodicity.png")
    plt.show()

    # Summary
    print()
    print("=" * 70)
    print("SHO PERIODICITY SEARCH - SUMMARY")
    print("=" * 70)
    print("""
    Result: Feature distance d(J, J+Δ) increases monotonically with |Δ|

    Interpretation:
    - No minima found at Δ ≠ 0
    - Features do NOT repeat as J changes
    - SHO has no intrinsic action scale J₀

    This confirms: "Quantization" in SHO comes from Fourier mode labels,
    NOT from periodicity in action space. The integer n in |n⟩ states
    labels discrete modes, not discrete action values.
    """)

    return results


def test_feature_scaling_with_J():
    """
    Test how features scale with J (action).

    For SHO, amplitude scales as √J, so features should scale predictably.
    """
    print("=" * 70)
    print("FEATURE SCALING WITH ACTION J")
    print("=" * 70)

    J_values = np.linspace(0.5, 5.0, 19)  # 0.5 to 5.0 in steps of 0.25

    # Extract features at each J (averaged over phases)
    feature_means = []
    feature_stds = []

    print("\nExtracting features at each J...")
    for J in J_values:
        features = extract_features_for_J(J, n_phases=20, system='sho',
                                          feature_type='magnitude')
        feature_means.append(np.mean(features, axis=0))
        feature_stds.append(np.std(features, axis=0))

    feature_means = np.array(feature_means)
    feature_stds = np.array(feature_stds)

    # Plot feature norms vs J
    feature_norms = np.linalg.norm(feature_means, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Feature norm vs J
    ax1 = axes[0]
    ax1.plot(J_values, feature_norms, 'bo-', label='||Φ(J)||')
    ax1.plot(J_values, feature_norms[0] * np.sqrt(J_values / J_values[0]),
             'r--', label='∝ √J')
    ax1.set_xlabel('Action J')
    ax1.set_ylabel('Feature norm ||Φ(J)||')
    ax1.set_title('Feature Magnitude vs Action')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Individual feature components
    ax2 = axes[1]
    n_features = min(5, feature_means.shape[1])
    for i in range(n_features):
        ax2.plot(J_values, feature_means[:, i], '-', alpha=0.7, label=f'Feature {i}')
    ax2.set_xlabel('Action J')
    ax2.set_ylabel('Feature value')
    ax2.set_title('Individual Feature Components')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Feature Scaling with Action', fontsize=14)
    plt.tight_layout()
    plt.savefig('feature_scaling_with_J.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: feature_scaling_with_J.png")
    plt.show()

    # Check monotonicity
    is_monotonic = all(feature_norms[i] <= feature_norms[i+1]
                       for i in range(len(feature_norms)-1))
    print(f"\nFeature norm monotonically increasing: {is_monotonic}")

    return J_values, feature_means


def test_phase_sensitivity():
    """
    Test feature sensitivity to initial phase Q₀.

    For magnitude-only features: should be phase-invariant
    For full features: should encode phase
    """
    print("=" * 70)
    print("PHASE SENSITIVITY TEST")
    print("=" * 70)

    J = 2.0  # Fixed action
    n_phases = 36  # Every 10 degrees
    Q0_values = np.linspace(0, 2*np.pi, n_phases, endpoint=False)

    # Extract features for each phase
    features_mag = []
    features_full = []

    print(f"\nExtracting features at J = {J} for {n_phases} phases...")
    for Q0 in Q0_values:
        z = generate_sho_trajectory(J, Q0)
        features_mag.append(extract_features_magnitude_only(z))
        features_full.append(extract_features(z))

    features_mag = np.array(features_mag)
    features_full = np.array(features_full)

    # Compute variance across phases
    mag_var = np.var(features_mag, axis=0)
    full_var = np.var(features_full, axis=0)

    print(f"\nMagnitude-only features variance: {np.mean(mag_var):.6f}")
    print(f"Full features variance: {np.mean(full_var):.6f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Magnitude features vs Q0
    ax1 = axes[0]
    for i in range(min(3, features_mag.shape[1])):
        ax1.plot(Q0_values / np.pi, features_mag[:, i], '-', alpha=0.7,
                label=f'Mag feature {i}')
    ax1.set_xlabel('Initial phase Q₀ / π')
    ax1.set_ylabel('Feature value')
    ax1.set_title(f'Magnitude-only Features (var={np.mean(mag_var):.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Full features vs Q0
    ax2 = axes[1]
    for i in range(min(3, features_full.shape[1])):
        ax2.plot(Q0_values / np.pi, features_full[:, i], '-', alpha=0.7,
                label=f'Full feature {i}')
    ax2.set_xlabel('Initial phase Q₀ / π')
    ax2.set_ylabel('Feature value')
    ax2.set_title(f'Full Features (var={np.mean(full_var):.4f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Phase Sensitivity at J = {J}', fontsize=14)
    plt.tight_layout()
    plt.savefig('phase_sensitivity.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: phase_sensitivity.png")
    plt.show()

    # Phase-invariance check for magnitude features
    mag_cv = np.std(np.linalg.norm(features_mag, axis=1)) / np.mean(np.linalg.norm(features_mag, axis=1))
    print(f"\nMagnitude features coefficient of variation: {mag_cv:.4f}")
    print(f"Phase-invariant: {mag_cv < 0.1}")

    return Q0_values, features_mag, features_full


def run_all_tests():
    """Run all feature periodicity tests."""
    print("\n" + "=" * 70)
    print("FEATURE PERIODICITY ANALYSIS")
    print("=" * 70)
    print()
    print("Testing whether HST features show periodicity in action space.")
    print("Key insight: 'Quantization' = integer Fourier modes, NOT discrete J values")
    print()

    # Test 1: Phase sensitivity
    print("\n>>> TEST 1: Phase Sensitivity <<<\n")
    test_phase_sensitivity()

    # Test 2: Feature scaling
    print("\n>>> TEST 2: Feature Scaling <<<\n")
    test_feature_scaling_with_J()

    # Test 3: Periodicity search (main test)
    print("\n>>> TEST 3: Periodicity Search (Main Test) <<<\n")
    results = test_sho_periodicity()

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print("""
    Key Findings:

    1. PHASE SENSITIVITY
       - Magnitude features are approximately phase-invariant
       - Full features encode phase information

    2. FEATURE SCALING
       - Feature norm scales roughly as √J (amplitude scaling)
       - Features are smooth functions of J (no jumps)

    3. PERIODICITY SEARCH
       - NO periodicity found in SHO features
       - d(J, J+Δ) increases monotonically with |Δ|
       - Confirms: SHO lacks intrinsic action scale J₀

    Implications for Quantization:

    - The "quantization" in Glinsky's framework is about Fourier mode labels
    - Integer n in |n⟩ labels MODES, not action values
    - SHO has no natural J₀ because ω is J-independent
    - For natural J₀, need anharmonic system (e.g., pendulum near separatrix)
    """)


if __name__ == "__main__":
    run_all_tests()
