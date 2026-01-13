"""
Pendulum Action-Angle Validation Tests

Comprehensive tests for pendulum action-angle coordinates in libration regime.

Tests:
1. Roundtrip: (q, p) → (J, Q) → (q', p')
2. J conservation along trajectory
3. Q linear evolution: dQ/dt = ω(J)
4. Energy consistency: E = p²/2 - cos(q)
5. Small amplitude limit: matches SHO
6. Frequency matches numerical integration
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from action_angle_utils import angular_distance, wrap_to_2pi, unwrap_angle
from pendulum_action_angle import (
    pendulum_energy, pendulum_modulus_from_energy,
    pendulum_action_from_energy, pendulum_energy_from_action,
    pendulum_omega_from_energy, pendulum_omega_from_action,
    pendulum_action_angle, pendulum_from_action_angle,
    generate_pendulum_trajectory, J_SEPARATRIX
)


def test_energy_roundtrip():
    """Test E → J → E roundtrip."""
    print("=" * 60)
    print("[Test 1] Energy Roundtrip: E → J → E")
    print("=" * 60)

    test_energies = [-0.99, -0.9, -0.5, 0.0, 0.5, 0.9, 0.95]
    all_passed = True

    for E in test_energies:
        J = pendulum_action_from_energy(E)
        E_back = pendulum_energy_from_action(J)
        err = abs(E - E_back)

        status = "✓" if err < 1e-8 else "✗"
        print(f"  E={E:+.2f} → J={J:.4f} → E'={E_back:+.6f}, err={err:.2e} {status}")

        if err >= 1e-8:
            all_passed = False

    if all_passed:
        print("  PASSED")
    else:
        print("  FAILED")

    return all_passed


def test_coordinate_roundtrip():
    """Test (q, p) → (J, Q) → (q', p') roundtrip."""
    print("\n" + "=" * 60)
    print("[Test 2] Coordinate Roundtrip: (q, p) → (J, Q) → (q', p')")
    print("=" * 60)

    # Test various (q, p) points
    test_cases = [
        # (q, p) - small oscillations
        (0.1, 0.0),
        (0.0, 0.1),
        (0.2, 0.1),
        # Medium oscillations
        (0.5, 0.3),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.5),
        # Larger oscillations
        (1.5, 0.2),
        (2.0, 0.0),
        # Negative momentum
        (0.5, -0.3),
        (1.0, -0.5),
    ]

    all_passed = True

    for q, p in test_cases:
        E = pendulum_energy(q, p)
        if E >= 1:
            print(f"  q={q:.1f}, p={p:+.1f}: SKIP (E={E:.2f} >= 1)")
            continue

        try:
            J, Q = pendulum_action_angle(q, p)
            q2, p2 = pendulum_from_action_angle(J, Q)

            q_err = abs(q - q2)
            p_err = abs(p - p2)
            max_err = max(q_err, p_err)

            status = "✓" if max_err < 1e-6 else "✗"
            print(f"  q={q:+.1f}, p={p:+.1f}: E={E:+.2f}, J={J:.3f}, Q={Q:.2f} "
                  f"→ err={max_err:.2e} {status}")

            if max_err >= 1e-6:
                all_passed = False

        except Exception as e:
            print(f"  q={q:.1f}, p={p:+.1f}: ERROR - {e}")
            all_passed = False

    if all_passed:
        print("  PASSED")
    else:
        print("  FAILED")

    return all_passed


def test_J_conservation():
    """Test that J is conserved along numerically integrated trajectory."""
    print("\n" + "=" * 60)
    print("[Test 3] J Conservation Along Trajectory")
    print("=" * 60)

    # Pendulum ODE
    def pendulum_ode(y, t):
        q, p = y
        return [p, -np.sin(q)]

    # Test at various initial conditions
    initial_conditions = [
        (0.5, 0.0),   # Small oscillation at turning point
        (1.0, 0.5),   # Medium oscillation
        (1.5, 0.2),   # Larger oscillation
        (2.0, 0.0),   # Near separatrix
    ]

    dt = 0.01
    T_max = 50.0
    t = np.arange(0, T_max, dt)

    all_passed = True

    for q0, p0 in initial_conditions:
        E0 = pendulum_energy(q0, p0)
        if E0 >= 1:
            print(f"  q0={q0:.1f}, p0={p0:.1f}: SKIP (E={E0:.2f} >= 1)")
            continue

        # Integrate trajectory
        y0 = [q0, p0]
        sol = odeint(pendulum_ode, y0, t)
        q_traj, p_traj = sol[:, 0], sol[:, 1]

        # Compute J at each point
        J_values = []
        for q, p in zip(q_traj, p_traj):
            E = pendulum_energy(q, p)
            if E < 1:
                J = pendulum_action_from_energy(E)
                J_values.append(J)

        J_values = np.array(J_values)

        if len(J_values) < 10:
            print(f"  q0={q0:.1f}, p0={p0:.1f}: SKIP (trajectory left libration)")
            continue

        J_mean = np.mean(J_values)
        J_std = np.std(J_values)
        J_rel_std = J_std / J_mean if J_mean > 0 else 0

        status = "✓" if J_rel_std < 1e-4 else "✗"
        print(f"  q0={q0:.1f}, p0={p0:.1f}: E={E0:.2f}, J_mean={J_mean:.4f}, "
              f"J_std/J={J_rel_std:.2e} {status}")

        if J_rel_std >= 1e-4:
            all_passed = False

    if all_passed:
        print("  PASSED")
    else:
        print("  FAILED")

    return all_passed


def test_Q_evolution():
    """Test that Q evolves linearly: dQ/dt = ω(J)."""
    print("\n" + "=" * 60)
    print("[Test 4] Q Linear Evolution: dQ/dt = ω(J)")
    print("=" * 60)

    # Test at various J values
    J_values = [0.1, 0.5, 1.0, 1.5, 2.0]

    dt = 0.01
    n_samples = 2000  # Cover multiple periods
    all_passed = True

    for J in J_values:
        if J >= J_SEPARATRIX:
            print(f"  J={J:.2f}: SKIP (>= J_sep)")
            continue

        omega = pendulum_omega_from_action(J)
        Q0 = 0.3  # Arbitrary initial phase

        t, q_traj, p_traj, Q_traj = generate_pendulum_trajectory(
            J, Q0, dt=dt, n_samples=n_samples
        )

        # Unwrap Q to measure slope
        Q_unwrapped = unwrap_angle(Q_traj)

        # Linear fit to unwrapped Q
        coeffs = np.polyfit(t, Q_unwrapped, 1)
        omega_measured = coeffs[0]

        omega_err = abs(omega_measured - omega) / omega

        status = "✓" if omega_err < 1e-3 else "✗"
        print(f"  J={J:.2f}: ω_theory={omega:.4f}, ω_measured={omega_measured:.4f}, "
              f"rel_err={omega_err:.2e} {status}")

        if omega_err >= 1e-3:
            all_passed = False

    if all_passed:
        print("  PASSED")
    else:
        print("  FAILED")

    return all_passed


def test_energy_consistency():
    """Test that reconstructed (q, p) has correct energy."""
    print("\n" + "=" * 60)
    print("[Test 5] Energy Consistency: E = p²/2 - cos(q)")
    print("=" * 60)

    # Test at various (J, Q) combinations
    J_values = [0.1, 0.5, 1.0, 1.5, 2.0]
    Q_values = [0.0, 0.5, 1.0, np.pi, 4.0, 5.5]

    all_passed = True

    for J in J_values:
        if J >= J_SEPARATRIX:
            continue

        E_expected = pendulum_energy_from_action(J)

        for Q in Q_values:
            q, p = pendulum_from_action_angle(J, Q)
            E_computed = pendulum_energy(q, p)

            E_err = abs(E_computed - E_expected)

            status = "✓" if E_err < 1e-8 else "✗"
            if E_err >= 1e-8 or J == J_values[0]:  # Print first J and all failures
                print(f"  J={J:.2f}, Q={Q:.1f}: E_theory={E_expected:.6f}, "
                      f"E_computed={E_computed:.6f}, err={E_err:.2e} {status}")

            if E_err >= 1e-8:
                all_passed = False

    if all_passed:
        print("  (showing first J only, all passed)")
        print("  PASSED")
    else:
        print("  FAILED")

    return all_passed


def test_small_amplitude_limit():
    """Test that small amplitude matches SHO."""
    print("\n" + "=" * 60)
    print("[Test 6] Small Amplitude Limit (Should Match SHO)")
    print("=" * 60)

    # For small oscillations: pendulum ≈ SHO
    # J_sho = E_sho / ω_sho = (p² + q²) / 2 / 1 = (p² + q²) / 2
    # ω_sho = 1

    small_amplitudes = [0.01, 0.05, 0.1, 0.2]
    all_passed = True

    for q0 in small_amplitudes:
        p0 = 0.0  # At turning point

        E = pendulum_energy(q0, p0)
        J = pendulum_action_from_energy(E)
        omega = pendulum_omega_from_energy(E)

        # SHO approximation: J ≈ q²/2 for small q at p=0
        J_sho = q0**2 / 2
        omega_sho = 1.0

        J_err = abs(J - J_sho) / J_sho if J_sho > 0 else abs(J - J_sho)
        omega_err = abs(omega - omega_sho)

        # Error should scale as q⁴ or better
        expected_err_scale = q0**2

        status = "✓" if J_err < 10 * expected_err_scale else "✗"
        print(f"  q0={q0:.2f}: J={J:.6f}, J_sho={J_sho:.6f}, J_err={J_err:.2e}, "
              f"ω={omega:.4f}, ω_sho=1.0 {status}")

        if J_err >= 10 * expected_err_scale:
            all_passed = False

    if all_passed:
        print("  PASSED")
    else:
        print("  FAILED")

    return all_passed


def test_frequency_vs_numerical():
    """Compare ω(J) with frequency from numerical integration."""
    print("\n" + "=" * 60)
    print("[Test 7] Frequency vs Numerical Integration")
    print("=" * 60)

    # Pendulum ODE
    def pendulum_ode(y, t):
        q, p = y
        return [p, -np.sin(q)]

    # Test at various energies
    test_energies = [-0.9, -0.5, 0.0, 0.5, 0.8]
    all_passed = True

    for E in test_energies:
        omega_theory = pendulum_omega_from_energy(E)
        T_theory = 2 * np.pi / omega_theory

        # Initial condition at q = q_max, p = 0
        # From E = -cos(q_max), get q_max = arccos(-E)
        q_max = np.arccos(-E)
        y0 = [q_max, 0.0]

        # Integrate for a few periods
        n_periods = 5
        dt = T_theory / 200
        t_max = n_periods * T_theory
        t = np.arange(0, t_max, dt)

        sol = odeint(pendulum_ode, y0, t)
        q_traj = sol[:, 0]

        # Find zero crossings to measure period
        zero_crossings = []
        for i in range(1, len(q_traj)):
            if q_traj[i-1] * q_traj[i] < 0 and q_traj[i-1] > 0:  # Downward crossing
                # Linear interpolation
                t_cross = t[i-1] + (t[i] - t[i-1]) * (-q_traj[i-1]) / (q_traj[i] - q_traj[i-1])
                zero_crossings.append(t_cross)

        if len(zero_crossings) >= 2:
            periods = np.diff(zero_crossings)
            T_measured = np.mean(periods)
            omega_measured = 2 * np.pi / T_measured

            omega_err = abs(omega_measured - omega_theory) / omega_theory

            status = "✓" if omega_err < 1e-3 else "✗"
            print(f"  E={E:+.1f}: ω_theory={omega_theory:.4f}, ω_num={omega_measured:.4f}, "
                  f"rel_err={omega_err:.2e} {status}")

            if omega_err >= 1e-3:
                all_passed = False
        else:
            print(f"  E={E:+.1f}: Not enough zero crossings")
            all_passed = False

    if all_passed:
        print("  PASSED")
    else:
        print("  FAILED")

    return all_passed


def plot_phase_space():
    """Generate phase space plot showing action-angle coordinates."""
    print("\n" + "=" * 60)
    print("Generating Phase Space Plot...")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Phase portrait with constant J curves
    ax1 = axes[0]

    # Draw separatrix
    q_sep = np.linspace(-np.pi, np.pi, 200)
    p_sep_upper = np.sqrt(2 * (1 + np.cos(q_sep)))
    p_sep_lower = -p_sep_upper
    ax1.plot(q_sep, p_sep_upper, 'r-', linewidth=2, label='Separatrix')
    ax1.plot(q_sep, p_sep_lower, 'r-', linewidth=2)

    # Draw constant J curves
    J_values = [0.2, 0.5, 1.0, 1.5, 2.0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(J_values)))

    for J, color in zip(J_values, colors):
        Q_range = np.linspace(0, 2*np.pi, 100)
        q_curve = []
        p_curve = []
        for Q in Q_range:
            q, p = pendulum_from_action_angle(J, Q)
            q_curve.append(q)
            p_curve.append(p)
        ax1.plot(q_curve, p_curve, color=color, linewidth=1.5, label=f'J={J:.1f}')

    ax1.set_xlabel('q (position)')
    ax1.set_ylabel('p (momentum)')
    ax1.set_title('Pendulum Phase Space')
    ax1.legend(loc='upper right')
    ax1.set_xlim(-3.5, 3.5)
    ax1.set_ylim(-2.5, 2.5)
    ax1.grid(True, alpha=0.3)

    # Right: J and ω vs E
    ax2 = axes[1]

    E_range = np.linspace(-0.99, 0.99, 100)
    J_range = [pendulum_action_from_energy(E) for E in E_range]
    omega_range = [pendulum_omega_from_energy(E) for E in E_range]

    ax2.plot(E_range, J_range, 'b-', linewidth=2, label='J(E)')
    ax2.axhline(y=J_SEPARATRIX, color='r', linestyle='--', label=f'J_sep = 8/π ≈ {J_SEPARATRIX:.3f}')

    ax2_twin = ax2.twinx()
    ax2_twin.plot(E_range, omega_range, 'g-', linewidth=2, label='ω(E)')
    ax2_twin.set_ylabel('ω (frequency)', color='g')
    ax2_twin.tick_params(axis='y', labelcolor='g')

    ax2.set_xlabel('E (energy)')
    ax2.set_ylabel('J (action)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.set_title('Action and Frequency vs Energy')
    ax2.axvline(x=1, color='r', linestyle=':', alpha=0.5, label='Separatrix (E=1)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pendulum_action_angle.png', dpi=150, bbox_inches='tight')
    print(f"Saved: pendulum_action_angle.png")
    plt.show()


def run_all_tests():
    """Run all pendulum action-angle tests."""
    print("\n" + "=" * 70)
    print("PENDULUM ACTION-ANGLE VALIDATION TESTS")
    print("=" * 70)
    print(f"\nSeparatrix action: J_sep = 8/π ≈ {J_SEPARATRIX:.4f}")
    print("Testing libration regime: E ∈ [-1, 1), J ∈ [0, 8/π)")
    print()

    results = {}

    results['energy_roundtrip'] = test_energy_roundtrip()
    results['coordinate_roundtrip'] = test_coordinate_roundtrip()
    results['J_conservation'] = test_J_conservation()
    results['Q_evolution'] = test_Q_evolution()
    results['energy_consistency'] = test_energy_consistency()
    results['small_amplitude'] = test_small_amplitude_limit()
    results['frequency_numerical'] = test_frequency_vs_numerical()

    # Generate plot
    plot_phase_space()

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:25s}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  🎉 ALL TESTS PASSED - Pendulum action-angle coordinates validated!")
    else:
        print("  ⚠ Some tests failed")

    print("\n" + "=" * 70)

    return all_passed


if __name__ == "__main__":
    run_all_tests()
