"""
Free Rotor Action-Angle Validation

Free rotor: H = L²/2I, particle on circle.

This is the trivial case:
- q = θ ∈ [0, 2π)  (angle on circle)
- p = L (angular momentum, conserved)
- P = L (action = angular momentum)
- Q = θ (angle variable = physical angle)
- S(θ, L) = L·θ (generating function)

No branch cuts! Perfect for validating the test harness.
"""

import numpy as np
from action_angle_utils import angular_distance, wrap_to_2pi, unwrap_angle


def rotor_action_angle(theta, L):
    """
    (θ, L) → (P, Q) for free rotor.
    Trivial: P = L, Q = θ
    """
    P = L
    Q = wrap_to_2pi(theta)
    return P, Q


def rotor_from_action_angle(P, Q):
    """
    (P, Q) → (θ, L) for free rotor.
    Trivial: θ = Q, L = P
    """
    theta = wrap_to_2pi(Q)
    L = P
    return theta, L


def rotor_generating_function(theta, L):
    """
    S(θ, L) = L·θ

    Check:
    - ∂S/∂θ = L = p  ✓
    - ∂S/∂L = θ = Q  ✓
    """
    return L * theta


def test_rotor():
    """Full test suite for rotor"""

    print("=" * 60)
    print("FREE ROTOR VALIDATION")
    print("=" * 60)

    # Test 1: Roundtrip (θ, L) → (P, Q) → (θ', L')
    print("\n[Test 1] Roundtrip")
    L_test = 2.5
    all_passed = True
    for theta in [0.0, 0.5, np.pi, 1.5*np.pi, 2*np.pi - 0.1]:
        P, Q = rotor_action_angle(theta, L_test)
        theta2, L2 = rotor_from_action_angle(P, Q)

        theta_err = angular_distance(theta, theta2)
        L_err = abs(L_test - L2)

        status = "✓" if (theta_err < 1e-10 and L_err < 1e-10) else "✗"
        print(f"  θ={theta:.3f}: θ_err={theta_err:.2e}, L_err={L_err:.2e} {status}")
        if theta_err >= 1e-10 or L_err >= 1e-10:
            all_passed = False

    if all_passed:
        print("  PASSED")
    else:
        print("  FAILED")
        return False

    # Test 2: Generating function derivatives
    print("\n[Test 2] ∂S/∂θ = L, ∂S/∂L = θ")
    dtheta = 1e-7
    dL = 1e-7
    all_passed = True

    for theta in [0.3, 1.0, 2.0, 4.0]:
        for L in [0.5, 1.0, 2.0]:
            S = rotor_generating_function(theta, L)

            # ∂S/∂θ = L
            dS_dtheta = (rotor_generating_function(theta + dtheta, L) - S) / dtheta
            err_p = abs(dS_dtheta - L)

            # ∂S/∂L = θ
            dS_dL = (rotor_generating_function(theta, L + dL) - S) / dL
            err_Q = abs(dS_dL - theta)

            status = "✓" if (err_p < 1e-6 and err_Q < 1e-6) else "✗"
            print(f"  θ={theta:.1f}, L={L:.1f}: |∂S/∂θ - L|={err_p:.2e}, |∂S/∂L - θ|={err_Q:.2e} {status}")
            if err_p >= 1e-6 or err_Q >= 1e-6:
                all_passed = False

    if all_passed:
        print("  PASSED")
    else:
        print("  FAILED")
        return False

    # Test 3: Trajectory (L conserved, θ advances linearly)
    print("\n[Test 3] Trajectory: L conserved, θ linear")
    L_true = 1.5
    I = 1.0  # Moment of inertia
    omega = L_true / I  # Angular velocity
    dt = 0.01
    t = np.arange(0, 10, dt)
    theta_traj = (omega * t) % (2 * np.pi)
    L_traj = np.full_like(t, L_true)

    # Extract P, Q along trajectory
    P_traj = []
    Q_traj = []
    for theta, L in zip(theta_traj, L_traj):
        P, Q = rotor_action_angle(theta, L)
        P_traj.append(P)
        Q_traj.append(Q)

    P_traj = np.array(P_traj)
    Q_traj = np.array(Q_traj)

    # P should be constant
    P_std = np.std(P_traj)
    print(f"  P std: {P_std:.2e} (should be ~0)")
    if P_std >= 1e-10:
        print("  FAILED: P not conserved")
        return False

    # Q should advance at rate ω
    Q_unwrapped = unwrap_angle(Q_traj)
    omega_measured = np.mean(np.diff(Q_unwrapped) / dt)
    omega_err = abs(omega_measured - omega)
    print(f"  ω measured: {omega_measured:.6f}, expected: {omega:.6f}, error: {omega_err:.2e}")
    if omega_err >= 1e-4:
        print("  FAILED: ω incorrect")
        return False

    print("  PASSED")

    # Test 4: Different L values (positive and negative)
    print("\n[Test 4] Sign of L (direction of rotation)")
    for L in [-2.0, -0.5, 0.5, 2.0]:
        I = 1.0
        omega = L / I
        dt = 0.01
        t = np.arange(0, 5, dt)

        theta_traj = wrap_to_2pi(omega * t)
        P_traj, Q_traj = [], []
        for theta in theta_traj:
            P, Q = rotor_action_angle(theta, L)
            P_traj.append(P)
            Q_traj.append(Q)

        P_traj = np.array(P_traj)
        Q_traj = np.array(Q_traj)

        # P = L should be preserved (including sign)
        P_err = np.max(np.abs(P_traj - L))
        print(f"  L={L:+.1f}: max|P - L| = {P_err:.2e}")
        if P_err >= 1e-10:
            print("  FAILED")
            return False

    print("  PASSED")

    print("\n" + "=" * 60)
    print("ROTOR: ALL TESTS PASSED ✓")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_rotor()
