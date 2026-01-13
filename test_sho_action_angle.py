"""
Simple Harmonic Oscillator Action-Angle Validation

SHO: H = p²/2m + mω²q²/2

Key formulas:
- P = (p² + m²ω²q²) / (2mω) = E/ω
- Q = arctan2(mωq, p)  ← Uses both q and p, no branch cut!
- S(q, P) = P·arcsin(q/q_max) + (q/2)·√(2mωP - m²ω²q²)

CRITICAL: The generating function has TWO branches (p = ±√...).
Must use sign(cos(Q)) to pick correct branch when comparing ∂S/∂q to p.
"""

import numpy as np
from action_angle_utils import angular_distance, wrap_to_2pi, unwrap_angle, safe_points_mask


def sho_action_angle(q, p, omega=1.0, m=1.0):
    """
    (q, p) → (P, Q) for SHO.

    P = E/ω = (p² + m²ω²q²) / (2mω)
    Q = arctan2(mωq, p)  ← Full 2π range, no singularity
    """
    P = (p**2 + m**2 * omega**2 * q**2) / (2 * m * omega)
    Q = np.arctan2(m * omega * q, p)
    Q = wrap_to_2pi(Q)  # Ensure [0, 2π)
    return P, Q


def sho_from_action_angle(P, Q, omega=1.0, m=1.0):
    """
    (P, Q) → (q, p) for SHO.

    q = √(2P/mω) · sin(Q)
    p = √(2mωP) · cos(Q)
    """
    # Handle P=0 case
    if np.isscalar(P):
        if P <= 0:
            return 0.0, 0.0
    else:
        P = np.maximum(P, 0)

    q_max = np.sqrt(2 * P / (m * omega))
    p_max = np.sqrt(2 * m * omega * P)

    q = q_max * np.sin(Q)
    p = p_max * np.cos(Q)
    return q, p


def sho_generating_function(q, P, omega=1.0, m=1.0):
    """
    Type-2 generating function S(q, P).

    S = P·arcsin(q/q_max) + (q/2)·√(2mωP - m²ω²q²)

    WARNING: Has branch cuts at q = ±q_max.
    Only use at safe points where |q| < (1-ε)·q_max.

    Returns S (scalar or array)
    """
    if P <= 0:
        return 0.0

    q_max = np.sqrt(2 * P / (m * omega))

    # NO CLAMPING - caller must ensure safe points
    q_normalized = q / q_max

    # Check for unsafe points
    if np.any(np.abs(q_normalized) >= 1.0):
        raise ValueError(f"q={q} too close to turning point q_max={q_max}. Use safe_points_mask().")

    arcsin_term = P * np.arcsin(q_normalized)
    sqrt_arg = 2 * m * omega * P - m**2 * omega**2 * q**2
    sqrt_term = (q / 2) * np.sqrt(sqrt_arg)

    S = arcsin_term + sqrt_term
    return S


def sho_dS_dq_expected(q, P, Q_true, omega=1.0, m=1.0):
    """
    Expected ∂S/∂q = p, with correct branch/sign.

    CRITICAL: p = sign(cos(Q)) · √(2mωP - m²ω²q²)

    The sign comes from which half of the orbit we're on.
    Use Q_true (from arctan2) to determine the sign.
    """
    sqrt_arg = 2 * m * omega * P - m**2 * omega**2 * q**2
    magnitude = np.sqrt(np.maximum(0, sqrt_arg))

    # Sign from cos(Q): positive for Q ∈ (-π/2, π/2), negative otherwise
    sign = np.sign(np.cos(Q_true))
    if sign == 0:
        sign = 1  # At turning point, arbitrary

    return sign * magnitude


def sho_dS_dP_expected(q, P, omega=1.0, m=1.0):
    """
    Expected ∂S/∂P = Q.

    Q = arcsin(q/q_max) on the principal branch.
    Compare using angular_distance, not raw subtraction.
    """
    if P <= 0:
        return 0.0

    q_max = np.sqrt(2 * P / (m * omega))
    q_normalized = np.clip(q / q_max, -0.999, 0.999)

    # Principal branch of arcsin gives Q ∈ [-π/2, π/2]
    Q = np.arcsin(q_normalized)

    # Wrap to [0, 2π) for comparison
    return wrap_to_2pi(Q)


def test_sho():
    """Full test suite for SHO"""

    print("=" * 60)
    print("SIMPLE HARMONIC OSCILLATOR VALIDATION")
    print("=" * 60)

    omega = 1.0
    m = 1.0

    # Test 1: Roundtrip (q, p) → (P, Q) → (q', p')
    print("\n[Test 1] Roundtrip")
    P_test = 2.0
    all_passed = True
    for Q in np.linspace(0.1, 2*np.pi - 0.1, 8):
        q, p = sho_from_action_angle(P_test, Q, omega, m)
        P2, Q2 = sho_action_angle(q, p, omega, m)
        q2, p2 = sho_from_action_angle(P2, Q2, omega, m)

        q_err = abs(q - q2)
        p_err = abs(p - p2)
        P_err = abs(P_test - P2)
        Q_err = angular_distance(Q, Q2)

        status = "✓" if max(q_err, p_err, P_err, Q_err) < 1e-10 else "✗"
        print(f"  Q={Q:.2f}: q_err={q_err:.2e}, p_err={p_err:.2e}, P_err={P_err:.2e}, Q_err={Q_err:.2e} {status}")
        if max(q_err, p_err, P_err, Q_err) >= 1e-10:
            all_passed = False

    if all_passed:
        print("  PASSED")
    else:
        print("  FAILED")
        return False

    # Test 2: P conservation along trajectory
    print("\n[Test 2] P conservation")
    P_true = 1.5
    Q0 = 0.3
    dt = 0.01
    t = np.arange(0, 20, dt)
    Q_traj = wrap_to_2pi(Q0 + omega * t)

    P_values = []
    for Q in Q_traj:
        q, p = sho_from_action_angle(P_true, Q, omega, m)
        P_measured, _ = sho_action_angle(q, p, omega, m)
        P_values.append(P_measured)

    P_values = np.array(P_values)
    P_err = np.std(P_values) / np.mean(P_values)
    print(f"  P relative std: {P_err:.2e} (should be ~0)")
    if P_err >= 1e-10:
        print("  FAILED")
        return False
    print("  PASSED")

    # Test 3: Q advances linearly at rate ω
    print("\n[Test 3] Q linear evolution")
    Q_measured = []
    for Q in Q_traj:
        q, p = sho_from_action_angle(P_true, Q, omega, m)
        _, Q_meas = sho_action_angle(q, p, omega, m)
        Q_measured.append(Q_meas)

    Q_measured = np.array(Q_measured)
    Q_unwrapped = unwrap_angle(Q_measured)
    omega_meas = np.mean(np.diff(Q_unwrapped) / dt)
    omega_err = abs(omega_meas - omega)
    print(f"  ω measured: {omega_meas:.6f}, expected: {omega:.6f}, error: {omega_err:.2e}")
    if omega_err >= 1e-6:
        print("  FAILED")
        return False
    print("  PASSED")

    # Test 4: Generating function derivatives at SAFE POINTS
    print("\n[Test 4] Generating function derivatives (safe points only)")
    print("         NOTE: Type-2 S(q,P) only covers positive-p branch (cos(Q) > 0)")

    epsilon = 1e-2  # Stay away from turning points
    dq = 1e-7
    dP = 1e-7
    all_passed = True

    for P in [0.5, 1.0, 2.0]:
        q_max = np.sqrt(2 * P / (m * omega))

        # CRITICAL: Only test Q values where cos(Q) > 0 (positive p branch)
        # The Type-2 generating function S(q,P) = P·arcsin(q/q_max) + ...
        # only covers the branch where p = +√(...), i.e., Q ∈ [0, π/2) ∪ (3π/2, 2π]
        # Values in [π/2, 3π/2] are on the negative-p branch → different S formula
        safe_Q_values = [0.2, 0.5, 1.0, 5.0, 5.5, 6.0]  # cos(Q) > 0 for all

        for Q_true in safe_Q_values:
            # Skip if on negative-p branch
            if np.cos(Q_true) <= 0.1:  # Small margin to avoid near-turning-point issues
                continue

            q, p = sho_from_action_angle(P, Q_true, omega, m)

            # Check if safe from turning point
            if abs(q) > (1 - epsilon) * q_max:
                continue

            try:
                S = sho_generating_function(q, P, omega, m)

                # ∂S/∂q vs p (on positive branch, dS/dq = +√(...) = p)
                S_plus_q = sho_generating_function(q + dq, P, omega, m)
                dS_dq = (S_plus_q - S) / dq

                # On positive-p branch, ∂S/∂q should equal p directly
                err_p_actual = abs(dS_dq - p)

                # ∂S/∂P vs Q (using angular distance!)
                S_plus_P = sho_generating_function(q, P + dP, omega, m)
                dS_dP = (S_plus_P - S) / dP
                Q_expected = sho_dS_dP_expected(q, P, omega, m)
                err_Q = angular_distance(dS_dP, Q_expected)

                status = "✓" if err_p_actual < 1e-4 else "✗"
                print(f"  P={P:.1f}, Q={Q_true:.1f}: |∂S/∂q - p|={err_p_actual:.2e}, |∂S/∂P - Q|={err_Q:.2e} {status}")

                if err_p_actual >= 1e-4:
                    all_passed = False

            except ValueError as e:
                print(f"  P={P:.1f}, Q={Q_true:.1f}: SKIPPED (near boundary)")

    if all_passed:
        print("  PASSED")
    else:
        print("  FAILED")
        return False

    # Test 5: Taylor expansion near Q=0 (local validation)
    print("\n[Test 5] Taylor expansion near Q=0")
    P = 1.0
    q_max = np.sqrt(2 * P / (m * omega))
    p_max = np.sqrt(2 * m * omega * P)

    all_passed = True
    # Near Q=0: q ≈ q_max·Q, p ≈ p_max·(1 - Q²/2)
    for Q_small in [0.01, 0.05, 0.1]:
        q_exact, p_exact = sho_from_action_angle(P, Q_small, omega, m)

        q_taylor = q_max * Q_small  # First order
        p_taylor = p_max * (1 - Q_small**2 / 2)  # Second order

        q_err = abs(q_exact - q_taylor) / abs(q_exact) if abs(q_exact) > 1e-10 else abs(q_exact - q_taylor)
        p_err = abs(p_exact - p_taylor) / abs(p_exact) if abs(p_exact) > 1e-10 else abs(p_exact - p_taylor)

        # Taylor should be accurate to O(Q³) for q, O(Q⁴) for p
        q_ok = q_err < Q_small**2
        p_ok = p_err < Q_small**2
        status = "✓" if (q_ok and p_ok) else "✗"

        print(f"  Q={Q_small:.2f}: q_rel_err={q_err:.2e}, p_rel_err={p_err:.2e} {status}")

        if not (q_ok and p_ok):
            all_passed = False

    if all_passed:
        print("  PASSED")
    else:
        print("  FAILED")
        return False

    # Test 6: Energy consistency
    print("\n[Test 6] Energy E = P·ω consistency")
    all_passed = True
    for P in [0.5, 1.0, 2.0, 5.0]:
        for Q in [0.3, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]:
            q, p = sho_from_action_angle(P, Q, omega, m)
            E = p**2 / (2*m) + m * omega**2 * q**2 / 2
            E_expected = P * omega

            err = abs(E - E_expected)
            status = "✓" if err < 1e-10 else "✗"
            print(f"  P={P:.1f}, Q={Q:.2f}: E={E:.6f}, P·ω={E_expected:.6f}, err={err:.2e} {status}")

            if err >= 1e-10:
                all_passed = False

    if all_passed:
        print("  PASSED")
    else:
        print("  FAILED")
        return False

    print("\n" + "=" * 60)
    print("SHO: ALL TESTS PASSED ✓")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_sho()
