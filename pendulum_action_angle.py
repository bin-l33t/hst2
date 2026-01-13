"""
Pendulum Action-Angle Coordinates (Libration Regime)

Hamiltonian: H = p²/2 - cos(q)  (mass = 1, length = 1, g = 1)

Libration regime: E ∈ [-1, 1)  (oscillates within well)
Separatrix: E = 1 (infinite period)

Key formulas:
- Modulus: m = k² = (1 + E) / 2, where E ∈ [-1, 1)
- Action: J(E) = (8/π) · [E(m) - (1-m)·K(m)]
- Frequency: ω(E) = π / (2·K(m))
- Position: sin(q/2) = k · sn(u, k) where u = (2K/π)·Q
- Momentum: p = 2k · cn(u, k)

Limits:
- E → -1: J → 0, ω → 1 (reduces to SHO)
- E → 1: J → 8/π ≈ 2.546, ω → 0 (separatrix)

The separatrix action J_sep = 8/π ≈ 2.546 is the natural scale.
"""

import numpy as np
from scipy.special import ellipk, ellipe, ellipj
from scipy.optimize import brentq
from action_angle_utils import wrap_to_2pi


# Separatrix action - the natural scale
J_SEPARATRIX = 8 / np.pi  # ≈ 2.5465


def pendulum_energy(q, p):
    """
    Compute pendulum energy E = p²/2 - cos(q).

    Parameters
    ----------
    q : float or array
        Position (angle) in radians
    p : float or array
        Momentum (angular velocity)

    Returns
    -------
    E : float or array
        Energy in range [-1, ∞)
        E < 1: libration (oscillation)
        E = 1: separatrix
        E > 1: rotation
    """
    return p**2 / 2 - np.cos(q)


def pendulum_modulus_from_energy(E):
    """
    Compute elliptic modulus m = k² from energy.

    For libration: m = (1 + E) / 2, where E ∈ [-1, 1)

    Parameters
    ----------
    E : float
        Energy in range [-1, 1) for libration

    Returns
    -------
    m : float
        Elliptic modulus (parameter for scipy functions)
        m ∈ [0, 1)
    """
    if np.any(E >= 1):
        raise ValueError(f"Energy {E} >= 1: not in libration regime")
    if np.any(E < -1):
        raise ValueError(f"Energy {E} < -1: below minimum")

    m = (1 + E) / 2
    return m


def pendulum_action_from_energy(E):
    """
    Compute action J from energy E.

    J(E) = (8/π) · [E(m) - (1-m)·K(m)]

    where m = (1 + E) / 2, and E(m), K(m) are complete elliptic integrals.

    Limits:
    - E → -1: J → 0
    - E → 1: J → 8/π ≈ 2.546

    Parameters
    ----------
    E : float
        Energy in range [-1, 1) for libration

    Returns
    -------
    J : float
        Action in range [0, 8/π)
    """
    m = pendulum_modulus_from_energy(E)

    # Handle m → 0 (E → -1) limit analytically
    if np.isscalar(m):
        if m < 1e-10:
            # Small m expansion: J ≈ (E + 1) (matches SHO)
            return E + 1
        if m > 1 - 1e-10:
            # Approaching separatrix
            return J_SEPARATRIX * (1 - 1e-10)

    K_val = ellipk(m)
    E_val = ellipe(m)

    J = (8 / np.pi) * (E_val - (1 - m) * K_val)

    return J


def pendulum_energy_from_action(J):
    """
    Compute energy E from action J by inverting J(E).

    Uses root finding since J(E) is monotonic.

    Parameters
    ----------
    J : float
        Action in range [0, 8/π)

    Returns
    -------
    E : float
        Energy in range [-1, 1)
    """
    if J < 0:
        raise ValueError(f"Action {J} < 0: invalid")
    if J >= J_SEPARATRIX:
        raise ValueError(f"Action {J} >= J_sep = {J_SEPARATRIX}: not in libration regime")

    # Handle small J analytically (SHO limit)
    if J < 1e-6:
        return -1 + J  # J ≈ E + 1 for small oscillations

    # Root finding: solve J(E) - J = 0
    def residual(E):
        return pendulum_action_from_energy(E) - J

    # Search in valid range, staying away from separatrix
    E_min = -1 + 1e-10
    E_max = 1 - 1e-8

    try:
        E = brentq(residual, E_min, E_max)
    except ValueError:
        # Fallback for edge cases
        E = -1 + J  # SHO approximation

    return E


def pendulum_omega_from_energy(E):
    """
    Compute angular frequency ω from energy.

    ω(E) = π / (2·K(m))

    Limits:
    - E → -1: ω → 1 (SHO frequency)
    - E → 1: ω → 0 (infinite period at separatrix)

    Parameters
    ----------
    E : float
        Energy in range [-1, 1)

    Returns
    -------
    omega : float
        Angular frequency in range (0, 1]
    """
    m = pendulum_modulus_from_energy(E)

    # Handle m → 0 limit
    if np.isscalar(m) and m < 1e-10:
        return 1.0  # SHO limit

    K_val = ellipk(m)
    omega = np.pi / (2 * K_val)

    return omega


def pendulum_omega_from_action(J):
    """
    Compute angular frequency ω from action J.

    Convenience function: ω(J) = ω(E(J))
    """
    E = pendulum_energy_from_action(J)
    return pendulum_omega_from_energy(E)


def pendulum_action_angle(q, p):
    """
    Transform (q, p) → (J, Q) for pendulum in libration regime.

    Parameters
    ----------
    q : float
        Position (angle) in radians, typically in [-π, π]
    p : float
        Momentum

    Returns
    -------
    J : float
        Action
    Q : float
        Angle variable in [0, 2π)
    """
    # Compute energy
    E = pendulum_energy(q, p)

    if E >= 1:
        raise ValueError(f"Energy {E} >= 1: rotation regime not implemented")

    # Compute action from energy
    J = pendulum_action_from_energy(E)

    # Compute angle Q
    # The angle variable is related to the Jacobi amplitude
    m = pendulum_modulus_from_energy(E)
    k = np.sqrt(m)

    # Handle small oscillations (SHO limit)
    if m < 1e-10:
        # SHO: Q = arctan2(ω·q, p) where ω = 1
        Q = np.arctan2(q, p)
        return J, wrap_to_2pi(Q)

    K_val = ellipk(m)

    # From sin(q/2) = k · sn(u, k), solve for u
    # Then Q = (π / 2K) · u
    sin_q2 = np.sin(q / 2)
    sn_val = sin_q2 / k if k > 1e-10 else 0

    # Handle turning points specially (p ≈ 0)
    # At turning points: u = K(m), Q = π/2 (positive q) or Q = 3π/2 (negative q)
    if abs(p) < 1e-10:
        if q > 0:
            Q = np.pi / 2  # Positive turning point
        elif q < 0:
            Q = 3 * np.pi / 2  # Negative turning point
        else:
            Q = 0  # At equilibrium
        return J, wrap_to_2pi(Q)

    # Clamp to valid range for arcsn
    sn_val = np.clip(sn_val, -1 + 1e-10, 1 - 1e-10)

    # Use scipy's ellipj to find u from sn
    # We need the inverse: u = F(arcsin(sn), k) where F is incomplete elliptic integral
    # But scipy doesn't have F directly. Use the Jacobi amplitude instead.
    from scipy.special import ellipkinc

    # arcsin(sn) gives the amplitude φ
    phi = np.arcsin(sn_val)

    # u = F(φ, m) where F is incomplete elliptic integral of first kind
    # scipy.special.ellipkinc(phi, m) computes F(phi, m)
    u = ellipkinc(phi, m)

    # Determine which half of the cycle we're in using momentum sign
    # For libration: p > 0 means Q ∈ [0, π), p < 0 means Q ∈ [π, 2π)
    if p < 0:
        u = 2 * K_val - u  # Reflect to second half

    # Convert u to Q
    Q = (np.pi / (2 * K_val)) * u

    return J, wrap_to_2pi(Q)


def pendulum_from_action_angle(J, Q):
    """
    Transform (J, Q) → (q, p) for pendulum in libration regime.

    Parameters
    ----------
    J : float
        Action in range [0, 8/π)
    Q : float
        Angle variable in [0, 2π)

    Returns
    -------
    q : float
        Position (angle) in radians
    p : float
        Momentum
    """
    if J < 0:
        raise ValueError(f"Action {J} < 0: invalid")
    if J >= J_SEPARATRIX:
        raise ValueError(f"Action {J} >= J_sep: not in libration regime")

    # Handle J → 0 (SHO limit)
    if J < 1e-10:
        q = np.sqrt(2 * J) * np.sin(Q)
        p = np.sqrt(2 * J) * np.cos(Q)
        return q, p

    # Get energy and modulus from action
    E = pendulum_energy_from_action(J)
    m = pendulum_modulus_from_energy(E)
    k = np.sqrt(m)
    K_val = ellipk(m)

    # Convert Q to u
    # u ∈ [0, 4K) for one full libration, Q ∈ [0, 2π)
    u = (2 * K_val / np.pi) * Q

    # Get Jacobi elliptic functions
    sn, cn, dn, ph = ellipj(u, m)

    # Position: sin(q/2) = k · sn(u, m)
    sin_q2 = k * sn
    sin_q2 = np.clip(sin_q2, -1, 1)  # Numerical safety
    q = 2 * np.arcsin(sin_q2)

    # Momentum: p = 2k · cn(u, m)
    # cn naturally handles the sign (positive for first half, negative for second half)
    p = 2 * k * cn

    return q, p


def generate_pendulum_trajectory(J, Q0, dt=0.01, n_samples=512):
    """
    Generate pendulum trajectory starting from (J, Q0).

    Parameters
    ----------
    J : float
        Action (constant of motion)
    Q0 : float
        Initial angle variable
    dt : float
        Time step
    n_samples : int
        Number of samples

    Returns
    -------
    t : np.ndarray
        Time array
    q : np.ndarray
        Position trajectory
    p : np.ndarray
        Momentum trajectory
    Q : np.ndarray
        Angle variable trajectory
    """
    # Get frequency for this action
    omega = pendulum_omega_from_action(J)

    t = np.arange(n_samples) * dt
    Q_traj = wrap_to_2pi(Q0 + omega * t)

    q = np.zeros(n_samples)
    p = np.zeros(n_samples)

    for i, Q in enumerate(Q_traj):
        q[i], p[i] = pendulum_from_action_angle(J, Q)

    return t, q, p, Q_traj


def validate_pendulum_formulas():
    """
    Quick validation of the pendulum action-angle formulas.
    """
    print("=" * 60)
    print("PENDULUM ACTION-ANGLE FORMULA VALIDATION")
    print("=" * 60)

    print(f"\nSeparatrix action: J_sep = 8/π = {J_SEPARATRIX:.6f}")

    # Test at various energies
    print("\n[Test 1] Action and frequency vs energy")
    print("  E        m        J        ω        (limits)")
    print("  -------- -------- -------- --------")

    test_energies = [-0.99, -0.9, -0.5, 0.0, 0.5, 0.9, 0.99]

    for E in test_energies:
        m = pendulum_modulus_from_energy(E)
        J = pendulum_action_from_energy(E)
        omega = pendulum_omega_from_energy(E)

        note = ""
        if E < -0.9:
            note = "(→ SHO)"
        elif E > 0.9:
            note = "(→ sep)"

        print(f"  {E:+.2f}     {m:.4f}   {J:.4f}   {omega:.4f}   {note}")

    # Test roundtrip E → J → E
    print("\n[Test 2] Energy roundtrip E → J → E")
    for E in test_energies[:-1]:  # Skip E=0.99 (too close to separatrix)
        J = pendulum_action_from_energy(E)
        E_back = pendulum_energy_from_action(J)
        err = abs(E - E_back)
        status = "✓" if err < 1e-8 else "✗"
        print(f"  E={E:+.2f}: J={J:.4f} → E'={E_back:+.6f}, err={err:.2e} {status}")

    # Test (q, p) → (J, Q) → (q', p') roundtrip
    print("\n[Test 3] Coordinate roundtrip (q, p) → (J, Q) → (q', p')")

    test_cases = [
        (0.5, 0.3),   # Small oscillation
        (1.0, 0.5),   # Medium oscillation
        (1.5, 0.2),   # Larger oscillation
        (0.3, -0.4),  # Negative momentum
    ]

    for q, p in test_cases:
        E = pendulum_energy(q, p)
        if E >= 1:
            print(f"  q={q}, p={p}: SKIP (E={E:.2f} >= 1, rotation)")
            continue

        try:
            J, Q = pendulum_action_angle(q, p)
            q2, p2 = pendulum_from_action_angle(J, Q)

            q_err = abs(q - q2)
            p_err = abs(p - p2)

            status = "✓" if max(q_err, p_err) < 1e-6 else "✗"
            print(f"  q={q:.1f}, p={p:+.1f}: J={J:.4f}, Q={Q:.2f} → "
                  f"q_err={q_err:.2e}, p_err={p_err:.2e} {status}")
        except Exception as e:
            print(f"  q={q:.1f}, p={p:+.1f}: ERROR - {e}")

    # Test small amplitude limit (should match SHO)
    print("\n[Test 4] Small amplitude limit (should match SHO)")
    for q_small in [0.01, 0.05, 0.1]:
        p_small = 0.0  # At turning point

        E = pendulum_energy(q_small, p_small)
        J = pendulum_action_from_energy(E)
        omega = pendulum_omega_from_energy(E)

        # SHO would have: J_sho = E_sho / ω = (p²/2 + q²/2) / 1 = q²/2
        J_sho = q_small**2 / 2

        print(f"  q={q_small:.2f}: J={J:.6f}, J_sho={J_sho:.6f}, "
              f"ω={omega:.4f}, rel_err={(J-J_sho)/J_sho:.2e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    validate_pendulum_formulas()
