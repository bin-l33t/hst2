"""
Hamiltonian Systems for Geodesic Property Testing

These systems have CONSERVED ENERGY, so different initial conditions give
different action values P. This allows non-trivial testing of the geodesic
property: ω = f(P).

Systems implemented:
1. Simple Harmonic Oscillator - degenerate (ω = const)
2. Anharmonic Oscillator (Duffing) - ω decreases with E
3. Pendulum - strong ω(E) dependence near separatrix

From Glinsky's transcripts:
"anharmonic phononic field based on the Hamiltonian of a pendulum clock"
The pendulum is his explicit example of a collective system!
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import ellipk


class HamiltonianSystem:
    """Base class for Hamiltonian systems."""

    def __init__(self, name):
        self.name = name

    def hamiltonian(self, q, p):
        """H(q, p) - should be conserved."""
        raise NotImplementedError

    def dynamics(self, t, state):
        """Hamilton's equations: q̇ = ∂H/∂p, ṗ = -∂H/∂q"""
        raise NotImplementedError

    def theoretical_period(self, E):
        """T(E) if analytically known."""
        return None

    def theoretical_omega(self, E):
        """ω(E) = 2π/T(E) if analytically known."""
        T = self.theoretical_period(E)
        if T is not None and T > 0 and np.isfinite(T):
            return 2 * np.pi / T
        return None

    def initial_condition_for_energy(self, E):
        """
        Return (q0, p0) that gives energy E.
        Default: start at q=0 with appropriate p.
        """
        V0 = self.hamiltonian(0, 0)  # Potential at q=0 with p=0
        if E > V0:
            p0 = np.sqrt(2 * (E - V0))
            return 0.0, p0
        else:
            return 0.1, 0.0  # Fallback


class SimpleHarmonicOscillator(HamiltonianSystem):
    """
    H = ½p² + ½ω₀²q²

    Properties:
    - ω = ω₀ (constant, independent of amplitude)
    - All orbits are ellipses in (q, p) space
    - Action: I = E/ω₀

    This is DEGENERATE: ω doesn't depend on I.
    But it's still a valid test - MLP should learn constant ω.
    """

    def __init__(self, omega0=1.0):
        super().__init__("Simple Harmonic Oscillator")
        self.omega0 = omega0

    def hamiltonian(self, q, p):
        return 0.5 * p**2 + 0.5 * self.omega0**2 * q**2

    def dynamics(self, t, state):
        q, p = state
        dq = p
        dp = -self.omega0**2 * q
        return [dq, dp]

    def theoretical_period(self, E):
        return 2 * np.pi / self.omega0

    def action_from_energy(self, E):
        """I = E/ω₀"""
        return E / self.omega0

    def initial_condition_for_energy(self, E):
        """Start at q=0, p=√(2E)"""
        if E > 0:
            return 0.0, np.sqrt(2 * E)
        return 0.1, 0.0


class AnharmonicOscillator(HamiltonianSystem):
    """
    H = ½p² + ½q² + (ε/4)q⁴

    This is the UNDAMPED DUFFING oscillator (no forcing, no damping).

    Properties:
    - Energy E is conserved
    - Period T(E) increases with E (harder spring at large amplitude)
    - ω(E) = 2π/T(E) DECREASES with E

    For small ε, perturbation theory gives:
    ω(E) ≈ 1 - (3ε/8)(2E) + O(ε²)

    This is the KEY TEST: ω genuinely depends on E (equivalently, on action I).
    """

    def __init__(self, epsilon=0.1):
        super().__init__("Anharmonic Oscillator")
        self.epsilon = epsilon

    def hamiltonian(self, q, p):
        return 0.5 * p**2 + 0.5 * q**2 + (self.epsilon / 4) * q**4

    def dynamics(self, t, state):
        q, p = state
        dq = p
        dp = -q - self.epsilon * q**3
        return [dq, dp]

    def theoretical_period(self, E):
        """
        Period via perturbation theory for small ε.
        T ≈ 2π(1 + (3ε/8)(2E))
        """
        # This is approximate but good for ε < 0.5
        return 2 * np.pi * (1 + (3 * self.epsilon / 8) * (2 * E))

    def initial_condition_for_energy(self, E):
        """Start at q=0, p determined from H(0,p) = E"""
        # H(0, p) = ½p² = E → p = √(2E)
        if E > 0:
            return 0.0, np.sqrt(2 * E)
        return 0.1, 0.0


class PendulumOscillator(HamiltonianSystem):
    """
    H = ½p² - cos(q)  (with m=g=L=1)

    Properties:
    - Libration (E < 1): oscillates back and forth
    - Rotation (E > 1): continuous rotation
    - Separatrix (E = 1): infinite period

    Period for libration:
    T(E) = 4K(k) where k² = (1+E)/2, K = complete elliptic integral

    This has STRONG ω(E) dependence near separatrix.
    """

    def __init__(self):
        super().__init__("Pendulum")

    def hamiltonian(self, q, p):
        return 0.5 * p**2 - np.cos(q)

    def dynamics(self, t, state):
        q, p = state
        dq = p
        dp = -np.sin(q)
        return [dq, dp]

    def theoretical_period(self, E):
        """Period via elliptic integral for libration."""
        if E >= 1:
            return np.inf  # Rotation or separatrix

        # k² = (1 + E) / 2
        k2 = (1 + E) / 2
        if k2 >= 1 or k2 < 0:
            return np.inf

        try:
            K = ellipk(k2)
            return 4 * K
        except:
            return np.inf

    def initial_condition_for_energy(self, E):
        """
        Start at q=0, p determined from H(0,p) = E.
        H(0, p) = ½p² - 1 = E → p = √(2(E+1))
        """
        if E > -1:
            return 0.0, np.sqrt(2 * (E + 1))
        return 0.1, 0.0


def simulate_hamiltonian(system, q0, p0, T, dt, rtol=1e-10, atol=1e-12):
    """
    Simulate Hamiltonian system, verify energy conservation.

    Parameters
    ----------
    system : HamiltonianSystem
        The system to simulate
    q0, p0 : float
        Initial conditions
    T : float
        Total time
    dt : float
        Output time step

    Returns
    -------
    t : array
        Time points
    q, p : arrays
        Position and momentum trajectories
    z : array (complex)
        Complex representation z = q + i*p
    E : float
        Energy (should be constant)
    """
    sol = solve_ivp(
        system.dynamics, (0, T), [q0, p0],
        t_eval=np.arange(0, T, dt),
        method='DOP853',  # High-order for Hamiltonian systems
        rtol=rtol, atol=atol
    )

    q = sol.y[0]
    p = sol.y[1]
    z = q + 1j * p

    # Verify energy conservation
    E_initial = system.hamiltonian(q0, p0)
    E_final = system.hamiltonian(q[-1], p[-1])
    E_drift = np.abs(E_final - E_initial) / (np.abs(E_initial) + 1e-10)

    if E_drift > 1e-6:
        print(f"Warning: Energy drift = {E_drift:.2e}")

    return sol.t, q, p, z, E_initial


def generate_ensemble_at_different_energies(system, energies, T=100, dt=0.01):
    """
    Generate trajectories at different energy levels.

    Parameters
    ----------
    system : HamiltonianSystem
        The system to simulate
    energies : array
        Target energy values
    T : float
        Simulation time per trajectory
    dt : float
        Time step

    Returns
    -------
    trajectories : list of complex arrays
        z(t) = q(t) + i*p(t) for each trajectory
    measured_periods : list of floats
        Measured period from zero-crossings
    actual_energies : list of floats
        Actual energy values
    """
    trajectories = []
    measured_periods = []
    actual_energies = []

    for E in energies:
        # Get initial condition for this energy
        q0, p0 = system.initial_condition_for_energy(E)

        try:
            t, q, p, z, E_actual = simulate_hamiltonian(system, q0, p0, T, dt)
        except Exception as e:
            print(f"  Simulation failed for E={E}: {e}")
            continue

        trajectories.append(z)
        actual_energies.append(E_actual)

        # Measure period from zero-crossings of q (positive direction)
        zero_crossings = np.where((q[:-1] < 0) & (q[1:] >= 0))[0]
        if len(zero_crossings) >= 2:
            periods = np.diff(t[zero_crossings])
            period = np.mean(periods)
        else:
            # Try from q maxima
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(q)
            if len(peaks) >= 2:
                period = np.mean(np.diff(t[peaks]))
            else:
                period = T  # Couldn't measure

        measured_periods.append(period)

    return trajectories, measured_periods, actual_energies


def test_system_basics(system, E_test=1.0, T=50):
    """Quick test of a Hamiltonian system."""
    print(f"\nTesting: {system.name}")
    print("-" * 40)

    q0, p0 = system.initial_condition_for_energy(E_test)
    print(f"  Initial condition for E={E_test}: q0={q0:.3f}, p0={p0:.3f}")

    t, q, p, z, E_actual = simulate_hamiltonian(system, q0, p0, T, dt=0.01)
    print(f"  Actual energy: {E_actual:.4f}")

    # Check energy conservation
    E_traj = np.array([system.hamiltonian(q[i], p[i]) for i in range(len(q))])
    E_drift = np.std(E_traj) / np.abs(E_actual)
    print(f"  Energy drift (std/mean): {E_drift:.2e}")

    # Measure period
    zero_crossings = np.where((q[:-1] < 0) & (q[1:] >= 0))[0]
    if len(zero_crossings) >= 2:
        measured_period = np.mean(np.diff(t[zero_crossings]))
    else:
        measured_period = np.nan

    theoretical_period = system.theoretical_period(E_actual)
    print(f"  Measured period: {measured_period:.4f}")
    print(f"  Theoretical period: {theoretical_period:.4f}" if theoretical_period else "  (no theory)")

    if theoretical_period and np.isfinite(theoretical_period):
        error = np.abs(measured_period - theoretical_period) / theoretical_period
        print(f"  Period error: {error:.2%}")

    return E_actual, measured_period


if __name__ == "__main__":
    print("=" * 60)
    print("HAMILTONIAN SYSTEMS - BASIC TESTS")
    print("=" * 60)

    # Test SHO
    sho = SimpleHarmonicOscillator(omega0=1.0)
    test_system_basics(sho, E_test=1.0)

    # Test Duffing
    duffing = AnharmonicOscillator(epsilon=0.3)
    test_system_basics(duffing, E_test=1.0)
    test_system_basics(duffing, E_test=3.0)

    # Test Pendulum (in libration regime)
    pendulum = PendulumOscillator()
    test_system_basics(pendulum, E_test=-0.5)  # Libration
    test_system_basics(pendulum, E_test=0.5)   # Closer to separatrix

    # Generate ensemble
    print("\n" + "=" * 60)
    print("ENSEMBLE GENERATION TEST")
    print("=" * 60)

    energies = np.linspace(0.5, 3.0, 10)
    trajectories, periods, actual_E = generate_ensemble_at_different_energies(
        duffing, energies, T=100, dt=0.01
    )

    print(f"\nGenerated {len(trajectories)} Duffing trajectories")
    print(f"Energy range: [{min(actual_E):.2f}, {max(actual_E):.2f}]")
    print(f"Period range: [{min(periods):.2f}, {max(periods):.2f}]")

    # Show ω vs E relationship
    omegas = 2 * np.pi / np.array(periods)
    print("\nω(E) relationship:")
    for E, omega in zip(actual_E[:5], omegas[:5]):
        print(f"  E={E:.2f}: ω={omega:.3f}")
