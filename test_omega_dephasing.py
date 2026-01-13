"""
Nonlinear ω(J) Dephasing Test (Landau Damping)

Key insight: Mode decay from action spread WITHOUT stochasticity.

Setup:
- Deterministic J grid (no RNG)
- Each trajectory: Q_k(t) = Q₀ + ω(J_k)·t  (fully deterministic)
- Observable: b_n(t) = (1/N) Σ_k exp(i n ω(J_k) t)

Expected result:
- |b_n(t)| decays due to dephasing from ω(J) spread
- Higher n decays faster
- This is "Landau damping" - phase mixing without dissipation

Comparison:
- Pendulum ω(J): nonlinear, strong dephasing
- SHO ω = const: NO dephasing (all phases stay coherent)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

from pendulum_action_angle import (
    pendulum_omega_from_action, J_SEPARATRIX
)


def compute_omega_grid_pendulum(J_min: float, J_max: float, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute deterministic grid of (J, ω) for pendulum.

    No RNG - purely deterministic.
    """
    J_grid = np.linspace(J_min, J_max, N)
    omega_grid = np.array([pendulum_omega_from_action(J) for J in J_grid])
    return J_grid, omega_grid


def compute_omega_grid_sho(J_min: float, J_max: float, N: int, omega0: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute (J, ω) for SHO - constant frequency.
    """
    J_grid = np.linspace(J_min, J_max, N)
    omega_grid = np.full(N, omega0)
    return J_grid, omega_grid


def compute_fourier_moment(omega_grid: np.ndarray, t: float, n: int, Q0: float = 0.0) -> complex:
    """
    Compute b_n(t) = (1/N) Σ_k exp(i n Q_k(t))

    where Q_k(t) = Q₀ + ω_k · t

    Fully deterministic - no averaging over initial conditions.
    """
    Q = Q0 + omega_grid * t
    return np.mean(np.exp(1j * n * Q))


def compute_dephasing_decay(omega_grid: np.ndarray, t_values: np.ndarray,
                            n_modes: int = 5, Q0: float = 0.0) -> np.ndarray:
    """
    Compute |b_n(t)| for multiple modes and times.

    Returns: array of shape (len(t_values), n_modes)
    """
    result = np.zeros((len(t_values), n_modes))

    for i, t in enumerate(t_values):
        for n in range(1, n_modes + 1):
            b_n = compute_fourier_moment(omega_grid, t, n, Q0)
            result[i, n-1] = np.abs(b_n)

    return result


def test_pendulum_dephasing():
    """
    Test dephasing from pendulum's nonlinear ω(J).
    """
    print("=" * 70)
    print("TEST 1: PENDULUM ω(J) DEPHASING")
    print("=" * 70)
    print("\nModel: Q_k(t) = Q₀ + ω(J_k)·t  (fully deterministic)")
    print("Observable: b_n(t) = (1/N) Σ_k exp(i n Q_k(t))")
    print("\nPendulum: ω(J) = π/(2K(m)) where m = (1+E)/2")
    print(f"          ω → 1 as J → 0, ω → 0 as J → J_sep = {J_SEPARATRIX:.3f}")

    # Deterministic J grid
    N = 200
    J_min, J_max = 0.1, 2.0  # Well within libration
    J_grid, omega_grid = compute_omega_grid_pendulum(J_min, J_max, N)

    print(f"\nAction grid: J ∈ [{J_min}, {J_max}], N = {N} points")
    print(f"Frequency range: ω ∈ [{omega_grid.min():.4f}, {omega_grid.max():.4f}]")
    print(f"Frequency spread: Δω = {omega_grid.max() - omega_grid.min():.4f}")

    # Characteristic dephasing time ~ 1/Δω
    delta_omega = omega_grid.max() - omega_grid.min()
    t_dephase = 2 * np.pi / delta_omega
    print(f"Dephasing timescale: t_dephase ≈ 2π/Δω = {t_dephase:.2f}")

    # Time grid
    t_values = np.linspace(0, 5 * t_dephase, 200)

    # Compute dephasing
    print("\nComputing Fourier moments...")
    b_n_pend = compute_dephasing_decay(omega_grid, t_values, n_modes=5)

    return J_grid, omega_grid, t_values, b_n_pend, t_dephase


def test_sho_no_dephasing():
    """
    Test SHO (constant ω) - should NOT dephase.
    """
    print("\n" + "=" * 70)
    print("TEST 2: SHO (CONSTANT ω) - NO DEPHASING")
    print("=" * 70)
    print("\nSHO: ω(J) = ω₀ = const (independent of action)")
    print("All trajectories rotate at same rate → no dephasing!")

    N = 200
    J_min, J_max = 0.1, 2.0
    omega0 = 1.0

    J_grid, omega_grid = compute_omega_grid_sho(J_min, J_max, N, omega0)

    print(f"\nAction grid: J ∈ [{J_min}, {J_max}], N = {N} points")
    print(f"Frequency: ω = {omega0} for all J (constant)")
    print("Expected: |b_n(t)| = 1 for all t (no decay)")

    # Use same time range
    delta_omega_ref = 0.5  # Reference spread
    t_dephase_ref = 2 * np.pi / delta_omega_ref
    t_values = np.linspace(0, 5 * t_dephase_ref, 200)

    # Compute (should stay at 1)
    b_n_sho = compute_dephasing_decay(omega_grid, t_values, n_modes=5)

    return J_grid, omega_grid, t_values, b_n_sho


def test_linear_omega_dephasing():
    """
    Test linear ω(J) = ω₀ + α·J for comparison.

    This gives predictable Gaussian-like decay.
    """
    print("\n" + "=" * 70)
    print("TEST 3: LINEAR ω(J) = ω₀ + α·J")
    print("=" * 70)

    N = 200
    J_min, J_max = 0.1, 2.0
    omega0 = 0.8
    alpha = 0.2  # dω/dJ

    J_grid = np.linspace(J_min, J_max, N)
    omega_grid = omega0 + alpha * J_grid

    print(f"\nω(J) = {omega0} + {alpha}·J")
    print(f"Frequency range: ω ∈ [{omega_grid.min():.4f}, {omega_grid.max():.4f}]")

    delta_omega = omega_grid.max() - omega_grid.min()
    t_dephase = 2 * np.pi / delta_omega
    print(f"Dephasing timescale: t_dephase ≈ {t_dephase:.2f}")

    t_values = np.linspace(0, 5 * t_dephase, 200)
    b_n_linear = compute_dephasing_decay(omega_grid, t_values, n_modes=5)

    return J_grid, omega_grid, t_values, b_n_linear, t_dephase


def analyze_dephasing_mechanism():
    """
    Analyze why nonlinear ω(J) causes mode decay.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS: WHY NONLINEAR ω(J) CAUSES MODE DECAY")
    print("=" * 70)

    print("""
    Physical picture:

    1. INITIAL STATE (t=0)
       All trajectories start at same phase Q₀
       b_n(0) = e^{inQ₀} → |b_n(0)| = 1

    2. EVOLUTION
       Q_k(t) = Q₀ + ω(J_k)·t
       Different J values have different ω → phases spread

    3. DEPHASING
       b_n(t) = (1/N) Σ_k exp(i n [Q₀ + ω_k·t])
             = e^{inQ₀} · (1/N) Σ_k exp(i n ω_k·t)

       The sum is over phases that spread due to ω variation.
       When phases span [0, 2π], they cancel → |b_n| → 0

    4. MODE DEPENDENCE
       Phase spread at time t: Δφ_n = n · Δω · t
       Mode n dephases when n·Δω·t ~ 2π
       → Higher modes dephase faster!

    This is LANDAU DAMPING:
    - No dissipation (each trajectory is reversible)
    - Apparent decay from phase mixing
    - Information moves to finer scales in J-space
    """)


def run_all_tests():
    """Run all dephasing tests and create comparison plots."""

    print("\n" + "=" * 70)
    print("NONLINEAR ω(J) DEPHASING (LANDAU DAMPING)")
    print("=" * 70)
    print("""
    Key insight: Mode decay WITHOUT stochasticity!

    Mechanism: Different actions have different frequencies.
    Result: Phases spread → Fourier moments decay.

    This is what the pendulum "should have shown" -
    not periodicity, but dephasing from ω(J) spread.
    """)

    # Run tests
    J_pend, omega_pend, t_pend, b_pend, t_deph = test_pendulum_dephasing()
    J_sho, omega_sho, t_sho, b_sho = test_sho_no_dephasing()
    J_lin, omega_lin, t_lin, b_lin, t_deph_lin = test_linear_omega_dephasing()

    analyze_dephasing_mechanism()

    # Create comprehensive plot
    fig = plt.figure(figsize=(16, 12))

    # Row 1: ω(J) relationships
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.plot(J_pend, omega_pend, 'b-', linewidth=2, label='Pendulum')
    ax1.axhline(y=1.0, color='r', linestyle='--', label='SHO (ω=1)')
    ax1.plot(J_lin, omega_lin, 'g-', linewidth=2, label='Linear')
    ax1.axvline(x=J_SEPARATRIX, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Action J')
    ax1.set_ylabel('Frequency ω(J)')
    ax1.set_title('Frequency-Action Relationships')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Row 1: dω/dJ (nonlinearity)
    ax2 = fig.add_subplot(3, 3, 2)
    # Numerical derivative
    dJ = J_pend[1] - J_pend[0]
    domega_dJ_pend = np.gradient(omega_pend, dJ)
    domega_dJ_lin = np.gradient(omega_lin, dJ)
    ax2.plot(J_pend, domega_dJ_pend, 'b-', linewidth=2, label='Pendulum')
    ax2.axhline(y=0, color='r', linestyle='--', label='SHO')
    ax2.plot(J_lin, domega_dJ_lin, 'g-', linewidth=2, label='Linear')
    ax2.set_xlabel('Action J')
    ax2.set_ylabel('dω/dJ')
    ax2.set_title('Nonlinearity: dω/dJ')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Row 1: Frequency histogram
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.hist(omega_pend, bins=30, alpha=0.6, label='Pendulum', density=True)
    ax3.axvline(x=omega_sho[0], color='r', linestyle='--', linewidth=2, label='SHO (δ-function)')
    ax3.hist(omega_lin, bins=30, alpha=0.6, label='Linear', density=True)
    ax3.set_xlabel('Frequency ω')
    ax3.set_ylabel('Density')
    ax3.set_title('Frequency Distribution\n(from uniform J distribution)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Row 2: |b_n(t)| for each system
    n_modes = 5
    colors = plt.cm.viridis(np.linspace(0, 0.8, n_modes))

    # Pendulum dephasing
    ax4 = fig.add_subplot(3, 3, 4)
    for n in range(n_modes):
        ax4.plot(t_pend / t_deph, b_pend[:, n], color=colors[n],
                linewidth=2, label=f'n={n+1}')
    ax4.set_xlabel('t / t_dephase')
    ax4.set_ylabel('|b_n(t)|')
    ax4.set_title('Pendulum: Strong Dephasing')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-0.1, 1.1)

    # SHO (no dephasing)
    ax5 = fig.add_subplot(3, 3, 5)
    for n in range(n_modes):
        ax5.plot(t_sho / t_deph, b_sho[:, n], color=colors[n],
                linewidth=2, label=f'n={n+1}')
    ax5.set_xlabel('t / t_dephase')
    ax5.set_ylabel('|b_n(t)|')
    ax5.set_title('SHO: NO Dephasing\n(ω = const)')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(-0.1, 1.1)

    # Linear dephasing
    ax6 = fig.add_subplot(3, 3, 6)
    for n in range(n_modes):
        ax6.plot(t_lin / t_deph_lin, b_lin[:, n], color=colors[n],
                linewidth=2, label=f'n={n+1}')
    ax6.set_xlabel('t / t_dephase')
    ax6.set_ylabel('|b_n(t)|')
    ax6.set_title('Linear ω(J): Dephasing')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(-0.1, 1.1)

    # Row 3: Comparison of n=1 decay
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.plot(t_pend / t_deph, b_pend[:, 0], 'b-', linewidth=2, label='Pendulum')
    ax7.plot(t_sho / t_deph, b_sho[:, 0], 'r--', linewidth=2, label='SHO')
    ax7.plot(t_lin / t_deph_lin, b_lin[:, 0], 'g-', linewidth=2, label='Linear')
    ax7.set_xlabel('t / t_dephase')
    ax7.set_ylabel('|b_1(t)|')
    ax7.set_title('Mode n=1 Comparison')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim(-0.1, 1.1)

    # Row 3: Dephasing time vs mode number
    ax8 = fig.add_subplot(3, 3, 8)

    # Find time when |b_n| first drops below 0.5
    def find_half_time(b_n_array, t_array, threshold=0.5):
        half_times = []
        for n in range(b_n_array.shape[1]):
            below = np.where(b_n_array[:, n] < threshold)[0]
            if len(below) > 0:
                half_times.append(t_array[below[0]])
            else:
                half_times.append(np.nan)
        return np.array(half_times)

    t_half_pend = find_half_time(b_pend, t_pend)
    t_half_lin = find_half_time(b_lin, t_lin)

    n_vals = np.arange(1, n_modes + 1)
    ax8.plot(n_vals, t_half_pend / t_deph, 'bo-', markersize=8, label='Pendulum')
    ax8.plot(n_vals, t_half_lin / t_deph_lin, 'gs-', markersize=8, label='Linear')
    # Theory: t_half ~ 1/n
    ax8.plot(n_vals, t_half_pend[0] / t_deph / n_vals, 'k--', alpha=0.5, label='∝ 1/n')
    ax8.set_xlabel('Mode number n')
    ax8.set_ylabel('t_half / t_dephase')
    ax8.set_title('Dephasing Time vs Mode\n(higher n → faster decay)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # Row 3: Key result text
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')
    ax9.text(0.1, 0.9, "KEY RESULTS", fontsize=14, fontweight='bold',
             transform=ax9.transAxes)
    ax9.text(0.1, 0.75, "1. Pendulum ω(J): Strong dephasing", fontsize=11,
             transform=ax9.transAxes)
    ax9.text(0.1, 0.62, "   • Modes decay due to frequency spread", fontsize=10,
             transform=ax9.transAxes)
    ax9.text(0.1, 0.52, "   • Higher modes decay faster", fontsize=10,
             transform=ax9.transAxes)
    ax9.text(0.1, 0.38, "2. SHO ω = const: NO dephasing", fontsize=11,
             transform=ax9.transAxes)
    ax9.text(0.1, 0.25, "   • All phases stay coherent forever", fontsize=10,
             transform=ax9.transAxes)
    ax9.text(0.1, 0.08, "3. This is LANDAU DAMPING:", fontsize=11,
             transform=ax9.transAxes, fontweight='bold')
    ax9.text(0.1, -0.05, "   Phase mixing without dissipation", fontsize=10,
             transform=ax9.transAxes)

    plt.suptitle('Nonlinear ω(J) Dephasing: Landau Damping Without Stochasticity',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('omega_dephasing.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: omega_dephasing.png")
    plt.close()

    # Verification
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    print("\n1. SHO should NOT dephase:")
    sho_final = b_sho[-1, :]
    print(f"   |b_n(t_final)| for SHO: {sho_final}")
    sho_ok = np.all(np.abs(sho_final - 1.0) < 0.01)
    print(f"   All ≈ 1? {sho_ok}")

    print("\n2. Pendulum SHOULD dephase:")
    pend_final = b_pend[-1, :]
    print(f"   |b_n(t_final)| for pendulum: {pend_final}")
    pend_ok = np.all(pend_final < 0.3)
    print(f"   All < 0.3? {pend_ok}")

    print("\n3. Higher modes decay faster:")
    t_half_valid = t_half_pend[~np.isnan(t_half_pend)]
    if len(t_half_valid) > 1:
        monotonic = np.all(np.diff(t_half_valid) < 0)
        print(f"   Half-times: {t_half_pend}")
        print(f"   Monotonically decreasing? {monotonic}")
    else:
        monotonic = True
        print("   (Insufficient data for monotonicity check)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_passed = sho_ok and pend_ok and monotonic

    if sho_ok:
        print("  SHO no-dephasing       : ✓ PASSED")
    else:
        print("  SHO no-dephasing       : ✗ FAILED")

    if pend_ok:
        print("  Pendulum dephasing     : ✓ PASSED")
    else:
        print("  Pendulum dephasing     : ✗ FAILED")

    if monotonic:
        print("  Mode hierarchy         : ✓ PASSED")
    else:
        print("  Mode hierarchy         : ✗ FAILED")

    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print("""
    1. NONLINEAR ω(J) CREATES MODE SELECTION
       • Pendulum's ω(J) varies with action
       • This causes dephasing even without noise
       • Fourier modes decay due to phase mixing

    2. THIS IS WHAT PENDULUM "SHOULD HAVE SHOWN"
       • We looked for periodicity - wrong question!
       • The right observation: dephasing from ω(J) spread
       • Mode decay is the signature of nonlinear dynamics

    3. LANDAU DAMPING = PHASE MIXING
       • No dissipation (each trajectory is reversible)
       • Apparent decay from ensemble dephasing
       • Information moves to finer scales, not lost

    4. CONNECTION TO GLINSKY
       • J uncertainty → ω spread → mode decay
       • Observation timescale selects visible modes
       • "Quantization" = mode selection by dephasing

    5. SHO IS SPECIAL
       • Constant ω means no dephasing
       • This is why SHO has no intrinsic J₀
       • Nonlinearity is essential for mode selection!
    """)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
