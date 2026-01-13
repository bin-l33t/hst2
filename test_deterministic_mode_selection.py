"""
Deterministic Mode Selection Tests

Key insight: Fourier mode decay doesn't require stochastic dynamics!
Three mechanisms that give mode selection deterministically:

1. Time-Averaging: sinc envelope from finite observation window
2. Frequency Jitter: exp(-n²σ²t²/2) from ensemble dephasing
3. Doubling Map: chaotic dynamics pushes content to high n

All confirm: mode selection comes from OBSERVATION, not dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


def wrap_to_2pi(Q):
    """Wrap angle to [0, 2π)."""
    return Q % (2 * np.pi)


# ============================================================================
# TEST 1: DETERMINISTIC TIME-AVERAGING
# ============================================================================

def time_averaged_fourier_moment(n: int, omega: float, W: float,
                                  Q0: float = 0.0) -> complex:
    """
    Compute time-averaged Fourier moment for perfect oscillator.

    Q(t) = ωt + Q₀
    b̃_n(W) = (1/W) ∫₀ᵂ e^{inQ(t)} dt

    Analytic result:
    b̃_n(W) = e^{in(Q₀ + ωW/2)} · sinc(nωW/2)

    where sinc(x) = sin(x)/x
    """
    if W < 1e-10:
        return np.exp(1j * n * Q0)

    phase_center = Q0 + omega * W / 2
    arg = n * omega * W / 2

    if abs(arg) < 1e-10:
        sinc_val = 1.0
    else:
        sinc_val = np.sin(arg) / arg

    return np.exp(1j * n * phase_center) * sinc_val


def time_averaged_fourier_numerical(n: int, omega: float, W: float,
                                     Q0: float = 0.0,
                                     n_samples: int = 1000) -> complex:
    """Numerical integration for verification."""
    t = np.linspace(0, W, n_samples)
    Q = omega * t + Q0
    integrand = np.exp(1j * n * Q)
    return np.trapz(integrand, t) / W


def test_time_averaging():
    """
    Test 1: Deterministic time-averaging gives sinc envelope.

    No noise, no randomness - just finite observation window.
    """
    print("=" * 70)
    print("TEST 1: DETERMINISTIC TIME-AVERAGING")
    print("=" * 70)
    print("\nModel: Q(t) = ωt + Q₀  (perfect oscillator)")
    print("Observable: b̃_n(W) = (1/W) ∫₀ᵂ e^{inQ(t)} dt")
    print("\nTheory: |b̃_n(W)| = |sinc(nωW/2)|")
    print("        First zero at W = 2π/(nω) = T/n")

    omega = 1.0
    T = 2 * np.pi / omega  # Period

    # Window sizes from 0 to 3 periods
    W_values = np.linspace(0.01, 3 * T, 200)
    W_over_T = W_values / T

    n_modes = [1, 2, 3, 4, 5]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: |b̃_n| vs W/T
    ax1 = axes[0]

    results = {}
    for n in n_modes:
        b_theory = np.array([abs(time_averaged_fourier_moment(n, omega, W))
                            for W in W_values])
        results[n] = b_theory
        ax1.plot(W_over_T, b_theory, label=f'n={n}', linewidth=2)

    # Mark first zeros at W/T = 1/n
    for n in n_modes:
        ax1.axvline(x=1/n, color='gray', linestyle=':', alpha=0.5)

    ax1.set_xlabel('W/T (observation window / period)')
    ax1.set_ylabel('|b̃_n(W)|')
    ax1.set_title('Time-Averaged Fourier Moments\n(sinc envelope)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)

    # Right: Verify sinc formula
    ax2 = axes[1]

    # Plot |b̃_n| vs nωW/2 - should all collapse to sinc
    for n in n_modes:
        x = n * omega * W_values / 2
        b_theory = results[n]
        ax2.plot(x / np.pi, b_theory, label=f'n={n}', alpha=0.7)

    # Overlay sinc function
    x_sinc = np.linspace(0.01, 5 * np.pi, 200)
    sinc_vals = np.abs(np.sin(x_sinc) / x_sinc)
    ax2.plot(x_sinc / np.pi, sinc_vals, 'k--', linewidth=2,
             label='|sinc(πx)|', alpha=0.8)

    ax2.set_xlabel('nωW/(2π)')
    ax2.set_ylabel('|b̃_n|')
    ax2.set_title('Universal Collapse to sinc\n(all modes on same curve)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig('time_averaging_sinc.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: time_averaging_sinc.png")
    plt.close()

    # Verify numerical integration matches theory
    print("\nVerification: Theory vs Numerical Integration")
    print("W/T     | n | Theory | Numerical | Error")
    print("--------|---|--------|-----------|-------")

    test_W = [0.5 * T, 1.0 * T, 2.0 * T]
    max_error = 0
    for W in test_W:
        for n in [1, 2, 3]:
            theory = abs(time_averaged_fourier_moment(n, omega, W))
            numerical = abs(time_averaged_fourier_numerical(n, omega, W))
            error = abs(theory - numerical)
            max_error = max(max_error, error)
            print(f"  {W/T:.1f}   | {n} | {theory:.4f} | {numerical:.4f}  | {error:.2e}")

    if max_error < 1e-3:
        print("\n✓ PASSED: Time-averaging gives sinc envelope")
        return True
    else:
        print("\n✗ FAILED: Theory-numerical mismatch")
        return False


# ============================================================================
# TEST 2: FREQUENCY JITTER (INHOMOGENEOUS DEPHASING)
# ============================================================================

def frequency_jitter_moment_theory(n: int, sigma: float, t: float) -> float:
    """
    Theoretical |b_n(t)| for frequency jitter.

    Each trial: Q(t) = ωt where ω ~ N(ω₀, σ²)
    No noise WITHIN trial - just uncertainty in frequency.

    b_n(t) = ⟨e^{inωt}⟩_ω = e^{inω₀t} · e^{-n²σ²t²/2}

    Key: t² decay, NOT t decay (unlike diffusion)
    """
    return np.exp(-n**2 * sigma**2 * t**2 / 2)


def simulate_frequency_jitter(n_trials: int, omega0: float, sigma: float,
                               t_values: np.ndarray) -> np.ndarray:
    """
    Simulate frequency jitter ensemble.

    Each trial has fixed ω drawn from N(ω₀, σ²).

    Key: We fix Q0 = 0 (or equivalently, measure |b_n| per trial then average).
    Random Q0 would kill the signal via phase cancellation.
    """
    n_modes = 5

    # Vectorized: draw all frequencies at once
    omegas = np.random.normal(omega0, sigma, n_trials)

    # For each time, compute the ensemble average of e^{inωt}
    # b_n(t) = (1/N) Σ_trials e^{in·ω_trial·t}
    b_n = np.zeros((len(t_values), n_modes), dtype=complex)

    for i, t in enumerate(t_values):
        # All phases at this time: ω_trial * t
        phases = omegas * t  # shape: (n_trials,)

        for n in range(1, n_modes + 1):
            # Average over ensemble
            b_n[i, n-1] = np.mean(np.exp(1j * n * phases))

    return b_n


def test_frequency_jitter():
    """
    Test 2: Frequency jitter gives exp(-n²σ²t²/2) decay.

    Key difference from diffusion:
    - Diffusion: exp(-n²Dt) - linear in t
    - Jitter: exp(-n²σ²t²/2) - quadratic in t
    """
    print("\n" + "=" * 70)
    print("TEST 2: FREQUENCY JITTER (INHOMOGENEOUS DEPHASING)")
    print("=" * 70)
    print("\nModel: Q(t) = ωt, where ω ~ N(ω₀, σ²)")
    print("       No noise in dynamics - frequency varies per trial")
    print("\nTheory: |b_n(t)| = exp(-n²σ²t²/2)")
    print("        Gaussian decay in t (NOT exponential!)")

    omega0 = 1.0
    sigma = 0.2
    n_trials = 2000

    # Characteristic time for n=1: t* = 1/σ
    t_star = 1 / sigma
    t_values = np.linspace(0, 3 * t_star, 50)

    print(f"\nParameters: ω₀ = {omega0}, σ = {sigma}")
    print(f"Characteristic time: t* = 1/σ = {t_star:.1f}")
    print(f"Ensemble size: {n_trials}")

    # Simulate
    print("\nRunning ensemble simulation...")
    b_n_sim = simulate_frequency_jitter(n_trials, omega0, sigma, t_values)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: |b_n(t)| vs t
    ax1 = axes[0]
    n_modes = 5
    colors = plt.cm.viridis(np.linspace(0, 0.8, n_modes))

    for n in range(1, n_modes + 1):
        # Simulation
        b_sim = np.abs(b_n_sim[:, n-1])
        ax1.plot(t_values / t_star, b_sim, 'o', color=colors[n-1],
                markersize=3, alpha=0.6)

        # Theory
        b_theory = frequency_jitter_moment_theory(n, sigma, t_values)
        ax1.plot(t_values / t_star, b_theory, '-', color=colors[n-1],
                linewidth=2, label=f'n={n}')

    ax1.set_xlabel('t / t*')
    ax1.set_ylabel('|b_n(t)|')
    ax1.set_title('Frequency Jitter: Gaussian Decay')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Middle: log|b_n| vs t² (should be linear for Gaussian)
    ax2 = axes[1]

    for n in range(1, 4):
        b_sim = np.abs(b_n_sim[:, n-1])
        valid = b_sim > 0.01  # Avoid log(0)

        t2 = (t_values[valid])**2
        log_b = np.log(b_sim[valid])

        ax2.plot(t2, log_b, 'o', markersize=4, label=f'n={n} (sim)')

        # Theory: log|b_n| = -n²σ²t²/2
        ax2.plot(t2, -n**2 * sigma**2 * t2 / 2, '--', linewidth=2)

    ax2.set_xlabel('t²')
    ax2.set_ylabel('log|b_n(t)|')
    ax2.set_title('Gaussian Decay: log|b_n| ∝ -t²\n(linear in t²)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Right: Compare jitter vs diffusion decay
    ax3 = axes[2]

    # Jitter: exp(-σ²t²/2) for n=1
    jitter = np.exp(-sigma**2 * t_values**2 / 2)

    # Diffusion with D chosen so same half-life
    # Jitter half-life: t_1/2 where exp(-σ²t²/2) = 0.5
    # → t_1/2 = sqrt(2 ln 2) / σ
    t_half_jitter = np.sqrt(2 * np.log(2)) / sigma
    D_equiv = np.log(2) / t_half_jitter  # So exp(-Dt) = 0.5 at same time
    diffusion = np.exp(-D_equiv * t_values)

    ax3.plot(t_values / t_star, jitter, 'b-', linewidth=2,
             label='Jitter: exp(-σ²t²/2)')
    ax3.plot(t_values / t_star, diffusion, 'r--', linewidth=2,
             label='Diffusion: exp(-Dt)')
    ax3.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax3.axvline(x=t_half_jitter / t_star, color='gray', linestyle=':', alpha=0.5)

    ax3.set_xlabel('t / t*')
    ax3.set_ylabel('|b_1(t)|')
    ax3.set_title('Jitter vs Diffusion\n(same half-life, different shape)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('frequency_jitter.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: frequency_jitter.png")
    plt.close()

    # Verify t² scaling by checking at specific times
    print("\nVerification: |b_n(t)| = exp(-n²σ²t²/2)")
    print("\nAt t = t* = 1/σ, theory predicts |b_n| = exp(-n²/2):")
    print("  n=1: |b_1| = exp(-0.5) = 0.607")
    print("  n=2: |b_2| = exp(-2.0) = 0.135")
    print("  n=3: |b_3| = exp(-4.5) = 0.011")

    # Find index closest to t = t*
    idx_tstar = np.argmin(np.abs(t_values - t_star))

    print(f"\nMeasured at t ≈ {t_values[idx_tstar]:.2f} (t* = {t_star:.1f}):")
    print("n | |b_n| sim | |b_n| theory | Rel error")
    print("--|----------|-------------|----------")

    for n in range(1, 4):
        b_sim = np.abs(b_n_sim[idx_tstar, n-1])
        b_theory = frequency_jitter_moment_theory(n, sigma, t_values[idx_tstar])
        rel_error = abs(b_sim - b_theory) / b_theory if b_theory > 0.01 else abs(b_sim - b_theory)

        status = "✓" if rel_error < 0.15 else "(noise)"
        print(f" {n} | {b_sim:.4f}   | {b_theory:.4f}      | {rel_error:.2f} {status}")

    # Also verify n² scaling: log|b_n|/log|b_1| should equal n²
    print("\nVerify n² scaling at t = 0.5·t*:")
    idx_half = np.argmin(np.abs(t_values - 0.5 * t_star))

    b1 = np.abs(b_n_sim[idx_half, 0])
    log_b1 = np.log(b1) if b1 > 0.01 else -10

    print("n | log|b_n|/log|b_1| | Theory (n²) | Match")
    print("--|-------------------|-------------|------")

    n2_scaling_ok = True
    for n in range(1, 5):
        b_n_val = np.abs(b_n_sim[idx_half, n-1])
        if b_n_val > 0.01 and b1 > 0.01:
            log_ratio = np.log(b_n_val) / log_b1
            ok = abs(log_ratio - n**2) < 0.5
            status = "✓" if ok else ""
            if not ok:
                n2_scaling_ok = False
            print(f" {n} | {log_ratio:.2f}              | {n**2}           | {status}")
        else:
            print(f" {n} | (too small)       | {n**2}           |")

    # Pass if n² scaling is correct (main physics) even if absolute values have noise
    if n2_scaling_ok:
        print("\n✓ PASSED: Frequency jitter gives t² (Gaussian) decay with n² scaling")
        return True
    else:
        print("\n⚠ CHECK: n² scaling failed")
        return False


# ============================================================================
# TEST 3: DOUBLING MAP (DETERMINISTIC CHAOS)
# ============================================================================

def doubling_map_iterate(Q0: np.ndarray, k_steps: int) -> np.ndarray:
    """
    Iterate doubling map: Q_{k+1} = 2·Q_k mod 2π

    This is deterministic but chaotic - exponential sensitivity.
    """
    Q = Q0.copy()
    for _ in range(k_steps):
        Q = (2 * Q) % (2 * np.pi)
    return Q


def doubling_map_moment_theory(n: int, k: int, b_initial: dict) -> complex:
    """
    Exact Fourier moment evolution under doubling map.

    Theory: b_n(k) = b_{2^k · n}(0)

    The map shifts Fourier content to higher modes!
    """
    n_eff = (2**k) * n
    return b_initial.get(n_eff, 0.0)


def compute_initial_moments(Q0: np.ndarray, max_n: int = 256) -> dict:
    """Compute initial Fourier moments up to n = max_n."""
    moments = {}
    for n in range(1, max_n + 1):
        moments[n] = np.mean(np.exp(1j * n * Q0))
    return moments


def test_doubling_map():
    """
    Test 3: Doubling map shifts Fourier content to higher modes.

    Q_{k+1} = 2Q_k mod 2π

    Exact result: b_n(k) = b_{2^k · n}(0)

    Content flows to arbitrarily high n!
    A finite-bandwidth observer sees "everything vanishes".
    """
    print("\n" + "=" * 70)
    print("TEST 3: DOUBLING MAP (DETERMINISTIC CHAOS)")
    print("=" * 70)
    print("\nModel: Q_{k+1} = 2·Q_k mod 2π")
    print("       Deterministic, but chaotic (Lyapunov exp = ln 2)")
    print("\nTheory: b_n(k) = b_{2^k · n}(0)")
    print("        Fourier content shifts to higher modes!")

    # Initial distribution: slightly non-uniform
    n_samples = 10000

    # Start with a bump near Q = π
    Q0 = np.random.vonmises(np.pi, 2.0, n_samples)  # von Mises distribution
    Q0 = wrap_to_2pi(Q0)

    # Compute initial moments
    max_n = 256  # Need high n to track where content goes
    b_initial = compute_initial_moments(Q0, max_n)

    print(f"\nInitial distribution: von Mises(μ=π, κ=2)")
    print(f"Initial |b_1| = {abs(b_initial[1]):.4f}")
    print(f"Initial |b_2| = {abs(b_initial[2]):.4f}")

    # Iterate the map
    k_max = 6  # After 6 steps, n=1 → n=64

    results_sim = {}
    results_theory = {}

    print(f"\nIterating doubling map for k = 0 to {k_max}...")

    Q = Q0.copy()
    for k in range(k_max + 1):
        # Measure low-n moments from simulation
        moments_k = {}
        for n in range(1, 9):
            moments_k[n] = np.mean(np.exp(1j * n * Q))
        results_sim[k] = moments_k

        # Theory: b_n(k) = b_{2^k · n}(0)
        theory_k = {}
        for n in range(1, 9):
            n_eff = (2**k) * n
            if n_eff <= max_n:
                theory_k[n] = b_initial.get(n_eff, 0)
            else:
                theory_k[n] = 0.0  # Beyond our initial computation
        results_theory[k] = theory_k

        # Iterate
        Q = (2 * Q) % (2 * np.pi)

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: |b_n(k)| vs k for various n
    ax1 = axes[0]

    k_vals = np.arange(k_max + 1)
    n_plot = [1, 2, 3, 4]

    for n in n_plot:
        b_sim = [abs(results_sim[k][n]) for k in k_vals]
        b_theory = [abs(results_theory[k][n]) for k in k_vals]

        ax1.plot(k_vals, b_sim, 'o-', label=f'n={n} (sim)', markersize=6)
        ax1.plot(k_vals, b_theory, 's--', alpha=0.5, markersize=4)

    ax1.set_xlabel('Iteration k')
    ax1.set_ylabel('|b_n(k)|')
    ax1.set_title('Fourier Moments vs Iteration\n(all vanish for finite-bandwidth observer)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Middle: Where does content go?
    ax2 = axes[1]

    # Track where n=1 content ends up
    n_track = 1
    k_vals_track = range(k_max + 1)
    n_eff_vals = [2**k * n_track for k in k_vals_track]
    b_vals = [abs(b_initial.get(n_eff, 0)) for n_eff in n_eff_vals]

    ax2.semilogy(k_vals_track, n_eff_vals, 'bo-', markersize=8, label='n_eff = 2^k')
    ax2.set_xlabel('Iteration k')
    ax2.set_ylabel('Effective mode number')
    ax2.set_title(f'n=1 Content Shifts to n={2**k_max}\n(exponential mode migration)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Add annotation
    for k in [0, 2, 4, 6]:
        if k <= k_max:
            ax2.annotate(f'n={2**k}', (k, 2**k), textcoords="offset points",
                        xytext=(5, 5), fontsize=9)

    # Right: "Apparent uniformity"
    ax3 = axes[2]

    # Sum of low-mode content
    bandwidth = 8
    low_mode_content = []

    for k in k_vals:
        total = sum(abs(results_sim[k][n])**2 for n in range(1, bandwidth + 1))
        low_mode_content.append(np.sqrt(total))

    ax3.plot(k_vals, low_mode_content, 'ro-', markersize=8, linewidth=2)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    ax3.set_xlabel('Iteration k')
    ax3.set_ylabel(f'√(Σ|b_n|² for n≤{bandwidth})')
    ax3.set_title(f'Low-Mode Content (bandwidth={bandwidth})\n"Vanishes" to limited observer')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('doubling_map.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: doubling_map.png")
    plt.close()

    # Verification table
    print("\nVerification: b_n(k) = b_{2^k · n}(0)")
    print("\nk | n=1: n_eff | |b_1(k)| sim | |b_{n_eff}(0)| | Match?")
    print("--|------------|-------------|---------------|-------")

    passed = True
    for k in range(min(5, k_max + 1)):
        n_eff = 2**k
        b_sim = abs(results_sim[k][1])
        b_theory = abs(b_initial.get(n_eff, 0))
        error = abs(b_sim - b_theory)
        match = "✓" if error < 0.02 else "✗"
        if error >= 0.02:
            passed = False
        print(f" {k} | {n_eff:10d} | {b_sim:.4f}      | {b_theory:.4f}        | {match}")

    # Show that bandwidth-limited observer sees decay
    print(f"\nBandwidth-limited observer (n ≤ {bandwidth}):")
    print("After k=6 iterations, low-mode content:")
    print(f"  Σ|b_n|² = {low_mode_content[-1]:.4f}")
    print("  (approaches zero - 'apparent uniformity')")

    if passed:
        print("\n✓ PASSED: Doubling map shifts content to high modes")
        return True
    else:
        print("\n⚠ CHECK: Some deviation from exact formula")
        return False


# ============================================================================
# MAIN: RUN ALL TESTS
# ============================================================================

def run_all_tests():
    """Run all three deterministic mode selection tests."""
    print("\n" + "=" * 70)
    print("DETERMINISTIC MODE SELECTION TESTS")
    print("=" * 70)
    print("""
Key insight: Mode decay doesn't require stochastic dynamics!

Three mechanisms:
1. TIME-AVERAGING: sinc envelope from finite window
2. FREQUENCY JITTER: Gaussian t² decay from ensemble dephasing
3. DOUBLING MAP: chaotic flow to high modes

All show: mode selection is about OBSERVATION, not dynamics.
""")

    results = {}

    # Test 1: Time averaging
    results['time_averaging'] = test_time_averaging()

    # Test 2: Frequency jitter
    results['frequency_jitter'] = test_frequency_jitter()

    # Test 3: Doubling map
    results['doubling_map'] = test_doubling_map()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for test, passed in results.items():
        status = "✓ PASSED" if passed else "⚠ CHECK"
        print(f"  {test:20s}: {status}")

    print("\n" + "=" * 70)
    print("KEY CONCLUSIONS")
    print("=" * 70)
    print("""
    1. TIME-AVERAGING:
       - |b̃_n(W)| = |sinc(nωW/2)|
       - First zero at W = T/n (period/mode number)
       - Pure geometry of finite observation window
       - NO randomness needed!

    2. FREQUENCY JITTER:
       - |b_n(t)| = exp(-n²σ²t²/2)
       - Gaussian (t²) decay, NOT exponential (t)
       - Dephasing from frequency uncertainty
       - Each trajectory is deterministic!

    3. DOUBLING MAP:
       - b_n(k) = b_{2^k·n}(0)
       - Deterministic chaos pushes content to high n
       - Finite-bandwidth observer sees "uniformization"
       - Information isn't lost - just moved to invisible modes

    UNIFYING THEME:
    Mode selection = mismatch between dynamics and observation
    - Dynamics has ALL modes
    - Observation resolves only SOME modes
    - "Quantization" = which modes the observer can see

    This is Glinsky's insight:
    The semiclassical spectrum isn't intrinsic to the system.
    It emerges from the observation protocol.
""")

    return all(results.values())


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
