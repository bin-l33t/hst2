"""
Back-Action Constraints on Energy Measurement Timescale

Key insight: Measuring E = H(q,p) requires knowing (q,p), but back-action
prevents precise simultaneous measurement. This forces period-based
measurement, which inherently requires time ≥ T.

Two methods compared:
1. INSTANTANEOUS: Measure (q,p) directly, compute E = H(q,p)
   - Fast (single instant)
   - But: back-action σ_q · σ_p_kick ≥ η corrupts E

2. PERIOD-BASED: Observe q(t), extract period T, compute E from ω = 2π/T
   - Slow (requires T_obs ≥ T)
   - But: averaging improves precision, no back-action on E

Result: Strong back-action forces period-based measurement,
which requires T_obs ≥ T, during which phase is lost.

This is WHY energy can be measured but not phase at long timescales.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

from hamiltonian_systems import PendulumOscillator, simulate_hamiltonian


def pendulum_omega(E):
    """Frequency ω(E) for pendulum"""
    if E >= 1 or E <= -1:
        return np.nan
    k2 = (1 + E) / 2
    if k2 <= 0 or k2 >= 1:
        return 1.0
    k = np.sqrt(k2)
    return np.pi / (2 * ellipk(k**2))


def omega_to_energy(omega_meas):
    """Invert ω(E) to get E from measured frequency (numerical)"""
    from scipy.optimize import brentq

    def residual(E):
        return pendulum_omega(E) - omega_meas

    try:
        # ω is monotonically decreasing with E for pendulum
        E_est = brentq(residual, -0.999, 0.999)
        return E_est
    except:
        return np.nan


def wrap_angle(angle):
    """Wrap angle to [-π, π]"""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


# ============================================================
# METHOD 1: INSTANTANEOUS MEASUREMENT WITH BACK-ACTION
# ============================================================

def instantaneous_energy_measurement(q_true, p_true, sigma_q, eta, seed=None):
    """
    Measure E instantaneously from (q, p) with back-action.

    Protocol:
    1. Measure q with precision σ_q → q_meas = q_true + noise(σ_q)
    2. Back-action kicks p: p_true → p_true + kick(η/σ_q)
    3. Measure p (already corrupted)
    4. Compute E_meas = p_meas²/2 - cos(q_meas)

    Parameters
    ----------
    q_true, p_true : float
        True phase space position
    sigma_q : float
        Position measurement precision
    eta : float
        Back-action coupling strength (like ℏ)

    Returns
    -------
    E_meas : float
        Measured energy
    q_meas, p_meas : float
        Measured values
    sigma_p_kick : float
        Back-action kick strength
    """
    if seed is not None:
        np.random.seed(seed)

    # Measure q
    q_meas = q_true + np.random.normal(0, sigma_q)

    # Back-action on p
    sigma_p_kick = eta / max(sigma_q, 1e-6)
    p_after_kick = p_true + np.random.normal(0, sigma_p_kick)

    # Measure p (with small additional noise)
    p_meas = p_after_kick + np.random.normal(0, 0.01)

    # Compute energy: H = p²/2 - cos(q)
    E_meas = p_meas**2 / 2 - np.cos(q_meas)

    return E_meas, q_meas, p_meas, sigma_p_kick


def optimal_sigma_q_for_energy(q_true, p_true, E_true, eta, n_trials=100):
    """
    Find optimal σ_q that minimizes energy measurement error.

    Trade-off:
    - Small σ_q: good q, but large p kick
    - Large σ_q: poor q, but small p kick
    """
    def mean_E_error(log_sigma_q):
        sigma_q = 10**log_sigma_q
        errors = []
        for i in range(n_trials):
            E_meas, _, _, _ = instantaneous_energy_measurement(
                q_true, p_true, sigma_q, eta, seed=i
            )
            errors.append(abs(E_meas - E_true))
        return np.mean(errors)

    # Search over log scale
    result = minimize_scalar(mean_E_error, bounds=(-3, 1), method='bounded')
    optimal_sigma_q = 10**result.x
    optimal_error = result.fun

    return optimal_sigma_q, optimal_error


# ============================================================
# METHOD 2: PERIOD-BASED MEASUREMENT
# ============================================================

def period_based_energy_measurement(z, t, T_obs, tau_det=0.1, sigma_obs=0.01, seed=None):
    """
    Measure E by observing oscillation and extracting period.

    Protocol:
    1. Observe q(t) over time T_obs with detector bandwidth τ_det
    2. Add observation noise σ_obs
    3. Detect zero-crossings or peaks to measure period
    4. Compute ω = 2π/T_measured
    5. Invert E(ω) to get E

    Parameters
    ----------
    z : array
        Complex trajectory z = q + ip
    t : array
        Time array
    T_obs : float
        Observation window length
    tau_det : float
        Detector averaging time (bandwidth)
    sigma_obs : float
        Observation noise

    Returns
    -------
    E_meas : float
        Measured energy from period
    T_meas : float
        Measured period
    omega_meas : float
        Measured frequency
    """
    if seed is not None:
        np.random.seed(seed)

    dt = t[1] - t[0]
    n_obs = int(T_obs / dt)

    if n_obs > len(z):
        n_obs = len(z)

    # Extract q (real part)
    q = np.real(z[:n_obs])
    t_obs = t[:n_obs]

    # Apply detector averaging (box kernel)
    n_avg = max(1, int(tau_det / dt))
    if n_avg > 1:
        q_smoothed = np.convolve(q, np.ones(n_avg)/n_avg, mode='same')
    else:
        q_smoothed = q

    # Add observation noise
    q_observed = q_smoothed + np.random.normal(0, sigma_obs, len(q_smoothed))

    # Find peaks to measure period
    peaks, _ = find_peaks(q_observed, height=0, distance=int(0.5/dt))

    if len(peaks) >= 2:
        # Period from peak-to-peak timing
        peak_times = t_obs[peaks]
        periods = np.diff(peak_times)
        T_meas = np.mean(periods)
    else:
        # Try zero-crossings
        zero_crossings = np.where(np.diff(np.sign(q_observed)) > 0)[0]
        if len(zero_crossings) >= 2:
            crossing_times = t_obs[zero_crossings]
            half_periods = np.diff(crossing_times)
            T_meas = 2 * np.mean(half_periods)
        else:
            return np.nan, np.nan, np.nan

    # Compute frequency and energy
    omega_meas = 2 * np.pi / T_meas
    E_meas = omega_to_energy(omega_meas)

    return E_meas, T_meas, omega_meas


# ============================================================
# COMPARISON TEST
# ============================================================

def compare_methods(E_true, eta_values, T_obs_ratios, n_trials=50, seed=42):
    """
    Compare instantaneous vs period-based methods across different
    back-action strengths and observation times.
    """
    np.random.seed(seed)

    # Generate trajectory
    pendulum = PendulumOscillator()
    omega_true = pendulum_omega(E_true)
    T_period = 2 * np.pi / omega_true

    q_max = np.arccos(-E_true) if E_true > -1 else np.pi * 0.95
    q0, p0 = q_max, 0.0

    T_sim = 50 * T_period
    dt = 0.01
    t, q, p, z, E_actual = simulate_hamiltonian(pendulum, q0, p0, T=T_sim, dt=dt)

    results = {
        'instant': {},  # eta -> {error, optimal_sigma_q, Q_error}
        'period': {},   # eta -> {T_obs_ratio -> error}
        'crossover': {} # eta -> T_crossover
    }

    for eta in eta_values:
        print(f"  Testing η = {eta:.3f}...")

        # Method 1: Instantaneous (at various points in trajectory)
        instant_E_errors = []
        instant_Q_errors = []

        for trial in range(n_trials):
            # Pick random point on trajectory
            idx = np.random.randint(len(q) // 4, 3 * len(q) // 4)
            q_true, p_true = q[idx], p[idx]

            # Find optimal sigma_q for this eta
            if trial == 0:
                opt_sigma_q, _ = optimal_sigma_q_for_energy(q_true, p_true, E_actual, eta, n_trials=30)

            # Measure
            E_meas, q_meas, p_meas, sigma_p_kick = instantaneous_energy_measurement(
                q_true, p_true, opt_sigma_q, eta, seed=seed + trial
            )

            instant_E_errors.append(abs(E_meas - E_actual))

            # Phase error: can we determine where in cycle we are?
            Q_true = np.arctan2(p_true, q_true)
            Q_meas = np.arctan2(p_meas, q_meas)
            instant_Q_errors.append(abs(wrap_angle(Q_meas - Q_true)))

        results['instant'][eta] = {
            'E_error': np.mean(instant_E_errors),
            'E_error_std': np.std(instant_E_errors),
            'optimal_sigma_q': opt_sigma_q,
            'sigma_p_kick': eta / opt_sigma_q,
            'Q_error': np.mean(instant_Q_errors)
        }

        # Method 2: Period-based at various T_obs
        period_results = {}

        for T_ratio in T_obs_ratios:
            T_obs = T_ratio * T_period

            E_errors = []
            for trial in range(n_trials):
                # Random start point
                start_idx = np.random.randint(0, len(z) // 2)
                z_window = z[start_idx:]
                t_window = t[start_idx:] - t[start_idx]

                E_meas, T_meas, omega_meas = period_based_energy_measurement(
                    z_window, t_window, T_obs, tau_det=0.1, sigma_obs=0.01, seed=seed + trial
                )

                if not np.isnan(E_meas):
                    E_errors.append(abs(E_meas - E_actual))

            if E_errors:
                period_results[T_ratio] = {
                    'E_error': np.mean(E_errors),
                    'E_error_std': np.std(E_errors),
                    'n_valid': len(E_errors)
                }
            else:
                period_results[T_ratio] = {'E_error': np.nan, 'n_valid': 0}

        results['period'][eta] = period_results

        # Find crossover point
        instant_error = results['instant'][eta]['E_error']
        crossover = None
        for T_ratio in T_obs_ratios:
            if T_ratio in period_results and not np.isnan(period_results[T_ratio]['E_error']):
                if period_results[T_ratio]['E_error'] < instant_error:
                    crossover = T_ratio
                    break
        results['crossover'][eta] = crossover

    return results, E_actual, T_period


def run_full_test():
    """Run full comparison and generate visualizations."""

    print("=" * 70)
    print("BACK-ACTION CONSTRAINTS ON ENERGY MEASUREMENT")
    print("=" * 70)
    print("\nComparing instantaneous vs period-based measurement...")

    E_true = 0.0
    eta_values = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5]
    T_obs_ratios = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0]

    results, E_actual, T_period = compare_methods(E_true, eta_values, T_obs_ratios)

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1a: E error vs T_obs for different η
    ax = axes[0, 0]
    for eta in eta_values:
        period_data = results['period'][eta]
        ratios = sorted([r for r in period_data.keys() if not np.isnan(period_data[r]['E_error'])])
        errors = [period_data[r]['E_error'] for r in ratios]

        if errors:
            ax.semilogy(ratios, errors, 'o-', label=f'η={eta}', markersize=4)

        # Add horizontal line for instantaneous error
        instant_error = results['instant'][eta]['E_error']
        ax.axhline(instant_error, linestyle='--', alpha=0.3)

    ax.axvline(1.0, color='red', linestyle=':', alpha=0.5, label='T_obs = T')
    ax.set_xlabel('T_obs / T')
    ax.set_ylabel('Energy error |E_meas - E_true|')
    ax.set_title('Period-Based Method\n(Dashed = Instantaneous)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 1b: Instantaneous E error vs η
    ax = axes[0, 1]
    etas = sorted(results['instant'].keys())
    E_errors = [results['instant'][eta]['E_error'] for eta in etas]
    Q_errors = [results['instant'][eta]['Q_error'] for eta in etas]

    ax.loglog(etas, E_errors, 'bo-', label='E error', markersize=8)
    ax.loglog(etas, Q_errors, 'rs-', label='Q error', markersize=8)
    ax.axhline(np.pi, color='red', linestyle=':', alpha=0.5, label='π (random)')
    ax.set_xlabel('Back-action strength η')
    ax.set_ylabel('Measurement error')
    ax.set_title('Instantaneous Method\n(Trade-off: E vs Q)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 1c: Optimal σ_q vs η
    ax = axes[0, 2]
    opt_sigmas = [results['instant'][eta]['optimal_sigma_q'] for eta in etas]
    sigma_kicks = [results['instant'][eta]['sigma_p_kick'] for eta in etas]

    ax.loglog(etas, opt_sigmas, 'go-', label='Optimal σ_q', markersize=8)
    ax.loglog(etas, sigma_kicks, 'm^-', label='σ_p (kick)', markersize=8)
    ax.loglog(etas, np.sqrt(etas), 'k--', alpha=0.5, label='√η')
    ax.set_xlabel('Back-action strength η')
    ax.set_ylabel('Measurement/kick precision')
    ax.set_title('Uncertainty Trade-off\n(σ_q · σ_p ≥ η)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2a: Phase diagram - which method wins?
    ax = axes[1, 0]

    # Create grid
    T_grid = np.array(T_obs_ratios)
    eta_grid = np.array(eta_values)

    # Compute best method for each cell
    best_error = np.zeros((len(eta_values), len(T_obs_ratios)))
    method_map = np.zeros((len(eta_values), len(T_obs_ratios)))  # 0=instant, 1=period

    for i, eta in enumerate(eta_values):
        instant_err = results['instant'][eta]['E_error']
        for j, T_ratio in enumerate(T_obs_ratios):
            period_data = results['period'][eta].get(T_ratio, {})
            period_err = period_data.get('E_error', np.inf)

            if np.isnan(period_err):
                period_err = np.inf

            best_error[i, j] = min(instant_err, period_err)
            method_map[i, j] = 0 if instant_err < period_err else 1

    im = ax.imshow(np.log10(best_error + 1e-10), aspect='auto', origin='lower',
                   extent=[np.log10(T_grid[0]), np.log10(T_grid[-1]),
                          np.log10(eta_grid[0]), np.log10(eta_grid[-1])],
                   cmap='viridis')
    ax.set_xlabel('log₁₀(T_obs / T)')
    ax.set_ylabel('log₁₀(η)')
    ax.set_title('Best Energy Error\n(log scale)')
    plt.colorbar(im, ax=ax, label='log₁₀(E error)')

    # 2b: Method dominance map
    ax = axes[1, 1]
    im2 = ax.imshow(method_map, aspect='auto', origin='lower',
                    extent=[np.log10(T_grid[0]), np.log10(T_grid[-1]),
                           np.log10(eta_grid[0]), np.log10(eta_grid[-1])],
                    cmap='RdYlBu', vmin=-0.5, vmax=1.5)
    ax.set_xlabel('log₁₀(T_obs / T)')
    ax.set_ylabel('log₁₀(η)')
    ax.set_title('Best Method\n(Blue=Instant, Red=Period)')

    # Add crossover line
    crossovers = results['crossover']
    cross_etas = [eta for eta in etas if crossovers[eta] is not None]
    cross_Ts = [crossovers[eta] for eta in cross_etas]
    if cross_etas and cross_Ts:
        ax.plot(np.log10(cross_Ts), np.log10(cross_etas), 'k-', linewidth=2, label='Crossover')
        ax.legend()

    # 2c: Summary
    ax = axes[1, 2]
    ax.axis('off')

    summary = """
    BACK-ACTION ENERGY MEASUREMENT
    ===============================

    TWO METHODS:

    1. INSTANTANEOUS (q,p) → E
       - Fast (single instant)
       - Back-action: σ_q·σ_p ≥ η
       - Trade-off: precise q → noisy p
       - Can measure E AND Q (if η small)

    2. PERIOD-BASED ω → E
       - Slow (requires T_obs ≥ T)
       - No back-action on E
       - Averages over cycle
       - E precise, but Q lost

    KEY RESULT:

    For small η (weak back-action):
      → Instantaneous works
      → Can know E AND Q
      → "Classical" regime

    For large η (strong back-action):
      → Must use period method
      → Requires T_obs ≥ T
      → E knowable, Q lost
      → "Quantum" regime

    The TIMESCALE of energy measurement
    determines whether phase is accessible!

    η sets the boundary between
    classical and quantum regimes.
    """
    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig('backaction_energy_measurement.png', dpi=150, bbox_inches='tight')
    print("\nSaved: backaction_energy_measurement.png")
    plt.show()

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    print(f"\n{'η':<10} {'E_err (inst)':<15} {'Q_err (inst)':<15} {'T_crossover':<15}")
    print("-" * 55)

    for eta in eta_values:
        E_err = results['instant'][eta]['E_error']
        Q_err = results['instant'][eta]['Q_error']
        crossover = results['crossover'][eta]
        cross_str = f"{crossover:.1f}T" if crossover else ">20T"
        print(f"{eta:<10.3f} {E_err:<15.4f} {Q_err:<15.4f} {cross_str:<15}")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print("""
    Back-action creates a fundamental link between:
    - MEASUREMENT PRECISION (σ_q, σ_p)
    - MEASUREMENT TIMESCALE (T_obs)
    - OBSERVABLE INFORMATION (E vs Q)

    Strong back-action (large η) forces period-based measurement,
    which requires T_obs ≥ T, during which phase averages out.

    This explains WHY:
    - Energy (action) can be measured at long timescales
    - Phase cannot be measured at long timescales

    The "quantum" regime emerges when η is large enough that
    instantaneous (q,p) measurement is worse than period-based.
    """)

    return results


if __name__ == "__main__":
    results = run_full_test()
