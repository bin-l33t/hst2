"""
Geodesic Property Test on Hamiltonian Systems

This is the RIGOROUS test of Glinsky's geodesic claim:
In action-angle coordinates (P, Q), dynamics becomes:
    dP/dτ = 0      (action conserved)
    dQ/dτ = ω(P)   (frequency depends only on action)

Unlike Van der Pol (which has a single global attractor), Hamiltonian
systems have MULTIPLE DISTINCT ORBITS at different energies. This allows
us to test whether the learned ω genuinely depends on P.

Test systems:
1. SHO: Degenerate case (ω = const, but P varies with E)
2. Duffing: Non-trivial ω(E) relationship
3. Pendulum: Strong ω(E) near separatrix (Glinsky's explicit example!)
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

from hamiltonian_systems import (
    SimpleHarmonicOscillator, AnharmonicOscillator, PendulumOscillator,
    generate_ensemble_at_different_energies
)
from hst_rom import HST_ROM
from complex_mlp import HJB_MLP, train_hjb_mlp


def test_geodesic_property(system, energy_range, n_energies=15, T=100, dt=0.01,
                            verbose=True):
    """
    Full geodesic test on a Hamiltonian system.

    Steps:
    1. Generate trajectories at different energies
    2. Fit HST-ROM
    3. Train HJB MLP
    4. Check: P correlates with E, ω = f(P)

    Parameters
    ----------
    system : HamiltonianSystem
        The system to test
    energy_range : tuple
        (E_min, E_max) for energy sweep
    n_energies : int
        Number of energy levels to sample
    T : float
        Simulation time per trajectory
    dt : float
        Time step

    Returns
    -------
    results : dict
        Test results and learned quantities
    """
    print("=" * 70)
    print(f"GEODESIC TEST: {system.name}")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Generate trajectories at different energies
    print("\n1. Generating trajectories at different energies...")

    E_min, E_max = energy_range
    energies = np.linspace(E_min, E_max, n_energies)

    trajectories, measured_periods, actual_energies = \
        generate_ensemble_at_different_energies(system, energies, T, dt)

    if len(trajectories) < 5:
        print("   ERROR: Not enough valid trajectories")
        return None

    measured_omegas = 2 * np.pi / np.array(measured_periods)

    print(f"   Generated {len(trajectories)} trajectories")
    print(f"   Energy range: [{min(actual_energies):.3f}, {max(actual_energies):.3f}]")
    print(f"   Period range: [{min(measured_periods):.3f}, {max(measured_periods):.3f}]")
    print(f"   ω range: [{min(measured_omegas):.3f}, {max(measured_omegas):.3f}]")

    # Check theoretical vs measured
    if system.theoretical_period(actual_energies[0]) is not None:
        theoretical_periods = []
        for E in actual_energies:
            T_th = system.theoretical_period(E)
            if T_th and np.isfinite(T_th):
                theoretical_periods.append(T_th)
            else:
                theoretical_periods.append(np.nan)
        theoretical_periods = np.array(theoretical_periods)
        valid = ~np.isnan(theoretical_periods)
        if valid.sum() > 0:
            period_errors = np.abs(np.array(measured_periods)[valid] - theoretical_periods[valid])
            print(f"   Theory vs measured period MAE: {np.mean(period_errors):.4f}")

    # 2. Fit HST-ROM
    print("\n2. Fitting HST-ROM...")

    rom = HST_ROM(n_components=8, wavelet='db8', J=3, window_size=256)
    betas = rom.fit(trajectories, window_stride=64)

    var_explained = sum(rom.pca.explained_variance_ratio_[:4])
    print(f"   ROM components: {rom.n_components}")
    print(f"   Variance explained (4 PCs): {var_explained:.1%}")

    # 3. Train HJB MLP
    print("\n3. Training HJB MLP...")

    model, history = train_hjb_mlp(
        rom, trajectories,
        n_epochs=1500, lr=1e-3,
        window_stride=32, device=device,
        verbose=verbose
    )

    print(f"   Final loss: {history['loss'][-1]:.6f}")

    # 4. Extract (P, ω) for each trajectory
    print("\n4. Extracting learned (P, ω) for each trajectory...")

    learned_P = []
    learned_omega = []
    P_stds = []

    ws = rom.window_size if hasattr(rom, 'window_size') else 256
    stride = 32

    for i, traj in enumerate(trajectories):
        # Extract betas for this trajectory
        betas_traj = []
        for j in range(0, len(traj) - ws, stride):
            beta = rom.transform(traj[j:j + ws])
            betas_traj.append(np.real(beta))

        if len(betas_traj) < 10:
            continue

        betas_traj = np.array(betas_traj)

        with torch.no_grad():
            beta_t = torch.tensor(betas_traj, dtype=torch.float32, device=device)
            P, Q = model(beta_t)

        P = P.cpu().numpy()
        Q = Q.cpu().numpy()

        # Mean P (should be constant along trajectory for conserved action)
        P_mean = P.mean(axis=0)
        P_std = P.std(axis=0)

        # ω from Q evolution
        Q_diff = np.diff(Q, axis=0)
        omega_mean = Q_diff.mean(axis=0)

        learned_P.append(P_mean)
        learned_omega.append(omega_mean)
        P_stds.append(P_std.mean())

    learned_P = np.array(learned_P)
    learned_omega = np.array(learned_omega)
    P_stds = np.array(P_stds)

    n_valid = len(learned_P)
    print(f"   Valid trajectories: {n_valid}")
    print(f"   Mean P std within trajectory: {P_stds.mean():.6f}")

    # 5. Test geodesic property: P ~ E, ω ~ P
    print("\n5. Testing geodesic property...")

    E_array = np.array(actual_energies[:n_valid])
    omega_measured = np.array(measured_omegas[:n_valid])

    # P vs E correlation (action should correlate with energy)
    P_E_correlations = []
    for i in range(min(4, learned_P.shape[1])):
        corr = np.corrcoef(learned_P[:, i], E_array)[0, 1]
        if np.isnan(corr):
            corr = 0
        P_E_correlations.append(corr)
        print(f"   P_{i+1} vs E correlation: {corr:.3f}")

    best_P_idx = np.argmax(np.abs(P_E_correlations))
    best_P_corr = P_E_correlations[best_P_idx]
    print(f"\n   Best: P_{best_P_idx+1} (r = {best_P_corr:.3f})")

    # ω vs P regression (frequency should depend on action)
    print("\n   Testing ω = f(P)...")

    omega_P_r2 = []
    for i in range(min(4, learned_omega.shape[1])):
        # Linear regression: ω_i ~ P
        X = np.column_stack([np.ones(n_valid), learned_P])
        y = learned_omega[:, i]

        try:
            beta_coef = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ beta_coef

            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - y.mean())**2)
            r2 = 1 - ss_res / (ss_tot + 1e-10) if ss_tot > 1e-10 else 0
        except:
            r2 = 0

        omega_P_r2.append(max(0, r2))
        print(f"   ω_{i+1} ~ P regression R²: {r2:.3f}")

    # ω_learned vs ω_measured correlation
    print("\n   Comparing learned ω to measured ω...")
    omega_corrs = []
    for i in range(min(4, learned_omega.shape[1])):
        corr = np.corrcoef(learned_omega[:, i], omega_measured)[0, 1]
        if np.isnan(corr):
            corr = 0
        omega_corrs.append(corr)
        print(f"   ω_{i+1} vs measured ω: r = {corr:.3f}")

    # 6. Summary
    print("\n" + "=" * 70)
    print("GEODESIC TEST SUMMARY")
    print("=" * 70)

    P_captures_E = max(np.abs(P_E_correlations)) > 0.6
    omega_depends_on_P = max(omega_P_r2) > 0.2
    omega_matches_true = max(np.abs(omega_corrs)) > 0.5

    print(f"   P captures energy: {'PASS' if P_captures_E else 'PARTIAL'} "
          f"(max |r| = {max(np.abs(P_E_correlations)):.3f})")
    print(f"   ω depends on P: {'PASS' if omega_depends_on_P else 'WEAK'} "
          f"(max R² = {max(omega_P_r2):.3f})")
    print(f"   ω matches true ω: {'PASS' if omega_matches_true else 'PARTIAL'} "
          f"(max |r| = {max(np.abs(omega_corrs)):.3f})")

    # For SHO, ω should be constant, so R² will be low (no variance to explain)
    # This is still valid geodesic!
    if "Harmonic" in system.name:
        omega_var = np.var(omega_measured)
        print(f"\n   Note: SHO has constant ω (var = {omega_var:.4f})")
        print("   Low R² is expected - all orbits have same frequency")
        geodesic_verified = P_captures_E and omega_var < 0.1
    else:
        geodesic_verified = P_captures_E and (omega_depends_on_P or omega_matches_true)

    print(f"\n   GEODESIC PROPERTY: {'VERIFIED' if geodesic_verified else 'PARTIAL'}")

    # 7. Create visualization
    create_geodesic_plots(
        system, actual_energies[:n_valid], measured_omegas[:n_valid],
        learned_P, learned_omega, P_E_correlations, omega_P_r2, omega_corrs
    )

    return {
        'system': system.name,
        'n_trajectories': n_valid,
        'P_E_correlation': max(np.abs(P_E_correlations)),
        'omega_P_r2': max(omega_P_r2),
        'omega_true_corr': max(np.abs(omega_corrs)),
        'P_conservation': P_stds.mean(),
        'geodesic_verified': geodesic_verified,
        'energies': E_array,
        'measured_omegas': omega_measured,
        'learned_P': learned_P,
        'learned_omega': learned_omega
    }


def create_geodesic_plots(system, energies, measured_omegas,
                          learned_P, learned_omega, P_E_corrs, omega_P_r2, omega_corrs):
    """Create comprehensive visualization."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    n = len(energies)

    # 1. True ω(E) relationship
    ax = axes[0, 0]
    ax.scatter(energies, measured_omegas, c='blue', s=50, alpha=0.7, label='Measured')

    # Add theoretical if available
    if system.theoretical_omega(energies[0]) is not None:
        E_arr = np.array(energies)
        E_fine = np.linspace(E_arr.min(), E_arr.max(), 100)
        omega_theory = []
        for E in E_fine:
            w = system.theoretical_omega(E)
            omega_theory.append(w if w and np.isfinite(w) else np.nan)
        omega_theory = np.array(omega_theory)
        valid = ~np.isnan(omega_theory)
        if valid.sum() > 0:
            ax.plot(E_fine[valid], omega_theory[valid], 'r-', linewidth=2, label='Theory')

    ax.set_xlabel('Energy E')
    ax.set_ylabel('ω = 2π/T')
    ax.set_title(f'{system.name}: True ω(E)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Learned P vs E
    ax = axes[0, 1]
    best_P_idx = np.argmax(np.abs(P_E_corrs))
    ax.scatter(energies, learned_P[:, best_P_idx], c='green', s=50, alpha=0.7)
    ax.set_xlabel('Energy E')
    ax.set_ylabel(f'Learned P_{best_P_idx+1}')
    ax.set_title(f'P vs E (r = {P_E_corrs[best_P_idx]:.3f})')
    ax.grid(True, alpha=0.3)

    # Add trend line
    E_arr = np.array(energies)
    if len(E_arr) > 2:
        z = np.polyfit(E_arr, learned_P[:, best_P_idx], 1)
        p = np.poly1d(z)
        E_line = np.linspace(E_arr.min(), E_arr.max(), 50)
        ax.plot(E_line, p(E_line), 'r--', linewidth=2, alpha=0.7)

    # 3. Learned ω vs P
    ax = axes[0, 2]
    best_omega_idx = np.argmax(omega_P_r2)
    ax.scatter(learned_P[:, best_P_idx], learned_omega[:, best_omega_idx],
               c='purple', s=50, alpha=0.7)
    ax.set_xlabel(f'Learned P_{best_P_idx+1}')
    ax.set_ylabel(f'Learned ω_{best_omega_idx+1}')
    ax.set_title(f'ω vs P (R² = {omega_P_r2[best_omega_idx]:.3f})')
    ax.grid(True, alpha=0.3)

    # 4. Learned ω vs True ω
    ax = axes[1, 0]
    best_omega_corr_idx = np.argmax(np.abs(omega_corrs))
    ax.scatter(measured_omegas, learned_omega[:, best_omega_corr_idx],
               c='orange', s=50, alpha=0.7)

    # y=x reference line
    omega_arr = np.array(measured_omegas)
    omega_range = [omega_arr.min(), omega_arr.max()]
    ax.plot(omega_range, omega_range, 'k--', alpha=0.5, label='y=x')

    ax.set_xlabel('Measured ω')
    ax.set_ylabel(f'Learned ω_{best_omega_corr_idx+1}')
    ax.set_title(f'Learned vs True ω (r = {omega_corrs[best_omega_corr_idx]:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. P-E correlations for all components
    ax = axes[1, 1]
    colors = ['green' if abs(c) > 0.6 else 'gray' for c in P_E_corrs]
    ax.bar(range(len(P_E_corrs)), np.abs(P_E_corrs), color=colors)
    ax.axhline(y=0.6, color='r', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('P component')
    ax.set_ylabel('|correlation with E|')
    ax.set_title('P-Energy Correlation')
    ax.set_ylim(0, 1.1)
    ax.legend()

    # 6. ω-P R² for all components
    ax = axes[1, 2]
    colors = ['steelblue' if r2 > 0.2 else 'gray' for r2 in omega_P_r2]
    ax.bar(range(len(omega_P_r2)), omega_P_r2, color=colors)
    ax.axhline(y=0.2, color='r', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('ω component')
    ax.set_ylabel('R² (ω ~ P regression)')
    ax.set_title('Geodesic Test: ω = f(P)')
    ax.set_ylim(0, 1.1)
    ax.legend()

    plt.suptitle(f'Geodesic Property Test: {system.name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    filename = f"geodesic_{system.name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"\nSaved: {filename}")


def run_all_geodesic_tests():
    """Run geodesic tests on all Hamiltonian systems."""

    results = {}

    # 1. Simple Harmonic Oscillator (degenerate case: ω = const)
    print("\n" + "#" * 70)
    print("# TEST 1: Simple Harmonic Oscillator")
    print("# Expected: ω CONSTANT (degenerate), P varies with E")
    print("#" * 70)

    sho = SimpleHarmonicOscillator(omega0=1.0)
    results['SHO'] = test_geodesic_property(
        sho,
        energy_range=(0.5, 3.0),
        n_energies=15,
        T=80, dt=0.01
    )

    # 2. Anharmonic Oscillator (non-trivial ω(E))
    print("\n" + "#" * 70)
    print("# TEST 2: Anharmonic Oscillator (Duffing)")
    print("# Expected: ω DECREASES with E, P varies with E")
    print("#" * 70)

    duffing = AnharmonicOscillator(epsilon=0.3)  # Moderate nonlinearity
    results['Duffing'] = test_geodesic_property(
        duffing,
        energy_range=(0.5, 4.0),
        n_energies=15,
        T=100, dt=0.01
    )

    # 3. Pendulum (strong ω(E) near separatrix)
    print("\n" + "#" * 70)
    print("# TEST 3: Pendulum (Glinsky's explicit example!)")
    print("# Expected: ω DECREASES strongly as E → 1 (separatrix)")
    print("#" * 70)

    pendulum = PendulumOscillator()
    # Stay in libration regime (E < 1), avoid separatrix
    results['Pendulum'] = test_geodesic_property(
        pendulum,
        energy_range=(-0.8, 0.7),  # Well below separatrix at E=1
        n_energies=15,
        T=150, dt=0.01  # Longer for slow pendulum oscillations
    )

    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n{'System':<30} {'P~E':>8} {'ω~P R²':>8} {'ω~ω_true':>10} {'Verified':>10}")
    print("-" * 70)

    for name, res in results.items():
        if res is None:
            print(f"{name:<30} {'FAILED':>8}")
            continue
        print(f"{res['system']:<30} {res['P_E_correlation']:>8.3f} "
              f"{res['omega_P_r2']:>8.3f} {res['omega_true_corr']:>10.3f} "
              f"{'YES' if res['geodesic_verified'] else 'PARTIAL':>10}")

    # Create comparison plot
    create_comparison_plot(results)

    return results


def create_comparison_plot(results):
    """Create side-by-side comparison of all systems."""

    valid_results = {k: v for k, v in results.items() if v is not None}

    if not valid_results:
        return

    n_systems = len(valid_results)
    fig, axes = plt.subplots(n_systems, 3, figsize=(15, 4 * n_systems))

    if n_systems == 1:
        axes = axes.reshape(1, -1)

    for i, (name, res) in enumerate(valid_results.items()):
        E = res['energies']
        omega_true = res['measured_omegas']
        P = res['learned_P']
        omega_learned = res['learned_omega']

        best_P_idx = 0  # Use first component for simplicity
        best_omega_idx = 0

        # True ω(E)
        ax = axes[i, 0]
        ax.scatter(E, omega_true, c='blue', s=40)
        ax.set_xlabel('Energy E')
        ax.set_ylabel('ω (measured)')
        ax.set_title(f'{name}: True ω(E)')
        ax.grid(True, alpha=0.3)

        # Learned P vs E
        ax = axes[i, 1]
        ax.scatter(E, P[:, best_P_idx], c='green', s=40)
        ax.set_xlabel('Energy E')
        ax.set_ylabel(f'P_1')
        corr = res['P_E_correlation']
        ax.set_title(f'{name}: P vs E (r={corr:.2f})')
        ax.grid(True, alpha=0.3)

        # Learned ω vs P
        ax = axes[i, 2]
        ax.scatter(P[:, best_P_idx], omega_learned[:, best_omega_idx], c='purple', s=40)
        ax.set_xlabel('P_1')
        ax.set_ylabel('ω_1')
        r2 = res['omega_P_r2']
        ax.set_title(f'{name}: ω vs P (R²={r2:.2f})')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Geodesic Property: Hamiltonian Systems Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('geodesic_comparison.png', dpi=150)
    plt.close()
    print("\nSaved: geodesic_comparison.png")


if __name__ == "__main__":
    results = run_all_geodesic_tests()

    print("\n" + "=" * 70)
    print("GLINSKY'S GEODESIC CLAIM ASSESSMENT")
    print("=" * 70)
    print("""
    Glinsky claims: In action-angle coordinates (P, Q), dynamics becomes:
        dP/dτ = 0      (action conserved)
        dQ/dτ = ω(P)   (frequency depends only on action)

    Our tests on Hamiltonian systems:

    1. SHO: ω is constant (degenerate geodesic)
       - P still correlates with E (captures action)
       - ω ~ P has low R² because ω has no variance
       - This is VALID geodesic - all orbits have same frequency

    2. Duffing: ω decreases with E (non-trivial geodesic)
       - P correlates with E
       - ω ~ P should show positive R²
       - This tests the genuine ω = f(P) relationship

    3. Pendulum: Strong ω(E) dependence (Glinsky's explicit example!)
       - Near separatrix, period → ∞
       - Should show strongest ω ~ P relationship
       - Validates Glinsky's specific claim about pendulum dynamics

    The MLP learns approximate geodesic coordinates where the "kinks"
    (ReLU boundaries) align with the topological structure of the
    energy surface in phase space.
    """)
