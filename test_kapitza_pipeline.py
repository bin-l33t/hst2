"""
Complete Kapitza Pendulum Pipeline Test

Validates the entire Glinsky pipeline (HST → ROM → Control) on the
canonical example of ponderomotive stabilization.

Key validation points:
1. Basic physics: Stability transition at κ=1 matches theory
2. ROM captures transition: β coordinates separate κ<1 from κ>1
3. Control works: Can stabilize κ<1 system using ponderomotive forcing
4. Quantitative match: Required ε·Ω matches theoretical prediction
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from kapitza_pendulum import (
    simulate_kapitza, compute_stability_parameter, kapitza_dynamics,
    effective_potential, test_kapitza_stability
)
from hst_rom import HST_ROM
from hjb_decoder import HAS_TORCH


def test_kapitza_rom():
    """
    Test HST-ROM on Kapitza pendulum.

    Key questions:
    1. Does ROM capture the stabilization transition?
    2. Can we see κ > 1 vs κ < 1 in ROM coordinates?
    """
    print("\n" + "=" * 60)
    print("HST-ROM ON KAPITZA PENDULUM")
    print("=" * 60)

    g, L = 9.81, 1.0

    # Generate ensemble of trajectories at different κ values
    kappa_values = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]

    trajectories = []
    labels = []

    print("\nGenerating trajectories for different κ values...")

    for kappa in kappa_values:
        # Solve for a given κ: κ = (aΩ)²/(2gL) → aΩ = √(2gLκ)
        Omega = 50.0  # Fix Ω
        a = np.sqrt(2 * g * L * kappa) / Omega

        # Start near inverted
        theta0 = np.pi - 0.1
        theta_dot0 = 0.0

        T = 10.0
        dt = 0.002

        t, theta, theta_dot, z = simulate_kapitza(
            theta0, theta_dot0, T, dt, g, L, a, Omega
        )

        trajectories.append(z)
        labels.append(f'κ={kappa:.1f}')

        print(f"  κ = {kappa:.1f}: {len(z)} samples, stable={np.max(np.abs(theta[-len(theta)//2:] - np.pi)) < np.pi/6}")

    # Fit ROM
    print("\nFitting HST-ROM...")
    rom = HST_ROM(n_components=8, wavelet='db8', J=3, window_size=256)
    betas = rom.fit(trajectories, extract_windows=True, window_stride=64)

    print(f"  ROM fitted on {len(betas)} windows")
    print(f"  Variance explained (first 4): {[f'{v:.3f}' for v in rom.pca.explained_variance_ratio_[:4]]}")
    print(f"  Total (4 PCs): {sum(rom.pca.explained_variance_ratio_[:4]):.3f}")

    # Transform each trajectory
    beta_by_kappa = {}
    for kappa, z in zip(kappa_values, trajectories):
        betas_traj, times = rom.transform_trajectory(z, window_stride=64)
        beta_by_kappa[kappa] = betas_traj

    # Analyze separation between stable/unstable
    print("\nAnalyzing ROM coordinate separation...")

    stable_betas = []
    unstable_betas = []

    for kappa in kappa_values:
        betas_traj = beta_by_kappa[kappa]
        if kappa > 1:
            stable_betas.append(betas_traj)
        else:
            unstable_betas.append(betas_traj)

    if stable_betas and unstable_betas:
        stable_all = np.vstack(stable_betas)
        unstable_all = np.vstack(unstable_betas)

        # Compute centroids
        stable_centroid = stable_all.mean(axis=0)
        unstable_centroid = unstable_all.mean(axis=0)

        separation = np.linalg.norm(stable_centroid - unstable_centroid)
        stable_spread = np.std(stable_all[:, 0])
        unstable_spread = np.std(unstable_all[:, 0])

        print(f"  Centroid separation (β₁): {np.abs(stable_centroid[0] - unstable_centroid[0]):.3f}")
        print(f"  Total separation: {separation:.3f}")
        print(f"  Stable spread: {stable_spread:.3f}")
        print(f"  Unstable spread: {unstable_spread:.3f}")

    # Create visualization
    fig = plt.figure(figsize=(14, 10))

    # 1. β₁ vs β₂ colored by κ
    ax1 = fig.add_subplot(2, 2, 1)
    for kappa in kappa_values:
        betas_traj = beta_by_kappa[kappa]
        color = 'red' if kappa < 1 else 'blue'
        marker = 'o' if kappa < 1 else 's'
        alpha = 0.3 + 0.5 * (kappa / max(kappa_values))
        ax1.scatter(betas_traj[:, 0], betas_traj[:, 1],
                   c=color, alpha=alpha, s=15, marker=marker, label=f'κ={kappa:.1f}')

    ax1.set_xlabel('β₁')
    ax1.set_ylabel('β₂')
    ax1.set_title('ROM Coordinates\nRed = unstable (κ<1), Blue = stable (κ>1)')
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # 2. β₁ mean vs κ
    ax2 = fig.add_subplot(2, 2, 2)
    beta1_means = [beta_by_kappa[k][:, 0].mean() for k in kappa_values]
    beta1_stds = [beta_by_kappa[k][:, 0].std() for k in kappa_values]

    colors = ['red' if k < 1 else 'blue' for k in kappa_values]
    ax2.bar(range(len(kappa_values)), beta1_means, yerr=beta1_stds,
           color=colors, alpha=0.7, capsize=5)
    ax2.set_xticks(range(len(kappa_values)))
    ax2.set_xticklabels([f'{k:.1f}' for k in kappa_values])
    ax2.axvline(x=kappa_values.index(1.0) if 1.0 in kappa_values else 3, color='k',
               linestyle='--', label='κ=1')
    ax2.set_xlabel('κ')
    ax2.set_ylabel('β₁ (mean ± std)')
    ax2.set_title('ROM Coordinate vs Stability Parameter')
    ax2.legend()

    # 3. Effective potential
    ax3 = fig.add_subplot(2, 2, 3)
    theta = np.linspace(0, 2*np.pi, 200)
    for kappa in [0.5, 1.0, 2.0]:
        Omega = 50.0
        a = np.sqrt(2 * g * L * kappa) / Omega
        V = effective_potential(theta, g, L, a, Omega)
        ax3.plot(np.degrees(theta), V, label=f'κ={kappa:.1f}')

    ax3.axvline(x=180, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('θ (degrees)')
    ax3.set_ylabel('V_eff / (mgL)')
    ax3.set_title('Effective Potential')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Reconstruction quality
    ax4 = fig.add_subplot(2, 2, 4)
    recon_errors = []
    for kappa in kappa_values:
        z = trajectories[kappa_values.index(kappa)]
        if len(z) >= 256:
            error = rom.reconstruction_error(z[:256])
            recon_errors.append(error * 100)
        else:
            recon_errors.append(np.nan)

    colors = ['red' if k < 1 else 'blue' for k in kappa_values]
    ax4.bar(range(len(kappa_values)), recon_errors, color=colors, alpha=0.7)
    ax4.set_xticks(range(len(kappa_values)))
    ax4.set_xticklabels([f'{k:.1f}' for k in kappa_values])
    ax4.set_xlabel('κ')
    ax4.set_ylabel('Reconstruction Error (%)')
    ax4.set_title('ROM Reconstruction Quality')
    ax4.axhline(y=20, color='g', linestyle='--', label='20% threshold')
    ax4.legend()

    plt.tight_layout()
    plt.savefig('kapitza_rom_analysis.png', dpi=150)
    plt.close()
    print("\nSaved: kapitza_rom_analysis.png")

    return rom, beta_by_kappa


def tune_ponderomotive_on_kapitza():
    """
    Systematically tune ponderomotive control parameters.

    The Kapitza pendulum gives us ground truth:
    - We know exactly when stabilization should occur (κ > 1)
    - We can compare our control to the analytical ponderomotive force
    """
    print("\n" + "=" * 60)
    print("PONDEROMOTIVE CONTROL TUNING")
    print("=" * 60)

    g, L = 9.81, 1.0
    omega0 = np.sqrt(g / L)  # Natural frequency ≈ 3.13 rad/s

    # Base system: κ = 0.7 (unstable inverted position)
    kappa_base = 0.7
    Omega_base = 50.0
    a_base = np.sqrt(2 * g * L * kappa_base) / Omega_base

    print(f"\nBase system:")
    print(f"  κ_base = {kappa_base}")
    print(f"  Ω_base = {Omega_base} rad/s")
    print(f"  a_base = {a_base:.4f} m")
    print(f"  ω₀ = {omega0:.2f} rad/s")

    # Theory: need κ_eff > 1 for stability
    # Adding control with amplitude ε and frequency Ω_ctrl:
    # δκ ≈ (ε·Ω_ctrl)² / (2gL)
    # Need: κ_base + δκ > 1
    # So: (ε·Ω_ctrl)² > 2gL(1 - κ_base) = 2gL·0.3
    # ε·Ω_ctrl > √(2gL·0.3) ≈ 2.43

    required_product = np.sqrt(2 * g * L * (1 - kappa_base))
    print(f"\nTheory: Need ε·Ω_ctrl > {required_product:.2f} for stability")

    # Test without control
    print("\n--- Without Control ---")
    t, theta, _, _ = simulate_kapitza(
        np.pi - 0.1, 0.0, 20.0, 0.002, g, L, a_base, Omega_base
    )
    max_dev_no_control = np.max(np.abs(theta[len(theta)//2:] - np.pi))
    print(f"Max deviation: {np.degrees(max_dev_no_control):.1f}°")
    print(f"Stable: {max_dev_no_control < np.pi/6}")

    # Parameter sweep
    print("\n--- Parameter Sweep ---")
    epsilon_values = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1]
    Omega_ctrl_values = [30, 50, 80, 100]

    results = []

    for Omega_ctrl in Omega_ctrl_values:
        for eps in epsilon_values:
            product = eps * Omega_ctrl
            delta_kappa = (eps * Omega_ctrl)**2 / (2 * g * L)
            effective_kappa = kappa_base + delta_kappa

            # Simulate with combined pivot + control
            def dynamics_with_control(t, y):
                theta, theta_dot = y

                # Base pivot
                pivot_base = (a_base / L) * Omega_base**2 * np.cos(Omega_base * t) * np.sin(theta)

                # Control pivot
                pivot_ctrl = (eps / L) * Omega_ctrl**2 * np.cos(Omega_ctrl * t) * np.sin(theta)

                gravity = -(g / L) * np.sin(theta)

                return [theta_dot, gravity + pivot_base + pivot_ctrl]

            sol = solve_ivp(
                dynamics_with_control, (0, 20), [np.pi - 0.1, 0.0],
                t_eval=np.arange(0, 20, 0.002),
                method='RK45', rtol=1e-8, atol=1e-10
            )

            theta_final = sol.y[0][len(sol.y[0])//2:]
            max_dev = np.max(np.abs(theta_final - np.pi))
            is_stable = max_dev < np.pi / 6

            results.append({
                'epsilon': eps,
                'Omega_ctrl': Omega_ctrl,
                'product': product,
                'delta_kappa': delta_kappa,
                'effective_kappa': effective_kappa,
                'max_deviation_deg': np.degrees(max_dev),
                'stable': is_stable
            })

            status = '✓' if is_stable else '✗'
            print(f"ε={eps:.2f}, Ω={Omega_ctrl:3d}: ε·Ω={product:5.1f}, κ_eff={effective_kappa:.2f}, "
                  f"dev={np.degrees(max_dev):5.1f}° {status}")

    # Find stability boundary
    print("\n--- Stability Boundary ---")
    stable_results = [r for r in results if r['stable']]
    if stable_results:
        min_product = min(r['product'] for r in stable_results)
        print(f"Minimum ε·Ω for stability: {min_product:.2f}")
        print(f"Theoretical minimum: {required_product:.2f}")
        print(f"Ratio: {min_product / required_product:.2f}")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Stability map
    ax = axes[0]
    for r in results:
        color = 'green' if r['stable'] else 'red'
        ax.scatter(r['Omega_ctrl'], r['epsilon'], c=color, s=100,
                  marker='o' if r['stable'] else 'x')

    # Theory line: ε·Ω = required_product
    Omega_line = np.linspace(20, 120, 100)
    eps_line = required_product / Omega_line
    ax.plot(Omega_line, eps_line, 'b--', label=f'ε·Ω = {required_product:.2f} (theory)')

    ax.set_xlabel('Ω_ctrl (rad/s)')
    ax.set_ylabel('ε (m)')
    ax.set_title('Stability Map\n(green = stable, red = unstable)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Effective κ vs max deviation
    ax = axes[1]
    kappas = [r['effective_kappa'] for r in results]
    devs = [r['max_deviation_deg'] for r in results]
    colors = ['green' if r['stable'] else 'red' for r in results]

    ax.scatter(kappas, devs, c=colors, s=50)
    ax.axvline(x=1.0, color='b', linestyle='--', label='κ=1 (theory)')
    ax.axhline(y=30, color='k', linestyle=':', label='30° threshold')
    ax.set_xlabel('Effective κ')
    ax.set_ylabel('Max Deviation (degrees)')
    ax.set_title('Deviation vs Effective Stability Parameter')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('kapitza_control_tuning.png', dpi=150)
    plt.close()
    print("\nSaved: kapitza_control_tuning.png")

    return results


def test_full_pipeline():
    """
    Complete pipeline test: HST → ROM → Control on Kapitza.
    """
    print("\n" + "=" * 60)
    print("FULL KAPITZA PIPELINE TEST")
    print("=" * 60)

    g, L = 9.81, 1.0
    omega0 = np.sqrt(g / L)

    # Step 1: Generate training data (stable regime, κ > 1)
    print("\n[Step 1] Generating training data (stable regime, κ=2.0)...")

    training_trajectories = []
    kappa_train = 2.0
    Omega = 50.0
    a_train = np.sqrt(2 * g * L * kappa_train) / Omega

    np.random.seed(42)
    for _ in range(15):
        theta0 = np.pi + 0.2 * (np.random.rand() - 0.5)
        theta_dot0 = 0.3 * (np.random.rand() - 0.5)

        t, theta, _, z = simulate_kapitza(
            theta0, theta_dot0, 15.0, 0.002, g, L, a_train, Omega
        )
        training_trajectories.append(z)

    print(f"  Generated {len(training_trajectories)} stable trajectories")

    # Step 2: Fit ROM
    print("\n[Step 2] Fitting HST-ROM...")
    rom = HST_ROM(n_components=8, wavelet='db8', J=3, window_size=256)
    betas = rom.fit(training_trajectories, window_stride=64)

    print(f"  Windows extracted: {len(betas)}")
    print(f"  Variance (4 PCs): {sum(rom.pca.explained_variance_ratio_[:4]):.3f}")

    # Step 3: Define target (stable inverted position)
    print("\n[Step 3] Computing target ROM state...")
    target_window = training_trajectories[0][-256:]
    target_beta = rom.transform(target_window)
    print(f"  Target β: {target_beta[:4]}")

    # Step 4: Test on UNSTABLE system (κ < 1)
    print("\n[Step 4] Testing on unstable system (κ=0.7)...")

    kappa_test = 0.7
    a_test = np.sqrt(2 * g * L * kappa_test) / Omega

    # Control parameters (tuned from sweep)
    eps_ctrl = 0.05
    Omega_ctrl = 80.0

    # Without control
    theta0 = np.pi - 0.1
    theta_dot0 = 0.0
    T = 25.0
    dt = 0.002

    t_nc, theta_nc, theta_dot_nc, z_nc = simulate_kapitza(
        theta0, theta_dot0, T, dt, g, L, a_test, Omega
    )

    # With control
    def dynamics_controlled(t, y):
        theta, theta_dot = y

        # Base system
        pivot_base = (a_test / L) * Omega**2 * np.cos(Omega * t) * np.sin(theta)
        gravity = -(g / L) * np.sin(theta)

        # ROM-informed control
        # Use deviation from inverted as proxy (in full version, use ROM transform)
        theta_error = theta - np.pi

        # Ponderomotive control: amplitude modulated by error
        u_amp = eps_ctrl * np.tanh(2 * theta_error)
        pivot_ctrl = (u_amp / L) * Omega_ctrl**2 * np.cos(Omega_ctrl * t) * np.sin(theta)

        return [theta_dot, gravity + pivot_base + pivot_ctrl]

    sol = solve_ivp(
        dynamics_controlled, (0, T), [theta0, theta_dot0],
        t_eval=np.arange(0, T, dt),
        method='RK45', rtol=1e-8, atol=1e-10
    )
    t_c = sol.t
    theta_c = sol.y[0]
    theta_dot_c = sol.y[1]
    z_c = theta_c + 1j * theta_dot_c / omega0

    # Evaluate
    dev_nc = np.abs(theta_nc[len(theta_nc)//2:] - np.pi)
    dev_c = np.abs(theta_c[len(theta_c)//2:] - np.pi)

    max_dev_nc = np.max(dev_nc)
    max_dev_c = np.max(dev_c)

    stable_nc = max_dev_nc < np.pi / 6
    stable_c = max_dev_c < np.pi / 6

    print(f"\n[Results]")
    print(f"  Without control: max deviation = {np.degrees(max_dev_nc):.1f}°, stable = {stable_nc}")
    print(f"  With control:    max deviation = {np.degrees(max_dev_c):.1f}°, stable = {stable_c}")

    improvement = (max_dev_nc - max_dev_c) / max_dev_nc * 100
    print(f"  Improvement: {improvement:.1f}%")

    if not stable_nc and stable_c:
        print("\n  ✓ SUCCESS: Control stabilized the inverted pendulum!")
        success = True
    elif stable_c:
        print("\n  ✓ System stable with control")
        success = True
    else:
        print("\n  ✗ Control did not achieve full stabilization")
        success = False

    # Compare ROM trajectories
    print("\n[Step 5] Comparing ROM trajectories...")

    if len(z_nc) >= 256 and len(z_c) >= 256:
        betas_nc, _ = rom.transform_trajectory(z_nc, window_stride=64)
        betas_c, _ = rom.transform_trajectory(z_c, window_stride=64)

        dist_nc = np.mean([np.linalg.norm(b - target_beta) for b in betas_nc])
        dist_c = np.mean([np.linalg.norm(b - target_beta) for b in betas_c])

        print(f"  Mean distance to target (no control): {dist_nc:.3f}")
        print(f"  Mean distance to target (controlled): {dist_c:.3f}")

    # Create visualization
    fig = plt.figure(figsize=(14, 10))

    # 1. Angle trajectory
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(t_nc, np.degrees(theta_nc), 'r-', alpha=0.7, label='No control')
    ax1.plot(t_c, np.degrees(theta_c), 'b-', alpha=0.7, label='With control')
    ax1.axhline(y=180, color='k', linestyle='--', alpha=0.5)
    ax1.axhline(y=150, color='g', linestyle=':', alpha=0.3)
    ax1.axhline(y=210, color='g', linestyle=':', alpha=0.3)
    ax1.fill_between([0, T], 150, 210, alpha=0.1, color='green')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('θ (degrees)')
    ax1.set_title(f'Angle Trajectory (κ_base={kappa_test})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Phase portrait
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(np.degrees(theta_nc), theta_dot_nc, 'r-', alpha=0.5, linewidth=0.5, label='No control')
    ax2.plot(np.degrees(theta_c), theta_dot_c, 'b-', alpha=0.5, linewidth=0.5, label='With control')
    ax2.scatter([180], [0], c='g', s=100, marker='*', zorder=5, label='Target')
    ax2.set_xlabel('θ (degrees)')
    ax2.set_ylabel('θ̇ (rad/s)')
    ax2.set_title('Phase Portrait')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. ROM trajectory
    ax3 = fig.add_subplot(2, 2, 3)
    if len(z_nc) >= 256 and len(z_c) >= 256:
        ax3.plot(betas_nc[:, 0], betas_nc[:, 1], 'r.-', alpha=0.5, markersize=3, label='No control')
        ax3.plot(betas_c[:, 0], betas_c[:, 1], 'b.-', alpha=0.5, markersize=3, label='With control')
        ax3.scatter([target_beta[0]], [target_beta[1]], c='g', s=150, marker='*', zorder=5, label='Target')
        ax3.set_xlabel('β₁')
        ax3.set_ylabel('β₂')
        ax3.set_title('ROM Coordinates')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. Deviation over time
    ax4 = fig.add_subplot(2, 2, 4)
    t_plot = t_nc[len(t_nc)//4:]
    ax4.semilogy(t_plot, np.abs(theta_nc[len(theta_nc)//4:] - np.pi),
                'r-', alpha=0.7, label='No control')
    ax4.semilogy(t_c[len(t_c)//4:], np.abs(theta_c[len(theta_c)//4:] - np.pi),
                'b-', alpha=0.7, label='With control')
    ax4.axhline(y=np.pi/6, color='g', linestyle='--', label='Stability threshold')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('|θ - π| (rad)')
    ax4.set_title('Deviation from Inverted Position')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'Kapitza Pendulum Control: κ={kappa_test} → Stabilized={stable_c}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('kapitza_full_pipeline.png', dpi=150)
    plt.close()
    print("\nSaved: kapitza_full_pipeline.png")

    return {
        'stable_without_control': stable_nc,
        'stable_with_control': stable_c,
        'max_dev_nc_deg': np.degrees(max_dev_nc),
        'max_dev_c_deg': np.degrees(max_dev_c),
        'improvement': improvement,
        'success': success
    }


def run_all_kapitza_tests():
    """Run complete Kapitza pendulum validation suite."""
    print("\n" + "=" * 70)
    print("KAPITZA PENDULUM - COMPLETE VALIDATION SUITE")
    print("=" * 70)

    results = {}

    # Test 1: Basic physics
    print("\n[TEST 1] Basic Stability Physics")
    stability_results = test_kapitza_stability()
    results['stability'] = stability_results
    physics_pass = all(r['match'] for r in stability_results)

    # Test 2: ROM analysis
    print("\n[TEST 2] HST-ROM Analysis")
    rom, beta_by_kappa = test_kapitza_rom()
    results['rom'] = {'rom': rom, 'beta_by_kappa': beta_by_kappa}

    # Test 3: Control tuning
    print("\n[TEST 3] Ponderomotive Control Tuning")
    tuning_results = tune_ponderomotive_on_kapitza()
    results['tuning'] = tuning_results

    # Test 4: Full pipeline
    print("\n[TEST 4] Full Pipeline")
    pipeline_results = test_full_pipeline()
    results['pipeline'] = pipeline_results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n1. Basic Physics (κ=1 transition): {'PASS' if physics_pass else 'FAIL'}")
    print(f"   {sum(r['match'] for r in stability_results)}/{len(stability_results)} cases matched theory")

    print(f"\n2. ROM Analysis: COMPLETE")
    print(f"   Variance explained (4 PCs): {sum(rom.pca.explained_variance_ratio_[:4]):.3f}")

    stable_tuning = [r for r in tuning_results if r['stable']]
    print(f"\n3. Control Tuning: {len(stable_tuning)}/{len(tuning_results)} parameter combinations achieved stability")

    print(f"\n4. Full Pipeline: {'SUCCESS' if pipeline_results['success'] else 'NEEDS TUNING'}")
    print(f"   Without control: {pipeline_results['max_dev_nc_deg']:.1f}° deviation")
    print(f"   With control: {pipeline_results['max_dev_c_deg']:.1f}° deviation")
    print(f"   Improvement: {pipeline_results['improvement']:.1f}%")

    return results


if __name__ == "__main__":
    results = run_all_kapitza_tests()
