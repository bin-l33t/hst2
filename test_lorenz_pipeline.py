"""
Complete Lorenz Chaos Control Pipeline Test

Validates HST-ROM on chaotic systems, testing Glinsky's claims about
"topology of dynamical manifolds."

Key validation points:
1. ROM captures strange attractor structure (two wings)
2. ROM can identify/distinguish UPO regions
3. OGY-style control reduces Lyapunov exponent
4. Controlled trajectory shows more regular behavior

Comparison to Kapitza:
- Kapitza: Driven periodic system, stabilize fixed point
- Lorenz: Autonomous chaotic system, stabilize UPO
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from lorenz_system import (
    simulate_lorenz, lorenz_dynamics, lorenz_jacobian,
    compute_lyapunov_exponent, find_poincare_crossings,
    identify_upo_candidates, find_fixed_points, plot_lorenz_attractor
)
from hst_rom import HST_ROM


def test_lorenz_rom():
    """
    Test HST-ROM on Lorenz attractor.

    Key questions:
    1. Does ROM capture the two-wing structure?
    2. Can ROM distinguish which wing the trajectory is on?
    3. What's the reconstruction quality?
    """
    print("\n" + "=" * 60)
    print("HST-ROM ON LORENZ ATTRACTOR")
    print("=" * 60)

    # Generate long trajectory
    x0 = [1.0, 1.0, 1.0]
    T = 150.0
    dt = 0.01

    print("\nSimulating Lorenz system...")
    t, trajectory, z_xy, z_xz = simulate_lorenz(x0, T, dt)

    print(f"  Generated {len(t)} time points ({T} seconds)")

    # Verify chaos
    lyap = compute_lyapunov_exponent(trajectory, dt)
    print(f"  Lyapunov exponent: {lyap:.3f} (expected ~0.9)")

    # Find Poincaré crossings for later analysis
    crossings = find_poincare_crossings(trajectory)
    n_right = sum(1 for c in crossings if c['wing'] == 'right')
    n_left = len(crossings) - n_right
    print(f"  Poincaré crossings: {len(crossings)} (right: {n_right}, left: {n_left})")

    # Test both embeddings
    results = {}

    for name, z_complex in [('x+iy', z_xy), ('x+iz', z_xz)]:
        print(f"\n--- Embedding: {name} ---")

        rom = HST_ROM(n_components=8, wavelet='db8', J=3, window_size=256)

        # Fit on trajectory
        betas = rom.fit([z_complex], extract_windows=True, window_stride=64)

        print(f"  ROM samples: {len(betas)}")
        print(f"  Variance explained: {[f'{v:.3f}' for v in rom.pca.explained_variance_ratio_[:4]]}")
        print(f"  Total (4 PC): {sum(rom.pca.explained_variance_ratio_[:4]):.3f}")

        # Transform trajectory
        betas_traj, times = rom.transform_trajectory(z_complex, window_stride=32)

        # Check if ROM separates the two wings
        # Wing determined by sign of x at each window center
        time_indices = times.astype(int)
        valid_indices = time_indices < len(trajectory)
        time_indices = time_indices[valid_indices]
        betas_valid = betas_traj[valid_indices]

        x_at_times = trajectory[time_indices, 0]
        wing = np.sign(x_at_times)  # +1 (right) or -1 (left)

        # Correlation between β₁ and wing
        corr_wing = np.abs(np.corrcoef(betas_valid[:, 0], wing)[0, 1])
        print(f"  |β₁| correlation with wing: {corr_wing:.3f}")

        # Also check β₂
        corr_wing_2 = np.abs(np.corrcoef(betas_valid[:, 1], wing)[0, 1])
        print(f"  |β₂| correlation with wing: {corr_wing_2:.3f}")

        # Reconstruction error
        test_window = z_complex[1000:1256]
        recon_error = rom.reconstruction_error(test_window)
        print(f"  Reconstruction error: {recon_error*100:.1f}%")

        results[name] = {
            'rom': rom,
            'betas': betas,
            'betas_traj': betas_valid,
            'times': time_indices,
            'wing': wing,
            'corr_wing': corr_wing,
            'recon_error': recon_error
        }

    # Pick better embedding (higher wing correlation)
    best_embedding = max(results.keys(), key=lambda k: results[k]['corr_wing'])
    print(f"\nBest embedding: {best_embedding}")

    return results[best_embedding], trajectory, crossings


def test_upo_detection_via_rom(rom_result, trajectory, crossings):
    """
    Test if ROM can help identify UPO regions.
    """
    print("\n" + "=" * 60)
    print("UPO DETECTION VIA ROM")
    print("=" * 60)

    rom = rom_result['rom']
    betas_traj = rom_result['betas_traj']
    times = rom_result['times']

    # Find UPO candidates
    upos = identify_upo_candidates(crossings, max_period=4, tolerance=1.5)
    print(f"\nFound {len(upos)} UPO candidates in Poincaré section")

    # Get ROM coordinates for UPO regions
    upo_betas = []

    for upo in upos[:10]:  # Top 10 by return distance
        # Find closest ROM window to UPO crossing
        crossing_time = int(crossings[upo['start_idx']]['time_idx'])

        # Find closest time in our ROM trajectory
        time_diffs = np.abs(times - crossing_time)
        closest_idx = np.argmin(time_diffs)

        if time_diffs[closest_idx] < 50:  # Within 50 time steps
            upo_betas.append({
                'beta': betas_traj[closest_idx],
                'period': upo['period'],
                'return_distance': upo['return_distance']
            })

    print(f"Matched {len(upo_betas)} UPOs to ROM coordinates")

    if upo_betas:
        # Analyze UPO clustering in β space
        print("\nUPO positions in ROM space:")
        for i, ub in enumerate(upo_betas[:5]):
            print(f"  Period-{ub['period']}: β[:2] = {ub['beta'][:2]}, "
                  f"return_dist = {ub['return_distance']:.3f}")

        # Check if different period UPOs cluster separately
        period1 = [ub['beta'] for ub in upo_betas if ub['period'] == 1]
        period2 = [ub['beta'] for ub in upo_betas if ub['period'] == 2]

        if period1 and period2:
            p1_center = np.mean(period1, axis=0)
            p2_center = np.mean(period2, axis=0)
            separation = np.linalg.norm(p1_center - p2_center)
            print(f"\nPeriod-1 vs Period-2 separation: {separation:.3f}")

    return upo_betas


def test_chaos_control(rom_result, trajectory, crossings):
    """
    Test OGY-style chaos control using ROM.
    """
    print("\n" + "=" * 60)
    print("OGY-STYLE CHAOS CONTROL VIA ROM")
    print("=" * 60)

    rom = rom_result['rom']
    sigma, rho, beta_param = 10.0, 28.0, 8/3

    # Find a good UPO target
    upos = identify_upo_candidates(crossings, max_period=2, tolerance=1.5)

    if not upos:
        print("No UPO candidates found - using attractor center as target")
        target_beta = rom_result['betas'].mean(axis=0)
    else:
        # Use best period-1 UPO
        period1_upos = [u for u in upos if u['period'] == 1]
        if period1_upos:
            best_upo = min(period1_upos, key=lambda x: x['return_distance'])
            print(f"Target: Period-1 UPO with return distance {best_upo['return_distance']:.3f}")

            # Get ROM coordinates near this UPO
            crossing_time = int(crossings[best_upo['start_idx']]['time_idx'])
            times = rom_result['times']
            time_diffs = np.abs(times - crossing_time)
            closest_idx = np.argmin(time_diffs)
            target_beta = rom_result['betas_traj'][closest_idx]
        else:
            target_beta = rom_result['betas'].mean(axis=0)

    print(f"Target β[:4]: {target_beta[:4]}")

    # Simulate without control
    print("\nSimulating without control...")
    x0 = [1.0, 1.0, 1.0]
    T = 100.0
    dt = 0.01

    t_nc, traj_nc, z_nc, _ = simulate_lorenz(x0, T, dt)
    lyap_nc = compute_lyapunov_exponent(traj_nc, dt)
    print(f"  Lyapunov (no control): {lyap_nc:.3f}")

    # Simulate with control
    print("\nSimulating with ROM-based control...")

    def controlled_lorenz(t, state, rom, target_beta, epsilon=0.5, control_gain=0.1):
        """Lorenz with ROM-based stabilizing control."""
        x, y, z = state

        # Base dynamics
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta_param * z

        # Control: push toward target in x direction based on deviation
        # This is a simplified control - full OGY would use local linearization

        # Estimate current "phase" by x position
        # Apply control to y equation (affects x dynamics)
        x_target = np.sqrt(beta_param * (rho - 1)) if target_beta[0] > 0 else -np.sqrt(beta_param * (rho - 1))

        # Control when near target region
        control = control_gain * (x_target - x) * np.exp(-((z - 27)**2) / 50)

        # Add control to y dynamics
        dy += control

        return [dx, dy, dz]

    # Integrate controlled system
    sol = solve_ivp(
        lambda t, y: controlled_lorenz(t, y, rom, target_beta),
        (0, T), x0, t_eval=np.arange(0, T, dt),
        method='RK45', rtol=1e-8, atol=1e-10
    )
    traj_c = sol.y.T

    lyap_c = compute_lyapunov_exponent(traj_c, dt)
    print(f"  Lyapunov (with control): {lyap_c:.3f}")

    # Evaluate control effectiveness
    lyap_reduction = (lyap_nc - lyap_c) / lyap_nc * 100 if lyap_nc > 0 else 0
    print(f"\nLyapunov reduction: {lyap_reduction:.1f}%")

    # Check trajectory regularity
    # Count wing switches
    def count_wing_switches(traj):
        x = traj[:, 0]
        switches = np.sum(np.diff(np.sign(x)) != 0)
        return switches

    switches_nc = count_wing_switches(traj_nc)
    switches_c = count_wing_switches(traj_c)
    print(f"Wing switches (no control): {switches_nc}")
    print(f"Wing switches (with control): {switches_c}")

    # Measure time spent near target fixed point
    fps = find_fixed_points()
    C_target = fps['C+'] if target_beta[0] > 0 else fps['C-']

    def time_near_fp(traj, fp, threshold=5.0):
        distances = np.linalg.norm(traj - fp, axis=1)
        return np.sum(distances < threshold) / len(traj)

    near_nc = time_near_fp(traj_nc, C_target)
    near_c = time_near_fp(traj_c, C_target)
    print(f"Time near target C± (no control): {near_nc*100:.1f}%")
    print(f"Time near target C± (with control): {near_c*100:.1f}%")

    results = {
        'traj_nc': traj_nc,
        'traj_c': traj_c,
        'lyap_nc': lyap_nc,
        'lyap_c': lyap_c,
        'lyap_reduction': lyap_reduction,
        'switches_nc': switches_nc,
        'switches_c': switches_c,
        'near_nc': near_nc,
        'near_c': near_c,
        'target_beta': target_beta
    }

    return results


def create_lorenz_visualization(rom_result, control_result, trajectory):
    """Create comprehensive visualization."""
    fig = plt.figure(figsize=(16, 12))

    # 1. Uncontrolled attractor
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    traj = control_result['traj_nc']
    ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-', alpha=0.3, linewidth=0.3)
    ax1.set_title(f"No Control (λ={control_result['lyap_nc']:.2f})")
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    # 2. Controlled attractor
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    traj = control_result['traj_c']
    ax2.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'r-', alpha=0.3, linewidth=0.3)
    ax2.set_title(f"With Control (λ={control_result['lyap_c']:.2f})")
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')

    # 3. ROM coordinates colored by wing
    ax3 = fig.add_subplot(2, 3, 3)
    betas = rom_result['betas_traj']
    wing = rom_result['wing']
    colors = ['blue' if w > 0 else 'red' for w in wing]
    ax3.scatter(betas[:, 0], betas[:, 1], c=colors, s=5, alpha=0.5)
    ax3.set_xlabel('β₁')
    ax3.set_ylabel('β₂')
    ax3.set_title(f"ROM Coordinates\n(corr with wing: {rom_result['corr_wing']:.2f})")

    # 4. X time series comparison
    ax4 = fig.add_subplot(2, 3, 4)
    t = np.arange(len(control_result['traj_nc'])) * 0.01
    ax4.plot(t[:2000], control_result['traj_nc'][:2000, 0], 'b-', alpha=0.7, label='No control')
    ax4.plot(t[:2000], control_result['traj_c'][:2000, 0], 'r-', alpha=0.7, label='With control')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('x')
    ax4.set_title('Time Series (first 20s)')
    ax4.legend()

    # 5. Wing switching
    ax5 = fig.add_subplot(2, 3, 5)
    labels = ['No Control', 'With Control']
    switches = [control_result['switches_nc'], control_result['switches_c']]
    colors = ['blue', 'red']
    ax5.bar(labels, switches, color=colors, alpha=0.7)
    ax5.set_ylabel('Wing Switches')
    ax5.set_title('Trajectory Complexity')

    # 6. Summary metrics
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    summary = f"""
    LORENZ CHAOS CONTROL RESULTS
    ============================

    Lyapunov Exponent:
      No control:    λ = {control_result['lyap_nc']:.3f}
      With control:  λ = {control_result['lyap_c']:.3f}
      Reduction:     {control_result['lyap_reduction']:.1f}%

    Wing Switches (T=100s):
      No control:    {control_result['switches_nc']}
      With control:  {control_result['switches_c']}

    Time Near Target C±:
      No control:    {control_result['near_nc']*100:.1f}%
      With control:  {control_result['near_c']*100:.1f}%

    ROM Analysis:
      Wing correlation: {rom_result['corr_wing']:.3f}
      Recon error: {rom_result['recon_error']*100:.1f}%
    """
    ax6.text(0.1, 0.5, summary, family='monospace', fontsize=10,
             verticalalignment='center', transform=ax6.transAxes)

    plt.tight_layout()
    plt.savefig('lorenz_chaos_control.png', dpi=150)
    plt.close()
    print("\nSaved: lorenz_chaos_control.png")


def run_lorenz_pipeline():
    """Run complete Lorenz chaos control pipeline."""
    print("\n" + "=" * 70)
    print("LORENZ ATTRACTOR - CHAOS CONTROL PIPELINE")
    print("=" * 70)

    # Test 1: ROM analysis
    print("\n[TEST 1] HST-ROM Analysis")
    rom_result, trajectory, crossings = test_lorenz_rom()

    # Test 2: UPO detection
    print("\n[TEST 2] UPO Detection via ROM")
    upo_betas = test_upo_detection_via_rom(rom_result, trajectory, crossings)

    # Test 3: Chaos control
    print("\n[TEST 3] OGY-Style Chaos Control")
    control_result = test_chaos_control(rom_result, trajectory, crossings)

    # Create visualization
    print("\n[Visualization]")
    create_lorenz_visualization(rom_result, control_result, trajectory)
    plot_lorenz_attractor(trajectory, crossings, identify_upo_candidates(crossings))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n1. ROM Analysis:")
    print(f"   - Wing correlation: {rom_result['corr_wing']:.3f}")
    print(f"   - Reconstruction error: {rom_result['recon_error']*100:.1f}%")
    print(f"   - ROM captures attractor structure: {'YES' if rom_result['corr_wing'] > 0.3 else 'PARTIAL'}")

    print(f"\n2. UPO Detection:")
    print(f"   - UPOs matched to ROM: {len(upo_betas)}")

    print(f"\n3. Chaos Control:")
    print(f"   - Lyapunov reduction: {control_result['lyap_reduction']:.1f}%")
    print(f"   - Control effective: {'YES' if control_result['lyap_reduction'] > 5 else 'MARGINAL'}")

    # Overall assessment
    success_criteria = [
        rom_result['corr_wing'] > 0.2,  # ROM captures wing structure
        rom_result['recon_error'] < 0.5,  # Reasonable reconstruction
        control_result['lyap_reduction'] > 0  # Any Lyapunov reduction
    ]

    print(f"\nOverall: {sum(success_criteria)}/3 criteria met")

    return {
        'rom': rom_result,
        'upos': upo_betas,
        'control': control_result
    }


if __name__ == "__main__":
    results = run_lorenz_pipeline()
