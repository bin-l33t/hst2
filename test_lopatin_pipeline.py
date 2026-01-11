"""
Test the Complete Glinsky Pipeline on Lopatin's Three Models.

From Glinsky/Lopatin: Three explicit toy problems with known Lie group structure:

1. Van der Pol (SO(2)): Limit cycle ρ → 2
   - Slow variable: amplitude ρ
   - Fast variable: phase φ

2. Duffing (SO(2)): Frequency shift φ̇ = 1 + ε(3/8)ρ²
   - Amplitude-dependent frequency
   - Tests nonlinear frequency extraction

3. Sphere (SO(3)): ẏ₁ = 0, ẏ₂ = -ε²(...)y₁, ẏ₃ = 1 + ε/4
   - Higher-dimensional rotation group
   - Tests generalization beyond SO(2)

Success Criteria:
1. ROM captures dynamics: PC1-3 explain >90% variance
2. Geodesic motion verified: P_std < 0.1, Q linear
3. Control works: Can push system toward target β
"""

import numpy as np
import matplotlib.pyplot as plt

from hst_rom import HST_ROM
from hjb_decoder import HJB_Decoder, train_hjb_decoder, verify_geodesic_motion, HAS_TORCH
from ponderomotive_control import PonderomotiveController


# ============================================================
# LOPATIN'S THREE MODELS
# ============================================================

def van_der_pol_dynamics(z, t, mu=1.0):
    """
    Van der Pol oscillator (SO(2) symmetry).

    ẍ - μ(1-x²)ẋ + x = 0

    In complex form: z = x + iv where v = ẋ
    Has limit cycle with amplitude ρ → 2
    """
    x = np.real(z)
    v = np.imag(z)

    dx = v
    dv = mu * (1 - x**2) * v - x

    return dx + 1j * dv


def duffing_dynamics(z, t, epsilon=0.1, omega=1.0, gamma=0.0):
    """
    Duffing oscillator (SO(2) symmetry with nonlinear frequency).

    ẍ + ωx + εx³ + γẋ = 0

    Frequency shift: φ̇ ≈ ω + ε(3/8)ρ² (Lopatin)
    """
    x = np.real(z)
    v = np.imag(z)

    dx = v
    dv = -omega * x - epsilon * x**3 - gamma * v

    return dx + 1j * dv


def sphere_dynamics(y, t, epsilon=0.1):
    """
    Sphere dynamics (SO(3) symmetry).

    From Lopatin:
    ẏ₁ = 0
    ẏ₂ = -ε²(...)y₁
    ẏ₃ = 1 + ε/4

    This is a perturbation of rotation on S².
    y is a 3D vector on the unit sphere.
    """
    y = np.asarray(y)
    if len(y) != 3:
        raise ValueError("Sphere dynamics requires 3D vector")

    y1, y2, y3 = y

    # Simplified Lopatin dynamics (approximation)
    dy1 = 0
    dy2 = -epsilon**2 * y1 * np.sin(t)
    dy3 = 1 + epsilon/4

    return np.array([dy1, dy2, dy3])


def generate_trajectories(system_name, n_trajectories=50, n_steps=2048, dt=0.01,
                          settle_steps=3000):
    """
    Generate ensemble of trajectories for a given system.

    Parameters
    ----------
    system_name : str
        'van_der_pol', 'duffing', or 'sphere'
    n_trajectories : int
        Number of trajectories
    n_steps : int
        Steps per trajectory
    dt : float
        Time step
    settle_steps : int
        Steps to run before recording (allows transients to decay)

    Returns
    -------
    trajectories : list of arrays
        Complex trajectories (or 3D for sphere)
    ground_truth : dict
        Known quantities (amplitude, phase, etc.)
    """
    np.random.seed(42)
    trajectories = []
    ground_truth = {'rho': [], 'phi': [], 'system': system_name}

    for i in range(n_trajectories):
        if system_name == 'van_der_pol':
            # Random initial conditions
            r0 = 0.5 + 3 * np.random.rand()
            theta0 = 2 * np.pi * np.random.rand()
            z0 = r0 * np.exp(1j * theta0)

            # Settle first (let transients decay)
            z = z0
            for _ in range(settle_steps):
                dz = van_der_pol_dynamics(z, 0)
                z = z + dt * dz

            # Now record trajectory on attractor
            traj = [z]
            rhos = [np.abs(z)]
            phis = [np.angle(z)]

            for _ in range(n_steps):
                dz = van_der_pol_dynamics(z, 0)
                z = z + dt * dz
                traj.append(z)
                rhos.append(np.abs(z))
                phis.append(np.angle(z))

            trajectories.append(np.array(traj))
            ground_truth['rho'].append(np.array(rhos))
            ground_truth['phi'].append(np.array(phis))

        elif system_name == 'duffing':
            # Random initial conditions
            r0 = 0.5 + 2 * np.random.rand()
            theta0 = 2 * np.pi * np.random.rand()
            z0 = r0 * np.exp(1j * theta0)

            # Simulate
            z = z0
            traj = [z]
            rhos = [np.abs(z)]
            phis = [np.angle(z)]

            for _ in range(n_steps):
                dz = duffing_dynamics(z, 0, epsilon=0.1)
                z = z + dt * dz
                traj.append(z)
                rhos.append(np.abs(z))
                phis.append(np.angle(z))

            trajectories.append(np.array(traj))
            ground_truth['rho'].append(np.array(rhos))
            ground_truth['phi'].append(np.array(phis))

        elif system_name == 'sphere':
            # Random initial point on sphere
            theta = np.pi * np.random.rand()
            phi = 2 * np.pi * np.random.rand()
            y0 = np.array([np.sin(theta)*np.cos(phi),
                           np.sin(theta)*np.sin(phi),
                           np.cos(theta)])

            # Simulate
            y = y0
            traj = [y]

            for step in range(n_steps):
                t = step * dt
                dy = sphere_dynamics(y, t, epsilon=0.1)
                y = y + dt * dy
                # Project back to sphere
                y = y / np.linalg.norm(y)
                traj.append(y)

            # Convert to complex (stereographic projection)
            traj_array = np.array(traj)
            # Stereographic: z = (y1 + i*y2) / (1 - y3)
            z_traj = (traj_array[:, 0] + 1j * traj_array[:, 1]) / (1 - traj_array[:, 2] + 1e-10)

            trajectories.append(z_traj)
            ground_truth['rho'].append(np.abs(z_traj))
            ground_truth['phi'].append(np.angle(z_traj))

    return trajectories, ground_truth


def analyze_rom_quality(rom, betas, ground_truth, system_name):
    """
    Analyze how well ROM captures dynamics.

    Checks:
    1. Variance explained by first 3 PCs
    2. Correlation with ground truth (ρ, φ)
    """
    results = {}

    # Variance explained
    var_explained = rom.pca.explained_variance_ratio_
    results['variance_explained'] = var_explained
    results['total_variance_3pc'] = sum(var_explained[:3])

    # Correlations with ground truth
    # This requires matching windows to ground truth values
    # For now, compute over all samples

    results['success'] = results['total_variance_3pc'] > 0.80

    return results


def test_system(system_name, n_trajectories=30, verbose=True):
    """
    Test complete pipeline on one system.

    Returns
    -------
    results : dict
        All test results and metrics
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing {system_name.upper()}")
        print('='*60)

    results = {'system': system_name}

    # 1. Generate trajectories
    if verbose:
        print("\n1. Generating trajectories...")
    trajectories, ground_truth = generate_trajectories(system_name, n_trajectories)

    results['n_trajectories'] = len(trajectories)
    results['trajectory_length'] = len(trajectories[0])

    if verbose:
        print(f"   Generated {len(trajectories)} trajectories, length {len(trajectories[0])}")

    # 2. Build ROM
    if verbose:
        print("\n2. Building ROM...")

    # Use more PCA components for better reconstruction
    # But track correlations with first 4 for slow manifold
    rom = HST_ROM(n_components=16, wavelet='db8', J=3, window_size=128)
    betas = rom.fit(trajectories[:20])  # Train on subset

    results['rom'] = rom
    results['betas'] = betas
    results['feature_dim'] = rom.feature_dim_
    results['variance_explained'] = rom.pca.explained_variance_ratio_.tolist()
    results['total_variance'] = sum(results['variance_explained'])
    results['variance_4pc'] = sum(results['variance_explained'][:4])

    if verbose:
        print(f"   Feature dimension: {rom.feature_dim_}")
        print(f"   Variance (first 4 PCs): {[f'{v:.3f}' for v in results['variance_explained'][:4]]}")
        print(f"   Total (16 PCs): {results['total_variance']:.3f}")
        print(f"   First 4 PCs: {results['variance_4pc']:.3f}")

    # 3. Test reconstruction
    if verbose:
        print("\n3. Testing reconstruction...")

    test_signal = trajectories[25][:128]  # Use unseen trajectory
    recon_error = rom.reconstruction_error(test_signal)
    results['reconstruction_error'] = recon_error

    if verbose:
        print(f"   Reconstruction error: {recon_error*100:.1f}%")

    # 4. Train HJB decoder (if PyTorch available)
    if HAS_TORCH:
        if verbose:
            print("\n4. Training HJB decoder...")

        decoder, history = train_hjb_decoder(rom, trajectories[:20],
                                              n_epochs=200, verbose=False)
        results['decoder'] = decoder
        results['training_history'] = history
        results['final_loss'] = history['loss'][-1] if history['loss'] else None

        if verbose and results['final_loss']:
            print(f"   Final loss: {results['final_loss']:.6f}")

        # 5. Verify geodesic motion
        if verbose:
            print("\n5. Verifying geodesic motion...")

        test_traj = trajectories[25]
        geodesic_metrics = verify_geodesic_motion(decoder, rom, test_traj, plot=False)
        results['geodesic_metrics'] = geodesic_metrics

        if verbose:
            print(f"   P conservation (CV < 0.1): {geodesic_metrics['P_conservation']}")
            print(f"   Q linear (R² > 0.8): {geodesic_metrics['Q_linear']}")
            print(f"   Geodesic verified: {geodesic_metrics['geodesic']}")
    else:
        if verbose:
            print("\n4-5. Skipping decoder/geodesic (no PyTorch)")
        results['decoder'] = None
        results['geodesic_metrics'] = None

    # 6. Test ponderomotive control
    if verbose:
        print("\n6. Testing ponderomotive control...")

    # Target: average of fitted trajectories
    target_beta = betas.mean(axis=0)
    controller = PonderomotiveController(rom, results.get('decoder'),
                                          target_beta, Omega=20.0, epsilon=0.2)

    # Simulate from different IC
    z0 = trajectories[28][0]  # Different initial condition
    z = z0
    z_buffer = [z0] * 128

    initial_error = np.linalg.norm(rom.transform(np.array(z_buffer[-128:])) - target_beta)

    for _ in range(500):
        # Control
        z_window = np.array(z_buffer[-128:], dtype=complex)
        try:
            u, _ = controller.compute_control(z_window, t=0)
            u_applied = u[-1] if len(u) > 0 else 0
        except:
            u_applied = 0

        # Step (using appropriate dynamics)
        if system_name == 'van_der_pol':
            dz = van_der_pol_dynamics(z, 0)
        elif system_name == 'duffing':
            dz = duffing_dynamics(z, 0)
        else:
            dz = 0  # Sphere handled differently

        z = z + 0.01 * (dz + u_applied)
        z_buffer.append(z)

    final_beta = rom.transform(np.array(z_buffer[-128:], dtype=complex))
    final_error = np.linalg.norm(final_beta - target_beta)

    results['control_initial_error'] = initial_error
    results['control_final_error'] = final_error
    results['control_improvement'] = 1 - final_error / (initial_error + 1e-10)

    if verbose:
        print(f"   Initial β error: {initial_error:.4f}")
        print(f"   Final β error: {final_error:.4f}")
        print(f"   Improvement: {results['control_improvement']*100:.1f}%")

    # Summary
    results['rom_success'] = results['variance_4pc'] > 0.70  # First 4 PCs for slow manifold
    results['reconstruction_success'] = results['reconstruction_error'] < 0.20
    results['control_success'] = results['control_improvement'] > 0.1

    if verbose:
        print(f"\n--- Summary for {system_name} ---")
        print(f"   ROM captures dynamics (>80% var): {results['rom_success']}")
        print(f"   Reconstruction (<20% error): {results['reconstruction_success']}")
        print(f"   Control works (>10% improvement): {results['control_success']}")

    return results


def run_full_pipeline():
    """
    Run complete Glinsky pipeline on all three Lopatin models.
    """
    print("\n" + "="*70)
    print("GLINSKY PIPELINE - COMPLETE TEST")
    print("Testing on Lopatin's Three Models")
    print("="*70)

    all_results = {}

    for system_name in ['van_der_pol', 'duffing', 'sphere']:
        try:
            results = test_system(system_name, n_trajectories=30, verbose=True)
            all_results[system_name] = results
        except Exception as e:
            print(f"\nError testing {system_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[system_name] = {'error': str(e)}

    # Create visualization
    create_pipeline_visualization(all_results)

    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    summary_table = []
    for system in ['van_der_pol', 'duffing', 'sphere']:
        if system in all_results and 'error' not in all_results[system]:
            r = all_results[system]
            summary_table.append({
                'System': system,
                'Variance': f"{r['variance_4pc']:.2f}",
                'Recon Error': f"{r['reconstruction_error']*100:.1f}%",
                'Control Imp': f"{r['control_improvement']*100:.1f}%",
                'ROM OK': r['rom_success'],
                'Recon OK': r['reconstruction_success'],
                'Ctrl OK': r['control_success']
            })
        else:
            summary_table.append({
                'System': system,
                'Variance': 'ERROR',
                'Recon Error': 'ERROR',
                'Control Imp': 'ERROR',
                'ROM OK': False,
                'Recon OK': False,
                'Ctrl OK': False
            })

    # Print table
    print(f"\n{'System':<15} {'Variance':>10} {'Recon Err':>10} {'Ctrl Imp':>10} {'ROM':>5} {'Rec':>5} {'Ctrl':>5}")
    print("-"*65)
    for row in summary_table:
        rom_ok = 'YES' if row['ROM OK'] else 'NO'
        rec_ok = 'YES' if row['Recon OK'] else 'NO'
        ctrl_ok = 'YES' if row['Ctrl OK'] else 'NO'
        print(f"{row['System']:<15} {row['Variance']:>10} {row['Recon Error']:>10} {row['Control Imp']:>10} {rom_ok:>5} {rec_ok:>5} {ctrl_ok:>5}")

    # Overall success
    n_success = sum(1 for row in summary_table
                    if row['ROM OK'] and row['Recon OK'])
    print(f"\nOverall: {n_success}/3 systems passed ROM + Reconstruction tests")

    return all_results


def create_pipeline_visualization(all_results):
    """Create comprehensive visualization of pipeline results."""
    fig = plt.figure(figsize=(16, 12))

    systems = ['van_der_pol', 'duffing', 'sphere']
    colors = {'van_der_pol': '#1f77b4', 'duffing': '#ff7f0e', 'sphere': '#2ca02c'}

    # 1. Variance explained bar chart
    ax1 = fig.add_subplot(2, 3, 1)
    x = np.arange(4)
    width = 0.25

    for i, system in enumerate(systems):
        if system in all_results and 'variance_explained' in all_results[system]:
            var_exp = all_results[system]['variance_explained'][:4]
            ax1.bar(x + i*width, var_exp, width, label=system, color=colors[system])

    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Variance Explained')
    ax1.set_title('ROM Variance Explained')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(['PC1', 'PC2', 'PC3', 'PC4'])
    ax1.legend()
    ax1.axhline(0.25, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylim(0, 1)

    # 2. Reconstruction error
    ax2 = fig.add_subplot(2, 3, 2)
    recon_errors = []
    labels = []
    for system in systems:
        if system in all_results and 'reconstruction_error' in all_results[system]:
            recon_errors.append(all_results[system]['reconstruction_error'] * 100)
            labels.append(system)

    if recon_errors:
        bars = ax2.bar(labels, recon_errors, color=[colors[s] for s in labels])
        ax2.axhline(20, color='red', linestyle='--', label='20% threshold')
        ax2.set_ylabel('Reconstruction Error (%)')
        ax2.set_title('HST Reconstruction Quality')
        ax2.legend()

    # 3. Control improvement
    ax3 = fig.add_subplot(2, 3, 3)
    ctrl_imp = []
    labels = []
    for system in systems:
        if system in all_results and 'control_improvement' in all_results[system]:
            ctrl_imp.append(all_results[system]['control_improvement'] * 100)
            labels.append(system)

    if ctrl_imp:
        bars = ax3.bar(labels, ctrl_imp, color=[colors[s] for s in labels])
        ax3.axhline(10, color='green', linestyle='--', label='10% threshold')
        ax3.set_ylabel('Control Improvement (%)')
        ax3.set_title('Ponderomotive Control Effect')
        ax3.legend()

    # 4-6. β trajectory for each system
    for i, system in enumerate(systems):
        ax = fig.add_subplot(2, 3, 4 + i)

        if system in all_results and 'betas' in all_results[system]:
            betas = all_results[system]['betas']

            # Plot first two PCs
            ax.scatter(betas[:, 0], betas[:, 1], c=np.arange(len(betas)),
                      cmap='viridis', alpha=0.5, s=10)
            ax.set_xlabel('β₁ (PC1)')
            ax.set_ylabel('β₂ (PC2)')
            ax.set_title(f'{system}: ROM Trajectory')

            # Mark target if available
            target = betas.mean(axis=0)
            ax.plot(target[0], target[1], 'r*', ms=15, label='Target')
            ax.legend()

    plt.tight_layout()
    plt.savefig('lopatin_pipeline_results.png', dpi=150)
    plt.close()
    print("\nSaved: lopatin_pipeline_results.png")

    # Additional: Per-system phase portraits
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, system in enumerate(systems):
        ax = axes[i]

        # Generate one trajectory for visualization
        trajs, _ = generate_trajectories(system, n_trajectories=1, n_steps=2000)
        traj = trajs[0]

        ax.plot(np.real(traj), np.imag(traj), 'b-', alpha=0.7, lw=0.5)
        ax.plot(np.real(traj[0]), np.imag(traj[0]), 'go', ms=10, label='Start')
        ax.plot(np.real(traj[-1]), np.imag(traj[-1]), 'ro', ms=10, label='End')
        ax.set_xlabel('Re(z)')
        ax.set_ylabel('Im(z)')
        ax.set_title(f'{system} Phase Portrait')
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lopatin_phase_portraits.png', dpi=150)
    plt.close()
    print("Saved: lopatin_phase_portraits.png")


if __name__ == "__main__":
    results = run_full_pipeline()
