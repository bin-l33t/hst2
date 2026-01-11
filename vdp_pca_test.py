"""
Van der Pol Oscillator + HST + PCA Test

Tests Glinsky's claim that HST + PCA extracts slow manifold coordinates
from a nonlinear dynamical system.

The Van der Pol oscillator has a slow manifold (limit cycle) with natural
coordinates (ρ, φ) representing amplitude and phase. If HST + PCA correctly
extracts slow manifold structure, PCA components should correlate with these.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from hst import hst_forward, measure_convergence_rate


# Simple PCA implementation using numpy (to avoid sklearn version issues)
class SimplePCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit_transform(self, X):
        # Center data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Explained variance
        n_samples = X.shape[0]
        self.explained_variance_ = (S ** 2) / (n_samples - 1)
        total_var = np.sum(self.explained_variance_)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var

        # Components
        if self.n_components is not None:
            self.components_ = Vt[:self.n_components]
            self.explained_variance_ = self.explained_variance_[:self.n_components]
            self.explained_variance_ratio_ = self.explained_variance_ratio_[:self.n_components]
            return U[:, :self.n_components] * S[:self.n_components]
        else:
            self.components_ = Vt
            return U * S


def standardize(X):
    """Standardize features to zero mean and unit variance."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std < 1e-10] = 1.0  # Avoid division by zero
    return (X - mean) / std

# ============================================================
# VAN DER POL OSCILLATOR
# ============================================================

def van_der_pol(t, y, eps=0.1):
    """
    Van der Pol oscillator: x'' - eps*(1-x^2)*x' + x = 0

    State: y = [x, v] where v = x'

    Parameters:
    - eps: nonlinearity parameter (small eps = weakly nonlinear)
    """
    x, v = y
    dxdt = v
    dvdt = eps * (1 - x**2) * v - x
    return [dxdt, dvdt]


def generate_vdp_trajectory(eps=0.1, T=100, N=4096, x0=0.1, v0=0):
    """
    Generate Van der Pol oscillator trajectory.

    Returns:
    - t: time array
    - z: complex signal z = x + i*v
    - rho: amplitude sqrt(x^2 + v^2)
    - phi: phase arctan2(v, x)
    """
    t_span = (0, T)
    t_eval = np.linspace(0, T, N)

    sol = solve_ivp(
        van_der_pol,
        t_span,
        [x0, v0],
        t_eval=t_eval,
        args=(eps,),
        method='RK45',
        rtol=1e-8
    )

    x = sol.y[0]
    v = sol.y[1]

    # Complex signal
    z = x + 1j * v

    # Slow manifold coordinates
    rho = np.sqrt(x**2 + v**2)  # amplitude
    phi = np.arctan2(v, x)       # phase

    return sol.t, z, rho, phi, x, v


def compute_instantaneous_phase(z):
    """Compute instantaneous phase using Hilbert transform."""
    from scipy.signal import hilbert
    analytic = hilbert(z.real)
    return np.angle(analytic)


# ============================================================
# HST + PCA ANALYSIS
# ============================================================

def extract_scattering_features(z, J=4, verbose=False):
    """
    Apply HST and extract features for PCA.

    Returns flattened scattering coefficients suitable for PCA.
    """
    S_coeffs, u_final = hst_forward(z, J=J, verbose=verbose)

    # Collect features at multiple scales
    features = []

    for j, S_j in enumerate(S_coeffs):
        # Use both real and imaginary parts (phase preservation!)
        features.append(S_j.real)
        features.append(S_j.imag)

        if verbose:
            print(f"Layer {j}: {len(S_j)} samples, "
                  f"|Re|_rms={np.sqrt(np.mean(S_j.real**2)):.4f}, "
                  f"|Im|_rms={np.sqrt(np.mean(S_j.imag**2)):.4f}")

    # Also include final layer
    features.append(u_final.real)
    features.append(u_final.imag)

    return features, S_coeffs, u_final


def create_feature_matrix(features, max_len=None):
    """
    Create feature matrix for PCA from multi-scale scattering coefficients.

    Each time point gets features from all scales (interpolated to same length).
    """
    if max_len is None:
        max_len = max(len(f) for f in features)

    # Interpolate all features to same length
    X = np.zeros((max_len, len(features)))

    for i, f in enumerate(features):
        # Linear interpolation to max_len
        x_orig = np.linspace(0, 1, len(f))
        x_new = np.linspace(0, 1, max_len)
        X[:, i] = np.interp(x_new, x_orig, f)

    return X


def sliding_window_features(z, window_size=256, step=64, J=3):
    """
    Create feature matrix using sliding windows.

    For each window position:
    1. Extract window of signal
    2. Apply HST
    3. Flatten scattering coefficients into feature vector

    Returns:
    - X: (n_windows, n_features) matrix
    - centers: center time index of each window
    """
    n_samples = len(z)
    n_windows = (n_samples - window_size) // step + 1

    feature_list = []
    centers = []

    for i in range(n_windows):
        start = i * step
        end = start + window_size
        center = (start + end) // 2

        window = z[start:end]

        # Apply HST to window
        S_coeffs, u_final = hst_forward(window, J=J, verbose=False)

        # Extract statistics from each layer as features
        features = []
        for S_j in S_coeffs:
            features.extend([
                np.mean(S_j.real),
                np.std(S_j.real),
                np.mean(S_j.imag),
                np.std(S_j.imag),
                np.sqrt(np.mean(np.abs(S_j)**2)),  # RMS magnitude
            ])

        # Final layer features
        features.extend([
            np.mean(u_final.real),
            np.std(u_final.real),
            np.mean(u_final.imag),
            np.std(u_final.imag),
        ])

        feature_list.append(features)
        centers.append(center)

    return np.array(feature_list), np.array(centers)


def analyze_pca_correlation(X, rho, phi, centers, n_components=5):
    """
    Apply PCA and analyze correlation with slow manifold coordinates.
    """
    # Standardize features
    X_scaled = standardize(X)

    # PCA
    pca = SimplePCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Get rho and phi at window centers
    rho_centers = rho[centers]
    phi_centers = phi[centers]

    # Compute correlations with slow manifold coordinates
    correlations = {
        'rho': [],
        'phi': [],
        'cos_phi': [],
        'sin_phi': [],
    }

    for i in range(n_components):
        pc = X_pca[:, i]
        correlations['rho'].append(np.corrcoef(pc, rho_centers)[0, 1])
        correlations['phi'].append(np.corrcoef(pc, phi_centers)[0, 1])
        correlations['cos_phi'].append(np.corrcoef(pc, np.cos(phi_centers))[0, 1])
        correlations['sin_phi'].append(np.corrcoef(pc, np.sin(phi_centers))[0, 1])

    return pca, X_pca, correlations, rho_centers, phi_centers


# ============================================================
# VISUALIZATION
# ============================================================

def plot_vdp_analysis(t, z, rho, phi, x, v, save_path='vdp_trajectory.png'):
    """Plot Van der Pol trajectory and slow manifold coordinates."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Van der Pol Oscillator Analysis', fontsize=14)

    # Phase portrait
    ax = axes[0, 0]
    ax.plot(x, v, 'b-', alpha=0.7, linewidth=0.5)
    ax.plot(x[0], v[0], 'go', markersize=10, label='Start')
    ax.plot(x[-1], v[-1], 'ro', markersize=10, label='End')
    ax.set_xlabel('x')
    ax.set_ylabel('v = dx/dt')
    ax.set_title('Phase Portrait')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Time series
    ax = axes[0, 1]
    ax.plot(t, x, 'b-', label='x(t)', alpha=0.8)
    ax.plot(t, v, 'r-', label='v(t)', alpha=0.8)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Time Series')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Amplitude (slow variable)
    ax = axes[1, 0]
    ax.plot(t, rho, 'g-', linewidth=0.5)
    ax.axhline(y=2.0, color='k', linestyle='--', alpha=0.5, label='Limit cycle ρ≈2')
    ax.set_xlabel('Time')
    ax.set_ylabel('ρ = √(x² + v²)')
    ax.set_title('Amplitude (Slow Manifold Coordinate)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Phase (fast variable)
    ax = axes[1, 1]
    ax.plot(t, phi, 'purple', linewidth=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('φ = arctan2(v, x)')
    ax.set_title('Phase (Fast Variable)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_pca_results(pca, X_pca, correlations, rho_centers, phi_centers,
                     save_path='vdp_pca_results.png'):
    """Plot PCA results and correlations with slow manifold."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('HST + PCA Analysis of Van der Pol Oscillator', fontsize=14)

    # Explained variance
    ax = axes[0, 0]
    ax.bar(range(1, len(pca.explained_variance_ratio_)+1),
           pca.explained_variance_ratio_ * 100)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('PCA Explained Variance')
    ax.grid(True, alpha=0.3)

    # Cumulative variance
    ax = axes[0, 1]
    cum_var = np.cumsum(pca.explained_variance_ratio_) * 100
    ax.plot(range(1, len(cum_var)+1), cum_var, 'bo-')
    ax.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='90%')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Variance (%)')
    ax.set_title('Cumulative Explained Variance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Correlations heatmap
    ax = axes[0, 2]
    corr_matrix = np.array([
        correlations['rho'],
        correlations['cos_phi'],
        correlations['sin_phi'],
    ])
    im = ax.imshow(np.abs(corr_matrix), cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(correlations['rho'])))
    ax.set_xticklabels([f'PC{i+1}' for i in range(len(correlations['rho']))])
    ax.set_yticks(range(3))
    ax.set_yticklabels(['ρ', 'cos(φ)', 'sin(φ)'])
    ax.set_title('|Correlation| with Slow Manifold')
    plt.colorbar(im, ax=ax)

    # Add correlation values as text
    for i in range(3):
        for j in range(len(correlations['rho'])):
            val = corr_matrix[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   color='white' if abs(val) > 0.5 else 'black')

    # PC1 vs rho
    ax = axes[1, 0]
    ax.scatter(X_pca[:, 0], rho_centers, alpha=0.5, s=10)
    ax.set_xlabel('PC1')
    ax.set_ylabel('ρ (amplitude)')
    r = correlations['rho'][0]
    ax.set_title(f'PC1 vs Amplitude (r = {r:.3f})')
    ax.grid(True, alpha=0.3)

    # PC1 vs PC2 colored by rho
    ax = axes[1, 1]
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=rho_centers, cmap='viridis',
                    alpha=0.6, s=10)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PC1 vs PC2 (colored by ρ)')
    plt.colorbar(sc, ax=ax, label='ρ')
    ax.grid(True, alpha=0.3)

    # PC1 vs PC2 colored by phase
    ax = axes[1, 2]
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=phi_centers, cmap='hsv',
                    alpha=0.6, s=10)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PC1 vs PC2 (colored by φ)')
    plt.colorbar(sc, ax=ax, label='φ')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_scattering_coefficients(S_coeffs, save_path='vdp_scattering.png'):
    """Plot scattering coefficients at each layer."""
    J = len(S_coeffs)
    fig, axes = plt.subplots(J, 2, figsize=(12, 3*J))
    fig.suptitle('HST Scattering Coefficients', fontsize=14)

    for j, S_j in enumerate(S_coeffs):
        # Real part
        ax = axes[j, 0] if J > 1 else axes[0]
        ax.plot(S_j.real, 'b-', alpha=0.7)
        ax.set_ylabel(f'Re(S_{j})')
        ax.set_title(f'Layer {j} Real (N={len(S_j)})')
        ax.grid(True, alpha=0.3)

        # Imaginary part
        ax = axes[j, 1] if J > 1 else axes[1]
        ax.plot(S_j.imag, 'r-', alpha=0.7)
        ax.set_ylabel(f'Im(S_{j})')
        ax.set_title(f'Layer {j} Imag (N={len(S_j)})')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# MAIN TEST
# ============================================================

def run_vdp_pca_test(eps=0.1, T=100, N=4096, J=4, window_size=256, step=32):
    """
    Complete Van der Pol + HST + PCA test.

    Parameters:
    - eps: VdP nonlinearity parameter
    - T: total simulation time
    - N: number of samples
    - J: HST layers
    - window_size: sliding window size for feature extraction
    - step: sliding window step
    """
    print("=" * 70)
    print("VAN DER POL + HST + PCA TEST")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  VdP epsilon: {eps}")
    print(f"  Simulation time: {T}")
    print(f"  Samples: {N}")
    print(f"  HST layers: {J}")
    print(f"  Window size: {window_size}")
    print(f"  Step size: {step}")

    # Generate trajectory
    print("\n[1] Generating Van der Pol trajectory...")
    t, z, rho, phi, x, v = generate_vdp_trajectory(eps=eps, T=T, N=N)
    print(f"    Trajectory length: {len(z)}")
    print(f"    Final amplitude (should be ~2): {rho[-1]:.4f}")
    print(f"    Mean amplitude (transient): {np.mean(rho):.4f}")

    # Plot trajectory
    plot_vdp_analysis(t, z, rho, phi, x, v,
                      save_path='/home/ubuntu/rectifier/vdp_trajectory.png')

    # Measure HST convergence on this signal
    print("\n[2] Measuring HST convergence rate...")
    conv_result = measure_convergence_rate(z, J=J+1, verbose=False)
    print(f"    Im(u) norms: {[f'{x:.4f}' for x in conv_result['u_im_norms']]}")
    if conv_result['lambda_empirical']:
        print(f"    Empirical λ: {conv_result['lambda_empirical']:.4f}")

    # Plot scattering coefficients
    features, S_coeffs, u_final = extract_scattering_features(z, J=J, verbose=True)
    plot_scattering_coefficients(S_coeffs,
                                  save_path='/home/ubuntu/rectifier/vdp_scattering.png')

    # Extract sliding window features
    print(f"\n[3] Extracting sliding window HST features...")
    X, centers = sliding_window_features(z, window_size=window_size, step=step, J=J)
    print(f"    Feature matrix shape: {X.shape}")
    print(f"    (n_windows x n_features)")

    # PCA analysis
    print("\n[4] Applying PCA...")
    n_components = min(10, X.shape[1])
    pca, X_pca, correlations, rho_centers, phi_centers = analyze_pca_correlation(
        X, rho, phi, centers, n_components=n_components
    )

    print(f"\n    Explained variance by component:")
    for i, var in enumerate(pca.explained_variance_ratio_[:5]):
        print(f"      PC{i+1}: {var*100:.1f}%")
    print(f"    Total (first 5): {sum(pca.explained_variance_ratio_[:5])*100:.1f}%")

    # Report correlations
    print("\n[5] Correlations with slow manifold coordinates:")
    print("\n    PC vs ρ (amplitude):")
    for i, r in enumerate(correlations['rho'][:5]):
        stars = '*' * min(3, int(abs(r) * 3 / 0.3))
        print(f"      PC{i+1}: r = {r:+.3f} {stars}")

    print("\n    PC vs cos(φ):")
    for i, r in enumerate(correlations['cos_phi'][:5]):
        stars = '*' * min(3, int(abs(r) * 3 / 0.3))
        print(f"      PC{i+1}: r = {r:+.3f} {stars}")

    print("\n    PC vs sin(φ):")
    for i, r in enumerate(correlations['sin_phi'][:5]):
        stars = '*' * min(3, int(abs(r) * 3 / 0.3))
        print(f"      PC{i+1}: r = {r:+.3f} {stars}")

    # Find best correlating components
    print("\n[6] Best correlating components:")

    rho_best = np.argmax(np.abs(correlations['rho']))
    print(f"    Best for ρ: PC{rho_best+1} (r = {correlations['rho'][rho_best]:.3f})")

    cos_best = np.argmax(np.abs(correlations['cos_phi']))
    print(f"    Best for cos(φ): PC{cos_best+1} (r = {correlations['cos_phi'][cos_best]:.3f})")

    sin_best = np.argmax(np.abs(correlations['sin_phi']))
    print(f"    Best for sin(φ): PC{sin_best+1} (r = {correlations['sin_phi'][sin_best]:.3f})")

    # Plot results
    plot_pca_results(pca, X_pca, correlations, rho_centers, phi_centers,
                     save_path='/home/ubuntu/rectifier/vdp_pca_results.png')

    # Summary assessment
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    max_rho_corr = max(abs(r) for r in correlations['rho'])
    max_cos_corr = max(abs(r) for r in correlations['cos_phi'])
    max_sin_corr = max(abs(r) for r in correlations['sin_phi'])

    print(f"\nMaximum correlations achieved:")
    print(f"  |r(PC, ρ)|     = {max_rho_corr:.3f}")
    print(f"  |r(PC, cos φ)| = {max_cos_corr:.3f}")
    print(f"  |r(PC, sin φ)| = {max_sin_corr:.3f}")

    # Glinsky's claim: HST + PCA extracts slow manifold coordinates
    threshold = 0.5
    rho_extracted = max_rho_corr > threshold
    phase_extracted = max(max_cos_corr, max_sin_corr) > threshold

    print(f"\nGlinsky claim assessment (threshold = {threshold}):")
    print(f"  Amplitude (ρ) extracted: {'YES' if rho_extracted else 'NO'} (r = {max_rho_corr:.3f})")
    print(f"  Phase (φ) extracted: {'YES' if phase_extracted else 'NO'} (r = {max(max_cos_corr, max_sin_corr):.3f})")

    if rho_extracted and phase_extracted:
        print("\n  ✓ CLAIM SUPPORTED: HST + PCA extracts slow manifold coordinates!")
    elif rho_extracted or phase_extracted:
        print("\n  ~ PARTIAL: One coordinate extracted, further investigation needed")
    else:
        print("\n  ✗ CLAIM NOT SUPPORTED with current parameters")

    return {
        't': t,
        'z': z,
        'rho': rho,
        'phi': phi,
        'X': X,
        'centers': centers,
        'pca': pca,
        'X_pca': X_pca,
        'correlations': correlations,
        'S_coeffs': S_coeffs,
    }


if __name__ == "__main__":
    results = run_vdp_pca_test()
