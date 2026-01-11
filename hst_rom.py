"""
HST Reduced Order Model (ROM)

The ROM lives in β_i coordinates (PCA components of HST output).
From Glinsky: β_i are solutions to Renormalization Group Equations.
The dynamics in β space is geodesic: dP/dτ = 0, dQ/dτ = ω(P)

This module provides:
1. HST_ROM class: fit/transform/inverse_transform for ROM coordinates
2. Feature extraction from HST coefficients
3. PCA-based dimensionality reduction
"""

import numpy as np
from hst import hst_forward_pywt, hst_inverse_pywt


class SimplePCA:
    """
    Simple PCA implementation using numpy SVD.
    Avoids sklearn dependency issues.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None

    def fit(self, X):
        """Fit PCA on data matrix X (n_samples, n_features)."""
        X = np.asarray(X)
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # SVD decomposition
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Determine number of components
        n_components = self.n_components
        if n_components is None:
            n_components = min(X.shape)
        n_components = min(n_components, min(X.shape))

        self.components_ = Vt[:n_components]
        self.singular_values_ = s[:n_components]

        # Explained variance
        total_var = np.sum(s**2)
        self.explained_variance_ratio_ = (s[:n_components]**2) / total_var

        return self

    def transform(self, X):
        """Project X onto principal components."""
        X = np.asarray(X)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        """Reconstruct from principal components."""
        X_transformed = np.asarray(X_transformed)
        return X_transformed @ self.components_ + self.mean_


class HST_ROM:
    """
    Reduced Order Model from HST + PCA.

    From Glinsky: β_i are solutions to Renormalization Group Equations.
    The dynamics in β space should be geodesic after proper transformation.

    Usage:
    ------
    rom = HST_ROM(n_components=4, wavelet='db8', J=4)

    # Fit from trajectory windows
    betas = rom.fit(trajectories)

    # Transform new signal to ROM coordinates
    beta = rom.transform(z)

    # Reconstruct from ROM
    z_rec = rom.inverse_transform(beta)
    """

    def __init__(self, n_components=4, wavelet='db8', J=4, window_size=None):
        """
        Initialize HST ROM.

        Parameters
        ----------
        n_components : int
            Number of PCA components (ROM dimension)
        wavelet : str
            PyWavelets wavelet name
        J : int
            Number of HST cascade layers
        window_size : int
            Size of signal windows for feature extraction
            If None, uses full signals
        """
        self.n_components = n_components
        self.wavelet = wavelet
        self.J = J
        self.window_size = window_size

        self.pca = None
        self.mean_ = None
        self.feature_dim_ = None
        self.reference_coeffs_ = None  # For inverse transform structure

    def _flatten_hst(self, coeffs):
        """
        Flatten HST coefficients into feature vector.

        We use magnitudes of detail coefficients at each level,
        plus the final approximation coefficients.
        """
        features = []

        # Detail coefficients at each level
        for cD in coeffs['cD']:
            # Use both real and imaginary parts (HST preserves phase!)
            features.extend(np.real(cD).flatten())
            features.extend(np.imag(cD).flatten())

        # Final approximation
        cA = coeffs['cA_final']
        features.extend(np.real(cA).flatten())
        features.extend(np.imag(cA).flatten())

        return np.array(features)

    def _unflatten_hst(self, features, reference_coeffs):
        """
        Reconstruct HST coefficient structure from feature vector.

        Parameters
        ----------
        features : array
            Flattened feature vector
        reference_coeffs : dict
            Reference HST coeffs for structure (lengths, etc.)

        Returns
        -------
        coeffs : dict
            HST coefficients in proper structure
        """
        coeffs = {
            'cD': [],
            'cA_final': None,
            'lengths': reference_coeffs['lengths'].copy(),
            'wavelet': reference_coeffs['wavelet'],
            'J': reference_coeffs['J']
        }

        idx = 0

        # Reconstruct detail coefficients
        for cD_ref in reference_coeffs['cD']:
            n = len(cD_ref)
            real_part = features[idx:idx+n]
            idx += n
            imag_part = features[idx:idx+n]
            idx += n
            coeffs['cD'].append(real_part + 1j * imag_part)

        # Reconstruct final approximation
        n = len(reference_coeffs['cA_final'])
        real_part = features[idx:idx+n]
        idx += n
        imag_part = features[idx:idx+n]
        idx += n
        coeffs['cA_final'] = real_part + 1j * imag_part

        return coeffs

    def fit(self, trajectories, extract_windows=True, window_stride=None):
        """
        Fit ROM from ensemble of trajectories.

        Parameters
        ----------
        trajectories : list of arrays
            Each trajectory is a complex signal z(t)
        extract_windows : bool
            If True, extract sliding windows from each trajectory
        window_stride : int
            Stride between windows. If None, uses window_size // 2

        Returns
        -------
        betas : array (n_samples, n_components)
            ROM coordinates for all samples
        """
        hst_features = []

        for z in trajectories:
            z = np.asarray(z, dtype=complex)

            if extract_windows and self.window_size is not None:
                # Extract sliding windows
                stride = window_stride or self.window_size // 2
                for start in range(0, len(z) - self.window_size + 1, stride):
                    window = z[start:start + self.window_size]
                    coeffs = hst_forward_pywt(window, J=self.J, wavelet_name=self.wavelet)

                    # Store reference for inverse transform
                    if self.reference_coeffs_ is None:
                        self.reference_coeffs_ = coeffs

                    features = self._flatten_hst(coeffs)
                    hst_features.append(features)
            else:
                # Use full signal
                coeffs = hst_forward_pywt(z, J=self.J, wavelet_name=self.wavelet)

                if self.reference_coeffs_ is None:
                    self.reference_coeffs_ = coeffs

                features = self._flatten_hst(coeffs)
                hst_features.append(features)

        # Stack into matrix
        X = np.vstack(hst_features)
        self.feature_dim_ = X.shape[1]

        # PCA to find β_i coordinates
        self.pca = SimplePCA(n_components=self.n_components)
        self.mean_ = X.mean(axis=0)
        betas = self.pca.fit_transform(X)

        return betas

    def transform(self, z):
        """
        Map signal to ROM coordinates β.

        Parameters
        ----------
        z : array
            Signal (length should match training windows)

        Returns
        -------
        beta : array (n_components,)
            ROM coordinates
        """
        if self.pca is None:
            raise ValueError("ROM not fitted. Call fit() first.")

        z = np.asarray(z, dtype=complex)
        coeffs = hst_forward_pywt(z, J=self.J, wavelet_name=self.wavelet)
        features = self._flatten_hst(coeffs)

        # Project onto PCA
        beta = self.pca.transform(features.reshape(1, -1))[0]

        return beta

    def inverse_transform(self, beta, original_length=None):
        """
        Map ROM coordinates back to signal.

        Parameters
        ----------
        beta : array (n_components,)
            ROM coordinates
        original_length : int
            Target signal length

        Returns
        -------
        z_rec : array
            Reconstructed signal
        """
        if self.pca is None:
            raise ValueError("ROM not fitted. Call fit() first.")

        if self.reference_coeffs_ is None:
            raise ValueError("No reference coefficients. fit() may have failed.")

        # Inverse PCA
        features = self.pca.inverse_transform(beta.reshape(1, -1))[0]

        # Unflatten to HST structure
        coeffs = self._unflatten_hst(features, self.reference_coeffs_)

        # Inverse HST
        z_rec = hst_inverse_pywt(coeffs, original_length=original_length)

        return z_rec

    def transform_trajectory(self, z, window_stride=None):
        """
        Transform full trajectory to sequence of β coordinates.

        Parameters
        ----------
        z : array
            Full trajectory
        window_stride : int
            Stride between windows

        Returns
        -------
        betas : array (n_windows, n_components)
            ROM coordinates for each window
        times : array (n_windows,)
            Center times of each window
        """
        if self.window_size is None:
            raise ValueError("window_size not set. Use transform() for single signals.")

        z = np.asarray(z, dtype=complex)
        stride = window_stride or self.window_size // 2

        betas = []
        times = []

        for start in range(0, len(z) - self.window_size + 1, stride):
            window = z[start:start + self.window_size]
            beta = self.transform(window)
            betas.append(beta)
            times.append(start + self.window_size // 2)

        return np.array(betas), np.array(times)

    def reconstruction_error(self, z):
        """
        Compute reconstruction error for a signal.

        Returns relative L2 error: ||z - z_rec|| / ||z||
        """
        z = np.asarray(z, dtype=complex)
        beta = self.transform(z)
        z_rec = self.inverse_transform(beta, original_length=len(z))

        return np.linalg.norm(z - z_rec) / np.linalg.norm(z)


def test_hst_rom():
    """Test HST_ROM on synthetic signals."""
    print("=" * 60)
    print("Testing HST_ROM")
    print("=" * 60)

    # Generate synthetic trajectories (oscillators with varying amplitude)
    np.random.seed(42)
    trajectories = []

    for i in range(20):
        t = np.linspace(0, 10, 512)
        amp = 1 + 0.5 * np.random.randn()
        freq = 2 + 0.5 * np.random.randn()
        phase = 2 * np.pi * np.random.rand()
        z = amp * np.exp(1j * (2 * np.pi * freq * t + phase))
        trajectories.append(z)

    print(f"Generated {len(trajectories)} trajectories, each length {len(trajectories[0])}")

    # Fit ROM
    rom = HST_ROM(n_components=4, wavelet='db8', J=3, window_size=128)
    betas = rom.fit(trajectories)

    print(f"\nROM fitted:")
    print(f"  Feature dimension: {rom.feature_dim_}")
    print(f"  n_components: {rom.n_components}")
    print(f"  Samples extracted: {len(betas)}")
    print(f"  Variance explained: {rom.pca.explained_variance_ratio_}")
    print(f"  Total variance: {sum(rom.pca.explained_variance_ratio_):.3f}")

    # Test transform/inverse_transform
    test_signal = trajectories[0][:128]
    beta = rom.transform(test_signal)
    z_rec = rom.inverse_transform(beta, original_length=128)

    error = np.linalg.norm(test_signal - z_rec) / np.linalg.norm(test_signal)
    print(f"\nReconstruction test:")
    print(f"  β shape: {beta.shape}")
    print(f"  Reconstruction error: {error:.4f} ({error*100:.1f}%)")

    # Test trajectory transform
    print("\nTrajectory transform test:")
    full_traj = trajectories[0]
    betas_traj, times = rom.transform_trajectory(full_traj, window_stride=32)
    print(f"  Trajectory length: {len(full_traj)}")
    print(f"  Windows extracted: {len(betas_traj)}")
    print(f"  β trajectory shape: {betas_traj.shape}")

    return rom, betas


if __name__ == "__main__":
    rom, betas = test_hst_rom()
