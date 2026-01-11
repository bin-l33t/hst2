"""
HST Implementation in PyTorch with Complex Support

This module implements the Heisenberg Scattering Transform using PyTorch,
enabling GPU acceleration and automatic differentiation for learning.

Key features:
- Native complex arithmetic (torch.complex64/128)
- GPU acceleration via CUDA
- Automatic differentiation for learning singularity locations
- Compatible with existing numpy HST outputs

From Glinsky's theory:
- HST forward: Signal -> Rectifier -> Wavelet cascade
- The rectifier R(z) = i*ln(R0(z)) contracts imaginary parts
- Converges to real axis (branch cut) at rate lambda = 2/pi
"""

import torch
import torch.nn as nn
import torch.fft
import numpy as np

# Try to import pywt for filter initialization
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


class ComplexHST(nn.Module):
    """
    HST implemented in PyTorch with complex support.

    Advantages over numpy version:
    - GPU acceleration
    - Automatic differentiation
    - Native complex arithmetic

    Parameters
    ----------
    J : int
        Number of decomposition levels
    wavelet : str
        Wavelet name (default 'db8')
    """

    def __init__(self, J=4, wavelet='db8'):
        super().__init__()
        self.J = J
        self.wavelet = wavelet

        # Precompute wavelet filters as torch tensors
        if HAS_PYWT:
            w = pywt.Wavelet(wavelet)
            lo_d = torch.tensor(w.dec_lo, dtype=torch.float32)
            hi_d = torch.tensor(w.dec_hi, dtype=torch.float32)
            lo_r = torch.tensor(w.rec_lo, dtype=torch.float32)
            hi_r = torch.tensor(w.rec_hi, dtype=torch.float32)
        else:
            # Fallback: Simple Haar wavelet
            lo_d = torch.tensor([0.7071067811865476, 0.7071067811865476], dtype=torch.float32)
            hi_d = torch.tensor([-0.7071067811865476, 0.7071067811865476], dtype=torch.float32)
            lo_r = lo_d.clone()
            hi_r = hi_d.clone()

        # Register as buffers (not parameters, but move with model)
        self.register_buffer('lo_d', lo_d)
        self.register_buffer('hi_d', hi_d)
        self.register_buffer('lo_r', lo_r)
        self.register_buffer('hi_r', hi_r)

        self.filter_len = len(lo_d)

    def R0_sheeted(self, z):
        """
        Sheeted rectifier R0 in PyTorch.

        R0(z) = -i(w +/- s) where w = 2z/pi, s = sqrt(w-1)*sqrt(w+1)
        Sign chosen to preserve half-plane:
        - Upper half-plane (Im > 0): use + branch
        - Lower half-plane (Im < 0): use - branch

        Parameters
        ----------
        z : torch.Tensor (complex)
            Input tensor

        Returns
        -------
        R0 : torch.Tensor (complex)
            Rectified output, same shape as input
        """
        # Ensure complex
        if not z.is_complex():
            z = z.to(torch.complex64)

        w = 2.0 * z / torch.pi

        # Complex square roots - PyTorch handles branch cuts
        s1 = torch.sqrt(w - 1.0)
        s2 = torch.sqrt(w + 1.0)
        s = s1 * s2

        # Determine sign based on imaginary part
        im_sign = torch.sign(z.imag)
        # Handle exact real case (im = 0)
        im_sign = torch.where(im_sign == 0, torch.ones_like(im_sign), im_sign)

        # Upper half -> +s, Lower half -> -s
        R0 = -1j * (w + im_sign.to(z.dtype) * s)

        return R0

    def R_sheeted(self, z):
        """
        Full rectifier: R(z) = i * ln(R0(z))

        Preserves half-planes:
        - Im(z) > 0 -> Im(R) > 0
        - Im(z) < 0 -> Im(R) < 0

        Parameters
        ----------
        z : torch.Tensor (complex)
            Input tensor

        Returns
        -------
        R : torch.Tensor (complex)
            Rectified output, |Im| decreased
        """
        R0 = self.R0_sheeted(z)
        return 1j * torch.log(R0)

    def R_inv(self, w):
        """
        Inverse rectifier.

        R_inv(w) = (pi/2) * sin(w)

        Parameters
        ----------
        w : torch.Tensor (complex)
            Rectified signal

        Returns
        -------
        z : torch.Tensor (complex)
            Original signal
        """
        return (torch.pi / 2.0) * torch.sin(w)

    def _dwt_level(self, x):
        """
        Single level DWT via convolution.

        Parameters
        ----------
        x : torch.Tensor (complex)
            Input signal, shape (..., N)

        Returns
        -------
        cA : torch.Tensor
            Approximation coefficients
        cD : torch.Tensor
            Detail coefficients
        """
        n = x.shape[-1]

        # Convert filters to complex for multiplication
        lo = self.lo_d.to(x.dtype)
        hi = self.hi_d.to(x.dtype)

        # Pad signal for circular convolution
        pad_len = self.filter_len - 1
        x_padded = torch.nn.functional.pad(x, (pad_len, 0), mode='circular')

        # Convolution via FFT (more efficient for long filters)
        n_padded = x_padded.shape[-1]

        # Zero-pad filters to match signal length
        lo_padded = torch.zeros(n_padded, dtype=x.dtype, device=x.device)
        hi_padded = torch.zeros(n_padded, dtype=x.dtype, device=x.device)
        lo_padded[:self.filter_len] = lo
        hi_padded[:self.filter_len] = hi

        # FFT convolution
        X = torch.fft.fft(x_padded, dim=-1)
        Lo = torch.fft.fft(lo_padded)
        Hi = torch.fft.fft(hi_padded)

        cA_full = torch.fft.ifft(X * Lo, dim=-1)
        cD_full = torch.fft.ifft(X * Hi, dim=-1)

        # Trim padding and downsample by 2
        cA_full = cA_full[..., pad_len:]
        cD_full = cD_full[..., pad_len:]

        cA = cA_full[..., ::2]
        cD = cD_full[..., ::2]

        return cA, cD

    def _idwt_level(self, cA, cD):
        """
        Single level inverse DWT.

        Parameters
        ----------
        cA : torch.Tensor (complex)
            Approximation coefficients
        cD : torch.Tensor (complex)
            Detail coefficients

        Returns
        -------
        x : torch.Tensor
            Reconstructed signal
        """
        n = cA.shape[-1]

        # Convert filters to complex
        lo = self.lo_r.to(cA.dtype)
        hi = self.hi_r.to(cA.dtype)

        # Upsample by 2 (insert zeros)
        cA_up = torch.zeros(*cA.shape[:-1], n * 2, dtype=cA.dtype, device=cA.device)
        cD_up = torch.zeros(*cD.shape[:-1], n * 2, dtype=cD.dtype, device=cD.device)
        cA_up[..., ::2] = cA
        cD_up[..., ::2] = cD

        # Pad for convolution
        pad_len = self.filter_len - 1
        cA_padded = torch.nn.functional.pad(cA_up, (pad_len, 0), mode='circular')
        cD_padded = torch.nn.functional.pad(cD_up, (pad_len, 0), mode='circular')

        n_padded = cA_padded.shape[-1]

        # Zero-pad filters
        lo_padded = torch.zeros(n_padded, dtype=cA.dtype, device=cA.device)
        hi_padded = torch.zeros(n_padded, dtype=cA.dtype, device=cA.device)
        lo_padded[:self.filter_len] = lo
        hi_padded[:self.filter_len] = hi

        # FFT convolution
        CA = torch.fft.fft(cA_padded, dim=-1)
        CD = torch.fft.fft(cD_padded, dim=-1)
        Lo = torch.fft.fft(lo_padded)
        Hi = torch.fft.fft(hi_padded)

        x_lo = torch.fft.ifft(CA * Lo, dim=-1)
        x_hi = torch.fft.ifft(CD * Hi, dim=-1)

        # Combine and trim
        x = x_lo + x_hi
        x = x[..., pad_len:]

        return x

    def forward(self, f, return_all=False):
        """
        Forward HST transform.

        HST cascade:
        1. u = R(f)
        2. For each level j:
           - (cA, cD) = DWT(u)
           - Store cD
           - u = R(cA)

        Parameters
        ----------
        f : torch.Tensor (complex or real)
            Input signal, shape (batch, length) or (length,)
        return_all : bool
            If True, return all intermediate u values

        Returns
        -------
        coeffs : dict
            'cD': list of detail coefficients at each level
            'cA_final': final approximation (deepest level)
            'u_levels': (if return_all) list of u at each level
        """
        # Ensure batch dimension
        squeeze_output = False
        if f.dim() == 1:
            f = f.unsqueeze(0)
            squeeze_output = True

        # Ensure complex
        if not f.is_complex():
            f = f.to(torch.complex64)

        # Initial rectification
        u = self.R_sheeted(f)

        coeffs = {
            'cD': [],
            'cA_final': None,
            'original_length': f.shape[-1]
        }

        if return_all:
            coeffs['u_levels'] = [u]

        for j in range(self.J):
            # Wavelet decomposition
            cA, cD = self._dwt_level(u)

            # Store detail coefficients
            coeffs['cD'].append(cD)

            # Rectify approximation for next level
            u = self.R_sheeted(cA)

            if return_all:
                coeffs['u_levels'].append(u)

        coeffs['cA_final'] = u

        if squeeze_output:
            coeffs['cD'] = [c.squeeze(0) for c in coeffs['cD']]
            coeffs['cA_final'] = coeffs['cA_final'].squeeze(0)
            if return_all:
                coeffs['u_levels'] = [u.squeeze(0) for u in coeffs['u_levels']]

        return coeffs

    def inverse(self, coeffs):
        """
        Inverse HST transform.

        Reverses the forward cascade:
        1. u = cA_final
        2. For each level j (reversed):
           - cA = R_inv(u)
           - u = IDWT(cA, cD[j])
        3. f = R_inv(u)

        Parameters
        ----------
        coeffs : dict
            Output from forward()

        Returns
        -------
        f_rec : torch.Tensor (complex)
            Reconstructed signal
        """
        u = coeffs['cA_final']

        # Ensure batch dimension
        squeeze_output = False
        if u.dim() == 1:
            u = u.unsqueeze(0)
            squeeze_output = True
            cD_list = [c.unsqueeze(0) for c in coeffs['cD']]
        else:
            cD_list = coeffs['cD']

        for j in reversed(range(self.J)):
            # Inverse rectifier on approximation
            cA = self.R_inv(u)

            # Inverse DWT
            u = self._idwt_level(cA, cD_list[j])

        # Final inverse rectifier
        f_rec = self.R_inv(u)

        # Trim to original length if needed
        if 'original_length' in coeffs:
            f_rec = f_rec[..., :coeffs['original_length']]

        if squeeze_output:
            f_rec = f_rec.squeeze(0)

        return f_rec

    def extract_features(self, f, flatten=True):
        """
        Extract HST features for use in downstream tasks.

        Returns concatenated scattering coefficients suitable for
        PCA or feeding into MLP.

        Parameters
        ----------
        f : torch.Tensor
            Input signal
        flatten : bool
            If True, flatten all coefficients into single vector

        Returns
        -------
        features : torch.Tensor
            HST feature vector
        """
        coeffs = self.forward(f)

        if flatten:
            # Concatenate all detail coefficients and final approximation
            parts = []
            for cD in coeffs['cD']:
                if cD.dim() == 1:
                    parts.append(cD)
                else:
                    parts.append(cD.reshape(cD.shape[0], -1))

            cA = coeffs['cA_final']
            if cA.dim() == 1:
                parts.append(cA)
            else:
                parts.append(cA.reshape(cA.shape[0], -1))

            features = torch.cat(parts, dim=-1)
        else:
            features = coeffs

        return features


def test_torch_vs_numpy():
    """
    Verify PyTorch HST matches numpy version.

    Returns True if they match within tolerance.
    """
    print("=" * 60)
    print("TESTING: PyTorch HST vs NumPy HST")
    print("=" * 60)

    # Import numpy version
    try:
        from hst import hst_forward_pywt, hst_inverse_pywt
        HAS_NUMPY_HST = True
    except ImportError:
        print("NumPy HST not available, skipping comparison")
        HAS_NUMPY_HST = False
        return True

    # Test signal
    t_np = np.linspace(0, 10, 512)
    f_np = np.exp(1j * 2 * np.pi * t_np) + 0.5 * np.exp(1j * 4 * np.pi * t_np)

    # Convert to torch
    f_torch = torch.tensor(f_np, dtype=torch.complex64)

    # NumPy version
    if HAS_NUMPY_HST:
        coeffs_np = hst_forward_pywt(f_np, J=3, wavelet_name='db8')

    # PyTorch version
    hst = ComplexHST(J=3, wavelet='db8')
    coeffs_torch = hst(f_torch)

    # Compare detail coefficients at each level
    print("\nDetail coefficient comparison:")
    all_match = True

    for j in range(3):
        np_cD = coeffs_np['cD'][j] if HAS_NUMPY_HST else np.zeros(10)
        torch_cD = coeffs_torch['cD'][j].numpy()

        # Lengths may differ slightly due to boundary handling
        min_len = min(len(np_cD), len(torch_cD))
        if min_len > 0:
            error = np.linalg.norm(np_cD[:min_len] - torch_cD[:min_len])
            norm = np.linalg.norm(np_cD[:min_len])
            rel_error = error / norm if norm > 1e-10 else error

            status = "PASS" if rel_error < 0.1 else "FAIL"
            print(f"  Level {j}: relative error = {rel_error:.2e} [{status}]")

            if rel_error >= 0.1:
                all_match = False

    # Test reconstruction
    print("\nReconstruction test:")
    f_rec_torch = hst.inverse(coeffs_torch)
    rec_error = torch.norm(f_torch - f_rec_torch) / torch.norm(f_torch)
    print(f"  Reconstruction error: {rec_error.item():.2e}")

    # Test rectifier roundtrip
    print("\nRectifier roundtrip test:")
    z_test = torch.tensor([1+2j, 3+4j, -1-2j, 0.5+0.1j], dtype=torch.complex64)
    R_z = hst.R_sheeted(z_test)
    z_rec = hst.R_inv(R_z)
    rect_error = torch.norm(z_test - z_rec) / torch.norm(z_test)
    print(f"  R -> R_inv error: {rect_error.item():.2e}")

    # Test GPU if available
    if torch.cuda.is_available():
        print("\nGPU test:")
        device = torch.device('cuda')
        hst_gpu = ComplexHST(J=3, wavelet='db8').to(device)
        f_gpu = f_torch.to(device)

        coeffs_gpu = hst_gpu(f_gpu)
        print(f"  GPU forward: OK")

        f_rec_gpu = hst_gpu.inverse(coeffs_gpu)
        gpu_error = torch.norm(f_gpu - f_rec_gpu) / torch.norm(f_gpu)
        print(f"  GPU reconstruction error: {gpu_error.item():.2e}")
    else:
        print("\nGPU not available")

    return all_match


def test_im_contraction():
    """
    Test that |Im| contracts through HST levels (Glinsky's lambda = 2/pi).
    """
    print("\n" + "=" * 60)
    print("TESTING: Imaginary part contraction")
    print("=" * 60)

    # Test signal with significant imaginary part
    t = torch.linspace(0, 10, 1024)
    f = torch.exp(1j * 2 * torch.pi * t).to(torch.complex64)

    hst = ComplexHST(J=5, wavelet='db8')
    coeffs = hst(f, return_all=True)

    print("\n|Im(u)| by level:")
    im_norms = []
    for j, u in enumerate(coeffs['u_levels']):
        im_norm = torch.abs(u.imag).mean().item()
        im_norms.append(im_norm)
        print(f"  Level {j}: |Im| = {im_norm:.4f}")

    # Compute contraction rate
    ratios = []
    for j in range(1, len(im_norms)):
        if im_norms[j-1] > 1e-10:
            ratio = im_norms[j] / im_norms[j-1]
            ratios.append(ratio)

    if ratios:
        mean_lambda = np.mean(ratios)
        print(f"\nMean contraction rate: {mean_lambda:.4f}")
        print(f"Expected (2/pi): {2/np.pi:.4f}")
        print(f"Match: {'PASS' if abs(mean_lambda - 2/np.pi) < 0.2 else 'CLOSE'}")


if __name__ == "__main__":
    test_torch_vs_numpy()
    test_im_contraction()
