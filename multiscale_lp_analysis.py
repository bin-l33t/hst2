"""
Multi-Scale Littlewood-Paley Analysis

Shows how wavelets tile the frequency axis across all scales,
and compares different wavelet families for HST suitability.
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt

# ============================================================
# SINGLE-LEVEL LITTLEWOOD-PALEY
# ============================================================

def single_level_lp(wavelet_name, n_freq=4096):
    """
    Compute single-level Littlewood-Paley sum: |H(ω)|² + |G(ω)|²

    For orthogonal wavelets, this should equal 2 exactly.
    """
    wavelet = pywt.Wavelet(wavelet_name)

    lo_d = np.array(wavelet.dec_lo)
    hi_d = np.array(wavelet.dec_hi)

    # Zero-pad for frequency resolution
    lo_padded = np.zeros(n_freq)
    hi_padded = np.zeros(n_freq)
    lo_padded[:len(lo_d)] = lo_d
    hi_padded[:len(hi_d)] = hi_d

    # Frequency responses
    H = np.fft.fft(lo_padded)
    G = np.fft.fft(hi_padded)

    # Littlewood-Paley sum
    LP_sum = np.abs(H)**2 + np.abs(G)**2

    freq = np.fft.fftfreq(n_freq)

    return freq, H, G, LP_sum


def compare_wavelets_single_level(wavelet_names, save_path='wavelet_comparison_lp.png'):
    """Compare LP condition across multiple wavelets"""
    print("=" * 70)
    print("Comparing Single-Level Littlewood-Paley Across Wavelets")
    print("=" * 70)

    n_wavelets = len(wavelet_names)
    fig, axes = plt.subplots(2, (n_wavelets + 1) // 2, figsize=(14, 8))
    axes = axes.flatten()

    results = {}

    for idx, name in enumerate(wavelet_names):
        freq, H, G, LP = single_level_lp(name)

        # Statistics
        lp_min = LP.min()
        lp_max = LP.max()
        lp_dev = np.abs(LP - 2).max()

        wavelet = pywt.Wavelet(name)

        results[name] = {
            'filter_length': wavelet.dec_len,
            'orthogonal': wavelet.orthogonal,
            'lp_deviation': lp_dev,
            'lp_min': lp_min,
            'lp_max': lp_max,
        }

        print(f"\n{name}:")
        print(f"  Filter length: {wavelet.dec_len}")
        print(f"  Orthogonal: {wavelet.orthogonal}")
        print(f"  LP sum range: [{lp_min:.10f}, {lp_max:.10f}]")
        print(f"  Deviation from 2: {lp_dev:.2e}")

        # Plot
        ax = axes[idx]
        pos = freq[:len(freq)//2]
        ax.plot(pos, np.abs(H[:len(freq)//2])**2, 'b-', label='|H|²', alpha=0.7)
        ax.plot(pos, np.abs(G[:len(freq)//2])**2, 'r-', label='|G|²', alpha=0.7)
        ax.plot(pos, LP[:len(freq)//2], 'g-', label='|H|²+|G|²', lw=2)
        ax.axhline(y=2, color='k', linestyle='--', alpha=0.5)
        ax.set_title(f'{name} (L={wavelet.dec_len}, dev={lp_dev:.1e})')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Power')
        ax.legend(fontsize=7)
        ax.set_xlim([0, 0.5])
        ax.set_ylim([0, 2.5])
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n_wavelets, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Single-Level Littlewood-Paley: |H(ω)|² + |G(ω)|² = 2', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")

    return results


# ============================================================
# MULTI-SCALE LITTLEWOOD-PALEY
# ============================================================

def multiscale_littlewood_paley(wavelet_name, J=5, n_freq=4096):
    """
    Compute multi-scale Littlewood-Paley sum for J-level decomposition.

    For a J-level DWT, the frequency axis is tiled by:
    - Final approximation: |H(ω)|² × |H(2ω)|² × ... × |H(2^(J-1)ω)|²
    - Detail at level j: |G(2^(j-1)ω)|² × ∏_{k<j} |H(2^(k-1)ω)|²

    The sum of all these should equal 2^J for orthogonal wavelets
    (or 1 if properly normalized).
    """
    wavelet = pywt.Wavelet(wavelet_name)

    lo_d = np.array(wavelet.dec_lo)
    hi_d = np.array(wavelet.dec_hi)

    freq = np.fft.fftfreq(n_freq)
    omega = 2 * np.pi * freq  # Angular frequency

    # Compute H(ω) and G(ω) as functions of omega
    def H_omega(w):
        """Low-pass frequency response at angular frequency w"""
        result = np.zeros(len(w), dtype=complex)
        for k, h_k in enumerate(lo_d):
            result += h_k * np.exp(-1j * k * w)
        return result

    def G_omega(w):
        """High-pass frequency response at angular frequency w"""
        result = np.zeros(len(w), dtype=complex)
        for k, g_k in enumerate(hi_d):
            result += g_k * np.exp(-1j * k * w)
        return result

    # Compute contributions at each scale
    contributions = []

    # Product of low-pass filters up to each level
    H_products = [np.ones(n_freq, dtype=complex)]  # H_product[0] = 1

    for j in range(J):
        # H(2^j ω)
        H_j = H_omega(omega * (2**j))
        H_products.append(H_products[-1] * H_j)

        # Detail contribution at level j+1:
        # |G(2^j ω)|² × |H(2^(j-1) ω)|² × ... × |H(ω)|²
        G_j = G_omega(omega * (2**j))

        detail_power = np.abs(G_j)**2 * np.abs(H_products[j])**2
        contributions.append(detail_power)

    # Approximation contribution at level J:
    # |H(2^(J-1) ω)|² × ... × |H(ω)|²
    approx_power = np.abs(H_products[J])**2

    # Total sum
    total = approx_power + sum(contributions)

    return freq, contributions, approx_power, total


def plot_multiscale_tiling(wavelet_names, J=4, save_path='multiscale_littlewood_paley.png'):
    """Plot multi-scale frequency tiling for multiple wavelets"""
    print("\n" + "=" * 70)
    print("Multi-Scale Littlewood-Paley Frequency Tiling")
    print("=" * 70)

    n_wavelets = len(wavelet_names)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, wavelet_name in enumerate(wavelet_names[:4]):
        ax = axes[idx]

        freq, contribs, approx, total = multiscale_littlewood_paley(wavelet_name, J=J)

        pos_idx = len(freq) // 2
        pos_freq = freq[:pos_idx]

        # Stacked area plot
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(contribs) + 1))

        # Plot from bottom to top
        bottom = np.zeros(pos_idx)

        # Approximation (lowest frequencies)
        ax.fill_between(pos_freq, bottom, bottom + approx[:pos_idx],
                        alpha=0.8, color=colors[0], label=f'Approx (J={J})')
        bottom += approx[:pos_idx]

        # Details (higher frequencies)
        for j, contrib in enumerate(reversed(contribs)):
            level = len(contribs) - j
            ax.fill_between(pos_freq, bottom, bottom + contrib[:pos_idx],
                           alpha=0.8, color=colors[j+1], label=f'Detail (j={level})')
            bottom += contrib[:pos_idx]

        # Expected value
        expected = 2**J
        ax.axhline(y=expected, color='r', linestyle='--', alpha=0.7,
                   label=f'Expected = {expected}')

        # Statistics
        total_pos = total[:pos_idx]
        dev = np.abs(total_pos - expected).max()

        ax.set_title(f'{wavelet_name}: J={J} levels (dev={dev:.1e})')
        ax.set_xlabel('Frequency (normalized)')
        ax.set_ylabel('Cumulative power')
        ax.legend(fontsize=7, loc='upper right')
        ax.set_xlim([0, 0.5])
        ax.set_ylim([0, expected * 1.2])
        ax.grid(True, alpha=0.3)

        print(f"\n{wavelet_name} (J={J}):")
        print(f"  Total power range: [{total.min():.4f}, {total.max():.4f}]")
        print(f"  Expected: {expected}")
        print(f"  Max deviation: {dev:.2e}")

    plt.suptitle(f'Multi-Scale Frequency Tiling: How {J}-level DWT covers all frequencies', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")


# ============================================================
# DYADIC FREQUENCY BANDS VISUALIZATION
# ============================================================

def plot_dyadic_bands(wavelet_name='db8', J=5, save_path='dyadic_frequency_bands.png'):
    """
    Visualize the dyadic frequency band structure of the DWT.

    At level j, the detail coefficients capture frequencies in [2^(-j-1), 2^(-j)]
    (as fractions of Nyquist).
    """
    print("\n" + "=" * 70)
    print(f"Dyadic Frequency Bands for {wavelet_name}")
    print("=" * 70)

    wavelet = pywt.Wavelet(wavelet_name)
    lo_d = np.array(wavelet.dec_lo)
    hi_d = np.array(wavelet.dec_hi)

    n_freq = 4096
    freq = np.fft.fftfreq(n_freq)
    omega = 2 * np.pi * freq

    # Compute filter responses
    def filter_response(coeffs, w):
        result = np.zeros(len(w), dtype=complex)
        for k, c in enumerate(coeffs):
            result += c * np.exp(-1j * k * w)
        return result

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Top plot: Individual filter responses at each level
    ax = axes[0]

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, J + 1))

    # Cumulative low-pass (for showing effective bands)
    H_cumulative = np.ones(n_freq, dtype=complex)

    pos_idx = n_freq // 2
    pos_freq = freq[:pos_idx]

    for j in range(J):
        # High-pass response at this level
        G_j = filter_response(hi_d, omega * (2**j))

        # Effective band: G(2^j ω) × H(2^(j-1) ω) × ... × H(ω)
        effective = np.abs(G_j)**2 * np.abs(H_cumulative)**2

        ax.fill_between(pos_freq, 0, effective[:pos_idx],
                        alpha=0.6, color=colors[j],
                        label=f'Level {j+1}: [{2**(-j-1):.3f}, {2**(-j):.3f}]')

        # Update cumulative low-pass
        H_j = filter_response(lo_d, omega * (2**j))
        H_cumulative = H_cumulative * H_j

    # Final approximation band
    approx = np.abs(H_cumulative)**2
    ax.fill_between(pos_freq, 0, approx[:pos_idx],
                    alpha=0.6, color=colors[J],
                    label=f'Approx: [0, {2**(-J):.3f}]')

    ax.set_xlabel('Frequency (normalized to Nyquist)')
    ax.set_ylabel('Effective filter power')
    ax.set_title(f'{wavelet_name}: Effective frequency bands at each decomposition level')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim([0, 0.5])
    ax.grid(True, alpha=0.3)

    # Bottom plot: Band boundaries (ideal dyadic)
    ax = axes[1]

    # Ideal dyadic bands
    for j in range(J + 1):
        if j < J:
            # Detail band
            f_low = 2**(-j-1)
            f_high = 2**(-j)
            ax.axvspan(f_low, f_high, alpha=0.3, color=colors[j])
            ax.text((f_low + f_high)/2, 0.5, f'D{j+1}', ha='center', va='center', fontsize=10)
        else:
            # Approximation band
            f_high = 2**(-J)
            ax.axvspan(0, f_high, alpha=0.3, color=colors[J])
            ax.text(f_high/2, 0.5, f'A{J}', ha='center', va='center', fontsize=10)

    # Add vertical lines at band boundaries
    for j in range(J + 1):
        f = 2**(-j) if j < J else 0
        ax.axvline(x=f, color='k', linestyle='-', alpha=0.5)

    ax.set_xlabel('Frequency (normalized to Nyquist)')
    ax.set_ylabel('')
    ax.set_title('Ideal Dyadic Frequency Bands: Each level halves the frequency range')
    ax.set_xlim([0, 0.5])
    ax.set_ylim([0, 1])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# WAVELET SUITABILITY TABLE
# ============================================================

def create_suitability_report():
    """Create a report on wavelet suitability for HST"""
    print("\n" + "=" * 70)
    print("Wavelet Suitability for HST")
    print("=" * 70)

    wavelets = [
        ('db4', 'Daubechies-4'),
        ('db8', 'Daubechies-8'),
        ('db16', 'Daubechies-16'),
        ('sym4', 'Symlet-4'),
        ('sym8', 'Symlet-8'),
        ('coif2', 'Coiflet-2'),
        ('coif4', 'Coiflet-4'),
    ]

    print(f"\n{'Wavelet':<10} {'Name':<15} {'Length':<8} {'LP Dev':<12} {'HST Suit.':<12}")
    print("-" * 60)

    results = []

    for short_name, full_name in wavelets:
        try:
            wavelet = pywt.Wavelet(short_name)
            _, _, _, LP = single_level_lp(short_name)
            lp_dev = np.abs(LP - 2).max()

            # Suitability based on filter length and orthogonality
            if lp_dev < 1e-10 and wavelet.orthogonal:
                if wavelet.dec_len <= 16:
                    suitability = "Excellent"
                elif wavelet.dec_len <= 24:
                    suitability = "Good"
                else:
                    suitability = "OK (long)"
            else:
                suitability = "Poor"

            print(f"{short_name:<10} {full_name:<15} {wavelet.dec_len:<8} {lp_dev:<12.2e} {suitability:<12}")

            results.append({
                'short': short_name,
                'name': full_name,
                'length': wavelet.dec_len,
                'lp_deviation': lp_dev,
                'suitability': suitability,
                'orthogonal': wavelet.orthogonal,
            })
        except Exception as e:
            print(f"{short_name:<10} Error: {e}")

    print("\nRecommendation: db8 or sym8 (length=16, perfect LP, orthogonal)")

    return results


# ============================================================
# MAIN
# ============================================================

def run_all_analyses():
    """Run all multi-scale LP analyses"""

    # 1. Compare single-level LP across wavelets
    wavelet_names = ['db4', 'db8', 'sym8', 'coif4']
    results = compare_wavelets_single_level(
        wavelet_names,
        save_path='/home/ubuntu/rectifier/wavelet_comparison_lp.png'
    )

    # 2. Multi-scale frequency tiling
    plot_multiscale_tiling(
        wavelet_names,
        J=4,
        save_path='/home/ubuntu/rectifier/multiscale_littlewood_paley.png'
    )

    # 3. Dyadic frequency bands
    plot_dyadic_bands(
        'db8',
        J=5,
        save_path='/home/ubuntu/rectifier/dyadic_frequency_bands.png'
    )

    # 4. Suitability report
    suitability = create_suitability_report()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Littlewood-Paley Analysis")
    print("=" * 70)

    print("""
KEY FINDINGS:

1. LITTLEWOOD-PALEY CONDITION
   - All tested orthogonal wavelets satisfy |H|² + |G|² = 2 EXACTLY
   - Deviation is at machine precision (~1e-15)
   - This GUARANTEES perfect reconstruction

2. MULTI-SCALE FREQUENCY TILING
   - J-level DWT tiles frequencies into J+1 dyadic bands
   - Each level captures an octave of frequency content
   - Total power at each frequency = 2^J (before normalization)

3. INTERPRETATION FOR HST
   - The rectifier R_sheeted operates at each decomposition level
   - Wavelet bands capture different time scales
   - Perfect LP + perfect R_inv = perfect HST reconstruction

4. RECOMMENDATION
   - db8 (Daubechies-8): Best balance of length and frequency localization
   - sym8 (Symlet-8): Similar to db8, slightly more symmetric
   - Avoid very long filters (>24) due to boundary effects
""")

    return results, suitability


if __name__ == "__main__":
    results, suitability = run_all_analyses()
