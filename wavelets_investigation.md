# Wavelet Investigation for HST

## Summary

This document investigates the wavelet requirements for Glinsky's Heisenberg Scattering Transformation (HST), drawing from the Glinsky 2025 paper and Ali et al.'s "Coherent States, Wavelets, and Their Generalizations" (2nd ed.).

---

## 1. Wavelet Choice

### Glinsky's Specification (Eq. 9-10)

The paper explicitly states wavelets must be:

> "**normalized, orthogonal, localized and harmonic (that is coherent states)**"

With the scaling relation:
```
ψ_k(x) ≡ k² ψ(kx)
```

This is the **L¹-normalization** (preserves integral), not the L²-normalization used in standard wavelet theory:
- L² normalization: `ψ_k(x) = k^(1/2) ψ(kx)` (preserves energy)
- L¹ normalization: `ψ_k(x) = k² ψ(kx)` (preserves integral, used in HST)

### Connection to Ali's Framework

From Ali Ch. 12, wavelets are **coherent states of the affine group** ("ax+b" group):
```
ψ_{b,a}(t) = |a|^(-1/2) ψ((t-b)/a)
```

The affine group acts by:
- **Translation** (b): position in signal
- **Dilation** (a): scale/frequency

Key insight: Morlet and Grossmann recognized from the beginning that "wavelets are simply coherent states associated to the affine group of the line."

### Specific Wavelet Families

**Candidates that satisfy "coherent state" requirement:**

1. **Morlet Wavelet** (complex, progressive)
   ```
   ψ(t) = exp(iξ₀t) exp(-t²/2) - correction term
   ```
   - Complex-valued → separates phase and modulus
   - Typical ξ₀ ≥ 5.5 for admissibility
   - Most commonly used in physics applications

2. **Mexican Hat / Marr Wavelet** (real)
   ```
   ψ(t) = (1 - t²) exp(-t²/2)
   ```
   - Second derivative of Gaussian
   - Real-valued → cannot separate phase

3. **Gabor Wavelets / Gaborettes** (STFT-based)
   - Related to Weyl-Heisenberg group, not affine group
   - Used for time-frequency (not time-scale)

**For HST**: The Morlet wavelet is likely preferred because:
- Complex-valued (HST works with complex fields throughout)
- Well-localized in both time and frequency
- Smooth and analytic

---

## 2. Orthonormal vs Overcomplete

### The Key Distinction

| Property | Overcomplete (Frames) | Orthonormal Basis |
|----------|----------------------|-------------------|
| Redundancy | High (continuous CWT) | None |
| Reconstruction | Via frame operator | Trivial |
| Standard wavelets | Morlet, Mexican Hat | Haar, Daubechies |
| Invertibility | Stable but redundant | Perfect |

### What Glinsky Says

From the implementation section:
> "orthogonal coherent states are used as described by Ali et al."

And critically:
> "Special care is taken that the set of Father Wavelets form a **partition-of-unity**, to preserve invertability."

### Partition of Unity Condition

The Father wavelets φ_k must satisfy:
```
Σ_k φ_k(x) = 1  (for all x)
```

This ensures:
1. **No information loss** in the transform
2. **Perfect reconstruction** is possible
3. The transform is **invertible**

### Littlewood-Paley Condition

Glinsky mentions ψ and φ must satisfy the "Littlewood-Pauley condition." From Ali Ch. 12, this is essentially:

```
Σ_j |ψ̂(2^j ξ)|² = 1  (almost everywhere)
```

This is the **frequency-domain partition of unity** - ensures all frequencies are covered exactly once.

### Resolution of Overcomplete vs Orthonormal

**HST uses a hybrid approach:**
1. The **continuous** coherent state framework (for theoretical justification)
2. Implemented with **discrete orthogonal** wavelets (for invertibility)
3. Father wavelets form **partition-of-unity** (for reconstruction)

This is possible because of the **bination** (downsampling by 2) at each stage - it effectively discretizes the continuous transform while preserving the coherent state structure.

---

## 3. The HST Cascade vs MST

### Standard MST (Mallat Scattering Transform)
```
S_m = |ψ_λm * ... |ψ_λ1 * x| ...|
```

- Uses **modulus** |·| as nonlinearity
- Discards phase information
- Loses invertibility (modulus is not injective)

### HST (Heisenberg Scattering Transform)
```
S_m[f(x)](z) = φ_kx ⋆ (∏_{n=1}^m i ln R₀ ψ_kn ⋆) i ln R₀ f(x)
```

- Uses **i ln R₀** as nonlinearity (the analytic rectifier)
- **Preserves phase** via complex logarithm: `ln(z) = ln|z| + i·arg(z)`
- Potentially invertible (conformal maps are locally invertible)

### Key Difference: Phase Preservation

The complex logarithm decomposes into:
```
i ln(z) = i ln|z| - arg(z)
```

So HST captures:
- **Magnitude** via `i ln|z|` (imaginary part)
- **Phase** via `-arg(z)` (real part)

While MST only keeps `|z|` and loses `arg(z)` entirely.

### The Rectifier's Role

From Glinsky:
> "The ln R₀ conformal (canonical) transformation is flattening the space onto the cylinder by transforming into polar coordinates about R₀"

The rectifier R₀:
1. Normalizes the complex logarithm's branch cut behavior
2. Ensures convergence to the real axis after iteration
3. Makes the transformation "compact" (bounded)

---

## 4. Bination and the Pyramid

### What is Bination?

**Bination** = downsampling by factor of 2 (decimation)

From Glinsky's implementation:
> "The Mother Wavelet is convolved with the signal. The signal is then binated. The convolution with the Mother Wavelet acts as an antialias filter for the bination."

### The Pyramid Structure

```
Level 0: signal s(x)           [N samples]
         ↓ ψ ⋆ (convolve)
         ↓ binate (↓2)
Level 1: coefficients          [N/2 samples]
         ↓ ψ ⋆
         ↓ binate
Level 2: coefficients          [N/4 samples]
         ...
```

**Key insight**: Same wavelet at each level, but bination doubles effective wavelength:
> "This has effectively doubled the wavelength of the filter without increasing the number of samples, leading to the N log N scaling."

### Mother vs Father Wavelet

| Wavelet | Role | Frequency | Symbol |
|---------|------|-----------|--------|
| Mother ψ | Extract details | Band-pass (high) | ψ_k |
| Father φ | Smooth/average | Low-pass | φ_kx |

From Ali Ch. 13:
- **Mother wavelet ψ**: Generates the detail spaces W_j
- **Scaling function φ** (Father): Generates approximation spaces V_j
- Relation: `V_{j+1} = V_j ⊕ W_j`

### Father Wavelet Specification

Glinsky:
> "The Father wavelet is matched to the Mother Wavelet and is a **Gaussian-like windowing function**. The size of the window is matched to the effective scale of the convolution to have maximum spatial resolution."

This suggests φ is essentially:
```
φ(x) ∝ exp(-x²/2σ²)
```
with σ matched to the scale k.

---

## 5. Implementation Requirements

### From Glinsky's Implementation Section

1. **Constant length in samples**: Mother wavelet has fixed number of samples
2. **N log N scaling**: Due to bination + fixed filter length
3. **GPU implementation**: PyTorch on GPU
4. **Time domain**: Convolution in time, not frequency
5. **Interleaved posting**: Transform interleaved by order m

### Wavelet Construction Recipe

Based on the analysis:

```python
def mother_wavelet(t, k, xi0=5.6):
    """
    Morlet-like mother wavelet with L1 normalization
    k: scale parameter
    xi0: center frequency
    """
    # L1 normalization: k² factor
    return k**2 * np.exp(1j * xi0 * k * t) * np.exp(-(k * t)**2 / 2)

def father_wavelet(t, k, x):
    """
    Gaussian father wavelet (windowing function)
    k: scale parameter
    x: position parameter
    """
    sigma = 1/k  # matched to scale
    return k**2 * np.exp(-((t - x) * k)**2 / 2)
```

### Partition of Unity Check

For invertibility, verify:
```python
def check_partition_of_unity(father_wavelets, x_range):
    """Sum of father wavelets should equal 1 everywhere"""
    total = sum(phi_k(x) for phi_k in father_wavelets)
    assert np.allclose(total, 1.0), "Partition of unity violated!"
```

---

## 6. Open Questions

### 6.1 Which specific wavelet does Glinsky use?

The paper doesn't give an explicit formula. Candidates:
- Modified Morlet (most likely, given complex requirement)
- Custom coherent state wavelet from Ali's construction

### 6.2 How is orthogonality achieved with continuous wavelets?

Standard CWT wavelets (Morlet, Mexican Hat) are **not orthogonal** - they form frames, not bases. Possible resolutions:
- Use discrete orthogonal wavelets (Daubechies) for implementation
- "Orthogonal" may mean orthogonality of the coherent state structure
- The bination + partition-of-unity may induce effective orthogonality

### 6.3 Invertibility claim

Glinsky claims HST is invertible with the same N log N complexity. Questions:
- Is this exact inversion or approximate?
- Does the rectifier preserve invertibility across iterations?
- What's the role of the Wick ordering (Eq. 19)?

### 6.4 Connection to Multiresolution Analysis

The HST structure mirrors MRA:
- Bination = dyadic downsampling
- Father/Mother wavelets = scaling/wavelet functions
- N log N = fast wavelet transform

But uses coherent states instead of orthonormal wavelets.

---

## 7. Recommendations for Implementation

1. **Start with Morlet wavelet** - matches "complex, localized, harmonic" requirements
2. **Use L¹ normalization** - `k² ψ(kx)` not `k^(1/2) ψ(kx)`
3. **Father wavelet = Gaussian window** - matched to scale
4. **Verify partition of unity** - critical for invertibility
5. **Bination after convolution** - antialias via wavelet filter
6. **Implement in time domain** - as Glinsky specifies

### Minimal Test

```python
# Test HST cascade structure
def hst_layer(signal, mother_psi, rectifier_R0):
    """Single HST layer: convolve, rectify, binate"""
    convolved = convolve(signal, mother_psi)
    rectified = 1j * np.log(rectifier_R0(convolved))
    binated = rectified[::2]  # downsample by 2
    return binated
```

---

## References

- Glinsky (2025): "A Collective Field Theory" - HST derivation
- Ali, Antoine, Gazeau (2014): "Coherent States, Wavelets, and Their Generalizations"
  - Ch. 12: Continuous Wavelet Transform
  - Ch. 13: Discrete Wavelet Transforms
- Mallat (2012): "Group Invariant Scattering" - original MST
