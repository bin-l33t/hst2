# Rectifier Project Context

## Overview

This project implements Glinsky's **Analytic Rectifier** R(z), a conformal mapping that iteratively converges complex numbers to the real axis. The rectifier is foundational to the **Heisenberg Scattering Transformation (HST)**, which generalizes wavelets for analyzing collective/complex systems.

---

## Verified Formulas

### Core Rectifier (Equations 14-15 from Glinsky paper)

```
h⁻¹(w) = w + √(w-1)·√(w+1)        # CRITICAL: NOT √(w²-1)!
R₀(z) = -i · h⁻¹(2z/π)            # Maps to exterior of unit disk
R(z)  = i · ln(R₀(z))             # The Analytic Rectifier
```

### Inverse Rectifier

```
R⁻¹(w) = (π/2) · sin(w)           # NOT cos(w)!
```

Verified: `R⁻¹(R(z)) = z` for all z outside branch cut.

### Constants

- **Convergence rate**: λ = 2/π ≈ 0.6366
- **Branch cut**: [-π/2, +π/2] on real axis
- **Fixed point**: z = 0

---

## Critical Implementation Detail

**MUST use `√(w-1)·√(w+1)`, NOT `√(w²-1)`!**

The latter gives incorrect branch cuts for Re(w) < 0. In NumPy:

```python
# CORRECT:
np.sqrt(w - 1) * np.sqrt(w + 1)

# WRONG (gives wrong branch):
np.sqrt(w**2 - 1)
```

---

## Test Points (Matching Paper's Figure 7)

| Point | Color | R₀(z) | R(z) |
|-------|-------|-------|------|
| z = -π/2 | Blue | +i | +π/2 |
| z = +π/2 | Blue | -i | -π/2 |
| z = +0.01i | Green | ≈ +1 | ≈ 0 |
| z = -0.01i | Red | ≈ -1 | ≈ -π |

**Geometric interpretation:**
- Blue points (±π/2) map to ±i on unit circle
- Green point (above cut) → +1 on unit circle
- Red point (below cut) → -1 on unit circle

---

## Sheeted Version (Half-Plane Preserving)

For applications requiring Im(z) > 0 → Im(R) > 0:

```python
def R0_sheeted(z):
    w = (2/π) * z
    s = √(w-1) · √(w+1)

    if Im(z) ≥ 0:
        return -i(w + s)    # |R₀| ≥ 1 (exterior)
    else:
        return -i(w - s)    # |R₀| ≤ 1 (interior)
```

**Properties:**
- Upper half-plane → Upper half-plane
- Lower half-plane → Lower half-plane
- Imaginary axis → Imaginary axis (Re(R) = 0)

---

## Convergence Behavior

All trajectories R^n(z₀) converge to z = 0:

| Start z₀ | Estimated λ | Final Re |
|----------|-------------|----------|
| 1+2j | 0.6366 | 0.0000 |
| -2+1j | 0.6366 | 0.0000 |
| 3j | 0.6366 | 0.0000 |
| -1-3j | 0.6366 | 0.0000 |
| 5+0.1j | 0.6366 | 0.0000 |

---

## Open Questions

### 1. λ Discrepancy
- **Paper shows**: λ ≈ 0.45 (in Figure 7)
- **We compute**: λ = 2/π ≈ 0.6366

Possible explanations:
- Different normalization in paper
- Effective λ for specific starting points
- λ for real part vs imaginary part convergence

### 2. Connection to Wavelets
- The rectifier should be the "activation function" in HST
- Need to understand: R(z) vs complex logarithm ln(z) = ln|z| + i·arg(z)
- How does iteration relate to wavelet scale?

### 3. Branch Structure
- Paper's Figure 7 shows specific branch cut handling
- Our sheeted version differs slightly from paper's color scheme
- Need to verify which convention matches HST requirements

---

## File Inventory

### Implementation
- `rectifier.py` - Verified implementation with visualization

### Reference Documents (PDF → TXT)
- `glinsky_collective.txt` - HST theory paper (2025)
- `glinsky_mhd_surrogate.txt` - MHD application (2023)
- `ali_ch3_pov_frames.pdf` - POV-frames theory
- `ali_ch8_square_integrable.pdf` - Square-integrable representations
- `ali_ch12_wavelets.pdf` - Wavelets and coherent states
- `ali_ch13_discrete_wt.pdf` - Discrete wavelet transforms
- `extracted_pages_351_360.pdf` - Additional reference

---

## Connection to HST

From Glinsky's 2025 paper:

> "The formula for this generating functional, the Heisenberg Scattering Transformation (HST), is derived in Sec. III... This gives the Taylor expansion of Sₚ(q), or the functional Taylor expansion of Sₚ[f(x)], or the S-matrix, or the Mayer Cluster expansion, or the Wigner-Weyl Transformation."

The HST uses:
1. **Wavelet convolution** (localized Fourier)
2. **Complex logarithm** as activation: ln(z) = ln|z| + i·arg(z)
3. **Principal components** give "singularity spectrums"

The rectifier R(z) = i·ln(R₀(z)) appears to be a normalized/regularized version of the complex logarithm that ensures convergence to the real axis.

---

## Next Steps

1. Understand wavelets ψ(x) in HST context (Ali chapters 12-13)
2. Clarify rectifier vs complex-log relationship
3. Implement HST with rectifier as activation
4. Test on synthetic signals to verify convergence properties
