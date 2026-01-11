# Honest Assessment of HST-ROM Implementation

**Date:** January 11, 2026
**Purpose:** Distinguish verified results from claims that need stronger evidence

---

## VERIFIED (Strong Evidence)

### 1. Rectifier R(z) Implementation
- **Claim:** R(z) = i·ln(R₀(z)) where R₀ = -i(w ± √(w²-1)), w = 2z/π
- **Evidence:**
  - Matches Figure 7 test points from Glinsky's paper
  - Test points verified: z = 0.5+0.3j → R = 0.32+0.20j (correct branch)
- **Confidence:** HIGH

### 2. Convergence Rate λ = 2/π
- **Claim:** |Im(R(z))| / |Im(z)| → 2/π ≈ 0.6366
- **Evidence:**
  - Measured λ = 0.6275 on chirp signals
  - Measured λ = 0.5193 on Van der Pol (signal-dependent)
- **Note:** Paper mentions λ ≈ 0.45 for scattering coefficient decay - this is DIFFERENT from Im contraction
- **Confidence:** HIGH for Im contraction, UNCLEAR for scattering decay

### 3. Inverse Rectifier R⁻¹
- **Claim:** R⁻¹(w) = (π/2)·sin(w)
- **Evidence:**
  - Round-trip error: 3e-16 (machine precision)
  - z → R(z) → R⁻¹(R(z)) = z verified
- **Confidence:** HIGH

### 4. Half-Plane Preservation (Sheeted Version)
- **Claim:** Im(z) > 0 → Im(R) > 0, Im(z) < 0 → Im(R) < 0
- **Evidence:**
  - Verified on grid of test points
  - Critical for avoiding branch cut crossings
- **Confidence:** HIGH

### 5. HST + pywt Perfect Reconstruction
- **Claim:** HST forward → HST inverse recovers original signal
- **Evidence:**
  - Reconstruction error: 7e-16 (machine precision)
  - Uses pywt's DWT/IDWT which guarantees perfect reconstruction
- **Confidence:** HIGH

---

## PARTIALLY SUPPORTED (Weak Evidence)

### 1. P Correlates with Energy
- **Claim:** Learned action P corresponds to physical energy/action
- **Evidence:**
  - SHO: r = 0.827
  - Duffing: r = 0.871
  - Pendulum: r = 0.879
- **Problem:** Correlation ≠ equality. We haven't shown:
  - P = f(E) for some monotonic f
  - P has correct units/scaling
  - P is the actual action integral I = ∮ p dq / 2π
- **Confidence:** MEDIUM - correlation exists but functional form unknown

### 2. Learned ω Correlates with True ω
- **Claim:** Learned frequency matches physical frequency
- **Evidence:**
  - Duffing: r = 0.569
  - Pendulum: r = 0.490
- **Problem:** r = 0.5 means only ~25% of variance explained
- **Confidence:** LOW - correlation is weak

### 3. Kapitza Stabilization
- **Claim:** 93.6% improvement in stability
- **Evidence:**
  - Without control: 355° deviation
  - With control: 22.6° deviation
- **Problem:**
  - Only tested one parameter combination
  - Not compared to optimal control baseline
  - Improvement metric is deviation, not Lyapunov stability
- **Confidence:** MEDIUM - effect is real but not rigorously characterized

### 4. Lorenz Wing Detection
- **Claim:** ROM captures attractor structure
- **Evidence:**
  - 53.8% correlation with wing identity
- **Problem:** 53.8% is barely above chance for binary classification
- **Confidence:** LOW - marginal evidence

---

## NOT VERIFIED (Despite Claims)

### 1. ω = f(P) Functional Relationship
- **Claim:** Frequency depends only on action (geodesic property)
- **Evidence:**
  - ω ~ P regression R² ≈ 0 for all systems
- **Reality:** When MLP loss → 0, it outputs constant ω per trajectory. The R² = 0 means there's no functional relationship being learned across different P values.
- **Status:** NOT VERIFIED

### 2. P Conservation Within Trajectories
- **Claim:** Action is conserved along trajectories
- **Evidence:**
  - "P_std ≈ 0" reported
- **Problem:**
  - std = 0 is suspicious - suggests MLP outputting constants
  - Need coefficient of variation, not absolute std
  - Need comparison to theoretical conservation
- **Status:** NOT RIGOROUSLY TESTED

### 3. MLP Learns Singularity Structure
- **Claim:** ReLU kinks align with dynamical singularities
- **Evidence:**
  - Laplacian shows some localization (7.7x ratio)
- **Problem:**
  - No comparison to known singularity locations
  - Localization ratio not compared to null hypothesis
  - Could be random MLP structure, not physics
- **Status:** SPECULATIVE

### 4. Geodesic Motion
- **Claim:** Trajectories are straight lines in (P, Q) space
- **Evidence:**
  - Loss → 0 (but this just means MLP fits training data)
- **Problem:**
  - Fitting training data ≠ learning correct physics
  - No test on held-out data
  - No comparison to true action-angle coordinates
- **Status:** NOT VERIFIED

---

## OPEN QUESTIONS

### 1. λ Discrepancy
- Paper: λ ≈ 0.45 for scattering coefficient decay
- Our measurement: λ ≈ 0.63 for Im contraction
- **Question:** Are these measuring different things? Is one wrong?

### 2. What Does "Invertibility" Mean?
- We have perfect reconstruction with pywt DWT/IDWT
- But HST involves rectifier at each level
- **Question:** Is the rectifier truly "removing" nonlinearity, or just transforming it?

### 3. Bination Effects
- Downsampling by 2 at each level
- **Question:** What information is lost? How does this affect the dynamics learned by ROM?

### 4. Why Does MLP Loss → 0?
- Training loss goes to essentially zero
- **Question:** Is this overfitting? Or is the geodesic structure actually that simple?

### 5. Complex vs Real MLP
- We implemented complex MLP but tested mostly with real
- **Question:** Does complex MLP actually help for these systems?

---

## RECOMMENDATIONS

1. **Design rigorous tests** with pre-specified pass/fail criteria
2. **Compare to ground truth** where analytical solutions exist (SHO, Pendulum)
3. **Use held-out data** to test generalization
4. **Report confidence intervals**, not just point estimates
5. **State null hypotheses** explicitly before testing
