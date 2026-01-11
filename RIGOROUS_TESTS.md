# Rigorous Tests for HST-ROM Implementation

**Date:** January 11, 2026
**Purpose:** Pre-specified tests with clear pass/fail criteria

---

## Test Philosophy

1. **State hypothesis BEFORE running test**
2. **Define pass/fail thresholds BEFORE seeing results**
3. **Use held-out data** for generalization tests
4. **Compare to ground truth** where analytical solutions exist
5. **Report confidence intervals**, not just point estimates

---

## TEST 1: Action Recovery (SHO)

### Background
For Simple Harmonic Oscillator, the action integral is analytically known:
```
I = E / ω₀
```
where E is energy and ω₀ is the natural frequency.

### Hypothesis
The learned action P should be a monotonic function of true action I.

### Protocol
1. Generate 50 SHO trajectories with E ∈ [0.5, 5.0]
2. Split: 40 training, 10 test (held-out)
3. Train HJB-MLP on training set
4. Compute learned P for all trajectories
5. Compute true I = E/ω₀ for all trajectories
6. Measure Pearson correlation r(P, I) on TEST SET

### Pass/Fail Criteria
| Result | Interpretation |
|--------|----------------|
| **PASS** | r > 0.99 on test set |
| **MARGINAL** | 0.95 < r < 0.99 |
| **FAIL** | r < 0.95 |

### Rationale
SHO is the simplest possible case. If we can't recover action here, the method doesn't work.

---

## TEST 2: P Conservation Within Trajectory

### Background
For integrable Hamiltonian systems, action is an adiabatic invariant - it should be approximately constant along each trajectory.

### Hypothesis
P should have low variation within each trajectory (coefficient of variation < 5%).

### Protocol
1. Generate 30 Duffing trajectories at different energies
2. For each trajectory of length N:
   - Compute P at each time step (sliding window HST)
   - Calculate coefficient of variation: CV = std(P) / |mean(P)|
3. Report fraction of trajectories with CV < threshold

### Pass/Fail Criteria
| Result | Interpretation |
|--------|----------------|
| **PASS** | >90% of trajectories have CV < 0.05 |
| **MARGINAL** | >80% have CV < 0.05, or >90% have CV < 0.10 |
| **FAIL** | <80% have CV < 0.10 |

### Rationale
If P isn't conserved within a trajectory, it's not capturing the action.

---

## TEST 3: ω(P) Functional Relationship (Pendulum)

### Background
For the pendulum in libration regime, frequency depends on energy:
```
ω(E) = π / (2 * K(k))  where k² = (1+E)/2
```
This gives a non-trivial ω(I) curve we can compare against.

### Hypothesis
The learned ω should match the theoretical ω(E) curve.

### Protocol
1. Generate 40 pendulum trajectories with E ∈ [-0.8, 0.8] (libration regime)
2. Split: 30 training, 10 test
3. Train HJB-MLP
4. For each test trajectory:
   - Compute learned (P, ω)
   - Compute theoretical ω from E using elliptic integral
5. Fit linear model: ω_learned = a * ω_theoretical + b
6. Report R² on test set

### Pass/Fail Criteria
| Result | Interpretation |
|--------|----------------|
| **PASS** | R² > 0.90 on test set |
| **MARGINAL** | 0.70 < R² < 0.90 |
| **FAIL** | R² < 0.70 |

### Note
We test ω_learned vs ω_theoretical directly (not ω vs P), because P may be a nonlinear transform of I.

---

## TEST 4: Synthetic Signal Reconstruction

### Background
HST should be invertible - we should be able to reconstruct signals from HST coefficients.

### Hypothesis
Reconstruction error should be at machine precision for the wavelet transform, and bounded for rectified transform.

### Protocol
1. Generate test signals:
   - Chirp: sin(t + 0.1*t²)
   - Sum of sines: sin(t) + 0.5*sin(2.3*t)
   - SHO trajectory
2. Apply HST forward transform
3. Apply HST inverse transform
4. Measure relative error: ||x - x_rec|| / ||x||

### Pass/Fail Criteria (Wavelet only, no rectifier)
| Result | Interpretation |
|--------|----------------|
| **PASS** | Error < 1e-10 (machine precision) |
| **FAIL** | Error > 1e-6 |

### Pass/Fail Criteria (Full HST with rectifier)
| Result | Interpretation |
|--------|----------------|
| **PASS** | Error < 0.01 (1%) |
| **MARGINAL** | 0.01 < Error < 0.05 |
| **FAIL** | Error > 0.10 (10%) |

### Rationale
If we can't reconstruct signals, we're losing information somewhere.

---

## TEST 5: Generalization to Unseen Energies

### Background
A good representation should generalize - interpolating to energies not seen during training.

### Hypothesis
MLP trained on sparse energy grid should predict accurately at intermediate energies.

### Protocol
1. Generate Duffing trajectories at E = [0.5, 1.5, 2.5, 3.5, 4.5] (5 energies)
2. Train HJB-MLP
3. Generate test trajectories at E = [1.0, 2.0, 3.0, 4.0] (4 intermediate energies)
4. Evaluate P~E correlation on test set only

### Pass/Fail Criteria
| Result | Interpretation |
|--------|----------------|
| **PASS** | r > 0.95 on interpolated test set |
| **MARGINAL** | 0.85 < r < 0.95 |
| **FAIL** | r < 0.85 |

---

## TEST 6: Comparison with Analytical Action-Angle

### Background
For SHO, we can compute exact action-angle coordinates:
```
I = (q² + p²) / 2    (action)
θ = arctan(p/q)       (angle)
```

### Hypothesis
Learned (P, Q) should be monotonically related to (I, θ).

### Protocol
1. Generate SHO trajectories
2. Compute true (I, θ) at each point
3. Train HJB-MLP to get (P, Q)
4. Compute correlation matrix between (P, Q) and (I, θ)
5. Check that P correlates with I, Q correlates with θ (modulo 2π)

### Pass/Fail Criteria
| Result | Interpretation |
|--------|----------------|
| **PASS** | |r(P, I)| > 0.99 AND circular correlation |r(Q, θ)| > 0.95 |
| **MARGINAL** | |r(P, I)| > 0.95 AND |r(Q, θ)| > 0.80 |
| **FAIL** | Either correlation below marginal threshold |

---

## Summary Table

| Test | System | Metric | Pass | Marginal | Fail |
|------|--------|--------|------|----------|------|
| 1 | SHO | r(P, I) on test | >0.99 | 0.95-0.99 | <0.95 |
| 2 | Duffing | %traj CV<0.05 | >90% | 80-90% | <80% |
| 3 | Pendulum | R²(ω, ω_true) | >0.90 | 0.70-0.90 | <0.70 |
| 4a | Synthetic | Reconstruction (wavelet) | <1e-10 | - | >1e-6 |
| 4b | Synthetic | Reconstruction (HST) | <1% | 1-5% | >10% |
| 5 | Duffing | r(P,E) interpolation | >0.95 | 0.85-0.95 | <0.85 |
| 6 | SHO | r(P,I) & r(Q,θ) | >0.99 & >0.95 | >0.95 & >0.80 | below |

---

## Implementation Notes

### Test Independence
Each test should be runnable independently with:
```bash
python rigorous_tests.py --test 1
python rigorous_tests.py --test all
```

### Reproducibility
- Set random seed at start of each test
- Report seed in results
- Results should be reproducible with same seed

### Reporting
For each test, report:
1. Pass/Marginal/Fail status
2. Actual metric value
3. 95% confidence interval (via bootstrap)
4. Number of samples used

### Example Output Format
```
TEST 1: Action Recovery (SHO)
=============================
Status: PASS
Metric: r(P, I) = 0.9923
95% CI: [0.9891, 0.9948]
Samples: 10 test trajectories
Seed: 42
```

---

## What Success Looks Like

If HST-ROM is working correctly:
- **TEST 1 PASS**: Method captures action in simplest case
- **TEST 2 PASS**: P is actually conserved (adiabatic invariant)
- **TEST 3 PASS**: ω(P) relationship is meaningful
- **TEST 4 PASS**: No information loss in transform
- **TEST 5 PASS**: Generalization works
- **TEST 6 PASS**: Coordinates match analytical solution

If we get FAIL on TEST 1 or TEST 4, there's a fundamental problem.
If we get FAIL on TEST 2 or TEST 6, the geodesic claims are not supported.
If we get FAIL on TEST 3 or TEST 5, the method may work but has limited utility.
