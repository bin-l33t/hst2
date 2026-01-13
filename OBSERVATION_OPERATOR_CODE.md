# Observation Operator Code - Exact Implementation

**Purpose:** Document the exact observation model so ChatGPT can verify it's not tautological.

---

## The Scenario

1. System starts at phase Q_original (varies across samples)
2. Observer receives window starting at UNKNOWN time t0 ~ Unif[0, Δτ]
3. Features extracted from observed window
4. Test: Can features predict Q_original?

---

## Key Definitions

### Q_original
The phase of the system at t=0 of the original trajectory.

```python
# Q_original is defined BEFORE observation
# It varies uniformly across samples
Q_original = np.random.uniform(0, 2 * np.pi)
```

### Q_observed
The phase at the (unknown) start of the observation window.

```python
# Q_observed = Q_original + offset_phase (mod 2π)
# where offset_phase comes from the random time offset
offset_phase = (omega * offset_samples * dt) % (2 * np.pi)
Q_observed = (Q_original + offset_phase) % (2 * np.pi)
```

### The Offset
```python
# Observation starts at unknown time offset
delta_tau = ratio * T_period  # e.g., ratio = 5.0 means Δτ = 5T
max_offset_samples = int(delta_tau / dt)

# Random offset within [0, Δτ]
offset_samples = np.random.randint(0, max_offset_samples)
```

---

## Full observe() Function

```python
def observe(z, t, T_period, mode='time_shift', delta_tau=None, seed=None):
    """
    Apply observation operator that makes phase unidentifiable.

    This is the PHYSICAL mechanism of coarse-graining:
    - The trajectory z(t) still contains phase information
    - But the observation process makes it impossible to determine

    Parameters
    ----------
    z : array
        Complex trajectory z = q + ip
    t : array
        Time array
    T_period : float
        Natural period of the system
    mode : str
        'time_shift': Random shift by t0 ~ Unif[0, T)
        'strobe': Sample at intervals delta_tau
        'cycle_average': Average over windows >> T
    delta_tau : float
        For 'strobe' mode: sampling interval
        For 'cycle_average': window size
    seed : int
        Random seed

    Returns
    -------
    z_observed : array
        Degraded observation
    metadata : dict
        Information about the observation process
    """
    if seed is not None:
        np.random.seed(seed)

    dt = t[1] - t[0]
    n = len(z)

    if mode == 'time_shift':
        # Random time shift: learner doesn't know when observation started
        shift_samples = np.random.randint(0, max(1, int(T_period / dt)))
        z_observed = np.roll(z, shift_samples)

        # Learner only sees a WINDOW of the trajectory
        window_size = min(512, len(z_observed))
        start = np.random.randint(0, max(1, len(z_observed) - window_size))
        z_observed = z_observed[start:start + window_size]

        return z_observed, {
            'mode': 'time_shift',
            'shift_samples': shift_samples,
            'shift_phase': 2 * np.pi * shift_samples * dt / T_period,
            'T_period': T_period,
            'window_start': start
        }

    elif mode == 'strobe':
        # Sample at intervals delta_tau
        if delta_tau is None:
            delta_tau = T_period

        sample_interval = max(1, int(delta_tau / dt))
        z_sampled = z[::sample_interval]

        # Pad/truncate to fixed size
        target_size = 512
        if len(z_sampled) < target_size:
            z_observed = np.interp(
                np.linspace(0, len(z_sampled)-1, target_size),
                np.arange(len(z_sampled)),
                z_sampled
            )
        else:
            z_observed = z_sampled[:target_size]

        return z_observed, {
            'mode': 'strobe',
            'delta_tau': delta_tau,
            'delta_tau_over_T': delta_tau / T_period,
            'sample_interval': sample_interval
        }

    elif mode == 'cycle_average':
        # Average over windows of size W
        if delta_tau is None:
            delta_tau = T_period

        window_samples = max(1, int(delta_tau / dt))
        n_windows = len(z) // window_samples

        z_averaged = []
        for i in range(n_windows):
            window = z[i * window_samples:(i + 1) * window_samples]
            z_averaged.append(np.mean(window))
        z_averaged = np.array(z_averaged)

        target_size = 512
        z_observed = np.interp(
            np.linspace(0, len(z_averaged)-1, target_size),
            np.arange(len(z_averaged)),
            z_averaged
        )

        return z_observed, {
            'mode': 'cycle_average',
            'window_size': delta_tau,
            'window_over_T': delta_tau / T_period
        }
```

---

## The Dataset Generation (Critical Part)

This is from `generate_dataset_physical()` in `test_observation_operator.py`:

```python
def generate_dataset_physical(n_energies=10, n_samples_per_E=20,
                               delta_tau_ratios=[0.1, 0.5, 1.0, 2.0, 5.0],
                               seed=42):
    """
    Generate dataset with PHYSICAL observation degradation.

    Key: The learner sees features from z_obs, but we ask:
    "Can you predict Q_original (not Q_observed)?"
    """
    np.random.seed(seed)
    E_values = np.linspace(-0.8, 0.7, n_energies)
    pendulum = PendulumOscillator()

    datasets = {ratio: {'features': [], 'P_true': [], 'Q_original': [],
                        'Q_observed': [], 'E': [], 'T_period': [], 'offset_phase': []}
                for ratio in delta_tau_ratios}

    for E_target in E_values:
        omega = pendulum_omega(E_target)
        T_period = 2 * np.pi / omega
        dt = 0.01

        if E_target >= 0.99:
            continue

        for sample_idx in range(n_samples_per_E):
            # DIFFERENT initial phase for each sample
            Q_original = np.random.uniform(0, 2 * np.pi)

            # Initial condition at turning point (Q=0 convention)
            q_max = np.arccos(-E_target) if E_target > -1 else np.pi * 0.95
            q0 = q_max
            p0 = 0.0

            # Generate trajectory starting at Q=0
            T_sim = max(30 * T_period, 150)
            t, q, p, z, E_actual = simulate_hamiltonian(pendulum, q0, p0, T=T_sim, dt=dt)

            # P is constant along orbit
            P_true = np.mean(np.abs(z))

            # The "original observation time" corresponds to Q_original
            t0_original = (Q_original / omega) % T_period
            idx_original = int(t0_original / dt)

            for ratio in delta_tau_ratios:
                # Observation time uncertainty
                delta_tau = ratio * T_period
                max_offset_samples = max(1, int(delta_tau / dt))

                # Random ADDITIONAL offset within [0, delta_tau]
                offset_samples = np.random.randint(0, max_offset_samples)

                # Total start index = original + offset
                start_idx = (idx_original + offset_samples) % max(1, len(z) - 600)

                window_size = 512
                if start_idx + window_size >= len(z):
                    start_idx = max(0, len(z) - window_size - 1)

                z_obs = z[start_idx:start_idx + window_size]

                # Offset phase (the unknown part)
                offset_phase = (omega * offset_samples * dt) % (2 * np.pi)

                # Q_observed = Q_original + offset_phase (mod 2pi)
                Q_observed = (Q_original + offset_phase) % (2 * np.pi)

                # The learner doesn't know the offset!
                # Features come from z_obs which starts at Q_observed
                feat = extract_features(z_obs, J=3, wavelet_name='db8')

                datasets[ratio]['features'].append(feat)
                datasets[ratio]['P_true'].append(P_true)
                datasets[ratio]['Q_original'].append(Q_original)  # TARGET
                datasets[ratio]['Q_observed'].append(Q_observed)   # In features
                datasets[ratio]['offset_phase'].append(offset_phase)
                datasets[ratio]['E'].append(E_actual)
                datasets[ratio]['T_period'].append(T_period)

    return datasets
```

---

## Features Used

From `hst.py:extract_features()`:

```python
def extract_features(z, J=3, wavelet_name='db8'):
    """
    Extract HST features from complex trajectory segment.

    INCLUDES BOTH MAGNITUDE AND PHASE FEATURES (21 dimensions total):

    For each of J=3 wavelet levels:
    - np.mean(np.abs(cD_complex))     # Magnitude mean
    - np.std(np.abs(cD_complex))      # Magnitude std
    - np.mean(np.cos(phases))         # Phase (circular mean, real)
    - np.mean(np.sin(phases))         # Phase (circular mean, imag)
    - np.std(phases)                  # Phase dispersion

    Plus final approximation and direct phase from z.
    """
```

The features INCLUDE phase information from the observed window. They CAN predict Q_observed.

---

## Why It's Not Tautological

The test asks: **Can features predict Q_original?**

1. Features encode Q_observed (the phase at observation start)
2. Q_original = Q_observed - offset_phase
3. offset_phase ~ Unif[0, 2π·Δτ/T]

At fine scales (Δτ << T):
- offset_phase is small
- Q_observed ≈ Q_original
- Features CAN predict Q_original

At coarse scales (Δτ >> T):
- offset_phase spans [0, 2π] multiple times
- Q_observed is uniformly distributed relative to Q_original
- Features CANNOT predict Q_original
- But they CAN still predict Q_observed (it's encoded in the features)

---

## Results Recap

```
Delta_tau/T    r(P)    r(Q_original)    r(Q_observed)
0.1            0.989   0.973            0.988
0.5            0.989   0.503            0.992
1.0            0.985   0.056            0.994
2.0            0.985   0.083            0.993
5.0            0.987   -0.109           0.992
10.0           0.985   -0.026           0.994
```

- **r(Q_observed) stays ~0.99**: Features encode the phase they see
- **r(Q_original) drops to ~0**: But original phase is unrecoverable
- **r(P) stays ~0.99**: Action is constant along orbit

This is NOT dropping phase from features. It's showing that the observation process makes the original phase unrecoverable.
