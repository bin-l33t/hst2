"""
Glinsky Clock Demo - Visual Illustration of Quantization Mechanism

This demo shows three key concepts:

1. PHASE MIXING (Linear vs Nonlinear)
   - Linear oscillator: ω constant → no phase mixing (all clocks stay together)
   - Nonlinear oscillator: ω(P) varies → phase spreads uniformly (clocks disperse)

2. ERGODICITY
   - Single clock observed many times ≡ many clocks observed once
   - Uncertainty in P → uncertainty in ω → phase randomizes over time

3. TOPOLOGICAL SELECTION
   - Single-valuedness on circle S¹ requires exp(i·P·2π) = 1
   - Only integer P values satisfy this → discrete spectrum emerges
   - Non-integer P has "topological defect" → suppressed

This is Glinsky's mechanism: dω/dP ≠ 0 + finite ε → phase mixing → quantization
"""

import numpy as np
import matplotlib.pyplot as plt

def glinsky_clock_demo_corrected():
    # --- CONFIGURATION ---
    # We simulate a "Swarm" of 5,000 clocks to represent the probability distribution.
    N_CLOCKS = 5000

    # 1. THE MEASUREMENT (Epistemic Limit)
    # We measure the energy (Action P), but have finite precision.
    # We define P as a narrow Gaussian distribution around P=5.5
    # (Note: 5.5 is NOT an integer, so we expect it to be "unstable")
    P_mean = 5.5
    P_uncertainty = 0.02  # Tiny uncertainty (0.4%)
    P_swarm = np.random.normal(P_mean, P_uncertainty, N_CLOCKS)

    # We know the initial Phase Q exactly (t=0)
    Q_start = np.zeros(N_CLOCKS) # All clocks start at "12 o'clock"

    # --- 2. THE DYNAMICS (Linear vs Nonlinear) ---

    # Case A: Ideal Harmonic Oscillator (Linear)
    # Frequency is constant, regardless of Energy. (Isochronous)
    w_linear = 1.0 * np.ones_like(P_swarm)

    # Case B: Real Pendulum (Nonlinear/Anharmonic)
    # Frequency depends on Energy. Higher Energy = Slower period.
    # This dependency (dw/dP) is the critical mechanism for phase mixing.
    # Approximation: w(P) = w0 - k*P
    w_nonlinear = 1.0 - 0.1 * P_swarm

    # --- SIMULATION (The "Black Box") ---
    # We let the clocks run for a "Long Time" (Delta Tau >> Period)
    T_long = 100.0  # e.g., 100 seconds

    # Calculate Phase Q at T_long
    # Q(t) = w * t
    Q_linear_end = w_linear * T_long
    Q_nonlinear_end = w_nonlinear * T_long

    # Wrap to circle [0, 2pi]
    Q_linear_mod = np.mod(Q_linear_end, 2*np.pi)
    Q_nonlinear_mod = np.mod(Q_nonlinear_end, 2*np.pi)

    # --- PLOTTING PART 1: Phase Mixing ---
    # Create a figure with 3 subplots: Linear, Nonlinear, Single Clock
    fig = plt.figure(figsize=(18, 6))

    # Linear System
    ax1 = fig.add_subplot(131, projection='polar')
    ax1.scatter(Q_linear_mod, np.ones_like(Q_linear_mod)*0.8, s=1, alpha=0.5, c='blue')
    ax1.set_title(f"Linear Oscillator\n(No Phase Mixing)\nInput Action P = {P_mean} (Non-Int)", y=1.1)
    ax1.set_yticks([])

    # Nonlinear System
    ax2 = fig.add_subplot(132, projection='polar')
    ax2.scatter(Q_nonlinear_mod, np.ones_like(Q_nonlinear_mod)*0.8, s=1, alpha=0.5, c='red')
    ax2.set_title(f"Nonlinear Pendulum\n(Ergodic Phase Mixing)\nInput Action P = {P_mean} (Non-Int)", y=1.1)
    ax2.set_yticks([])

    # Single Clock Ergodic Demo
    T_total = 5000  # Long observation time
    N_samples = 2000 # Number of random observations

    # True Action P has slight drift/jitter (Brownian motion in Action)
    t_eval = np.sort(np.random.uniform(0, T_total, N_samples))
    P_drift = 5.5 + np.cumsum(np.random.normal(0, 0.001, N_samples))

    # Frequency depends on P (Nonlinear)
    w_nonlinear_single = 1.0 - 0.1 * P_drift

    # Phase accumulation: Integral of w(t) dt
    Q_accumulated = np.cumsum(w_nonlinear_single * (np.diff(t_eval, prepend=0)))
    Q_mod_single = np.mod(Q_accumulated, 2*np.pi)

    ax3 = fig.add_subplot(133, projection='polar')
    ax3.scatter(Q_mod_single, np.ones_like(Q_mod_single)*0.8, s=5, alpha=0.3, c='purple')
    ax3.set_title(f"Single Nonlinear Clock\n(Observed {N_samples} times over T={T_total})\nAction P(t) = 5.5 (Drifting)", y=1.1)
    ax3.set_yticks([])

    plt.tight_layout()
    plt.savefig('glinsky_clock_phase_mixing.png', dpi=150, bbox_inches='tight')
    print("Saved: glinsky_clock_phase_mixing.png")
    plt.show()

    # --- 3. THE SELECTION RULE (The Fix) ---
    # We test which Actions P are "valid" on a circle.
    # Validity = Single-valuedness (The wave must match itself after 2pi).

    test_Ps = np.linspace(0, 6, 500)
    topological_cost = []

    for P_val in test_Ps:
        # 1. Define the endpoints of the loop
        psi_start = np.exp(1j * P_val * 0)         # Angle = 0
        psi_end   = np.exp(1j * P_val * 2*np.pi)   # Angle = 2pi

        # 2. Calculate the Mismatch (The Topological Defect)
        # If P is integer, mismatch is 0. If P=1.5, mismatch is large.
        defect = np.abs(psi_end - psi_start)**2

        # 3. Convert to "Survival Probability" (Boltzmann weight)
        # States with high defects are exponentially suppressed.
        prob = np.exp(-5 * defect)
        topological_cost.append(prob)

    # Plot the Selection Rule
    plt.figure(figsize=(10, 4))
    plt.plot(test_Ps, topological_cost, color='green', linewidth=2)
    plt.title("The Emergence of Quantization (Topological Selection)", fontsize=14)
    plt.xlabel("Action P (Continuous Input)", fontsize=12)
    plt.ylabel("Observer Confidence", fontsize=12)
    plt.grid(True, alpha=0.3)

    # Highlight Integers
    for i in range(7):
        plt.axvline(x=i, color='red', linestyle='--', alpha=0.3)
        plt.text(i, 1.05, f'n={i}', ha='center', fontsize=10, color='red')

    plt.ylim(0, 1.15)
    plt.tight_layout()
    plt.savefig('glinsky_topological_selection.png', dpi=150, bbox_inches='tight')
    print("Saved: glinsky_topological_selection.png")
    plt.show()

    # --- SUMMARY ---
    print("\n" + "="*70)
    print("GLINSKY CLOCK DEMO - SUMMARY")
    print("="*70)
    print("""
    Three key insights demonstrated:

    1. PHASE MIXING (Plot 1, panels 1-2)
       - Linear oscillator: w = constant → phases stay clustered
       - Nonlinear oscillator: w(P) varies → phases spread uniformly
       - Key requirement: dw/dP != 0 (anharmonicity)

    2. ERGODICITY (Plot 1, panel 3)
       - Single clock with drifting P → phase fills circle uniformly
       - Ensemble average = Time average (ergodic hypothesis)
       - Finite precision in P → eventual phase randomization

    3. TOPOLOGICAL SELECTION (Plot 2)
       - Single-valuedness: psi(Q + 2pi) = psi(Q)
       - Requires exp(i*P*2pi) = 1 → P must be integer
       - Non-integer P has "topological defect" → suppressed
       - This is Bohr-Sommerfeld quantization from topology!

    KEY FORMULA:
        Delta_tau_critical ~ 2*pi / (|dw/dP| * epsilon)

    When observation time exceeds this, phase is effectively random,
    and only integer actions "survive" the topological constraint.
    """)


if __name__ == "__main__":
    glinsky_clock_demo_corrected()
