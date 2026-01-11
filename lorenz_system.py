"""
Lorenz System - Chaos Control Benchmark

The Lorenz system is the canonical example of deterministic chaos.
We use it to validate that HST-ROM can capture the topology of
chaotic attractors and enable OGY-style control.

Lorenz equations:
    ẋ = σ(y - x)
    ẏ = x(ρ - z) - y
    ż = xy - βz

Standard parameters: σ=10, ρ=28, β=8/3

Properties:
- Chaotic for these parameters (positive Lyapunov exponent λ₁ ≈ 0.9)
- Two unstable fixed points C± = (±√(β(ρ-1)), ±√(β(ρ-1)), ρ-1)
- Strange attractor with Hausdorff dimension ≈ 2.06
- Contains infinitely many unstable periodic orbits (UPOs)

References:
- Lorenz, E.N. (1963) "Deterministic Nonperiodic Flow"
- Ott, Grebogi, Yorke (1990) "Controlling Chaos"
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def lorenz_dynamics(t, state, sigma=10.0, rho=28.0, beta=8/3):
    """
    Lorenz system equations.

    ẋ = σ(y - x)
    ẏ = x(ρ - z) - y
    ż = xy - βz
    """
    x, y, z = state
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]


def lorenz_jacobian(state, sigma=10.0, rho=28.0, beta=8/3):
    """Jacobian matrix of Lorenz system."""
    x, y, z = state
    return np.array([
        [-sigma, sigma, 0],
        [rho - z, -1, -x],
        [y, x, -beta]
    ])


def simulate_lorenz(x0, T, dt, sigma=10.0, rho=28.0, beta=8/3,
                    rtol=1e-10, atol=1e-12):
    """
    Simulate Lorenz system.

    Parameters
    ----------
    x0 : array-like
        Initial state [x, y, z]
    T : float
        Total simulation time
    dt : float
        Time step for output
    sigma, rho, beta : float
        Lorenz parameters

    Returns
    -------
    t : array
        Time points
    trajectory : array (N, 3)
        State trajectory [x, y, z]
    z_xy : array (complex)
        Complex embedding z = x + i*y
    z_xz : array (complex)
        Complex embedding z = x + i*z
    """
    t_span = (0, T)
    t_eval = np.arange(0, T, dt)

    sol = solve_ivp(
        lorenz_dynamics, t_span, x0,
        args=(sigma, rho, beta),
        t_eval=t_eval,
        method='RK45', rtol=rtol, atol=atol
    )

    trajectory = sol.y.T  # Shape (N, 3)

    # Complex embeddings for HST
    z_xy = trajectory[:, 0] + 1j * trajectory[:, 1]
    z_xz = trajectory[:, 0] + 1j * trajectory[:, 2]

    return sol.t, trajectory, z_xy, z_xz


def find_fixed_points(sigma=10.0, rho=28.0, beta=8/3):
    """
    Find fixed points of Lorenz system.

    For ρ > 1:
    - Origin (0, 0, 0) - always unstable (saddle)
    - C+ = (+√(β(ρ-1)), +√(β(ρ-1)), ρ-1) - unstable for ρ > 24.74
    - C- = (-√(β(ρ-1)), -√(β(ρ-1)), ρ-1) - unstable for ρ > 24.74
    """
    origin = np.array([0.0, 0.0, 0.0])

    if rho > 1:
        sqrt_term = np.sqrt(beta * (rho - 1))
        C_plus = np.array([sqrt_term, sqrt_term, rho - 1])
        C_minus = np.array([-sqrt_term, -sqrt_term, rho - 1])
        return {'origin': origin, 'C+': C_plus, 'C-': C_minus}
    else:
        return {'origin': origin}


def compute_lyapunov_exponent(trajectory, dt, sigma=10.0, rho=28.0, beta=8/3,
                               warmup=1000):
    """
    Estimate largest Lyapunov exponent from trajectory.

    For standard Lorenz: λ₁ ≈ 0.9

    Uses the standard algorithm:
    1. Follow trajectory
    2. Evolve infinitesimal perturbation using linearized dynamics
    3. Periodically renormalize and accumulate log(stretch)
    """
    n_steps = len(trajectory) - 1
    if n_steps < warmup + 100:
        return np.nan

    # Initialize perturbation
    delta = np.array([1e-10, 0, 0])

    lyap_sum = 0
    n_renorm = 0

    for i in range(warmup, n_steps):
        J = lorenz_jacobian(trajectory[i], sigma, rho, beta)
        delta = delta + dt * J @ delta

        # Renormalize every 10 steps
        if i % 10 == 0:
            norm = np.linalg.norm(delta)
            if norm > 1e-15:
                lyap_sum += np.log(norm)
                delta = delta / norm
                n_renorm += 1

    if n_renorm > 0:
        lyap_exponent = lyap_sum / (n_renorm * 10 * dt)
    else:
        lyap_exponent = np.nan

    return lyap_exponent


def find_poincare_crossings(trajectory, z_threshold=27.0, direction='up'):
    """
    Find Poincaré section crossings at z = z_threshold.

    The Lorenz attractor crosses z ≈ 27 frequently as it
    switches between the two wings.

    Parameters
    ----------
    trajectory : array (N, 3)
        State trajectory
    z_threshold : float
        z-value for Poincaré section
    direction : str
        'up' for z increasing, 'down' for z decreasing

    Returns
    -------
    crossings : list of dicts
        Each contains 'time_idx', 'state', 'wing'
    """
    z = trajectory[:, 2]
    x = trajectory[:, 0]

    crossings = []
    for i in range(1, len(z)):
        crossed = False
        if direction == 'up' and z[i-1] < z_threshold <= z[i]:
            crossed = True
        elif direction == 'down' and z[i-1] > z_threshold >= z[i]:
            crossed = True

        if crossed:
            # Linear interpolation for precise crossing
            alpha = (z_threshold - z[i-1]) / (z[i] - z[i-1] + 1e-15)
            crossing_state = trajectory[i-1] + alpha * (trajectory[i] - trajectory[i-1])
            crossing_time = i - 1 + alpha

            # Determine which wing (by sign of x)
            wing = 'right' if crossing_state[0] > 0 else 'left'

            crossings.append({
                'time_idx': crossing_time,
                'state': crossing_state,
                'wing': wing,
                'x': crossing_state[0],
                'y': crossing_state[1]
            })

    return crossings


def identify_upo_candidates(crossings, max_period=4, tolerance=1.0):
    """
    Find approximate UPOs by looking for near-returns in Poincaré section.

    A period-n UPO appears as a point that returns close to itself
    after n crossings.

    Parameters
    ----------
    crossings : list
        Poincaré section crossings
    max_period : int
        Maximum period to search for
    tolerance : float
        Distance threshold for "close" return

    Returns
    -------
    upo_candidates : list of dicts
        Each contains 'period', 'start_idx', 'state', 'return_distance'
    """
    states = np.array([c['state'] for c in crossings])
    n_crossings = len(states)

    upo_candidates = []

    for period in range(1, max_period + 1):
        for i in range(n_crossings - period):
            dist = np.linalg.norm(states[i] - states[i + period])
            if dist < tolerance:
                upo_candidates.append({
                    'period': period,
                    'start_idx': i,
                    'state': states[i],
                    'return_distance': dist,
                    'crossing_idx': i
                })

    # Sort by return distance
    upo_candidates.sort(key=lambda x: x['return_distance'])

    return upo_candidates


def extract_upo_window(trajectory, upo, crossings, dt, window_before=50, window_after=50):
    """
    Extract trajectory segment around a UPO.

    Parameters
    ----------
    trajectory : array
        Full trajectory
    upo : dict
        UPO candidate from identify_upo_candidates
    crossings : list
        Poincaré crossings
    dt : float
        Time step
    window_before, window_after : int
        Number of time steps before/after crossing

    Returns
    -------
    upo_segment : array
        Trajectory segment around UPO
    """
    crossing_idx = int(crossings[upo['start_idx']]['time_idx'])
    start_idx = max(0, crossing_idx - window_before)
    end_idx = min(len(trajectory), crossing_idx + window_after)

    return trajectory[start_idx:end_idx]


def test_lorenz_basics():
    """Test basic Lorenz system properties."""
    print("=" * 60)
    print("LORENZ SYSTEM - BASIC TESTS")
    print("=" * 60)

    sigma, rho, beta = 10.0, 28.0, 8/3

    # Find fixed points
    fps = find_fixed_points(sigma, rho, beta)
    print("\nFixed points:")
    for name, fp in fps.items():
        print(f"  {name}: {fp}")

    # Simulate
    print("\nSimulating...")
    x0 = [1.0, 1.0, 1.0]
    T = 100.0
    dt = 0.01

    t, trajectory, z_xy, z_xz = simulate_lorenz(x0, T, dt, sigma, rho, beta)

    print(f"  Trajectory shape: {trajectory.shape}")
    print(f"  x range: [{trajectory[:, 0].min():.2f}, {trajectory[:, 0].max():.2f}]")
    print(f"  y range: [{trajectory[:, 1].min():.2f}, {trajectory[:, 1].max():.2f}]")
    print(f"  z range: [{trajectory[:, 2].min():.2f}, {trajectory[:, 2].max():.2f}]")

    # Lyapunov exponent
    print("\nComputing Lyapunov exponent...")
    lyap = compute_lyapunov_exponent(trajectory, dt, sigma, rho, beta)
    print(f"  λ₁ = {lyap:.3f} (expected ≈ 0.9)")

    # Poincaré section
    print("\nFinding Poincaré crossings...")
    crossings = find_poincare_crossings(trajectory)
    print(f"  Found {len(crossings)} crossings")

    # Count wings
    n_right = sum(1 for c in crossings if c['wing'] == 'right')
    n_left = len(crossings) - n_right
    print(f"  Right wing: {n_right}, Left wing: {n_left}")

    # Find UPOs
    print("\nSearching for UPO candidates...")
    upos = identify_upo_candidates(crossings, max_period=4, tolerance=2.0)
    print(f"  Found {len(upos)} candidates")

    for period in range(1, 5):
        period_upos = [u for u in upos if u['period'] == period]
        if period_upos:
            best = min(period_upos, key=lambda x: x['return_distance'])
            print(f"  Period-{period}: {len(period_upos)} candidates, "
                  f"best return distance = {best['return_distance']:.3f}")

    return trajectory, crossings, upos


def plot_lorenz_attractor(trajectory, crossings=None, upos=None, title="Lorenz Attractor"):
    """Create visualization of Lorenz attractor."""
    fig = plt.figure(figsize=(14, 5))

    # 3D view
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
             'b-', alpha=0.3, linewidth=0.3)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('3D View')

    # Mark fixed points
    fps = find_fixed_points()
    for name, fp in fps.items():
        marker = 'ro' if name == 'origin' else 'g*'
        ax1.scatter(*fp, s=100, c='red' if name == 'origin' else 'green')

    # X-Z projection
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(trajectory[:, 0], trajectory[:, 2], 'b-', alpha=0.2, linewidth=0.3)
    ax2.set_xlabel('x')
    ax2.set_ylabel('z')
    ax2.set_title('X-Z Projection')

    # Poincaré section
    ax3 = fig.add_subplot(1, 3, 3)
    if crossings:
        x_cross = [c['x'] for c in crossings]
        y_cross = [c['y'] for c in crossings]
        colors = ['blue' if c['wing'] == 'right' else 'red' for c in crossings]
        ax3.scatter(x_cross, y_cross, c=colors, s=5, alpha=0.5)
        ax3.set_xlabel('x at crossing')
        ax3.set_ylabel('y at crossing')
        ax3.set_title('Poincaré Section (z=27)')

        if upos:
            # Mark best UPO
            best_upo = min(upos, key=lambda x: x['return_distance'])
            ax3.scatter(best_upo['state'][0], best_upo['state'][1],
                       c='green', s=200, marker='*', label=f"Period-{best_upo['period']} UPO")
            ax3.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('lorenz_attractor.png', dpi=150)
    plt.close()
    print("Saved: lorenz_attractor.png")


if __name__ == "__main__":
    trajectory, crossings, upos = test_lorenz_basics()
    plot_lorenz_attractor(trajectory, crossings, upos)
