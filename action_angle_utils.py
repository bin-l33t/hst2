"""
Action-Angle Utility Functions

Common utilities for action-angle coordinate validation tests.
These handle the tricky parts: angular distances, branch cuts, wrapping.
"""

import numpy as np


def angular_distance(Q1, Q2):
    """
    Proper angular distance on circle.
    Returns value in [0, π].

    This is the geodesic distance on S¹, handling wraparound correctly.
    """
    diff = (Q1 - Q2) % (2 * np.pi)
    return np.minimum(diff, 2 * np.pi - diff)


def wrap_to_2pi(Q):
    """Wrap angle to [0, 2π)"""
    return Q % (2 * np.pi)


def wrap_to_pi(Q):
    """Wrap angle to [-π, π)"""
    return ((Q + np.pi) % (2 * np.pi)) - np.pi


def unwrap_angle(Q_series):
    """
    Unwrap angle series for slope calculation.
    Removes discontinuities at 2π boundaries.
    """
    return np.unwrap(Q_series)


def safe_points_mask(q, q_max, epsilon=1e-3):
    """
    Return mask for points safely away from turning points.
    Used to avoid branch cut singularities in derivative tests.

    Parameters:
    - q: position values
    - q_max: maximum amplitude (turning point)
    - epsilon: safety margin (default 0.1%)

    Returns:
    - Boolean mask, True where |q| < (1-ε)·q_max
    """
    return np.abs(q) <= (1 - epsilon) * q_max


def circular_mean(angles):
    """
    Compute mean of angles on circle.

    Standard mean fails for angles near 0/2π boundary.
    Use vector average instead.
    """
    x = np.mean(np.cos(angles))
    y = np.mean(np.sin(angles))
    return np.arctan2(y, x) % (2 * np.pi)


def circular_std(angles):
    """
    Compute circular standard deviation.

    Uses resultant length R = |mean(exp(iθ))|.
    σ_circular = √(-2·ln(R))

    Returns value in [0, ∞), with 0 meaning all angles identical.
    """
    x = np.mean(np.cos(angles))
    y = np.mean(np.sin(angles))
    R = np.sqrt(x**2 + y**2)

    # Clamp R to avoid log(0)
    R = np.clip(R, 1e-10, 1.0)

    return np.sqrt(-2 * np.log(R))


def is_uniform_on_circle(angles, n_bins=12, significance=0.05):
    """
    Test if angles are uniformly distributed on circle.
    Uses chi-squared test against uniform distribution.

    Parameters:
    - angles: array of angles in [0, 2π)
    - n_bins: number of bins for histogram
    - significance: p-value threshold

    Returns:
    - is_uniform: True if consistent with uniform
    - p_value: actual p-value from chi-squared test
    """
    from scipy import stats

    # Bin the angles
    bins = np.linspace(0, 2*np.pi, n_bins + 1)
    observed, _ = np.histogram(wrap_to_2pi(angles), bins=bins)

    # Expected counts for uniform
    expected = np.full(n_bins, len(angles) / n_bins)

    # Chi-squared test
    chi2, p_value = stats.chisquare(observed, expected)

    return p_value > significance, p_value


if __name__ == "__main__":
    # Quick self-test
    print("Testing action_angle_utils.py...")

    # Angular distance
    assert abs(angular_distance(0.1, 0.2) - 0.1) < 1e-10
    assert abs(angular_distance(0.1, 2*np.pi - 0.1) - 0.2) < 1e-10
    assert abs(angular_distance(0, np.pi) - np.pi) < 1e-10
    print("  angular_distance: OK")

    # Wrap
    assert abs(wrap_to_2pi(2.5*np.pi) - 0.5*np.pi) < 1e-10
    assert abs(wrap_to_2pi(-0.5*np.pi) - 1.5*np.pi) < 1e-10
    print("  wrap_to_2pi: OK")

    # Circular mean
    angles = np.array([0.1, -0.1, 0.05, -0.05])  # Near 0
    mean = circular_mean(angles)
    assert angular_distance(mean, 0) < 0.1
    print("  circular_mean: OK")

    # Uniform test
    uniform_angles = np.random.uniform(0, 2*np.pi, 1000)
    is_unif, p = is_uniform_on_circle(uniform_angles)
    print(f"  is_uniform_on_circle: p={p:.3f}, uniform={is_unif}")

    print("\nAll utility tests passed!")
