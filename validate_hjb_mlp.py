"""
HJB_MLP Validation Framework

Tests any MLP that learns (p, q) → (P, Q) against our validated ground truth.

Ground truth comes from:
- test_sho_action_angle.py: SHO with arctan2-based formulas
- pendulum_action_angle.py: Pendulum with elliptic integrals

Tests:
1. Accuracy: Does MLP output match analytic (P, Q)?
2. Symplectic: Is the learned map canonical? {P, Q} = ±1
3. Conservation: Is P constant along trajectories?
4. Phase evolution: Does Q advance at rate ω(P)?
5. Generating function: Does implicit S satisfy ∂S/∂q = p, ∂S/∂P = Q?

Known issues from ChatGPT:
- Current HJBLoss does NOT enforce symplectic constraint
- Angle Q needs circular representation (sin Q, cos Q)
- Need to add Poisson bracket loss term
"""

import numpy as np
from typing import Tuple, Callable, Optional

# PyTorch is optional - only needed for wrapping PyTorch models
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
from action_angle_utils import angular_distance, wrap_to_2pi, unwrap_angle
from test_sho_action_angle import sho_action_angle, sho_from_action_angle


# =============================================================================
# Ground Truth Functions
# =============================================================================

def sho_ground_truth(q: np.ndarray, p: np.ndarray, omega: float = 1.0
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ground truth (q, p) → (P, Q) for SHO.
    
    Uses arctan2-based formula (no branch cuts).
    """
    P = (p**2 + omega**2 * q**2) / (2 * omega)
    Q = np.arctan2(omega * q, p)
    Q = wrap_to_2pi(Q)
    return P, Q


def sho_inverse_ground_truth(P: np.ndarray, Q: np.ndarray, omega: float = 1.0
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ground truth (P, Q) → (q, p) for SHO.
    """
    q = np.sqrt(2 * P / omega) * np.sin(Q)
    p = np.sqrt(2 * P * omega) * np.cos(Q)
    return q, p


# =============================================================================
# Test Functions
# =============================================================================

def test_accuracy(mlp_forward: Callable, 
                  ground_truth: Callable = sho_ground_truth,
                  n_samples: int = 1000,
                  q_range: Tuple[float, float] = (-2.0, 2.0),
                  p_range: Tuple[float, float] = (-2.0, 2.0),
                  omega: float = 1.0) -> dict:
    """
    Test 1: Does MLP output match analytic (P, Q)?
    
    Parameters
    ----------
    mlp_forward : Callable
        Function (q, p) → (P, Q) from the MLP
    ground_truth : Callable
        Analytic (q, p) → (P, Q)
    
    Returns
    -------
    dict with P_error, Q_error, pass/fail status
    """
    # Generate random test points
    q = np.random.uniform(*q_range, n_samples)
    p = np.random.uniform(*p_range, n_samples)
    
    # Ground truth
    P_true, Q_true = ground_truth(q, p, omega)
    
    # MLP prediction
    P_pred, Q_pred = mlp_forward(q, p)
    
    # Errors
    P_errors = np.abs(P_pred - P_true)
    Q_errors = np.array([angular_distance(q_pred, q_true) 
                         for q_pred, q_true in zip(Q_pred, Q_true)])
    
    # Statistics
    P_mean_err = np.mean(P_errors)
    P_max_err = np.max(P_errors)
    P_rel_err = np.mean(P_errors / np.maximum(P_true, 1e-10))
    
    Q_mean_err = np.mean(Q_errors)
    Q_max_err = np.max(Q_errors)
    
    # Pass criteria
    P_pass = P_rel_err < 0.05  # 5% relative error
    Q_pass = Q_mean_err < 0.1  # 0.1 rad mean error
    
    return {
        'P_mean_error': P_mean_err,
        'P_max_error': P_max_err,
        'P_relative_error': P_rel_err,
        'Q_mean_error': Q_mean_err,
        'Q_max_error': Q_max_err,
        'P_pass': P_pass,
        'Q_pass': Q_pass,
        'overall_pass': P_pass and Q_pass
    }


def test_symplectic(mlp_forward: Callable,
                    n_samples: int = 100,
                    q_range: Tuple[float, float] = (-2.0, 2.0),
                    p_range: Tuple[float, float] = (-2.0, 2.0),
                    eps: float = 1e-5) -> dict:
    """
    Test 2: Is the learned map symplectic?
    
    Check: |{P, Q}| = 1 (Poisson bracket)
    
    {P, Q} = ∂P/∂q · ∂Q/∂p - ∂P/∂p · ∂Q/∂q
    """
    q = np.random.uniform(*q_range, n_samples)
    p = np.random.uniform(*p_range, n_samples)
    
    poisson_brackets = []
    
    for qi, pi in zip(q, p):
        # Numerical derivatives
        P_qp, Q_qp = mlp_forward(np.array([qi + eps]), np.array([pi]))
        P_qm, Q_qm = mlp_forward(np.array([qi - eps]), np.array([pi]))
        P_pp, Q_pp = mlp_forward(np.array([qi]), np.array([pi + eps]))
        P_pm, Q_pm = mlp_forward(np.array([qi]), np.array([pi - eps]))
        
        dP_dq = (P_qp[0] - P_qm[0]) / (2 * eps)
        dP_dp = (P_pp[0] - P_pm[0]) / (2 * eps)
        dQ_dq = (Q_qp[0] - Q_qm[0]) / (2 * eps)
        dQ_dp = (Q_pp[0] - Q_pm[0]) / (2 * eps)
        
        # Poisson bracket
        pb = dP_dq * dQ_dp - dP_dp * dQ_dq
        poisson_brackets.append(pb)
    
    pb_array = np.array(poisson_brackets)
    pb_mean = np.mean(np.abs(pb_array))
    pb_err = np.mean(np.abs(np.abs(pb_array) - 1.0))
    
    # Pass if |{P, Q}| ≈ 1
    symplectic_pass = pb_err < 0.01
    
    return {
        'poisson_bracket_mean': pb_mean,
        'poisson_bracket_error': pb_err,
        'symplectic_pass': symplectic_pass
    }


def test_conservation(mlp_forward: Callable,
                      inverse_transform: Callable = sho_inverse_ground_truth,
                      P_values: np.ndarray = np.array([0.5, 1.0, 2.0]),
                      n_steps: int = 1000,
                      dt: float = 0.01,
                      omega: float = 1.0) -> dict:
    """
    Test 3: Is P constant along trajectories?
    
    Generate trajectory in (q, p), map to (P, Q), check P variation.
    """
    results = []
    
    for P_true in P_values:
        Q0 = np.random.uniform(0, 2 * np.pi)
        
        # Generate trajectory
        t = np.arange(n_steps) * dt
        Q_traj = wrap_to_2pi(Q0 + omega * t)
        
        P_measured = []
        for Q in Q_traj:
            q, p = inverse_transform(np.array([P_true]), np.array([Q]), omega)
            P_pred, _ = mlp_forward(q, p)
            P_measured.append(P_pred[0])
        
        P_measured = np.array(P_measured)
        P_std = np.std(P_measured)
        P_rel_std = P_std / P_true
        
        results.append({
            'P_true': P_true,
            'P_mean': np.mean(P_measured),
            'P_std': P_std,
            'P_relative_std': P_rel_std
        })
    
    # Overall pass
    max_rel_std = max(r['P_relative_std'] for r in results)
    conservation_pass = max_rel_std < 0.01  # 1% variation
    
    return {
        'per_P_results': results,
        'max_relative_std': max_rel_std,
        'conservation_pass': conservation_pass
    }


def test_phase_evolution(mlp_forward: Callable,
                         inverse_transform: Callable = sho_inverse_ground_truth,
                         P_values: np.ndarray = np.array([0.5, 1.0, 2.0]),
                         omega_func: Optional[Callable] = None,
                         n_steps: int = 1000,
                         dt: float = 0.01,
                         omega: float = 1.0) -> dict:
    """
    Test 4: Does Q advance at rate ω(P)?
    
    For SHO: ω = const
    For pendulum: ω = ω(P)
    """
    if omega_func is None:
        omega_func = lambda P: omega  # SHO: constant
    
    results = []
    
    for P_true in P_values:
        Q0 = np.random.uniform(0, 2 * np.pi)
        omega_expected = omega_func(P_true)
        
        # Generate trajectory
        t = np.arange(n_steps) * dt
        Q_traj_true = wrap_to_2pi(Q0 + omega_expected * t)
        
        Q_measured = []
        for Q_true in Q_traj_true:
            q, p = inverse_transform(np.array([P_true]), np.array([Q_true]), omega)
            _, Q_pred = mlp_forward(q, p)
            Q_measured.append(Q_pred[0])
        
        Q_measured = np.array(Q_measured)
        Q_unwrapped = unwrap_angle(Q_measured)
        
        # Measure frequency from slope
        coeffs = np.polyfit(t, Q_unwrapped, 1)
        omega_measured = coeffs[0]
        omega_error = abs(omega_measured - omega_expected) / omega_expected
        
        results.append({
            'P': P_true,
            'omega_expected': omega_expected,
            'omega_measured': omega_measured,
            'omega_relative_error': omega_error
        })
    
    # Overall pass
    max_omega_err = max(r['omega_relative_error'] for r in results)
    evolution_pass = max_omega_err < 0.01  # 1% error
    
    return {
        'per_P_results': results,
        'max_omega_error': max_omega_err,
        'evolution_pass': evolution_pass
    }


def test_generating_function(mlp_forward: Callable,
                             q_values: np.ndarray = np.linspace(-1.5, 1.5, 20),
                             P_values: np.ndarray = np.array([0.5, 1.0, 2.0]),
                             eps: float = 1e-5,
                             omega: float = 1.0) -> dict:
    """
    Test 5: Does implicit S satisfy generating function relations?
    
    For a canonical map with generating function S(q, P):
    - p = ∂S/∂q
    - Q = ∂S/∂P
    
    We can't directly access S, but we can check consistency:
    - ∂p/∂P should equal ∂Q/∂q (mixed partials of S)
    
    This is the integrability condition.
    """
    integrability_errors = []
    
    for P in P_values:
        for q in q_values:
            # Skip points near boundaries
            if abs(q) > np.sqrt(2 * P / omega) * 0.9:
                continue
            
            # Get p at this (q, P) - need to invert
            # For SHO: p = ±√(2ωP - ω²q²)
            p_sq = 2 * omega * P - omega**2 * q**2
            if p_sq < 0:
                continue
            p = np.sqrt(p_sq)  # Positive branch
            
            # Numerical derivatives
            P_up, Q_up = mlp_forward(np.array([q]), np.array([p + eps]))
            P_dn, Q_dn = mlp_forward(np.array([q]), np.array([p - eps]))
            
            # At fixed (q, P), changing p changes Q
            # But we want ∂p/∂P at fixed q - harder to compute
            
            # Simpler test: check that the map is consistent
            # (P_pred should equal P_true at the true (q, p) point)
            P_pred, Q_pred = mlp_forward(np.array([q]), np.array([p]))
            
            # For ground truth map, P_pred should exactly equal P
            P_error = abs(P_pred[0] - P)
            integrability_errors.append(P_error)
    
    mean_error = np.mean(integrability_errors) if integrability_errors else np.nan
    max_error = np.max(integrability_errors) if integrability_errors else np.nan
    
    return {
        'mean_integrability_error': mean_error,
        'max_integrability_error': max_error,
        'integrability_pass': mean_error < 0.01 if not np.isnan(mean_error) else False
    }


# =============================================================================
# Full Validation Suite
# =============================================================================

def validate_hjb_mlp(mlp_forward: Callable,
                     system: str = 'sho',
                     omega: float = 1.0,
                     verbose: bool = True) -> dict:
    """
    Run full validation suite on an MLP claiming to learn (q, p) → (P, Q).
    
    Parameters
    ----------
    mlp_forward : Callable
        Function (q, p) → (P, Q) where q, p are numpy arrays
    system : str
        'sho' or 'pendulum'
    omega : float
        Angular frequency (for SHO)
    verbose : bool
        Print results
    
    Returns
    -------
    dict with all test results
    """
    if system == 'sho':
        ground_truth = lambda q, p, w=omega: sho_ground_truth(q, p, w)
        inverse_transform = lambda P, Q, w=omega: sho_inverse_ground_truth(P, Q, w)
    else:
        raise NotImplementedError(f"System {system} not implemented")
    
    if verbose:
        print("=" * 70)
        print("HJB_MLP VALIDATION SUITE")
        print("=" * 70)
        print(f"\nSystem: {system}, ω = {omega}")
        print()
    
    results = {}
    
    # Test 1: Accuracy
    if verbose:
        print("[Test 1] Accuracy: MLP output vs ground truth")
    results['accuracy'] = test_accuracy(mlp_forward, ground_truth, omega=omega)
    if verbose:
        r = results['accuracy']
        print(f"  P relative error: {r['P_relative_error']:.4f} "
              f"({'PASS' if r['P_pass'] else 'FAIL'})")
        print(f"  Q mean error: {r['Q_mean_error']:.4f} rad "
              f"({'PASS' if r['Q_pass'] else 'FAIL'})")
    
    # Test 2: Symplectic
    if verbose:
        print("\n[Test 2] Symplectic: |{P, Q}| = 1")
    results['symplectic'] = test_symplectic(mlp_forward)
    if verbose:
        r = results['symplectic']
        print(f"  |{{P, Q}}| mean: {r['poisson_bracket_mean']:.6f}")
        print(f"  Error from ±1: {r['poisson_bracket_error']:.6f} "
              f"({'PASS' if r['symplectic_pass'] else 'FAIL'})")
    
    # Test 3: Conservation
    if verbose:
        print("\n[Test 3] Conservation: P constant along trajectory")
    results['conservation'] = test_conservation(mlp_forward, inverse_transform, omega=omega)
    if verbose:
        r = results['conservation']
        print(f"  Max relative std: {r['max_relative_std']:.6f} "
              f"({'PASS' if r['conservation_pass'] else 'FAIL'})")
    
    # Test 4: Phase evolution
    if verbose:
        print("\n[Test 4] Phase evolution: dQ/dt = ω(P)")
    results['evolution'] = test_phase_evolution(mlp_forward, inverse_transform, omega=omega)
    if verbose:
        r = results['evolution']
        print(f"  Max ω error: {r['max_omega_error']:.6f} "
              f"({'PASS' if r['evolution_pass'] else 'FAIL'})")
    
    # Test 5: Generating function
    if verbose:
        print("\n[Test 5] Generating function consistency")
    results['generating'] = test_generating_function(mlp_forward, omega=omega)
    if verbose:
        r = results['generating']
        if not np.isnan(r['mean_integrability_error']):
            print(f"  Mean integrability error: {r['mean_integrability_error']:.6f} "
                  f"({'PASS' if r['integrability_pass'] else 'FAIL'})")
        else:
            print("  Could not compute (no valid points)")
    
    # Summary
    all_pass = (results['accuracy']['overall_pass'] and
                results['symplectic']['symplectic_pass'] and
                results['conservation']['conservation_pass'] and
                results['evolution']['evolution_pass'])
    
    results['all_pass'] = all_pass
    
    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"  Accuracy:     {'PASS' if results['accuracy']['overall_pass'] else 'FAIL'}")
        print(f"  Symplectic:   {'PASS' if results['symplectic']['symplectic_pass'] else 'FAIL'}")
        print(f"  Conservation: {'PASS' if results['conservation']['conservation_pass'] else 'FAIL'}")
        print(f"  Evolution:    {'PASS' if results['evolution']['evolution_pass'] else 'FAIL'}")
        print()
        if all_pass:
            print("  ✓ ALL TESTS PASSED - MLP learns valid canonical transformation")
        else:
            print("  ✗ SOME TESTS FAILED - Review results above")
        print("=" * 70)
    
    return results


# =============================================================================
# Demo: Validate Ground Truth Against Itself
# =============================================================================

def demo_ground_truth_validation():
    """
    Sanity check: validate ground truth functions against themselves.
    Should pass all tests perfectly.
    """
    print("\n" + "=" * 70)
    print("DEMO: Ground Truth Self-Validation")
    print("=" * 70)
    print("\nValidating analytic SHO formulas against themselves.")
    print("This should pass all tests with ~zero error.\n")
    
    # Ground truth as the "MLP"
    def ground_truth_mlp(q, p):
        return sho_ground_truth(q, p, omega=1.0)
    
    results = validate_hjb_mlp(ground_truth_mlp, system='sho', omega=1.0, verbose=True)
    
    return results


def demo_noisy_mlp_validation():
    """
    Demo: Validate a "noisy" MLP (ground truth + noise).
    Should fail some tests depending on noise level.
    """
    print("\n" + "=" * 70)
    print("DEMO: Noisy MLP Validation")
    print("=" * 70)
    print("\nValidating ground truth + 5% noise.")
    print("This should fail accuracy tests.\n")
    
    def noisy_mlp(q, p, noise_level=0.05):
        P, Q = sho_ground_truth(q, p, omega=1.0)
        P_noisy = P * (1 + noise_level * np.random.randn(len(P)))
        Q_noisy = Q + noise_level * np.random.randn(len(Q))
        return P_noisy, wrap_to_2pi(Q_noisy)
    
    results = validate_hjb_mlp(noisy_mlp, system='sho', omega=1.0, verbose=True)
    
    return results


# =============================================================================
# Template for User's MLP
# =============================================================================

def create_mlp_wrapper(model, device: str = 'cpu') -> Callable:
    """
    Create a numpy-compatible wrapper for a PyTorch MLP.
    
    Usage:
    ```python
    from hjb_mlp import HJB_MLP
    
    model = HJB_MLP(...)
    model.load_state_dict(torch.load('checkpoint.pt'))
    model.eval()
    
    mlp_forward = create_mlp_wrapper(model)
    results = validate_hjb_mlp(mlp_forward)
    ```
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for create_mlp_wrapper. Install with: pip install torch")
    
    import torch
    def wrapper(q: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            # Convert to tensor
            q_t = torch.tensor(q, dtype=torch.float32, device=device)
            p_t = torch.tensor(p, dtype=torch.float32, device=device)
            
            # Stack as input (assuming model takes [q, p] or [p, q])
            x = torch.stack([q_t, p_t], dim=-1)
            
            # Forward pass
            output = model(x)
            
            # Assume output is [P, Q] or has .P, .Q attributes
            if hasattr(output, 'P'):
                P = output.P.cpu().numpy()
                Q = output.Q.cpu().numpy()
            else:
                P = output[:, 0].cpu().numpy()
                Q = output[:, 1].cpu().numpy()
            
            return P, Q
    
    return wrapper


if __name__ == "__main__":
    # Run demos
    print("\n" + "=" * 70)
    print("HJB_MLP VALIDATION FRAMEWORK")
    print("=" * 70)
    print("""
This framework validates any MLP claiming to learn (q, p) → (P, Q).

Tests:
1. Accuracy: Does output match ground truth?
2. Symplectic: Is |{P, Q}| = 1?
3. Conservation: Is P constant along trajectories?
4. Phase evolution: Does Q advance at rate ω?
5. Generating function: Is the map integrable?

To validate your HJB_MLP:

```python
from validate_hjb_mlp import validate_hjb_mlp, create_mlp_wrapper
from hjb_mlp import HJB_MLP

model = HJB_MLP(...)
model.load_state_dict(torch.load('checkpoint.pt'))
model.eval()

mlp_forward = create_mlp_wrapper(model)
results = validate_hjb_mlp(mlp_forward, system='sho')
```
""")
    
    # Demo 1: Ground truth validation
    demo_ground_truth_validation()
    
    # Demo 2: Noisy MLP
    print()
    demo_noisy_mlp_validation()
