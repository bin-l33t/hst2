"""
Manifold diagnostics for validating single-chart MLP approach.

ChatGPT-suggested diagnostics to check before training:
1. Local intrinsic dimension (is β consistently ~2D?)
2. Fold/self-intersection detection (Euclidean-close but graph-far?)
3. S¹ factor detection (circular topology requiring sin/cos?)

These diagnostics help decide if a single-chart MLP is sufficient
or if an atlas/multi-chart approach is needed.

Note: Uses numpy-only implementations for compatibility.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra


def _knn_numpy(X, k):
    """
    Simple k-nearest neighbors using numpy.

    Args:
        X: Data matrix (n_samples, n_features)
        k: Number of neighbors

    Returns:
        distances: (n_samples, k) distances to neighbors
        indices: (n_samples, k) indices of neighbors
    """
    n = len(X)
    # Compute pairwise distances
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    sq_norms = np.sum(X**2, axis=1)
    distances_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * X @ X.T
    distances_sq = np.maximum(distances_sq, 0)  # Numerical stability
    distances = np.sqrt(distances_sq)

    # Get k+1 nearest (including self)
    idx = np.argsort(distances, axis=1)[:, :k+1]
    # Exclude self (index 0)
    indices = idx[:, 1:]

    # Get distances for these indices
    dist = np.array([distances[i, indices[i]] for i in range(n)])

    return dist, indices


def diagnostic_local_dimension(beta, k=30):
    """
    Check if β is consistently ~2D everywhere.

    Uses local PCA on k-nearest neighbors to estimate tangent space dimension.

    Args:
        beta: Feature matrix (n_samples, n_features)
        k: Number of neighbors for local PCA

    Returns:
        gaps: λ₂/λ₃ ratio per point (large = confidently 2D)
        curv: λ₃/λ₁ ratio per point (small = well-approximated by 2D tangent)

    Interpretation:
        - gaps uniformly large → single 2D manifold plausible
        - gaps collapse in regions → hitting singularities, need atlas
    """
    _, idx = _knn_numpy(beta, k)

    gaps = []
    curv = []
    for i in range(len(beta)):
        X = beta[idx[i]] - beta[i]
        C = (X.T @ X) / k
        w = np.linalg.eigvalsh(C)[::-1]  # descending
        # Avoid division by zero
        gaps.append(w[1] / (w[2] + 1e-12))
        curv.append((w[2] + 1e-12) / (w[0] + 1e-12))

    return np.array(gaps), np.array(curv)


def diagnostic_fold_detection(beta, k=15, m=3):
    """
    Detect folds/self-intersections: points Euclidean-close but graph-far.

    Builds a k-NN graph and computes shortest path distances. Points that
    are close in Euclidean space but far on the graph indicate folding.

    Args:
        beta: Feature matrix (n_samples, n_features)
        k: Number of neighbors for graph construction
        m: Number of closest neighbors to check for fold score

    Returns:
        scores: Fold score per point (high = potential multi-sheet)

    Interpretation:
        - Few/no high scores → injective embedding, single chart OK
        - Many high scores → folding, need atlas/gating
    """
    dist, idx = _knn_numpy(beta, k)

    # kNN graph weighted by Euclidean distance
    n = len(beta)
    rows = np.repeat(np.arange(n), k)
    cols = idx.reshape(-1)
    data = dist.reshape(-1)
    G = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    G = 0.5 * (G + G.T)  # Symmetrize

    # Graph distances via Dijkstra
    D = dijkstra(G, directed=False, indices=np.arange(n))

    # Fold score: how often "very close in Euclid" is "far in graph"
    scores = []
    for i in range(n):
        close = idx[i][:m]
        local_scale = np.median(dist[i]) + 1e-12
        scores.append(np.median(D[i, close]) / local_scale)

    return np.array(scores)


def diagnostic_circle_factor(beta, n_components=3, k=20):
    """
    Detect S¹ factor via spectral embedding (Laplacian eigenmaps).

    If first 2 eigenvectors form a ring → angle variable present.

    Args:
        beta: Feature matrix (n_samples, n_features)
        n_components: Number of spectral embedding dimensions
        k: Number of neighbors for graph construction

    Returns:
        has_circle: bool - True if S¹ factor detected
        ring_thinness: std(r) / mean(r) - small = thin annulus = circle
        coords: Spectral embedding coordinates

    Interpretation:
        - Thin annulus (ring_thinness < 0.3) suggests S¹ factor
        - If S¹ present, must use (sin, cos) embedding for angle variables
    """
    n = len(beta)
    k = min(k, n - 1)

    # Build kNN graph with Gaussian weights
    dist, idx = _knn_numpy(beta, k)

    # Gaussian kernel width (median distance)
    sigma = np.median(dist) + 1e-12

    # Build affinity matrix with Gaussian weights
    rows = np.repeat(np.arange(n), k)
    cols = idx.reshape(-1)
    weights = np.exp(-dist.reshape(-1)**2 / (2 * sigma**2))
    W = sp.csr_matrix((weights, (rows, cols)), shape=(n, n))
    W = 0.5 * (W + W.T)  # Symmetrize

    # Degree matrix
    D = np.array(W.sum(axis=1)).flatten()
    D_sqrt_inv = np.diag(1.0 / np.sqrt(D + 1e-12))

    # Normalized Laplacian: L = I - D^{-1/2} W D^{-1/2}
    W_dense = W.toarray()
    L_sym = np.eye(n) - D_sqrt_inv @ W_dense @ D_sqrt_inv

    # Get smallest eigenvectors (skip first, which is constant)
    eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
    # Skip first eigenvector (trivial)
    coords = eigenvectors[:, 1:n_components+1]

    # Scale by inverse sqrt of eigenvalues for spectral embedding
    for i in range(n_components):
        if eigenvalues[i+1] > 1e-10:
            coords[:, i] /= np.sqrt(eigenvalues[i+1])

    # Check if first 2 components form a ring
    u1, u2 = coords[:, 0], coords[:, 1]
    r = np.sqrt(u1**2 + u2**2)

    ring_thinness = np.std(r) / (np.mean(r) + 1e-12)

    # Thin annulus (ring_thinness < 0.3) suggests S¹ factor
    has_circle = ring_thinness < 0.3

    return has_circle, ring_thinness, coords


def run_manifold_diagnostics(beta, name=""):
    """
    Run all 3 diagnostics and report.

    Decision rule:
        Single-chart MLP is enough if:
        1. Local dimension consistently ~2 (median gap > 5)
        2. Fold scores low (median < 2)
        3. Any circle factor represented via sin/cos

    Args:
        beta: Feature matrix (n_samples, n_features)
        name: Optional name for logging

    Returns:
        dict with diagnostic results and recommendations
    """
    header = f" MANIFOLD DIAGNOSTICS{f' ({name})' if name else ''} "
    print("\n" + "=" * 60)
    print(header.center(60, "="))
    print("=" * 60)

    # Diagnostic 1: Local dimension
    print("\n[1] Local Intrinsic Dimension")
    print("-" * 40)
    gaps, curv = diagnostic_local_dimension(beta, k=min(30, len(beta) - 1))
    print(f"    λ₂/λ₃ gap: median={np.median(gaps):.2f}, "
          f"min={np.min(gaps):.2f}, max={np.max(gaps):.2f}")
    print(f"    Curvature (λ₃/λ₁): median={np.median(curv):.4f}")
    dim_ok = np.median(gaps) > 5
    print(f"    Status: {'OK' if dim_ok else 'WARN'} (need median gap > 5)")

    # Diagnostic 2: Fold detection
    print("\n[2] Fold/Self-Intersection Detection")
    print("-" * 40)
    fold_scores = diagnostic_fold_detection(beta, k=min(15, len(beta) - 1), m=3)
    print(f"    Fold scores: median={np.median(fold_scores):.2f}, "
          f"max={np.max(fold_scores):.2f}")
    n_high_fold = np.sum(fold_scores > 3)
    print(f"    Points with high fold score (>3): {n_high_fold}/{len(beta)}")
    fold_ok = np.median(fold_scores) < 2
    print(f"    Status: {'OK' if fold_ok else 'WARN'} (need median < 2)")

    # Diagnostic 3: Circle factor
    print("\n[3] S¹ Factor Detection (Spectral Embedding)")
    print("-" * 40)
    has_circle, thinness, coords = diagnostic_circle_factor(beta)
    print(f"    Ring thinness: {thinness:.3f}")
    print(f"    Has circle factor: {has_circle}")
    if has_circle:
        print(f"    → Must use (sin, cos) embedding for angle variable")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    single_chart_ok = dim_ok and fold_ok

    print(f"\n    Local dimension OK:    {'✓' if dim_ok else '✗'}")
    print(f"    No folds detected:     {'✓' if fold_ok else '✗'}")
    print(f"    S¹ factor present:     {'✓' if has_circle else '✗'}")

    print("\n" + "-" * 60)
    if single_chart_ok:
        print("CONCLUSION: Single-chart MLP is appropriate")
        if has_circle:
            print("            (use sin/cos embedding for angle variables)")
    else:
        print("CONCLUSION: May need atlas/multi-chart approach")
        if not dim_ok:
            print("            - Local dimension inconsistent (gap too small)")
        if not fold_ok:
            print("            - Detected folds/self-intersections")
    print("-" * 60)

    return {
        'dim_ok': dim_ok,
        'fold_ok': fold_ok,
        'has_circle': has_circle,
        'single_chart_ok': single_chart_ok,
        'gap_median': np.median(gaps),
        'gap_min': np.min(gaps),
        'fold_median': np.median(fold_scores),
        'fold_max': np.max(fold_scores),
        'ring_thinness': thinness,
        'spectral_coords': coords,
    }


def compare_manifolds(results_dict):
    """
    Compare manifold diagnostics across multiple systems.

    Args:
        results_dict: dict mapping name -> diagnostic results
    """
    print("\n" + "=" * 70)
    print("MANIFOLD COMPARISON")
    print("=" * 70)

    header = f"{'System':<15} | {'Gap(med)':<10} | {'Fold(med)':<10} | {'Ring':<8} | {'S¹?':<5} | {'1-Chart?':<8}"
    print(f"\n{header}")
    print("-" * 70)

    for name, r in results_dict.items():
        s1 = "Yes" if r['has_circle'] else "No"
        chart = "OK" if r['single_chart_ok'] else "WARN"
        print(f"{name:<15} | {r['gap_median']:<10.2f} | {r['fold_median']:<10.2f} | "
              f"{r['ring_thinness']:<8.3f} | {s1:<5} | {chart:<8}")

    print("-" * 70)


# =============================================================================
# NEW: Tangent Plane Twist Diagnostic (for detecting manifold curvature)
# =============================================================================

def compute_tangent_twist(beta, k=30):
    """
    Measure how much local tangent planes rotate across the manifold.

    High twist = curved manifold = linear map fails, need MLP
    Low twist = nearly flat = linear map works

    Method:
    1. At each point, compute local 2D tangent basis U_i via kNN PCA
    2. For each neighbor j, compute principal angle between U_i and U_j
    3. twist(i) = 1 - σ_min(U_i.T @ U_j), averaged over neighbors

    Args:
        beta: Feature matrix (n_samples, n_features)
        k: Number of neighbors for local PCA

    Returns:
        twists: array of twist scores per point (0 = flat, 1 = maximally twisted)
    """
    n = len(beta)
    k = min(k, n - 1)

    _, idx = _knn_numpy(beta, k)

    # Compute local tangent bases (top 2 eigenvectors of local covariance)
    tangent_bases = []
    for i in range(n):
        X = beta[idx[i]] - beta[idx[i]].mean(axis=0, keepdims=True)
        C = (X.T @ X) / max(k - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(C)
        # Top 2 eigenvectors = tangent basis (eigh returns ascending order)
        U_i = eigvecs[:, -2:]  # Shape: (d, 2)
        tangent_bases.append(U_i)

    tangent_bases = np.array(tangent_bases)  # (N, d, 2)

    # Compute twist scores
    twists = []
    n_compare = min(10, k)  # Use 10 nearest neighbors for twist
    for i in range(n):
        U_i = tangent_bases[i]
        neighbor_twists = []
        for j in idx[i][:n_compare]:
            U_j = tangent_bases[j]
            # Principal angle via SVD of U_i.T @ U_j
            M = U_i.T @ U_j  # (2, 2)
            sigma_min = np.linalg.svd(M, compute_uv=False).min()
            twist_ij = 1 - sigma_min  # 0 = parallel planes, 1 = orthogonal planes
            neighbor_twists.append(twist_ij)
        twists.append(np.mean(neighbor_twists))

    return np.array(twists)


def compute_local_pca_scores(beta, k=30):
    """
    Compute local PCA scores for 2D-ness and thickness.

    Args:
        beta: Feature matrix (n_samples, n_features)
        k: Number of neighbors

    Returns:
        rhos: λ₂/(λ₁+λ₂+λ₃) - fraction of variance in top 2 dims (2D-ness)
        taus: λ₃/λ₁ - thickness ratio (small = thin/flat)
        kappas: λ₁/λ₂ - anisotropy (how elongated the local neighborhood is)
    """
    n = len(beta)
    k = min(k, n - 1)

    _, idx = _knn_numpy(beta, k)

    rhos = []
    taus = []
    kappas = []

    for i in range(n):
        X = beta[idx[i]] - beta[idx[i]].mean(axis=0, keepdims=True)
        C = (X.T @ X) / max(k - 1, 1)
        w = np.linalg.eigvalsh(C)[::-1]  # descending order

        # 2D-ness: fraction of variance in top 2 eigenvalues
        total = np.sum(w[:3]) + 1e-12
        rho = (w[0] + w[1]) / total
        rhos.append(rho)

        # Thickness: ratio of 3rd to 1st eigenvalue
        tau = (w[2] + 1e-12) / (w[0] + 1e-12)
        taus.append(tau)

        # Anisotropy
        kappa = (w[0] + 1e-12) / (w[1] + 1e-12)
        kappas.append(kappa)

    return np.array(rhos), np.array(taus), np.array(kappas)


def run_full_diagnostics(beta, name=""):
    """
    Run complete diagnostics including twist (curvature) metric.

    This is an enhanced version of run_manifold_diagnostics that adds
    the twist diagnostic to detect manifold curvature.

    Args:
        beta: Feature matrix (n_samples, n_features)
        name: Optional name for logging

    Returns:
        dict with all diagnostic results
    """
    print(f"\n{'=' * 60}")
    print(f"MANIFOLD DIAGNOSTICS: {name}" if name else "MANIFOLD DIAGNOSTICS")
    print(f"{'=' * 60}")

    k = min(30, len(beta) - 1)

    # [1] Local dimension (2D-ness and thickness)
    print(f"\n[1] Local Intrinsic Dimension:")
    print("-" * 40)
    rhos, taus, kappas = compute_local_pca_scores(beta, k=k)
    gaps, curv = diagnostic_local_dimension(beta, k=k)
    print(f"    2D-ness (ρ): median={np.median(rhos):.3f}, min={np.min(rhos):.3f}")
    print(f"    Thickness (τ): median={np.median(taus):.4f}, 95%={np.percentile(taus, 95):.4f}")
    print(f"    λ₂/λ₃ gap: median={np.median(gaps):.2f}")
    dim_ok = np.median(gaps) > 5

    # [2] Tangent twist (NEW - key diagnostic for curvature)
    print(f"\n[2] Tangent Plane Twist (CURVATURE):")
    print("-" * 40)
    twists = compute_tangent_twist(beta, k=k)
    print(f"    Twist: median={np.median(twists):.4f}, 95%={np.percentile(twists, 95):.4f}")
    print(f"    Twist: min={np.min(twists):.4f}, max={np.max(twists):.4f}")
    # High twist (> 0.1) suggests curved manifold where linear maps fail
    twist_high = np.median(twists) > 0.1
    if twist_high:
        print(f"    Status: HIGH TWIST - manifold is curved, linear map will struggle")
    else:
        print(f"    Status: LOW TWIST - manifold is nearly flat, linear map OK")

    # [3] Fold detection
    print(f"\n[3] Fold/Self-Overlap:")
    print("-" * 40)
    fold_scores = diagnostic_fold_detection(beta, k=min(15, len(beta) - 1), m=3)
    print(f"    Fold score: median={np.median(fold_scores):.2f}, "
          f"95%={np.percentile(fold_scores, 95):.2f}")
    fold_ok = np.median(fold_scores) < 2

    # [4] S¹ detection
    print(f"\n[4] S¹ Factor Detection:")
    print("-" * 40)
    has_circle, ring_thin, coords = diagnostic_circle_factor(beta)
    print(f"    Ring thinness: {ring_thin:.3f}")
    print(f"    Has circle factor: {has_circle}")

    return {
        'rho_median': np.median(rhos),
        'rho_min': np.min(rhos),
        'tau_median': np.median(taus),
        'tau_95': np.percentile(taus, 95),
        'gap_median': np.median(gaps),
        'twist_median': np.median(twists),
        'twist_95': np.percentile(twists, 95),
        'twist_max': np.max(twists),
        'fold_median': np.median(fold_scores),
        'fold_95': np.percentile(fold_scores, 95),
        'has_circle': has_circle,
        'ring_thinness': ring_thin,
        'dim_ok': dim_ok,
        'fold_ok': fold_ok,
        'single_chart_ok': dim_ok and fold_ok,
    }


def compare_and_recommend(sho_results, pend_results):
    """
    Compare SHO vs Pendulum manifolds and give architecture recommendation.

    Key insight: If pendulum has higher twist than SHO, the linear β→(p,q) map
    fails due to manifold curvature, and an MLP encoder is needed.

    Args:
        sho_results: Results from run_full_diagnostics on SHO
        pend_results: Results from run_full_diagnostics on Pendulum

    Returns:
        dict with recommendation
    """
    print("\n" + "=" * 60)
    print("COMPARISON: SHO vs PENDULUM")
    print("=" * 60)

    print(f"\n{'Metric':<20} | {'SHO':<12} | {'Pendulum':<12} | {'Interpretation'}")
    print("-" * 75)

    # 2D-ness
    print(f"{'ρ (2D-ness)':<20} | {sho_results['rho_median']:<12.3f} | "
          f"{pend_results['rho_median']:<12.3f} | ", end="")
    if pend_results['rho_median'] < 0.9:
        print("Pendulum needs more dims")
        case_a = True
    else:
        print("Both are 2D ✓")
        case_a = False

    # Thickness
    print(f"{'τ (thickness)':<20} | {sho_results['tau_median']:<12.4f} | "
          f"{pend_results['tau_median']:<12.4f} | ", end="")
    if pend_results['tau_median'] > 2 * sho_results['tau_median']:
        print("Pendulum manifold thicker")
    else:
        print("Similar thickness ✓")

    # TWIST - the key metric for curvature
    twist_ratio = pend_results['twist_median'] / (sho_results['twist_median'] + 1e-6)
    print(f"{'TWIST (curvature)':<20} | {sho_results['twist_median']:<12.4f} | "
          f"{pend_results['twist_median']:<12.4f} | ", end="")
    if twist_ratio > 2:
        print(f"PENDULUM {twist_ratio:.1f}x MORE CURVED!")
        case_b = True
    else:
        print("Similar curvature")
        case_b = False

    # Fold
    print(f"{'Fold score':<20} | {sho_results['fold_median']:<12.2f} | "
          f"{pend_results['fold_median']:<12.2f} | ", end="")
    if pend_results['fold_95'] > 3:
        print("Pendulum has folds")
        case_c = True
    else:
        print("No folding ✓")
        case_c = False

    # S¹ factor
    print(f"{'S¹ factor':<20} | {'Yes' if sho_results['has_circle'] else 'No':<12} | "
          f"{'Yes' if pend_results['has_circle'] else 'No':<12} | ", end="")
    if pend_results['has_circle']:
        print("Use (cos q, sin q) embedding")
    else:
        print("No special handling needed")

    # RECOMMENDATION
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)

    recommendation = None

    if case_a:
        print("\n→ CASE A: Intrinsic dim > 2")
        print("  Action: Increase PCA components or add latent variables")
        recommendation = 'increase_dim'
    elif case_b:
        print("\n→ CASE B: 2D but CURVED (most likely for pendulum)")
        print("  Action: Replace linear W with small MLP for β → (p, cos q, sin q)")
        print("  This explains the ~0.3 forward error - linear can't fit curved manifold")
        recommendation = 'mlp_encoder'
    elif case_c:
        print("\n→ CASE C: Folding/overlap detected")
        print("  Action: Need atlas/mixture model or regime classifier")
        recommendation = 'atlas'
    else:
        print("\n→ Manifolds are similar - issue may be elsewhere")
        print("  Consider: decoder architecture, training hyperparameters, etc.")
        recommendation = 'other'

    print("=" * 60)

    return {
        'twist_ratio': twist_ratio,
        'case_a': case_a,
        'case_b': case_b,
        'case_c': case_c,
        'recommendation': recommendation,
    }


# =============================================================================
# NEW: Seam and Holonomy Tests (for detecting branch cuts and atlas needs)
# =============================================================================

def seam_test(beta, k=20):
    """
    Detect branch-cut seams via angle jumps in 2D embedding.

    If max |Δθ| ≈ π on many edges → periodic coordinate has a cut
    → need (sin, cos) representation or two-chart atlas

    Args:
        beta: Feature matrix (n_samples, n_features)
        k: Number of neighbors for graph construction

    Returns:
        dict with seam detection results:
        - max_jump: maximum angle jump across any edge
        - p95_jump: 95th percentile of angle jumps
        - n_large_jumps: number of edges with |Δθ| > π/2
        - has_seam: True if significant seam detected
    """
    n = len(beta)
    k = min(k, n - 1)

    # Global 2D embedding via PCA (fine since manifold is flat)
    # Center the data
    beta_centered = beta - beta.mean(axis=0)
    # SVD for PCA
    U, S, Vt = np.linalg.svd(beta_centered, full_matrices=False)
    u = U[:, :2] * S[:2]  # Project onto top 2 PCs

    # Angle-like coordinate
    theta = np.arctan2(u[:, 1], u[:, 0])  # [-π, π]

    # Build kNN graph
    _, idx = _knn_numpy(beta, k)

    # Compute angle jumps on all edges
    jumps = []
    for i in range(n):
        for j in idx[i]:
            delta = theta[j] - theta[i]
            # Wrap to [-π, π]
            delta = np.arctan2(np.sin(delta), np.cos(delta))
            jumps.append(abs(delta))

    jumps = np.array(jumps)

    max_jump = np.max(jumps)
    p95_jump = np.percentile(jumps, 95)
    n_large_jumps = np.sum(jumps > np.pi / 2)

    return {
        'max_jump': max_jump,
        'p95_jump': p95_jump,
        'n_large_jumps': n_large_jumps,
        'total_edges': len(jumps),
        'has_seam': p95_jump > np.pi / 3,  # Threshold: 60 degrees
    }


def holonomy_test(beta, k=20):
    """
    Test if tangent bases can be globally consistently oriented.

    Procedure:
    1. Compute local 2D tangent basis at each point
    2. BFS from root, aligning bases by Procrustes
    3. Measure inconsistency when paths meet

    High inconsistency → need atlas/covering

    Args:
        beta: Feature matrix (n_samples, n_features)
        k: Number of neighbors for graph construction

    Returns:
        dict with holonomy test results:
        - mean_inconsistency: average orientation conflict
        - max_inconsistency: maximum orientation conflict
        - needs_atlas: True if atlas/multi-chart approach needed
    """
    from collections import deque

    n = len(beta)
    k = min(k, n - 1)

    # Compute local tangent bases
    _, idx = _knn_numpy(beta, k)

    tangent_bases = []
    for i in range(n):
        X = beta[idx[i]] - beta[idx[i]].mean(axis=0, keepdims=True)
        C = (X.T @ X) / max(k - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(C)
        U_i = eigvecs[:, -2:]  # Top 2 eigenvectors
        tangent_bases.append(U_i)

    tangent_bases = np.array(tangent_bases)  # (N, d, 2)

    # BFS to propagate orientation
    visited = np.zeros(n, dtype=bool)
    assigned_bases = [None] * n

    # Start from node 0
    assigned_bases[0] = tangent_bases[0].copy()
    visited[0] = True
    queue = deque([0])

    inconsistencies = []

    while queue:
        i = queue.popleft()
        U_i = assigned_bases[i]

        for j in idx[i]:
            if not visited[j]:
                # Align U_j to U_i via Procrustes
                U_j = tangent_bases[j]
                M = U_i.T @ U_j  # (2, 2)

                # SVD for optimal rotation
                u, s, vh = np.linalg.svd(M)
                R = u @ vh

                # Check for reflection (det < 0)
                if np.linalg.det(R) < 0:
                    R[:, 1] *= -1

                assigned_bases[j] = U_j @ R.T
                visited[j] = True
                queue.append(j)
            else:
                # Already visited - check consistency
                U_j_assigned = assigned_bases[j]

                # How much would we need to rotate to match?
                M = U_i.T @ U_j_assigned
                u, s, vh = np.linalg.svd(M)

                # Inconsistency = 1 - min singular value
                inconsistency = 1 - s.min()
                inconsistencies.append(inconsistency)

    inconsistencies = np.array(inconsistencies) if inconsistencies else np.array([0])

    return {
        'mean_inconsistency': np.mean(inconsistencies),
        'max_inconsistency': np.max(inconsistencies),
        'p95_inconsistency': np.percentile(inconsistencies, 95) if len(inconsistencies) > 0 else 0,
        'needs_atlas': np.mean(inconsistencies) > 0.1,
    }


def run_extended_diagnostics(beta, name=""):
    """
    Run all diagnostics including seam and holonomy tests.

    This is the most comprehensive diagnostic suite.

    Args:
        beta: Feature matrix (n_samples, n_features)
        name: Optional name for logging

    Returns:
        dict with all diagnostic results
    """
    print(f"\n{'=' * 60}")
    print(f"EXTENDED MANIFOLD DIAGNOSTICS: {name}" if name else "EXTENDED MANIFOLD DIAGNOSTICS")
    print(f"{'=' * 60}")

    # Run full diagnostics (includes twist)
    results = run_full_diagnostics(beta, name="")

    # Add seam test
    print(f"\n[5] Seam/Branch-Cut Detection:")
    print("-" * 40)
    seam = seam_test(beta)
    print(f"    Max angle jump: {seam['max_jump']:.3f} rad ({np.degrees(seam['max_jump']):.1f} deg)")
    print(f"    95th percentile: {seam['p95_jump']:.3f} rad ({np.degrees(seam['p95_jump']):.1f} deg)")
    print(f"    Large jumps (>π/2): {seam['n_large_jumps']}/{seam['total_edges']}")
    print(f"    Has seam: {seam['has_seam']}")
    if seam['has_seam']:
        print(f"    → Use (sin, cos) embedding for angle coordinates")

    # Add holonomy test
    print(f"\n[6] Holonomy/Orientation Consistency:")
    print("-" * 40)
    holonomy = holonomy_test(beta)
    print(f"    Mean inconsistency: {holonomy['mean_inconsistency']:.4f}")
    print(f"    Max inconsistency: {holonomy['max_inconsistency']:.4f}")
    print(f"    Needs atlas: {holonomy['needs_atlas']}")
    if holonomy['needs_atlas']:
        print(f"    → Consider multi-chart/atlas approach")

    # Merge results
    results.update({
        'seam_max_jump': seam['max_jump'],
        'seam_p95_jump': seam['p95_jump'],
        'seam_has_seam': seam['has_seam'],
        'holonomy_mean': holonomy['mean_inconsistency'],
        'holonomy_max': holonomy['max_inconsistency'],
        'holonomy_needs_atlas': holonomy['needs_atlas'],
    })

    # Summary
    print(f"\n{'=' * 60}")
    print("EXTENDED SUMMARY")
    print(f"{'=' * 60}")
    print(f"    Single-chart OK: {results['single_chart_ok']}")
    print(f"    Has seam (use sin/cos): {seam['has_seam']}")
    print(f"    Needs atlas: {holonomy['needs_atlas']}")

    return results
