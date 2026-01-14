"""
Test Strong Forcing to Break Identifiability Gap

Previous experiment: F_scale=0.3 → |ΔP| ~ 0.09 → trivial solution survives
This experiment: F_scale=3.0 → |ΔP| ~ 0.9 → trivial solution should fail

Also test normalized action loss that penalizes dP=0 regardless of scale.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from scipy.stats import pearsonr, spearmanr
from dataclasses import dataclass

from fixed_hjb_loss import ImprovedHJB_MLP
from action_angle_utils import wrap_to_2pi


@dataclass
class ForcingResult:
    """Results from one training run."""
    F_scale: float
    seed: int
    P_spearman: float
    P_pearson: float
    final_loss: float
    converged: bool


def generate_forced_sho_batch(n_traj: int, dt: float, omega: float = 1.0,
                               F_scale: float = 3.0, device: str = 'cpu'):
    """Generate SHO trajectories with strong forcing."""

    # Initial action-angle
    P0 = torch.rand(n_traj, device=device) * 2 + 0.5  # P ∈ [0.5, 2.5]
    Q0 = torch.rand(n_traj, device=device) * 2 * np.pi

    # Strong random forcing
    F_ext = (torch.rand(n_traj, device=device) - 0.5) * 2 * F_scale

    # Evolve with forcing
    P1 = P0 + F_ext * dt
    P1 = torch.clamp(P1, min=0.1)  # Keep positive
    Q1 = torch.remainder(Q0 + omega * dt, 2 * np.pi)

    # Convert to (p, q)
    p0 = torch.sqrt(2 * P0 * omega) * torch.sin(Q0)
    q0 = torch.sqrt(2 * P0 / omega) * torch.cos(Q0)
    p1 = torch.sqrt(2 * P1 * omega) * torch.sin(Q1)
    q1 = torch.sqrt(2 * P1 / omega) * torch.cos(Q1)

    return p0, q0, p1, q1, F_ext, P0, Q0, P1, Q1


class StrongForcingLoss(nn.Module):
    """
    Loss with normalized action term.

    Key: Normalize action loss so dP=0 gives loss≈1 regardless of F_scale.
    """

    def __init__(self,
                 recon_weight: float = 1.0,
                 action_weight: float = 10.0,
                 evolution_weight: float = 5.0,
                 symplectic_weight: float = 0.1,
                 use_normalized: bool = True):
        super().__init__()
        self.weights = {
            'recon': recon_weight,
            'action': action_weight,
            'evolution': evolution_weight,
            'symplectic': symplectic_weight
        }
        self.use_normalized = use_normalized

    def forward(self, model, p0, q0, p1, q1, F_ext, omega, dt):
        P0, Q0 = model.encode(p0, q0)
        P1, Q1 = model.encode(p1, q1)

        # Reconstruction
        p0_rec, q0_rec = model.decode(P0, Q0)
        recon_loss = torch.mean((p0 - p0_rec)**2 + (q0 - q0_rec)**2)

        # Action response (normalized or raw)
        dP_expected = F_ext * dt
        dP_actual = P1 - P0

        if self.use_normalized:
            # Normalized: dP=0 gives loss≈1 regardless of scale
            mse = torch.mean((dP_actual - dP_expected)**2)
            normalizer = torch.mean(dP_expected**2) + 1e-6
            action_loss = mse / normalizer
        else:
            action_loss = torch.mean((dP_actual - dP_expected)**2)

        # Evolution
        dQ_expected = omega * dt
        dQ_actual = Q1 - Q0
        evolution_loss = torch.mean(1 - torch.cos(dQ_actual - dQ_expected))

        # Symplectic
        symplectic_loss = self.compute_symplectic_loss(model, p0, q0)

        total = (self.weights['recon'] * recon_loss +
                 self.weights['action'] * action_loss +
                 self.weights['evolution'] * evolution_loss +
                 self.weights['symplectic'] * symplectic_loss)

        return {
            'total': total,
            'recon': recon_loss,
            'action': action_loss,
            'evolution': evolution_loss,
            'symplectic': symplectic_loss
        }

    def compute_symplectic_loss(self, model, p, q, eps=1e-4):
        n = min(len(p), 32)
        pb_values = []
        for i in range(n):
            pi, qi = p[i:i+1], q[i:i+1]
            P_pp, Q_pp = model.encode(pi + eps, qi)
            P_pm, Q_pm = model.encode(pi - eps, qi)
            P_qp, Q_qp = model.encode(pi, qi + eps)
            P_qm, Q_qm = model.encode(pi, qi - eps)
            dP_dp = (P_pp - P_pm) / (2 * eps)
            dP_dq = (P_qp - P_qm) / (2 * eps)
            dQ_dp = (Q_pp - Q_pm) / (2 * eps)
            dQ_dq = (Q_qp - Q_qm) / (2 * eps)
            pb = dP_dq * dQ_dp - dP_dp * dQ_dq
            pb_values.append(pb)
        pb_tensor = torch.cat(pb_values)
        return torch.mean((torch.abs(pb_tensor) - 1.0)**2)


def train_strong_forcing(model, n_epochs=3000, n_traj=100, omega=1.0,
                          dt=0.3, F_scale=3.0, lr=1e-3, device=None,
                          use_normalized=True, verbose=False):
    """Train with strong forcing."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = StrongForcingLoss(use_normalized=use_normalized)

    losses = []
    for epoch in range(n_epochs):
        p0, q0, p1, q1, F_ext, _, _, _, _ = \
            generate_forced_sho_batch(n_traj, dt, omega, F_scale, device)

        optimizer.zero_grad()
        loss_dict = loss_fn(model, p0, q0, p1, q1, F_ext, omega, dt)
        loss_dict['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append({k: v.item() for k, v in loss_dict.items()})

        if verbose and epoch % 500 == 0:
            ld = losses[-1]
            print(f"  Epoch {epoch:4d}: total={ld['total']:.4f}, "
                  f"action={ld['action']:.4f}, evol={ld['evolution']:.4f}")

    return losses


def evaluate_model(model, omega=1.0, device='cpu'):
    """Evaluate P correlation with ground truth."""
    model.eval()

    # Generate test data
    np.random.seed(9999)
    n_test = 200
    P_true = np.random.uniform(0.5, 2.5, n_test)
    Q_true = np.random.uniform(0, 2*np.pi, n_test)

    p = np.sqrt(2 * P_true * omega) * np.sin(Q_true)
    q = np.sqrt(2 * P_true / omega) * np.cos(Q_true)

    with torch.no_grad():
        p_t = torch.tensor(p, dtype=torch.float32, device=device)
        q_t = torch.tensor(q, dtype=torch.float32, device=device)
        P_pred, Q_pred = model.encode(p_t, q_t)
        P_pred = P_pred.cpu().numpy()

    r_P, _ = pearsonr(P_pred, P_true)
    rho_P, _ = spearmanr(P_pred, P_true)

    return {'P_pearson': r_P, 'P_spearman': rho_P}


def run_sweep_experiment():
    """Sweep F_scale to find threshold for convergence."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    F_scales = [0.3, 1.0, 3.0, 10.0]
    n_seeds = 5
    n_epochs = 3000

    results = []

    print("=" * 70)
    print("STRONG FORCING SWEEP EXPERIMENT")
    print("=" * 70)
    print(f"\nTesting F_scale = {F_scales}")
    print(f"Each with {n_seeds} seeds, {n_epochs} epochs")
    print("NO gauge supervision - physics + forcing only\n")

    for F_scale in F_scales:
        print(f"\n{'='*60}")
        print(f"F_SCALE = {F_scale} (|ΔP| ~ {F_scale * 0.3:.2f})")
        print(f"{'='*60}")

        converged_count = 0
        spearmans = []

        for seed in range(n_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = ImprovedHJB_MLP(hidden_dim=64, num_layers=3)

            print(f"\n  Seed {seed}:", end=" ")
            losses = train_strong_forcing(
                model, n_epochs=n_epochs, F_scale=F_scale,
                device=device, use_normalized=True, verbose=False
            )

            eval_result = evaluate_model(model, device=device)
            spearman = eval_result['P_spearman']
            pearson = eval_result['P_pearson']

            converged = abs(spearman) > 0.9
            if converged:
                converged_count += 1

            spearmans.append(spearman)

            result = ForcingResult(
                F_scale=F_scale,
                seed=seed,
                P_spearman=spearman,
                P_pearson=pearson,
                final_loss=losses[-1]['total'],
                converged=converged
            )
            results.append(result)

            status = "✓" if converged else "✗"
            print(f"Spearman={spearman:.4f} [{status}]", end="")

        print(f"\n\n  Summary: {converged_count}/{n_seeds} converged")
        print(f"  Spearmans: {[f'{s:.3f}' for s in spearmans]}")

    return results


def analyze_sweep_results(results: List[ForcingResult]):
    """Analyze sweep results."""

    print("\n" + "=" * 70)
    print("SWEEP ANALYSIS")
    print("=" * 70)

    # Group by F_scale
    from collections import defaultdict
    by_scale = defaultdict(list)
    for r in results:
        by_scale[r.F_scale].append(r)

    print("\n| F_scale | |ΔP| est | Converged | Mean |ρ| |")
    print("|---------|---------|-----------|----------|")

    for F_scale in sorted(by_scale.keys()):
        runs = by_scale[F_scale]
        n_conv = sum(r.converged for r in runs)
        mean_rho = np.mean([abs(r.P_spearman) for r in runs])
        dP_est = F_scale * 0.3
        print(f"| {F_scale:7.1f} | {dP_est:7.2f} | {n_conv}/5       | {mean_rho:.4f}   |")

    # Find threshold
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    full_convergence_scales = [
        F for F in by_scale.keys()
        if sum(r.converged for r in by_scale[F]) == 5
    ]

    if full_convergence_scales:
        threshold = min(full_convergence_scales)
        print(f"\n✓ FULL CONVERGENCE at F_scale ≥ {threshold}")
        print(f"  Physics + forcing DOES determine (P, Q) without gauge!")
        print(f"  Glinsky's framework VALIDATED")
    else:
        best_scale = max(by_scale.keys(),
                         key=lambda F: sum(r.converged for r in by_scale[F]))
        best_conv = sum(r.converged for r in by_scale[best_scale])
        print(f"\n◐ Best: {best_conv}/5 at F_scale={best_scale}")
        print(f"  May need even stronger forcing or more epochs")


def create_sweep_visualization(results: List[ForcingResult],
                                save_path='forcing_sweep.png'):
    """Visualize F_scale sweep."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        from collections import defaultdict
        by_scale = defaultdict(list)
        for r in results:
            by_scale[r.F_scale].append(r)

        F_scales = sorted(by_scale.keys())

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. Convergence rate vs F_scale
        ax = axes[0]
        conv_rates = [sum(r.converged for r in by_scale[F])/5 for F in F_scales]
        ax.bar(range(len(F_scales)), conv_rates, color='steelblue', alpha=0.7)
        ax.set_xticks(range(len(F_scales)))
        ax.set_xticklabels([str(F) for F in F_scales])
        ax.set_xlabel('F_scale')
        ax.set_ylabel('Convergence Rate')
        ax.set_title('Convergence vs Forcing Strength')
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
        ax.set_ylim(0, 1.1)

        # 2. Mean |Spearman| vs F_scale
        ax = axes[1]
        mean_rhos = [np.mean([abs(r.P_spearman) for r in by_scale[F]]) for F in F_scales]
        ax.bar(range(len(F_scales)), mean_rhos, color='darkorange', alpha=0.7)
        ax.set_xticks(range(len(F_scales)))
        ax.set_xticklabels([str(F) for F in F_scales])
        ax.set_xlabel('F_scale')
        ax.set_ylabel('Mean |Spearman|')
        ax.set_title('Correlation Quality vs Forcing')
        ax.axhline(y=0.95, color='green', linestyle='--', label='Threshold')
        ax.set_ylim(0, 1.1)
        ax.legend()

        # 3. All individual Spearmans
        ax = axes[2]
        x_offset = 0
        colors = ['coral', 'steelblue', 'forestgreen', 'purple']
        for i, F in enumerate(F_scales):
            spearmans = [abs(r.P_spearman) for r in by_scale[F]]
            x_pos = [x_offset + j*0.15 for j in range(5)]
            ax.scatter(x_pos, spearmans, c=colors[i % len(colors)],
                       label=f'F={F}', s=100, alpha=0.7)
            x_offset += 1
        ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5)
        ax.set_xlabel('F_scale groups')
        ax.set_ylabel('|Spearman|')
        ax.set_title('Individual Run Results')
        ax.legend()
        ax.set_ylim(0, 1.1)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"\n[Saved sweep visualization to {save_path}]")

    except ImportError:
        print("\n[matplotlib not available]")


if __name__ == "__main__":
    results = run_sweep_experiment()
    analyze_sweep_results(results)
    create_sweep_visualization(results)
