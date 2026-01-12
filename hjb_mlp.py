"""
HJB-MLP: Learning Action-Angle Coordinates from HST Output

Based on Glinsky Figure 9: The neural network architecture to estimate
the solution of the Hamilton-Jacobi-Bellman equation.

The key insight: HST gives you (p,q), the MLP learns (p,q) → (P,Q).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional


class HJB_MLP(nn.Module):
    """
    MLP that learns the canonical transformation from basic (p,q) to 
    fundamental action-angle (P,Q) coordinates.
    
    Architecture from Glinsky Figure 9:
    - Decoder: (p,q) → (P,Q) via S_P(q)
    - Propagator: (P₀,Q₀) → (P,Q) via analytic equations
    - Encoder: (P,Q) → (p,q) via S_p(Q)
    
    The network implicitly learns:
    - S_P(q): generating function / action
    - E(P): energy as function of action  
    - ω_Q(P) = ∂E/∂P: frequency
    - π(q,P) = ∂S_P/∂q: policy (momentum)
    """
    
    def __init__(self, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        
        # Encoder network: (p,q) → hidden representation
        encoder_layers = [nn.Linear(2, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            encoder_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Action head: hidden → P (should be conserved)
        self.action_head = nn.Linear(hidden_dim, 1)
        
        # Angle head: hidden → Q (mod 2π)
        self.angle_head = nn.Linear(hidden_dim, 1)
        
        # Energy head: P → E(P)
        self.energy_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Decoder: (P,Q) → (p,q)
        decoder_layers = [nn.Linear(2, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            decoder_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        decoder_layers.append(nn.Linear(hidden_dim, 2))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, p: torch.Tensor, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map basic coordinates to action-angle: (p,q) → (P,Q)"""
        x = torch.stack([p, q], dim=-1)
        h = self.encoder(x)
        P = self.action_head(h).squeeze(-1)
        Q = self.angle_head(h).squeeze(-1)
        return P, Q
    
    def decode(self, P: torch.Tensor, Q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map action-angle to basic coordinates: (P,Q) → (p,q)"""
        x = torch.stack([P, Q], dim=-1)
        out = self.decoder(x)
        p = out[..., 0]
        q = out[..., 1]
        return p, q
    
    def energy(self, P: torch.Tensor) -> torch.Tensor:
        """Compute E(P) from action"""
        return self.energy_net(P.unsqueeze(-1)).squeeze(-1)
    
    def frequency(self, P: torch.Tensor) -> torch.Tensor:
        """Compute ω_Q(P) = ∂E/∂P using autograd"""
        P_var = P.clone().requires_grad_(True)
        E = self.energy(P_var)
        omega = torch.autograd.grad(E.sum(), P_var, create_graph=True)[0]
        return omega
    
    def propagate(self, P0: torch.Tensor, Q0: torch.Tensor, 
                  dt: float, F_ext: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Analytic propagation in action-angle space.
        
        dP/dτ = F_ext  (only external force changes action)
        dQ/dτ = ω_Q(P)
        
        For conservative system (F_ext = 0): P is conserved!
        """
        omega = self.frequency(P0)
        P = P0 + F_ext * dt
        Q = Q0 + omega * dt
        return P, Q
    
    def forward(self, p0: torch.Tensor, q0: torch.Tensor, 
                dt: float = 0.0, F_ext: float = 0.0) -> dict:
        """
        Full forward pass:
        1. Encode (p0,q0) → (P0,Q0)
        2. Propagate to (P,Q)
        3. Decode (P,Q) → (p,q)
        
        Returns dict with all intermediate values for loss computation.
        """
        # Encode
        P0, Q0 = self.encode(p0, q0)
        
        # Propagate
        P, Q = self.propagate(P0, Q0, dt, F_ext)
        
        # Decode
        p, q = self.decode(P, Q)
        
        # Compute energy and frequency
        E = self.energy(P)
        omega = self.frequency(P)
        
        return {
            'P0': P0, 'Q0': Q0,
            'P': P, 'Q': Q,
            'p': p, 'q': q,
            'E': E, 'omega': omega
        }


class HJBLoss(nn.Module):
    """
    Loss function for training the HJB-MLP.
    
    Components:
    1. Reconstruction loss: (p,q) → (P,Q) → (p,q) should be identity
    2. Conservation loss: P should be constant along trajectory
    3. Consistency loss: predicted (p,q) should match target
    4. (Optional) Angle advancement: Q should increase by ω·dt
    """
    
    def __init__(self, 
                 recon_weight: float = 1.0,
                 conservation_weight: float = 10.0,
                 consistency_weight: float = 1.0):
        super().__init__()
        self.recon_weight = recon_weight
        self.conservation_weight = conservation_weight  
        self.consistency_weight = consistency_weight
        
    def forward(self, 
                model: HJB_MLP,
                p0: torch.Tensor, q0: torch.Tensor,
                p_target: torch.Tensor, q_target: torch.Tensor,
                dt: float) -> Tuple[torch.Tensor, dict]:
        """
        Compute total loss.
        
        Args:
            model: HJB_MLP model
            p0, q0: Initial basic coordinates
            p_target, q_target: Target basic coordinates at time dt
            dt: Time step
        """
        # Forward pass
        out = model(p0, q0, dt)
        
        # 1. Reconstruction loss: encode → decode should recover input
        p_recon, q_recon = model.decode(out['P0'], out['Q0'])
        recon_loss = torch.mean((p_recon - p0)**2 + (q_recon - q0)**2)
        
        # 2. Conservation loss: P should not change (for conservative system)
        conservation_loss = torch.mean((out['P'] - out['P0'])**2)
        
        # 3. Consistency loss: predicted (p,q) should match target
        consistency_loss = torch.mean((out['p'] - p_target)**2 + (out['q'] - q_target)**2)
        
        # Total loss
        total_loss = (self.recon_weight * recon_loss + 
                      self.conservation_weight * conservation_loss +
                      self.consistency_weight * consistency_loss)
        
        loss_dict = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'conservation': conservation_loss.item(),
            'consistency': consistency_loss.item()
        }
        
        return total_loss, loss_dict


def train_on_sho(model: HJB_MLP, 
                 n_epochs: int = 1000,
                 n_trajectories: int = 50,
                 n_points_per_traj: int = 100,
                 omega0: float = 1.0,
                 lr: float = 1e-3) -> list:
    """
    Train HJB-MLP on Simple Harmonic Oscillator trajectories.
    
    For SHO:
    - True action: I = E / ω₀ = (p² + ω₀²q²) / (2ω₀)
    - True angle: θ = arctan(p / (ω₀q))
    - Dynamics: p(t) = p₀ cos(ω₀t) - ω₀q₀ sin(ω₀t)
                q(t) = q₀ cos(ω₀t) + p₀/ω₀ sin(ω₀t)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = HJBLoss(conservation_weight=10.0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    for epoch in range(n_epochs):
        # Generate SHO trajectories
        E_range = np.random.uniform(0.5, 5.0, n_trajectories)
        theta_init = np.random.uniform(0, 2*np.pi, n_trajectories)
        
        # Initial conditions from E and theta
        # For SHO: p = √(2E) sin(θ), q = √(2E/ω₀²) cos(θ)
        p0_np = np.sqrt(2 * E_range) * np.sin(theta_init)
        q0_np = np.sqrt(2 * E_range / omega0**2) * np.cos(theta_init)
        
        # Random time step
        dt = np.random.uniform(0.1, 1.0)
        
        # Analytical evolution for SHO
        p_target_np = p0_np * np.cos(omega0 * dt) - omega0 * q0_np * np.sin(omega0 * dt)
        q_target_np = q0_np * np.cos(omega0 * dt) + p0_np / omega0 * np.sin(omega0 * dt)
        
        # To tensors
        p0 = torch.tensor(p0_np, dtype=torch.float32, device=device)
        q0 = torch.tensor(q0_np, dtype=torch.float32, device=device)
        p_target = torch.tensor(p_target_np, dtype=torch.float32, device=device)
        q_target = torch.tensor(q_target_np, dtype=torch.float32, device=device)
        
        # Forward and backward
        optimizer.zero_grad()
        loss, loss_dict = criterion(model, p0, q0, p_target, q_target, dt)
        loss.backward()
        optimizer.step()
        
        losses.append(loss_dict)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss_dict['total']:.6f}, "
                  f"Recon = {loss_dict['recon']:.6f}, "
                  f"Conserv = {loss_dict['conservation']:.6f}, "
                  f"Consist = {loss_dict['consistency']:.6f}")
    
    return losses


def evaluate_action_angle(model: HJB_MLP, omega0: float = 1.0, n_test: int = 100):
    """
    Evaluate how well learned (P,Q) match true action-angle (I,θ).
    
    For SHO:
    - True action: I = E / ω₀
    - True angle: θ = arctan(p / (ω₀q))
    """
    device = next(model.parameters()).device
    
    # Generate test trajectories
    E_range = np.random.uniform(0.5, 5.0, n_test)
    theta_true = np.random.uniform(0, 2*np.pi, n_test)
    
    # True action
    I_true = E_range / omega0
    
    # Initial conditions
    p_np = np.sqrt(2 * E_range) * np.sin(theta_true)
    q_np = np.sqrt(2 * E_range / omega0**2) * np.cos(theta_true)
    
    p = torch.tensor(p_np, dtype=torch.float32, device=device)
    q = torch.tensor(q_np, dtype=torch.float32, device=device)
    
    # Get learned action-angle
    with torch.no_grad():
        P, Q = model.encode(p, q)
        P_np = P.cpu().numpy()
        Q_np = Q.cpu().numpy()
    
    # Compute correlations
    from scipy.stats import pearsonr
    
    r_action, _ = pearsonr(P_np, I_true)
    
    # For angle, use circular correlation
    # Map both to unit circle and compute
    theta_true_wrapped = theta_true % (2 * np.pi)
    Q_wrapped = Q_np % (2 * np.pi)
    
    # Simple linear correlation on unwrapped angles (for debugging)
    r_angle_linear, _ = pearsonr(Q_np, theta_true)
    
    # Check conservation along trajectory
    dt = 1.0
    p_evolved = p_np * np.cos(omega0 * dt) - omega0 * q_np * np.sin(omega0 * dt)
    q_evolved = q_np * np.cos(omega0 * dt) + p_np / omega0 * np.sin(omega0 * dt)
    
    p_ev = torch.tensor(p_evolved, dtype=torch.float32, device=device)
    q_ev = torch.tensor(q_evolved, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        P_ev, Q_ev = model.encode(p_ev, q_ev)
        P_ev_np = P_ev.cpu().numpy()
    
    # P should be conserved
    P_change = np.abs(P_ev_np - P_np)
    P_rel_change = P_change / (np.abs(P_np) + 1e-10)
    
    results = {
        'r_action': r_action,
        'r_angle_linear': r_angle_linear,
        'P_mean_change': np.mean(P_change),
        'P_rel_change': np.mean(P_rel_change),
        'P_values': P_np,
        'I_true': I_true,
        'Q_values': Q_np,
        'theta_true': theta_true
    }
    
    print(f"\nEvaluation Results:")
    print(f"  Correlation r(P_learned, I_true) = {r_action:.4f}")
    print(f"  Correlation r(Q_learned, θ_true) = {r_angle_linear:.4f}")
    print(f"  Mean |ΔP| along trajectory = {np.mean(P_change):.6f}")
    print(f"  Mean |ΔP|/|P| = {np.mean(P_rel_change):.6f}")
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("HJB-MLP: Learning Action-Angle from Basic Coordinates")
    print("=" * 70)
    
    # Create model
    model = HJB_MLP(hidden_dim=64, num_layers=3)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train on SHO
    print("\n" + "-" * 70)
    print("Training on Simple Harmonic Oscillator...")
    print("-" * 70)
    
    losses = train_on_sho(model, n_epochs=2000, n_trajectories=100, lr=1e-3)
    
    # Evaluate
    print("\n" + "-" * 70)
    print("Evaluating learned action-angle coordinates...")
    print("-" * 70)
    
    results = evaluate_action_angle(model, omega0=1.0, n_test=200)
    
    # The key test: does P correlate with true action I?
    print("\n" + "=" * 70)
    print("KEY RESULT:")
    print("=" * 70)
    if results['r_action'] > 0.95:
        print(f"✓ SUCCESS: r(P_learned, I_true) = {results['r_action']:.4f} > 0.95")
        print("  The MLP successfully learned to extract action!")
    else:
        print(f"✗ NEEDS WORK: r(P_learned, I_true) = {results['r_action']:.4f} < 0.95")
        print("  May need more training or architecture adjustment.")
