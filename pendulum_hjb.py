"""
Pendulum HJB-MLP for rotation regime.

Key differences from SHO:
1. ω depends on action P (energy-dependent frequency)
2. Uses (cos q, sin q) embedding for angle variable q ∈ S¹
3. Frequency supervision during training

Pendulum physics:
    H = p²/2 - ω₀² cos(q)
    Separatrix: E_sep = ω₀²
    Rotation regime: E > E_sep
    Frequency: ω(E) = π√(E/2) / (2K(k)) where k² = 2ω₀²/E
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.integrate import solve_ivp
from scipy.special import ellipk


def pendulum_dynamics(t, y, omega0=1.0):
    """Pendulum ODE: dq/dt = p, dp/dt = -ω₀² sin(q)"""
    q, p = y
    return [p, -omega0**2 * np.sin(q)]


def generate_pendulum_signal(p0, q0, omega0=1.0, dt=0.01, n_points=128):
    """
    Generate pendulum trajectory.

    Args:
        p0, q0: Initial momentum and position
        omega0: Natural frequency parameter
        dt: Time step
        n_points: Number of time points

    Returns:
        z: Complex signal encoding both position AND momentum:
           z = q_wrapped + i*p where q_wrapped = q mod 2π
           The HST extracts features that can then be mapped to (p, cos q, sin q)
        p_t: Momentum trajectory
        q_t: Position trajectory
    """
    t_span = (0, (n_points - 1) * dt)
    t_eval = np.linspace(0, (n_points - 1) * dt, n_points)

    sol = solve_ivp(
        pendulum_dynamics, t_span, [q0, p0],
        t_eval=t_eval, args=(omega0,), method='DOP853', rtol=1e-10
    )

    q_t = sol.y[0]
    p_t = sol.y[1]

    # Use q + i*p signal (similar to SHO)
    # The decoder will handle q's circular nature via (cos q, sin q) embedding
    z = q_t + 1j * p_t
    return z, p_t, q_t


def generate_pendulum_signal_extended(p0, q0, omega0=1.0, dt=0.01, n_points=128):
    """
    Generate extended pendulum signal with both p and q information.

    Returns a 2-channel signal: [cos(q) + i*sin(q), p/p_scale + i*0]
    This preserves both position (topology-aware) and momentum information.
    """
    t_span = (0, (n_points - 1) * dt)
    t_eval = np.linspace(0, (n_points - 1) * dt, n_points)

    sol = solve_ivp(
        pendulum_dynamics, t_span, [q0, p0],
        t_eval=t_eval, args=(omega0,), method='DOP853', rtol=1e-10
    )

    q_t = sol.y[0]
    p_t = sol.y[1]

    # Two-channel signal
    z_q = np.cos(q_t) + 1j * np.sin(q_t)  # Position channel (S¹ embedded)
    z_p = p_t + 1j * 0  # Momentum channel (real-valued)

    # Combine into single complex signal that preserves both
    # Use interleaving: even samples = q info, odd = p info
    # Or concatenate: first half = q, second half = p
    z = np.concatenate([z_q, z_p / 3.0])  # Scale p to similar magnitude

    return z, p_t, q_t


def generate_rotation_ic(E_min=1.5, E_max=3.0, omega0=1.0):
    """
    Generate initial conditions in rotation regime.

    E > ω₀² ensures rotation (not libration).

    Returns:
        p0, q0, E: Initial conditions and energy
    """
    E = np.random.uniform(E_min, E_max) * omega0**2
    q0 = np.random.uniform(0, 2 * np.pi)
    # p from energy: E = p²/2 - ω₀² cos(q) → p = √(2(E + ω₀² cos(q)))
    p0 = np.sqrt(2 * (E + omega0**2 * np.cos(q0)))
    # Randomly choose direction
    if np.random.random() < 0.5:
        p0 = -p0
    return p0, q0, E


def pendulum_energy(p, q, omega0=1.0):
    """Compute pendulum energy H = p²/2 - ω₀² cos(q)"""
    return 0.5 * p**2 - omega0**2 * np.cos(q)


def pendulum_frequency(E, omega0=1.0):
    """
    Compute frequency ω(E) for rotation regime.

    For H = p²/2 - ω₀² cos(q), separatrix at E_sep = ω₀².

    For rotation (E > E_sep):
        Period T = (4/ω₀) * K(k) / √((E + ω₀²)/(2ω₀²))
        where k² = 2ω₀² / (E + ω₀²)

    Reference: Goldstein, Classical Mechanics

    Returns:
        omega: Frequency, or None if E ≤ E_sep (libration regime)
    """
    E_sep = omega0**2
    if E <= E_sep:
        return None

    # Modulus for rotation regime
    k2 = 2 * omega0**2 / (E + omega0**2)
    if k2 >= 1 or k2 <= 0:
        return None

    K = ellipk(k2)
    # Period for rotation
    # T = (4/sqrt(g/l)) * K(k) / sqrt((E + ω₀²)/(2ω₀²))
    # Here g/l = ω₀²
    T = (4 / omega0) * K / np.sqrt((E + omega0**2) / (2 * omega0**2))
    omega = 2 * np.pi / T
    return omega


def pendulum_action(E, omega0=1.0):
    """
    Compute action J(E) for rotation regime.

    For rotation: J = (1/π) √(2E) E(k) where E(k) is elliptic integral

    This is approximate - exact formula involves elliptic integrals.
    """
    from scipy.special import ellipe

    E_sep = omega0**2
    if E <= E_sep:
        return None

    k2 = 2 * omega0**2 / E
    E_ellip = ellipe(k2)  # Complete elliptic integral of second kind

    # Action for rotation regime
    J = (2 / np.pi) * np.sqrt(2 * E) * E_ellip
    return J


class PendulumHJB_MLP(nn.Module):
    """
    HJB-MLP that learns ω(P) for pendulum (energy-dependent frequency).

    Key differences from SHO:
    1. ω depends on action P
    2. Uses (cos q, sin q) embedding for angle q ∈ S¹
    3. Includes frequency predictor network
    """

    def __init__(self, hidden_dim=64, num_layers=3, omega0=1.0):
        super().__init__()
        self.omega0 = omega0

        # Encoder: (p, cos q, sin q) → hidden → (P, sin Q, cos Q)
        encoder_layers = []
        in_dim = 3  # p, cos(q), sin(q)
        for i in range(num_layers - 1):
            encoder_layers.extend([
                nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                nn.Tanh()
            ])
        self.encoder_net = nn.Sequential(*encoder_layers)

        # Separate heads for P and Q
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # P > 0
        )
        self.angle_head = nn.Linear(hidden_dim, 2)  # (sin Q, cos Q)

        # Decoder: (P, sin Q, cos Q) → hidden → (p, cos q, sin q)
        decoder_layers = []
        for i in range(num_layers - 1):
            decoder_layers.extend([
                nn.Linear(3 if i == 0 else hidden_dim, hidden_dim),
                nn.Tanh()
            ])
        decoder_layers.append(nn.Linear(hidden_dim, 3))  # (p, cos q, sin q)
        self.decoder_net = nn.Sequential(*decoder_layers)

        # Frequency predictor: P → ω(P)
        self.omega_net = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # ω > 0
        )

    def encode(self, p, q):
        """
        Map (p, q) → (P, Q) with proper S¹ embedding.

        Args:
            p: Momentum (batch,)
            q: Position/angle (batch,)

        Returns:
            P: Action (batch,)
            Q: Angle (batch,)
        """
        if p.dim() == 0:
            p = p.unsqueeze(0)
            q = q.unsqueeze(0)

        # Embed q ∈ S¹ into ℝ²
        features = torch.stack([p, torch.cos(q), torch.sin(q)], dim=-1)

        h = self.encoder_net(features)

        # P via softplus (always positive)
        P = self.action_head(h).squeeze(-1)

        # Q via atan2 of learned (sin, cos)
        sc = self.angle_head(h)
        sin_Q = sc[..., 0]
        cos_Q = sc[..., 1]
        Q = torch.atan2(sin_Q, cos_Q)

        return P, Q

    def decode(self, P, Q):
        """
        Map (P, Q) → (p, q) with proper S¹ embedding.

        Args:
            P: Action (batch,)
            Q: Angle (batch,)

        Returns:
            p: Momentum (batch,)
            q: Position/angle (batch,)
        """
        if P.dim() == 0:
            P = P.unsqueeze(0)
            Q = Q.unsqueeze(0)

        features = torch.stack([P, torch.sin(Q), torch.cos(Q)], dim=-1)
        out = self.decoder_net(features)

        p = out[..., 0]
        cos_q = out[..., 1]
        sin_q = out[..., 2]
        q = torch.atan2(sin_q, cos_q)

        return p, q

    def get_omega(self, P):
        """Predict frequency from action."""
        if P.dim() == 0:
            P = P.unsqueeze(0)
        return self.omega_net(P.unsqueeze(-1)).squeeze(-1)

    def propagate(self, P, Q, dt):
        """
        Propagate with energy-dependent ω(P).

        dQ/dt = ω(P), dP/dt = 0 (action conserved)
        """
        omega = self.get_omega(P)
        Q_new = Q + omega * dt
        return P, Q_new


def train_pendulum_hjb(model, omega0=1.0, n_epochs=1000, dt=0.1,
                       n_batch=100, lr=1e-3, device='cpu', verbose=True):
    """
    Train HJB-MLP on pendulum rotation data.

    Losses:
    1. Reconstruction: (p,q) → (P,Q) → (p,q)
    2. Action conservation: P₀ ≈ P₁ for same trajectory
    3. Frequency supervision: learned ω(P) ≈ true ω(E)
    4. Evolution: dQ ≈ ω(P) · dt
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses_history = []

    for epoch in range(n_epochs):
        # Generate batch of rotation trajectories
        p0_list, q0_list, E_list = [], [], []
        p1_list, q1_list = [], []

        for _ in range(n_batch):
            p0, q0, E = generate_rotation_ic(E_min=1.5, E_max=3.0, omega0=omega0)

            # Evolve by dt using ODE solver
            sol = solve_ivp(
                pendulum_dynamics, (0, dt), [q0, p0],
                args=(omega0,), method='DOP853', rtol=1e-10
            )
            q1, p1 = sol.y[0, -1], sol.y[1, -1]

            p0_list.append(p0)
            q0_list.append(q0)
            p1_list.append(p1)
            q1_list.append(q1)
            E_list.append(E)

        p0 = torch.tensor(p0_list, dtype=torch.float32, device=device)
        q0 = torch.tensor(q0_list, dtype=torch.float32, device=device)
        p1 = torch.tensor(p1_list, dtype=torch.float32, device=device)
        q1 = torch.tensor(q1_list, dtype=torch.float32, device=device)
        E_true = torch.tensor(E_list, dtype=torch.float32, device=device)

        # Encode both states
        P0, Q0 = model.encode(p0, q0)
        P1, Q1 = model.encode(p1, q1)

        # Loss 1: Reconstruction (using circular loss for angles)
        p0_rec, q0_rec = model.decode(P0, Q0)
        loss_recon_p = F.mse_loss(p0_rec, p0)
        loss_recon_q = torch.mean(1 - torch.cos(q0_rec - q0))  # Circular loss
        loss_recon = loss_recon_p + loss_recon_q

        # Loss 2: Action conservation (P should be same on trajectory)
        loss_action = F.mse_loss(P0, P1)

        # Loss 3: P should correlate with E (scaled)
        # Normalize E to similar range as P
        E_min, E_max = omega0**2 * 1.5, omega0**2 * 3.0  # Training range
        E_normalized = (E_true - E_min) / (E_max - E_min + 1e-8)
        P_mean = P0.mean()
        P_std = P0.std() + 1e-8
        P_normalized = (P0 - P_mean) / P_std
        E_norm_std = (E_normalized - E_normalized.mean()) / (E_normalized.std() + 1e-8)
        loss_energy = F.mse_loss(P_normalized, E_norm_std)  # Correlation loss

        # Loss 4: Frequency supervision
        omega_true_list = []
        for E in E_list:
            omega_val = pendulum_frequency(E, omega0)
            omega_true_list.append(omega_val if omega_val is not None else 0.0)
        omega_true = torch.tensor(omega_true_list, dtype=torch.float32, device=device)
        omega_pred = model.get_omega(P0)
        loss_omega = F.mse_loss(omega_pred, omega_true)

        # Loss 5: Evolution (Q advances by ω·dt)
        dQ_pred = omega_pred * dt
        dQ_actual = Q1 - Q0
        # Circular loss for angle evolution
        loss_evol = torch.mean(1 - torch.cos(dQ_pred - dQ_actual))

        # Combined loss (with energy correlation)
        # Higher weight on reconstruction to ensure accurate encode-decode cycle
        loss = 20 * loss_recon + 10 * loss_action + 10 * loss_energy + 5 * loss_omega + 5 * loss_evol

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses_history.append({
            'total': loss.item(),
            'recon': loss_recon.item(),
            'action': loss_action.item(),
            'energy': loss_energy.item(),
            'omega': loss_omega.item(),
            'evol': loss_evol.item()
        })

        if verbose and (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, "
                  f"recon={loss_recon.item():.4f}, action={loss_action.item():.4f}, "
                  f"energy={loss_energy.item():.4f}, omega={loss_omega.item():.4f}")

    return losses_history


class PendulumNullspaceDecoder(nn.Module):
    """
    Nullspace decoder with proper S¹ embedding for pendulum angle variable.

    Key insight: q ∈ S¹, so we embed via (cos q, sin q) to avoid
    topological obstruction.

    Also supports ReLU vs Tanh activations for comparison.
    """

    ACTIVATIONS = {
        'tanh': nn.Tanh,
        'relu': nn.ReLU,
    }

    def __init__(self, W_beta_to_pcs, hjb_encoder, activation='tanh', hidden_dim=64):
        """
        Args:
            W_beta_to_pcs: Linear map β → (p, cos q, sin q), shape (n_beta, 3)
            hjb_encoder: Pre-trained PendulumHJB_MLP
            activation: 'tanh' or 'relu'
            hidden_dim: Hidden layer size
        """
        super().__init__()

        self.activation_name = activation
        act_fn = self.ACTIVATIONS[activation]

        n_beta = W_beta_to_pcs.shape[0]
        n_out = W_beta_to_pcs.shape[1]  # Should be 3: (p, cos q, sin q)

        # Register buffers
        self.register_buffer('W', torch.tensor(W_beta_to_pcs, dtype=torch.float32))
        W_pinv = np.linalg.pinv(W_beta_to_pcs)
        self.register_buffer('W_pinv', torch.tensor(W_pinv, dtype=torch.float32))

        # Nullspace projector
        P_parallel = W_beta_to_pcs @ W_pinv
        P_perp = np.eye(n_beta) - P_parallel
        self.register_buffer('P_perp', torch.tensor(P_perp, dtype=torch.float32))

        # Frozen HJB encoder
        self.hjb_encoder = hjb_encoder
        for param in self.hjb_encoder.parameters():
            param.requires_grad = False

        self.n_beta = n_beta

        # Input: (p, cos(q), sin(q), P, sin(Q), cos(Q)) = 6 features
        # Includes both phase space and action-angle coordinates
        n_features = 6

        # Deeper network with residual-like structure
        self.g = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, n_beta)
        )

        # Small initialization
        for m in self.g.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, p, q):
        """
        Decode (p, q) → β using S¹-aware embedding and HJB action-angle features.

        q is treated as an angle: we use (cos q, sin q) not q directly.
        We also include (P, sin Q, cos Q) from HJB encoder for richer features.
        """
        if p.dim() == 0:
            p = p.unsqueeze(0)
            q = q.unsqueeze(0)

        # Embed q ∈ S¹ into ℝ²
        cos_q = torch.cos(q)
        sin_q = torch.sin(q)

        # Linear part: uses (p, cos q, sin q) directly
        pcs = torch.stack([p, cos_q, sin_q], dim=-1)
        beta_linear = pcs @ self.W_pinv

        # Get action-angle coordinates from HJB encoder
        with torch.no_grad():
            P, Q = self.hjb_encoder.encode(p, q)

        # Rich feature set for correction: (p, cos q, sin q, P, sin Q, cos Q)
        features = torch.stack([
            p, cos_q, sin_q,
            P, torch.sin(Q), torch.cos(Q)
        ], dim=-1)

        g_output = self.g(features)
        beta_correction = g_output @ self.P_perp.T

        return beta_linear + beta_correction


def train_pendulum_nullspace_decoder(decoder, p_train, q_train, beta_train,
                                      epochs=1000, lr=1e-3, verbose=True):
    """Train pendulum nullspace decoder."""
    p_train = torch.tensor(p_train, dtype=torch.float32)
    q_train = torch.tensor(q_train, dtype=torch.float32)
    beta_train = torch.tensor(beta_train, dtype=torch.float32)

    optimizer = torch.optim.Adam(decoder.g.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        decoder.train()
        beta_pred = decoder(p_train, q_train)
        loss = F.mse_loss(beta_pred, beta_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if verbose and (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f}")

    return losses


def diagnose_forecast_error(hjb_encoder, p0, q0, T, omega0, dt=0.01):
    """
    Decompose forecast error into components.

    Helps identify whether error comes from:
    1. (p,q) prediction error at time T (HJB decode inaccuracy)
    2. Phase (Q) drift from ω(P) bias (frequency estimation error)
    3. Action (P) drift (conservation violation)

    Args:
        hjb_encoder: Trained PendulumHJB_MLP
        p0, q0: Initial state (center of window)
        T: Propagation time
        omega0: Natural frequency parameter
        dt: Time step for ODE integration

    Returns:
        dict with error decomposition
    """
    import torch

    # Ground truth at T via ODE
    z_true, p_t, q_t = generate_pendulum_signal(p0, q0, omega0, dt,
                                                 n_points=int(T/dt)+1)
    p_T_true = p_t[-1]
    q_T_true = q_t[-1]

    # Predicted (p,q) at T via HJB propagation
    hjb_encoder.eval()
    with torch.no_grad():
        p0_t = torch.tensor([p0], dtype=torch.float32)
        q0_t = torch.tensor([q0], dtype=torch.float32)

        # Encode initial state
        P0, Q0 = hjb_encoder.encode(p0_t, q0_t)

        # Propagate
        P_T, Q_T = hjb_encoder.propagate(P0, Q0, T)

        # Decode
        p_T_pred, q_T_pred = hjb_encoder.decode(P_T, Q_T)

        p_T_pred = p_T_pred.item()
        q_T_pred = q_T_pred.item()
        P0_val = P0.item()
        Q0_val = Q0.item()
        P_T_val = P_T.item()
        Q_T_val = Q_T.item()

    # Error decomposition
    p_error = abs(p_T_pred - p_T_true)
    q_error = abs(np.sin(q_T_pred - q_T_true))  # Circular distance

    # ω bias check
    E0 = pendulum_energy(p0, q0, omega0)
    omega_true = pendulum_frequency(E0, omega0)
    if omega_true is None:
        omega_true = 0.0

    omega_pred = hjb_encoder.get_omega(torch.tensor([P0_val])).item()
    omega_bias = omega_pred - omega_true
    expected_Q_drift = omega_bias * T

    # Action conservation check
    P_drift = abs(P_T_val - P0_val)

    return {
        'p_error': p_error,
        'q_error': q_error,
        'omega_true': omega_true,
        'omega_pred': omega_pred,
        'omega_bias': omega_bias,
        'omega_rel_error': abs(omega_bias) / (omega_true + 1e-12),
        'expected_Q_drift': expected_Q_drift,
        'P_drift': P_drift,
        'T': T,
    }
