"""
Complex-Valued MLP for HJB Decoder

From Glinsky's theory:
- H(beta) is ANALYTIC (holomorphic) -> minimal surface (Laplace: nabla^2 H = 0)
- Minimal surfaces are FLAT except at SINGULARITIES beta*
- ReLU networks are PIECE-WISE LINEAR = flat regions + kinks
- The KINKS should align with the SINGULARITIES

This module implements:
1. ComplexLinear: Complex-valued linear layer
2. ComplexReLU: Complex activation (modReLU, CReLU, zReLU)
3. HJB_MLP: MLP that learns geodesic coordinates (P, Q)

The HJB decoder learns the canonical transformation:
    beta (ROM coords) -> (P, Q) (action-angle coords)

Where:
    dP/dtau = 0       (P conserved - action variables)
    dQ/dtau = omega(P) (Q evolves linearly - angle variables)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ComplexLinear(nn.Module):
    """
    Complex-valued linear layer.

    For complex weight W = A + iB and input z = x + iy:
    W*z = (Ax - By) + i(Bx + Ay)

    Parameters
    ----------
    in_features : int
        Input dimension
    out_features : int
        Output dimension
    bias : bool
        Include bias term
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # Real and imaginary parts of the weight matrix
        self.weight_real = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_imag = nn.Parameter(torch.randn(out_features, in_features) * 0.1)

        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_features))
            self.bias_imag = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)

    def forward(self, z):
        """
        Parameters
        ----------
        z : torch.Tensor (complex64 or complex128)
            Input tensor

        Returns
        -------
        out : torch.Tensor (complex)
            Complex linear transformation
        """
        x, y = z.real, z.imag

        # Complex matrix multiplication: (A + iB)(x + iy) = (Ax - By) + i(Bx + Ay)
        real_out = F.linear(x, self.weight_real) - F.linear(y, self.weight_imag)
        imag_out = F.linear(x, self.weight_imag) + F.linear(y, self.weight_real)

        if self.bias_real is not None:
            real_out = real_out + self.bias_real
            imag_out = imag_out + self.bias_imag

        return torch.complex(real_out, imag_out)


class ComplexReLU(nn.Module):
    """
    Complex ReLU activation.

    Options:
    1. modReLU: ReLU(|z| - b) * z/|z| (Arjovsky et al.)
       - Preserves phase
       - Creates magnitude threshold at |z| = b

    2. CReLU: ReLU(Re(z)) + i*ReLU(Im(z))
       - Separable (acts on real/imag independently)
       - Kinks at Re=0 and Im=0 axes

    3. zReLU: z if Re(z) > 0 and Im(z) > 0, else 0
       - First quadrant only
       - Creates kink at both axes

    For Glinsky's theory, we want piece-wise linear in COMPLEX plane.
    modReLU preserves phase, CReLU is separable.

    Parameters
    ----------
    mode : str
        'modReLU', 'CReLU', or 'zReLU'
    bias : float
        Bias for modReLU threshold
    learnable_bias : bool
        If True, make bias a learnable parameter
    """

    def __init__(self, mode='modReLU', bias=0.0, learnable_bias=False):
        super().__init__()
        self.mode = mode

        if learnable_bias:
            self.bias = nn.Parameter(torch.tensor(bias))
        else:
            self.register_buffer('bias', torch.tensor(bias))

    def forward(self, z):
        """
        Parameters
        ----------
        z : torch.Tensor (complex)
            Input tensor

        Returns
        -------
        out : torch.Tensor (complex)
            Activated output
        """
        if self.mode == 'modReLU':
            # ReLU on magnitude, preserve phase
            magnitude = torch.abs(z)
            phase = z / (magnitude + 1e-8)
            activated_mag = F.relu(magnitude - self.bias)
            return activated_mag * phase

        elif self.mode == 'CReLU':
            # Apply ReLU to real and imaginary separately
            return torch.complex(
                F.relu(z.real),
                F.relu(z.imag)
            )

        elif self.mode == 'zReLU':
            # Zero unless both Re and Im are positive
            mask = (z.real > 0) & (z.imag > 0)
            return z * mask.float()

        elif self.mode == 'cardioid':
            # Cardioid activation: z * (1 + cos(phase))/2
            # Smooth alternative to zReLU
            phase = torch.angle(z)
            scale = 0.5 * (1 + torch.cos(phase))
            return z * scale

        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class RealLinear(nn.Module):
    """
    Linear layer that operates on stacked real/imag representation.

    This allows using standard ReLU which creates piece-wise linear
    structure in the full (real, imag) space.
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # Input: 2*in_features (real, imag stacked)
        # Output: 2*out_features
        self.linear = nn.Linear(2 * in_features, 2 * out_features, bias=bias)
        self.out_features = out_features

    def forward(self, z):
        """
        Parameters
        ----------
        z : torch.Tensor (complex)
            Input tensor

        Returns
        -------
        out : torch.Tensor (complex)
            Output tensor
        """
        # Stack real and imaginary
        x = torch.cat([z.real, z.imag], dim=-1)

        # Linear transform
        y = self.linear(x)

        # Split back to complex
        return torch.complex(y[..., :self.out_features], y[..., self.out_features:])


class HJB_MLP(nn.Module):
    """
    MLP that learns the HJB generating function S(P;q).

    From Glinsky:
    - H(beta) is analytic (minimal surface)
    - Singularities beta* define topology
    - MLP with ReLU matches this structure (piece-wise linear = flat + kinks)

    The network learns:
    - Input: ROM coordinates beta (from HST+PCA)
    - Output: Geodesic coordinates (P, Q)
    - Constraint: dP/dtau = 0 (P conserved), dQ/dtau = omega(P) (linear in time)

    Parameters
    ----------
    input_dim : int
        Dimension of input (ROM beta coordinates)
    hidden_dims : list of int
        Hidden layer dimensions
    output_dim : int
        Output dimension (defaults to input_dim)
    complex_mode : bool
        Use complex-valued layers
    activation : str
        Activation type ('modReLU', 'CReLU', 'ReLU')
    """

    def __init__(self, input_dim, hidden_dims=[64, 64], output_dim=None,
                 complex_mode=False, activation='modReLU'):
        super().__init__()

        if output_dim is None:
            output_dim = input_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.complex_mode = complex_mode

        # Build network
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            if complex_mode:
                layers.append(ComplexLinear(prev_dim, h_dim))
                layers.append(ComplexReLU(mode=activation, learnable_bias=True))
            else:
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(nn.ReLU())
            prev_dim = h_dim

        # Output layer (no activation)
        if complex_mode:
            layers.append(ComplexLinear(prev_dim, output_dim))
        else:
            layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Separate network for frequency omega(P)
        omega_input = output_dim // 2
        self.omega_net = nn.Sequential(
            nn.Linear(omega_input, 32),
            nn.ReLU(),
            nn.Linear(32, omega_input)
        )

    def forward(self, beta):
        """
        Map ROM coordinates beta to geodesic coordinates (P, Q).

        Parameters
        ----------
        beta : torch.Tensor
            ROM coordinates, shape (..., input_dim)

        Returns
        -------
        P : torch.Tensor
            Action variables (should be conserved)
        Q : torch.Tensor
            Angle variables (should evolve as dQ/dt = omega(P))
        """
        # Forward through network
        PQ = self.network(beta)

        # Split into P and Q
        dim = PQ.shape[-1] // 2
        P = PQ[..., :dim]
        Q = PQ[..., dim:]

        return P, Q

    def compute_omega(self, P):
        """
        Compute frequency omega(P) for angle evolution.

        Parameters
        ----------
        P : torch.Tensor
            Action variables

        Returns
        -------
        omega : torch.Tensor
            Frequencies for each action
        """
        if self.complex_mode and P.is_complex():
            # Use magnitude for frequency computation
            P_real = torch.cat([P.real, P.imag], dim=-1)
        else:
            P_real = P if P.is_floating_point() else P.real

        omega = self.omega_net(P_real)
        return omega


def geodesic_loss(P_t, P_t1, Q_t, Q_t1, omega, dt=1.0):
    """
    Loss function enforcing geodesic motion.

    Geodesic constraint (Hamilton-Jacobi):
    - dP/dtau = 0  ->  P_{t+1} = P_t  (action conserved)
    - dQ/dtau = omega(P)  ->  Q_{t+1} = Q_t + omega(P)*dt  (angle linear)

    Parameters
    ----------
    P_t, P_t1 : torch.Tensor
        Action variables at t and t+1
    Q_t, Q_t1 : torch.Tensor
        Angle variables at t and t+1
    omega : torch.Tensor
        Frequencies omega(P)
    dt : float
        Time step

    Returns
    -------
    loss : torch.Tensor
        Total loss
    loss_dict : dict
        Individual loss components
    """
    # Handle complex tensors
    if P_t.is_complex():
        # Use magnitude for loss computation
        P_diff = torch.abs(P_t1 - P_t)
        Q_predicted = Q_t + omega * dt
        Q_diff = torch.abs(Q_t1 - Q_predicted)
    else:
        P_diff = P_t1 - P_t
        Q_predicted = Q_t + omega * dt
        Q_diff = Q_t1 - Q_predicted

    # P conservation loss (should be zero)
    loss_P = torch.mean(P_diff ** 2)

    # Q linear evolution loss
    loss_Q = torch.mean(Q_diff ** 2)

    # Total loss
    loss = loss_P + loss_Q

    return loss, {'loss_P': loss_P.item(), 'loss_Q': loss_Q.item()}


def smoothness_regularization(model, beta_samples):
    """
    Regularization to encourage smooth mapping (reduce spurious kinks).

    Compute gradient norm to penalize overly jagged piece-wise linear regions.

    Parameters
    ----------
    model : HJB_MLP
        The model
    beta_samples : torch.Tensor
        Sample points in beta space

    Returns
    -------
    reg_loss : torch.Tensor
        Regularization loss
    """
    beta_samples.requires_grad_(True)
    P, Q = model(beta_samples)

    # Compute gradients
    grad_P = torch.autograd.grad(P.sum(), beta_samples, create_graph=True)[0]
    grad_Q = torch.autograd.grad(Q.sum(), beta_samples, create_graph=True)[0]

    # Penalize large gradient changes (second derivative)
    # This encourages larger flat regions with fewer kinks
    reg_P = torch.mean(grad_P ** 2)
    reg_Q = torch.mean(grad_Q ** 2)

    return reg_P + reg_Q


def train_hjb_mlp(rom, trajectories, n_epochs=1000, lr=1e-3,
                  window_stride=32, device='cpu', complex_mode=False,
                  verbose=True):
    """
    Train HJB MLP to learn geodesic coordinates.

    Parameters
    ----------
    rom : HST_ROM
        Fitted ROM (provides transform to beta coordinates)
    trajectories : list of arrays
        Training trajectories
    n_epochs : int
        Number of training epochs
    lr : float
        Learning rate
    window_stride : int
        Stride for extracting windows
    device : str
        'cpu' or 'cuda'
    complex_mode : bool
        Use complex-valued network
    verbose : bool
        Print progress

    Returns
    -------
    model : HJB_MLP
        Trained model
    history : dict
        Training history
    """
    device = torch.device(device)

    # Extract beta trajectories from all training trajectories
    all_betas = []
    for traj in trajectories:
        # Use ROM's transform_trajectory if available
        if hasattr(rom, 'transform_trajectory'):
            betas, times = rom.transform_trajectory(traj, window_stride=window_stride)
        else:
            # Simple windowed transform
            betas = []
            ws = rom.window_size if hasattr(rom, 'window_size') else 128
            for i in range(0, len(traj) - ws, window_stride):
                beta = rom.transform(traj[i:i + ws])
                betas.append(beta)
            betas = np.array(betas)

        if len(betas) > 1:
            all_betas.append(betas)

    if not all_betas:
        raise ValueError("No valid beta trajectories extracted")

    # Determine input dimension
    input_dim = all_betas[0].shape[-1]

    # Create model
    model = HJB_MLP(
        input_dim=input_dim,
        hidden_dims=[128, 128, 64],
        output_dim=input_dim,
        complex_mode=complex_mode,
        activation='modReLU' if complex_mode else 'ReLU'
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=100
    )

    history = {'loss': [], 'loss_P': [], 'loss_Q': []}

    for epoch in range(n_epochs):
        total_loss = 0
        total_loss_P = 0
        total_loss_Q = 0
        n_batches = 0

        for betas in all_betas:
            if len(betas) < 2:
                continue

            # Convert to torch
            if complex_mode and np.iscomplexobj(betas):
                beta_t = torch.tensor(betas[:-1], dtype=torch.complex64, device=device)
                beta_t1 = torch.tensor(betas[1:], dtype=torch.complex64, device=device)
            else:
                # Use real part only
                beta_t = torch.tensor(np.real(betas[:-1]), dtype=torch.float32, device=device)
                beta_t1 = torch.tensor(np.real(betas[1:]), dtype=torch.float32, device=device)

            # Forward pass
            P_t, Q_t = model(beta_t)
            P_t1, Q_t1 = model(beta_t1)

            # Compute frequency
            P_for_omega = P_t.real if P_t.is_complex() else P_t
            omega = model.compute_omega(P_for_omega)

            # Compute loss
            loss, loss_dict = geodesic_loss(P_t, P_t1, Q_t, Q_t1, omega)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_P += loss_dict['loss_P']
            total_loss_Q += loss_dict['loss_Q']
            n_batches += 1

        # Record history
        avg_loss = total_loss / max(n_batches, 1)
        scheduler.step(avg_loss)

        history['loss'].append(avg_loss)
        history['loss_P'].append(total_loss_P / max(n_batches, 1))
        history['loss_Q'].append(total_loss_Q / max(n_batches, 1))

        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss={avg_loss:.6f}, "
                  f"P_loss={history['loss_P'][-1]:.6f}, "
                  f"Q_loss={history['loss_Q'][-1]:.6f}")

    return model, history


def visualize_mlp_singularities(model, input_dim=2, beta_range=(-3, 3),
                                resolution=100, device='cpu'):
    """
    Visualize where the MLP has "kinks" (ReLU transitions).

    These should correspond to the singularities beta* of the analytic Hamiltonian.

    Parameters
    ----------
    model : HJB_MLP
        Trained model
    input_dim : int
        Input dimension (only uses first 2 for visualization)
    beta_range : tuple
        (min, max) range for beta values
    resolution : int
        Grid resolution
    device : str
        Computation device

    Returns
    -------
    results : dict
        Contains P, Q grids and singularity detection metrics
    """
    import matplotlib.pyplot as plt

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    # Create grid in beta_1 - beta_2 plane
    b = torch.linspace(beta_range[0], beta_range[1], resolution)
    B1, B2 = torch.meshgrid(b, b, indexing='ij')

    # Stack into input (other beta components are 0)
    beta_grid = torch.zeros(resolution, resolution, model.input_dim, device=device)
    beta_grid[..., 0] = B1.to(device)
    if model.input_dim > 1:
        beta_grid[..., 1] = B2.to(device)
    beta_flat = beta_grid.reshape(-1, model.input_dim)

    # Forward pass
    with torch.no_grad():
        P, Q = model(beta_flat)

    P = P.cpu().numpy().reshape(resolution, resolution, -1)
    Q = Q.cpu().numpy().reshape(resolution, resolution, -1)
    B1 = B1.numpy()
    B2 = B2.numpy()

    # Compute gradient magnitude (high gradient = steep region)
    dP_db1 = np.gradient(P[..., 0], axis=0)
    dP_db2 = np.gradient(P[..., 0], axis=1)
    grad_mag = np.sqrt(dP_db1**2 + dP_db2**2)

    # Second derivative (Laplacian) to find kinks
    # For piece-wise linear, gradient is constant within regions
    # but changes at boundaries (kinks)
    d2P_db1 = np.gradient(dP_db1, axis=0)
    d2P_db2 = np.gradient(dP_db2, axis=1)
    laplacian = np.abs(d2P_db1) + np.abs(d2P_db2)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    im = ax.contourf(B1, B2, P[..., 0], levels=50, cmap='viridis')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel(r'$\beta_1$')
    ax.set_ylabel(r'$\beta_2$')
    ax.set_title(r'$P_1$ (action variable)')

    ax = axes[0, 1]
    im = ax.contourf(B1, B2, Q[..., 0], levels=50, cmap='plasma')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel(r'$\beta_1$')
    ax.set_ylabel(r'$\beta_2$')
    ax.set_title(r'$Q_1$ (angle variable)')

    ax = axes[1, 0]
    im = ax.contourf(B1, B2, grad_mag, levels=50, cmap='hot')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel(r'$\beta_1$')
    ax.set_ylabel(r'$\beta_2$')
    ax.set_title(r'$|\nabla P_1|$ (gradient magnitude)')

    ax = axes[1, 1]
    im = ax.contourf(B1, B2, laplacian, levels=50, cmap='hot')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel(r'$\beta_1$')
    ax.set_ylabel(r'$\beta_2$')
    ax.set_title(r'$|\nabla^2 P_1|$ (kinks/singularities)')

    plt.suptitle('MLP Structure: ReLU Kinks as Learned Singularities', fontsize=14)
    plt.tight_layout()
    plt.savefig('mlp_singularities.png', dpi=150)
    plt.close()
    print("Saved: mlp_singularities.png")

    return {
        'P': P,
        'Q': Q,
        'grad_mag': grad_mag,
        'laplacian': laplacian,
        'B1': B1,
        'B2': B2
    }


def test_complex_layers():
    """Test complex layer operations."""
    print("=" * 60)
    print("TESTING: Complex Layer Operations")
    print("=" * 60)

    # Test ComplexLinear
    print("\n1. ComplexLinear layer:")
    layer = ComplexLinear(4, 8)
    z = torch.randn(2, 4) + 1j * torch.randn(2, 4)
    out = layer(z)
    print(f"   Input: {z.shape}, Output: {out.shape}")
    print(f"   Output is complex: {out.is_complex()}")

    # Test ComplexReLU (modReLU)
    print("\n2. ComplexReLU (modReLU):")
    relu = ComplexReLU(mode='modReLU', bias=0.5)
    z = torch.tensor([1+1j, 0.3+0.3j, -1-1j, 2+0j], dtype=torch.complex64)
    out = relu(z)
    print(f"   Input magnitudes: {torch.abs(z).tolist()}")
    print(f"   Output magnitudes: {torch.abs(out).tolist()}")
    print(f"   Phases preserved: {torch.allclose(torch.angle(z[torch.abs(z)>0.5]), torch.angle(out[torch.abs(out)>0]))}")

    # Test ComplexReLU (CReLU)
    print("\n3. ComplexReLU (CReLU):")
    relu_c = ComplexReLU(mode='CReLU')
    z = torch.tensor([1+1j, -1+1j, 1-1j, -1-1j], dtype=torch.complex64)
    out = relu_c(z)
    print(f"   Input: {z.tolist()}")
    print(f"   Output: {out.tolist()}")

    # Test HJB_MLP forward
    print("\n4. HJB_MLP (real mode):")
    model = HJB_MLP(input_dim=8, hidden_dims=[32, 16], output_dim=8, complex_mode=False)
    beta = torch.randn(5, 8)
    P, Q = model(beta)
    print(f"   Input: {beta.shape}")
    print(f"   P: {P.shape}, Q: {Q.shape}")

    # Test HJB_MLP (complex mode)
    print("\n5. HJB_MLP (complex mode):")
    model_c = HJB_MLP(input_dim=8, hidden_dims=[32, 16], output_dim=8, complex_mode=True)
    beta_c = torch.randn(5, 8) + 1j * torch.randn(5, 8)
    P_c, Q_c = model_c(beta_c)
    print(f"   Input: {beta_c.shape}, complex: {beta_c.is_complex()}")
    print(f"   P: {P_c.shape}, Q: {Q_c.shape}")
    print(f"   Output is complex: {P_c.is_complex()}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_complex_layers()
