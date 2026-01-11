"""
Ponderomotive Control in HST/ROM Coordinates.

From Glinsky: "Ponderomotive control can be implemented to stabilize
the optimal system design from disruption."

The ponderomotive effect: High-frequency modulation creates an effective
potential that can stabilize unstable equilibria (Kapitza pendulum effect).

In the HST/ROM framework:
1. System state is represented in β (ROM) coordinates
2. Control operates by modulating the effective Hamiltonian
3. High-frequency forcing creates effective restoring force toward target

Key equations:
    F_eff ≈ -ε²/(4Ω²) ∇V_eff

where:
    ε = modulation amplitude
    Ω = modulation frequency (>> system frequency)
    V_eff = effective potential shaped by control
"""

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class PonderomotiveController:
    """
    Ponderomotive control in HST/ROM coordinates.

    The idea: High-frequency modulation creates an effective potential
    that can stabilize unstable equilibria or push system toward target.

    Usage:
    ------
    controller = PonderomotiveController(rom, decoder, target_beta)

    # In simulation loop:
    u_control = controller.compute_control(z_current)
    z_next = system_step(z_current + u_control)
    """

    def __init__(self, rom, decoder, target_beta=None, Omega=10.0, epsilon=0.1):
        """
        Initialize ponderomotive controller.

        Parameters
        ----------
        rom : HST_ROM
            Fitted ROM for coordinate transformation
        decoder : HJB_Decoder
            Trained decoder for geodesic coordinates (optional)
        target_beta : array
            Target ROM state to stabilize toward
        Omega : float
            High-frequency modulation frequency (>> system frequency)
        epsilon : float
            Modulation amplitude
        """
        self.rom = rom
        self.decoder = decoder
        self.target_beta = target_beta
        self.Omega = Omega
        self.epsilon = epsilon

        # Cache for Jacobian computation
        self._jacobian_cache = {}

    def set_target(self, target_beta):
        """Set new target ROM state."""
        self.target_beta = np.asarray(target_beta)

    def set_target_from_signal(self, z_target):
        """Set target from a signal."""
        self.target_beta = self.rom.transform(z_target)

    def compute_control(self, current_z, t=0.0, use_decoder=True):
        """
        Compute ponderomotive control signal.

        Parameters
        ----------
        current_z : array
            Current signal in physical domain
        t : float
            Current time (for high-frequency modulation)
        use_decoder : bool
            If True, use decoder for geodesic-aware control

        Returns
        -------
        u : array
            Control signal to add to system
        control_info : dict
            Diagnostic information
        """
        current_z = np.asarray(current_z, dtype=complex)

        if self.target_beta is None:
            return np.zeros_like(current_z), {'status': 'no_target'}

        # Get current ROM state
        beta_current = self.rom.transform(current_z)

        # Compute error in ROM space
        delta_beta = beta_current - self.target_beta

        if use_decoder and self.decoder is not None and HAS_TORCH:
            # Use decoder to work in geodesic coordinates
            control_direction = self._geodesic_control(beta_current, delta_beta)
        else:
            # Direct control in β space
            control_direction = -delta_beta

        # Ponderomotive force magnitude
        # F_eff ≈ -ε²/(4Ω²) ∇V_eff
        # We approximate ∇V_eff ∝ delta_beta
        control_strength = self.epsilon**2 / (4 * self.Omega**2)

        # High-frequency modulation
        modulation = np.cos(self.Omega * t)

        # Scale control
        control_beta = control_strength * control_direction * modulation

        # Map control back to physical domain
        u = self._map_control_to_physical(control_beta, current_z)

        # Diagnostic info
        control_info = {
            'status': 'active',
            'delta_beta': delta_beta,
            'control_direction': control_direction,
            'control_strength': control_strength,
            'beta_error_norm': np.linalg.norm(delta_beta),
        }

        return u, control_info

    def _geodesic_control(self, beta_current, delta_beta):
        """
        Compute control direction using geodesic (P, Q) coordinates.

        In geodesic coordinates:
        - Control P to change which orbit we're on
        - Control Q doesn't matter (phase along orbit)
        """
        if not HAS_TORCH:
            return -delta_beta

        beta_t = torch.tensor(beta_current.reshape(1, -1), dtype=torch.float32)

        with torch.no_grad():
            PQ_current = self.decoder(beta_t)
            P_current, Q_current = self.decoder.split_PQ(PQ_current)
            P_current = P_current.numpy().flatten()

        # Target in P coordinates
        target_t = torch.tensor(self.target_beta.reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            PQ_target = self.decoder(target_t)
            P_target, Q_target = self.decoder.split_PQ(PQ_target)
            P_target = P_target.numpy().flatten()

        # Control only the P (action) variables
        # Q (angle) will naturally evolve
        delta_P = P_current - P_target

        # Map back to β space (approximate)
        # Use numerical gradient of P w.r.t. β
        dP_dbeta = self._compute_jacobian_P(beta_current)

        # Control direction in β that reduces |δP|
        # Solve: dP_dbeta @ control_beta = -delta_P
        control_direction = -np.linalg.lstsq(dP_dbeta.T, delta_P, rcond=None)[0]

        return control_direction

    def _compute_jacobian_P(self, beta, eps=1e-5):
        """Compute Jacobian dP/dβ numerically."""
        n = len(beta)
        n_P = n // 2

        jacobian = np.zeros((n_P, n))

        beta_t = torch.tensor(beta.reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            PQ_base = self.decoder(beta_t)
            P_base, _ = self.decoder.split_PQ(PQ_base)
            P_base = P_base.numpy().flatten()

        for i in range(n):
            beta_pert = beta.copy()
            beta_pert[i] += eps

            beta_pert_t = torch.tensor(beta_pert.reshape(1, -1), dtype=torch.float32)
            with torch.no_grad():
                PQ_pert = self.decoder(beta_pert_t)
                P_pert, _ = self.decoder.split_PQ(PQ_pert)
                P_pert = P_pert.numpy().flatten()

            jacobian[:, i] = (P_pert - P_base) / eps

        return jacobian

    def _map_control_to_physical(self, control_beta, z):
        """
        Map control in β space to physical signal space.

        Uses perturbation approach: small change in β → small change in z
        """
        # Perturbation approach
        beta_current = self.rom.transform(z)
        beta_controlled = beta_current + control_beta

        # Reconstruct both
        z_rec = self.rom.inverse_transform(beta_current, original_length=len(z))
        z_controlled = self.rom.inverse_transform(beta_controlled, original_length=len(z))

        # Control signal
        u = z_controlled - z_rec

        # Scale to reasonable magnitude
        u_norm = np.linalg.norm(u)
        if u_norm > self.epsilon:
            u = u * self.epsilon / u_norm

        return u

    def simulate_with_control(self, system_dynamics, z0, T, dt=0.01):
        """
        Simulate system with ponderomotive control.

        Parameters
        ----------
        system_dynamics : callable
            Function(z, t) that returns dz/dt
        z0 : array
            Initial condition
        T : float
            Total simulation time
        dt : float
            Time step

        Returns
        -------
        t_hist : array
            Time points
        z_hist : array
            Trajectory
        control_hist : list
            Control info at each step
        """
        n_steps = int(T / dt)
        t_hist = np.linspace(0, T, n_steps)

        # Initialize with enough samples for ROM window
        if self.rom.window_size is not None:
            z_buffer = [z0] * self.rom.window_size
        else:
            z_buffer = [z0]

        z_hist = []
        control_hist = []

        z = z0

        for i, t in enumerate(t_hist):
            # Update buffer
            z_buffer.append(z)
            if len(z_buffer) > self.rom.window_size:
                z_buffer.pop(0)

            # Get current window for ROM
            z_window = np.array(z_buffer, dtype=complex)

            # Compute control
            if len(z_window) >= self.rom.window_size:
                u, info = self.compute_control(z_window, t=t)
                # Apply control to last sample
                u_applied = u[-1] if len(u) > 0 else 0
            else:
                u_applied = 0
                info = {'status': 'warming_up'}

            # System dynamics + control
            dz = system_dynamics(z, t)
            z = z + dt * (dz + u_applied)

            z_hist.append(z)
            control_hist.append(info)

        return t_hist, np.array(z_hist), control_hist


class AdaptiveController(PonderomotiveController):
    """
    Adaptive ponderomotive controller that adjusts parameters.

    Adapts ε and Ω based on tracking error.
    """

    def __init__(self, rom, decoder, target_beta=None,
                 Omega_range=(5, 50), epsilon_range=(0.01, 0.5)):
        super().__init__(rom, decoder, target_beta)
        self.Omega_range = Omega_range
        self.epsilon_range = epsilon_range

        self.error_history = []
        self.adaptation_rate = 0.1

    def adapt_parameters(self, error_norm):
        """Adapt control parameters based on error."""
        self.error_history.append(error_norm)

        if len(self.error_history) > 10:
            recent_error = np.mean(self.error_history[-10:])
            older_error = np.mean(self.error_history[-20:-10]) if len(self.error_history) > 20 else recent_error

            # If error increasing, increase epsilon
            if recent_error > older_error * 1.1:
                self.epsilon = min(self.epsilon * (1 + self.adaptation_rate),
                                   self.epsilon_range[1])
            # If error small and decreasing, decrease epsilon
            elif recent_error < older_error * 0.9 and recent_error < 0.1:
                self.epsilon = max(self.epsilon * (1 - self.adaptation_rate),
                                   self.epsilon_range[0])

    def compute_control(self, current_z, t=0.0, use_decoder=True):
        u, info = super().compute_control(current_z, t, use_decoder)

        # Adapt based on error
        if 'beta_error_norm' in info:
            self.adapt_parameters(info['beta_error_norm'])
            info['epsilon'] = self.epsilon
            info['Omega'] = self.Omega

        return u, info


def test_ponderomotive_control():
    """Test ponderomotive controller on Van der Pol."""
    print("=" * 60)
    print("Testing Ponderomotive Control")
    print("=" * 60)

    from hst_rom import HST_ROM

    # Generate training trajectories (Van der Pol)
    def van_der_pol(z, t, mu=1.0):
        x, v = np.real(z), np.imag(z)
        dx = v
        dv = mu * (1 - x**2) * v - x
        return dx + 1j * dv

    print("Generating training trajectories...")
    np.random.seed(42)
    trajectories = []

    for _ in range(20):
        x0 = 3 * np.random.randn()
        v0 = 3 * np.random.randn()
        z0 = x0 + 1j * v0

        # Simulate
        z = z0
        traj = [z]
        for _ in range(1000):
            dz = van_der_pol(z, 0)
            z = z + 0.01 * dz
            traj.append(z)

        trajectories.append(np.array(traj))

    # Fit ROM
    print("Fitting ROM...")
    rom = HST_ROM(n_components=4, wavelet='db8', J=3, window_size=64)
    betas = rom.fit(trajectories)

    print(f"ROM fitted with {len(betas)} samples")
    print(f"Variance explained: {rom.pca.explained_variance_ratio_}")

    # Compute target (limit cycle attractor)
    # Take a point on the limit cycle
    z_limit = trajectories[0][-64:]  # Last window of settled trajectory
    target_beta = rom.transform(z_limit)

    print(f"\nTarget β: {target_beta}")

    # Create controller
    controller = PonderomotiveController(rom, None, target_beta,
                                          Omega=20.0, epsilon=0.2)

    # Simulate with and without control from different initial condition
    z0 = 4.0 + 4.0j  # Far from limit cycle

    print("\nSimulating without control...")
    traj_uncontrolled = [z0]
    z = z0
    for _ in range(2000):
        dz = van_der_pol(z, 0)
        z = z + 0.01 * dz
        traj_uncontrolled.append(z)
    traj_uncontrolled = np.array(traj_uncontrolled)

    print("Simulating with control...")
    # Need enough initial samples for ROM
    z_buffer = [z0] * 64
    traj_controlled = list(z_buffer)
    z = z0

    for i in range(2000):
        # Get window
        z_window = np.array(z_buffer[-64:], dtype=complex)

        # Compute control
        if len(z_window) >= 64:
            try:
                u, info = controller.compute_control(z_window, t=i*0.01)
                u_applied = u[-1] if len(u) > 0 else 0
            except:
                u_applied = 0
        else:
            u_applied = 0

        # Step
        dz = van_der_pol(z, 0)
        z = z + 0.01 * (dz + u_applied)

        z_buffer.append(z)
        traj_controlled.append(z)

    traj_controlled = np.array(traj_controlled)

    # Compare
    print("\nResults:")
    print(f"Uncontrolled final |z|: {np.abs(traj_uncontrolled[-1]):.3f}")
    print(f"Controlled final |z|: {np.abs(traj_controlled[-1]):.3f}")

    # Plot
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.plot(np.real(traj_uncontrolled), np.imag(traj_uncontrolled),
                'b-', alpha=0.5, label='Uncontrolled')
        ax.plot(np.real(traj_controlled[64:]), np.imag(traj_controlled[64:]),
                'r-', alpha=0.5, label='Controlled')
        ax.plot(np.real(z0), np.imag(z0), 'ko', ms=10, label='Start')
        ax.set_xlabel('x')
        ax.set_ylabel('v')
        ax.set_title('Phase Space')
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        t = np.arange(len(traj_uncontrolled)) * 0.01
        ax.plot(t, np.abs(traj_uncontrolled), 'b-', label='Uncontrolled')
        t_c = np.arange(len(traj_controlled) - 64) * 0.01
        ax.plot(t_c, np.abs(traj_controlled[64:]), 'r-', label='Controlled')
        ax.axhline(2, color='g', ls='--', label='Limit cycle |z|')
        ax.set_xlabel('Time')
        ax.set_ylabel('|z|')
        ax.set_title('Amplitude Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('ponderomotive_control_test.png', dpi=150)
        plt.close()
        print("\nSaved: ponderomotive_control_test.png")

    except Exception as e:
        print(f"Plotting failed: {e}")

    return controller, rom


if __name__ == "__main__":
    controller, rom = test_ponderomotive_control()
