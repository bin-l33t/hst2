"""
HJB Decoder - MLP that learns geodesic coordinates.

From Glinsky: The MLP approximates the generating function S(P;q) that
transforms to geodesic coordinates where:
    dP/dτ = 0  (P conserved - action variables)
    dQ/dτ = ω(P)  (Q evolves linearly - angle variables)

This is also the "action", "entropy", "log-likelihood", "Q-function of DRL",
and "value function" - all the same object in different contexts.

The key insight: Since H(β) is an analytic function, it will be well
approximated by MLPs with ReLU (piece-wise linear universal function approximators).
"""

import numpy as np

# Try to import torch, but provide numpy fallback
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available. Using numpy-based decoder.")


if HAS_TORCH:
    class HJB_Decoder(nn.Module):
        """
        MLP that approximates the generating function S(P;q).

        Maps ROM coordinates q → (P, Q) where:
            P: action variables (conserved along trajectories)
            Q: angle variables (evolve linearly in time)

        Architecture:
            Input: q (ROM coordinates, dimension n)
            Hidden: ReLU layers (piece-wise linear!)
            Output: (P, Q) where dim(P) = dim(Q) = n/2 or n

        The network learns the canonical transformation to action-angle
        coordinates where the dynamics becomes trivially integrable.
        """

        def __init__(self, input_dim, hidden_dims=[64, 64], output_dim=None):
            """
            Initialize HJB decoder.

            Parameters
            ----------
            input_dim : int
                Dimension of ROM coordinates (β)
            hidden_dims : list of int
                Hidden layer dimensions
            output_dim : int
                Output dimension. If None, equals input_dim
            """
            super().__init__()

            if output_dim is None:
                output_dim = input_dim

            self.input_dim = input_dim
            self.output_dim = output_dim

            # Build network
            layers = []
            prev_dim = input_dim

            for h_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(nn.ReLU())  # Piece-wise linear!
                prev_dim = h_dim

            layers.append(nn.Linear(prev_dim, output_dim))

            self.network = nn.Sequential(*layers)

            # Separate network to predict ω(P) for geodesic constraint
            self.omega_net = nn.Sequential(
                nn.Linear(output_dim // 2, 32),
                nn.ReLU(),
                nn.Linear(32, output_dim // 2)
            )

        def forward(self, q):
            """
            Compute (P, Q) from ROM coordinates q.

            Parameters
            ----------
            q : tensor (batch, input_dim)
                ROM coordinates

            Returns
            -------
            PQ : tensor (batch, output_dim)
                Geodesic coordinates [P, Q]
            """
            return self.network(q)

        def split_PQ(self, PQ):
            """Split output into P (action) and Q (angle) components."""
            n = PQ.shape[-1] // 2
            P = PQ[..., :n]
            Q = PQ[..., n:]
            return P, Q

        def predict_omega(self, P):
            """Predict frequency ω(P) for geodesic motion."""
            return self.omega_net(P)


    def train_hjb_decoder(rom, trajectories, n_epochs=500, lr=1e-3,
                          verbose=True, patience=50):
        """
        Train the HJB decoder to predict geodesic motion.

        The key insight: In (P, Q) coordinates, the dynamics should be:
            dP/dτ = 0  (P is constant along trajectories)
            dQ/dτ = ω(P)  (Q evolves linearly in time)

        We train by enforcing these constraints on trajectory data.

        Parameters
        ----------
        rom : HST_ROM
            Fitted ROM object
        trajectories : list of arrays
            Raw signal trajectories
        n_epochs : int
            Training epochs
        lr : float
            Learning rate
        verbose : bool
            Print progress
        patience : int
            Early stopping patience

        Returns
        -------
        decoder : HJB_Decoder
            Trained decoder
        history : dict
            Training history
        """
        # Convert trajectories to ROM coordinates
        print("Extracting ROM trajectories...")
        beta_trajectories = []

        for z in trajectories:
            z = np.asarray(z, dtype=complex)
            if rom.window_size is not None:
                betas, times = rom.transform_trajectory(z)
                if len(betas) > 1:
                    beta_trajectories.append(betas)
            else:
                # Single window per trajectory
                beta = rom.transform(z)
                beta_trajectories.append(beta.reshape(1, -1))

        print(f"Extracted {len(beta_trajectories)} ROM trajectories")

        # Create decoder
        decoder = HJB_Decoder(input_dim=rom.n_components)
        optimizer = optim.Adam(decoder.parameters(), lr=lr)

        history = {'loss': [], 'loss_P': [], 'loss_Q': []}
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(n_epochs):
            total_loss = 0
            total_loss_P = 0
            total_loss_Q = 0
            n_samples = 0

            for betas in beta_trajectories:
                if len(betas) < 2:
                    continue

                # Convert to tensors
                betas_t = torch.tensor(betas[:-1], dtype=torch.float32)
                betas_next = torch.tensor(betas[1:], dtype=torch.float32)

                # Get (P, Q) coordinates
                PQ = decoder(betas_t)
                PQ_next = decoder(betas_next)

                P, Q = decoder.split_PQ(PQ)
                P_next, Q_next = decoder.split_PQ(PQ_next)

                # Loss 1: P should be constant (dP/dτ = 0)
                # This encodes that action variables are conserved
                loss_P = ((P_next - P)**2).mean()

                # Loss 2: Q should evolve linearly (dQ/dτ = ω(P))
                # Predict ω from P and check consistency
                dQ = Q_next - Q
                omega_pred = decoder.predict_omega(P)
                loss_Q = ((dQ - omega_pred)**2).mean()

                # Combined loss
                loss = loss_P + 0.5 * loss_Q

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(betas_t)
                total_loss_P += loss_P.item() * len(betas_t)
                total_loss_Q += loss_Q.item() * len(betas_t)
                n_samples += len(betas_t)

            if n_samples > 0:
                avg_loss = total_loss / n_samples
                avg_loss_P = total_loss_P / n_samples
                avg_loss_Q = total_loss_Q / n_samples

                history['loss'].append(avg_loss)
                history['loss_P'].append(avg_loss_P)
                history['loss_Q'].append(avg_loss_Q)

                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

                if verbose and epoch % 50 == 0:
                    print(f"Epoch {epoch:4d}: Loss = {avg_loss:.6f} "
                          f"(P: {avg_loss_P:.6f}, Q: {avg_loss_Q:.6f})")

        return decoder, history


    def verify_geodesic_motion(decoder, rom, trajectory, plot=False):
        """
        Verify that the decoder produces geodesic motion.

        Checks:
        1. P should be approximately constant (std(P) small)
        2. Q should evolve approximately linearly (R² high)

        Parameters
        ----------
        decoder : HJB_Decoder
            Trained decoder
        rom : HST_ROM
            ROM object
        trajectory : array
            Test trajectory
        plot : bool
            Create visualization

        Returns
        -------
        metrics : dict
            Verification metrics
        """
        # Get ROM trajectory
        betas, times = rom.transform_trajectory(trajectory)
        betas_t = torch.tensor(betas, dtype=torch.float32)

        with torch.no_grad():
            PQ = decoder(betas_t)
            P, Q = decoder.split_PQ(PQ)
            P = P.numpy()
            Q = Q.numpy()

        # Metric 1: P conservation (should be constant)
        P_std = np.std(P, axis=0)
        P_mean = np.mean(P, axis=0)
        P_cv = P_std / (np.abs(P_mean) + 1e-10)  # Coefficient of variation

        # Metric 2: Q linearity (should be linear in time)
        # Fit linear regression to each Q component
        Q_r2_scores = []
        Q_slopes = []

        t_normalized = (times - times[0]) / (times[-1] - times[0] + 1e-10)

        for i in range(Q.shape[1]):
            # Linear fit
            A = np.vstack([t_normalized, np.ones_like(t_normalized)]).T
            slope, intercept = np.linalg.lstsq(A, Q[:, i], rcond=None)[0]

            # R² score
            Q_pred = slope * t_normalized + intercept
            ss_res = np.sum((Q[:, i] - Q_pred)**2)
            ss_tot = np.sum((Q[:, i] - Q[:, i].mean())**2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)

            Q_r2_scores.append(r2)
            Q_slopes.append(slope)

        metrics = {
            'P_std': P_std,
            'P_cv': P_cv,
            'P_conservation': np.mean(P_cv) < 0.1,  # CV < 10%
            'Q_r2': np.array(Q_r2_scores),
            'Q_slopes': np.array(Q_slopes),
            'Q_linear': np.mean(Q_r2_scores) > 0.8,  # R² > 80%
            'geodesic': np.mean(P_cv) < 0.1 and np.mean(Q_r2_scores) > 0.8
        }

        if plot:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # P evolution
            ax = axes[0, 0]
            for i in range(P.shape[1]):
                ax.plot(times, P[:, i], label=f'P_{i}')
            ax.set_xlabel('Time')
            ax.set_ylabel('P (action)')
            ax.set_title(f'Action Variables (should be constant)\nstd = {P_std}')
            ax.legend()

            # Q evolution
            ax = axes[0, 1]
            for i in range(Q.shape[1]):
                ax.plot(times, Q[:, i], label=f'Q_{i} (R²={Q_r2_scores[i]:.2f})')
            ax.set_xlabel('Time')
            ax.set_ylabel('Q (angle)')
            ax.set_title('Angle Variables (should be linear)')
            ax.legend()

            # P-Q phase space
            ax = axes[1, 0]
            if P.shape[1] >= 1 and Q.shape[1] >= 1:
                ax.scatter(P[:, 0], Q[:, 0], c=times, cmap='viridis')
                ax.set_xlabel('P_0')
                ax.set_ylabel('Q_0')
                ax.set_title('Phase Space (P_0, Q_0)')

            # Summary
            ax = axes[1, 1]
            ax.axis('off')
            summary = f"""
            Geodesic Motion Verification
            ============================

            P Conservation:
              std(P) = {P_std}
              CV(P) = {P_cv}
              Conserved: {metrics['P_conservation']}

            Q Linearity:
              R² = {Q_r2_scores}
              slopes = {Q_slopes}
              Linear: {metrics['Q_linear']}

            GEODESIC: {metrics['geodesic']}
            """
            ax.text(0.1, 0.5, summary, family='monospace', fontsize=10,
                    verticalalignment='center')

            plt.tight_layout()
            plt.savefig('geodesic_verification.png', dpi=150)
            plt.close()

        return metrics

else:
    # Numpy fallback when PyTorch not available
    class HJB_Decoder:
        """Simple numpy-based decoder (for testing without torch)."""

        def __init__(self, input_dim, hidden_dims=[64, 64], output_dim=None):
            self.input_dim = input_dim
            self.output_dim = output_dim or input_dim

            # Initialize random weights
            np.random.seed(42)
            self.weights = []
            self.biases = []

            prev_dim = input_dim
            for h_dim in hidden_dims:
                W = np.random.randn(prev_dim, h_dim) * 0.1
                b = np.zeros(h_dim)
                self.weights.append(W)
                self.biases.append(b)
                prev_dim = h_dim

            # Output layer
            W = np.random.randn(prev_dim, self.output_dim) * 0.1
            b = np.zeros(self.output_dim)
            self.weights.append(W)
            self.biases.append(b)

        def forward(self, q):
            """Forward pass with ReLU activation."""
            x = np.asarray(q)
            for i, (W, b) in enumerate(zip(self.weights, self.biases)):
                x = x @ W + b
                if i < len(self.weights) - 1:
                    x = np.maximum(0, x)  # ReLU
            return x

        def __call__(self, q):
            return self.forward(q)

        def split_PQ(self, PQ):
            n = PQ.shape[-1] // 2
            return PQ[..., :n], PQ[..., n:]


    def train_hjb_decoder(rom, trajectories, n_epochs=500, lr=1e-3,
                          verbose=True, patience=50):
        """Placeholder training (without torch, just returns initialized decoder)."""
        print("Warning: PyTorch not available. Returning untrained decoder.")
        decoder = HJB_Decoder(input_dim=rom.n_components)
        history = {'loss': [], 'loss_P': [], 'loss_Q': []}
        return decoder, history


    def verify_geodesic_motion(decoder, rom, trajectory, plot=False):
        """Verification without proper training."""
        metrics = {
            'P_std': np.zeros(rom.n_components // 2),
            'P_cv': np.zeros(rom.n_components // 2),
            'P_conservation': False,
            'Q_r2': np.zeros(rom.n_components // 2),
            'Q_slopes': np.zeros(rom.n_components // 2),
            'Q_linear': False,
            'geodesic': False
        }
        return metrics


def test_hjb_decoder():
    """Test HJB decoder."""
    print("=" * 60)
    print("Testing HJB Decoder")
    print("=" * 60)

    if not HAS_TORCH:
        print("Skipping test: PyTorch not available")
        return None, None

    from hst_rom import HST_ROM

    # Generate test trajectories
    np.random.seed(42)
    trajectories = []

    for i in range(30):
        t = np.linspace(0, 20, 1024)
        amp = 1 + 0.3 * np.random.randn()
        freq = 1 + 0.2 * np.random.randn()
        z = amp * np.exp(1j * 2 * np.pi * freq * t)
        trajectories.append(z)

    print(f"Generated {len(trajectories)} trajectories")

    # Fit ROM
    rom = HST_ROM(n_components=4, wavelet='db8', J=3, window_size=128)
    rom.fit(trajectories[:20])

    print(f"ROM variance explained: {rom.pca.explained_variance_ratio_}")

    # Train decoder
    decoder, history = train_hjb_decoder(rom, trajectories[:20],
                                          n_epochs=200, verbose=True)

    print(f"\nFinal loss: {history['loss'][-1]:.6f}")

    # Verify on test trajectory
    test_traj = trajectories[25]
    metrics = verify_geodesic_motion(decoder, rom, test_traj, plot=True)

    print(f"\nGeodesic verification:")
    print(f"  P conservation: {metrics['P_conservation']}")
    print(f"  Q linear: {metrics['Q_linear']}")
    print(f"  Geodesic: {metrics['geodesic']}")

    return decoder, rom


if __name__ == "__main__":
    decoder, rom = test_hjb_decoder()
