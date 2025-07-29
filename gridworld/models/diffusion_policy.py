import torch
import torch.nn.functional as F
from .network import FeedForwardNN
import numpy as np

class DiffusionPolicy(FeedForwardNN):
    """
    Extends FeedForwardNN for diffusion-based sampling with reproducible inference noise.
    """
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 device: torch.device = None,
                 num_steps: int = 10,
                 fixed_noise_inference: bool = False):
        super().__init__(in_dim, out_dim)
        # select device and move model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        # store whether to use fixed noise during inference
        self.fixed_noise_inference = fixed_noise_inference
        # pre-sample a single noise vector for inference
        self.init_noise = torch.randn(1, 2, device=self.device)
        # num sampling step
        self.num_steps = num_steps
        self.in_dim = in_dim
        self.out_dim = out_dim


    def sample_action(self, state_norm: torch.Tensor) -> torch.Tensor:
        """
        Run manual Euler diffusion to denoise initial noise into action delta.

        Args:
            state_norm: Tensor of shape (2,), normalized state in [-1,1]
            num_steps: Number of Euler integration steps
            dt: Step size (defaults to 1/num_steps)
            debug: If True, returns the full trajectory list instead of final action

        Returns:
            If debug=False: Tensor of shape (2,) representing Δ action
            If debug=True: List of numpy arrays for each intermediate x_t
        """
        # time step size
        state_norm = state_norm.unsqueeze(0)
        num_steps = self.num_steps
        dt = (1.0 / num_steps)
        # initialize x_t: use fixed or random noise for inference
        if self.fixed_noise_inference:
            x_t = self.init_noise.clone()
        else:
            x_t = torch.randn(1, 2, device=self.device)

        # perform Euler integration
        for step in range(num_steps):
            t_val = step * dt
            t_tensor = torch.full((1,1), t_val, device=self.device)
            inp = torch.cat([state_norm.to(self.device), x_t, t_tensor], dim=1)
            
            with torch.no_grad():
                velocity = self(inp)
            x_t = x_t + dt * velocity

        x_t_final = x_t[0]

        return x_t_final

    def sample_action_with_info(self, state_norm: torch.Tensor, num_train_samples: int = 100, include_inference_eps: bool = False):
        """
        Run Euler diffusion with tracking and return action + loss info.
        state_norm is (2,).. output is same..

        Returns:
            pred_action: final denoised action
            x_t_path: all intermediate x_t steps [1, T+1, D]
            eps: sampled eps used for initial noise [1, D]
            t: sampled time step [1, 1]
            initial_cfm_loss: scalar [1]
        """
        if state_norm.ndim == 1:
            state_norm = state_norm.unsqueeze(0)

        dt = 1.0 / self.num_steps
        state_norm = state_norm.to(self.device)
        eps = self.init_noise.clone() if self.fixed_noise_inference else torch.randn(1, self.out_dim, device=self.device)
        x_t = eps.clone()
        x_t_path = [x_t.detach().clone()]

        
        for step in range(self.num_steps):
            t_val = step * dt
            t_tensor = torch.full((1,1), t_val, device=self.device)
            inp = torch.cat([state_norm, x_t, t_tensor], dim=1)
            velocity = self(inp)
            x_t = x_t + dt * velocity
            x_t_path.append(x_t.detach().clone())

        x_t_path = torch.stack(x_t_path, dim=1)

        # Mine samples for training
        eps_sample = torch.randn(num_train_samples, self.out_dim, device=self.device)  # [N, D_a]
            
        t = torch.rand(num_train_samples, 1, device=self.device)  # [N, 1]
        x1 = x_t.repeat(num_train_samples, 1).detach()  # [N, D_a]
        state_tile = state_norm.expand(num_train_samples, -1)  # [N, D_s]

        initial_cfm_loss = self.compute_cfm_loss(state_tile, x1, eps_sample, t)  # [N]

        return x_t[0], x_t_path, eps_sample, t, initial_cfm_loss.detach()
        
    def compute_cfm_loss(self, state_norm: torch.Tensor,
                         x1: torch.Tensor,
                         eps: torch.Tensor,
                         t: torch.Tensor) -> torch.Tensor:
        """
        Compute conditional flow matching loss.

        Args:
            state_norm: [B, D_s] normalized input state
            x1: [B, D_a] final denoised action
            eps: [B, D_a] sampled noise
            t: [B, 1] time steps

        Returns:
            loss: [B] per-sample loss
        """
        B, D_a = eps.shape
        assert x1.shape == (B, D_a), f"x1 must be [B, D_a], got {x1.shape}"
        assert state_norm.shape[0] == B, f"state_norm must have batch size {B}, got {state_norm.shape[0]}"
        assert t.shape == (B, 1), f"t must be [B, 1], got {t.shape}"

        x_t = (1 - t) * eps + t * x1  # [B, D_a]
        inp = torch.cat([state_norm, x_t, t], dim=1)  # [B, D_s + D_a + 1]
        velocity_pred = self(inp)  # [B, D_a]

        return F.mse_loss(velocity_pred, x1 - eps, reduction='none').mean(dim=1)  # [B]

