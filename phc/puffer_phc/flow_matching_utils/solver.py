# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torchdiffeq import odeint

from abc import ABC, abstractmethod

from torch import nn, Tensor

def gradient(
    output: Tensor,
    x: Tensor,
    grad_outputs: Optional[Tensor] = None,
    create_graph: bool = False,
) -> Tensor:
    """
    Compute the gradient of the inner product of output and grad_outputs w.r.t :math:`x`.

    Args:
        output (Tensor): [N, D] Output of the function.
        x (Tensor): [N, d_1, d_2, ... ] input
        grad_outputs (Optional[Tensor]): [N, D] Gradient of outputs, if `None`,
            then will use a tensor of ones
        create_graph (bool): If True, graph of the derivative will be constructed, allowing
            to compute higher order derivative products. Defaults to False.
    Returns:
        Tensor: [N, d_1, d_2, ... ]. the gradient w.r.t x.
    """

    if grad_outputs is None:
        grad_outputs = torch.ones_like(output).detach()
    grad = torch.autograd.grad(
        output, x, grad_outputs=grad_outputs, create_graph=create_graph
    )[0]
    return grad


class Solver(ABC, nn.Module):
    """Abstract base class for solvers."""

    @abstractmethod
    def sample(self, x_0: Tensor = None) -> Tensor:
        ...


class ODESolver(Solver):
    """A class to solve ordinary differential equations (ODEs) using a specified velocity model.

    This class utilizes a velocity field model to solve ODEs over a given time grid using numerical ode solvers.

    Args:
        velocity_model (Union[ModelWrapper, Callable]): a velocity field model receiving :math:`(x,t)` and returning :math:`u_t(x)`
    """

    def __init__(self):
        super().__init__()

    def sample(
        self,
        velocity_model: Callable,
        x_init: Tensor,
        step_size: Optional[float],
        method: str = "euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        enable_grad: bool = False,
        **model_extras,
    ) -> Union[Tensor, Sequence[Tensor]]:
        r"""Solve the ODE with the velocity field.

        Example:

        .. code-block:: python

            import torch
            from flow_matching.utils import ModelWrapper
            from flow_matching.solver import ODESolver

            class DummyModel(ModelWrapper):
                def __init__(self):
                    super().__init__(None)

                def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
                    return torch.ones_like(x) * 3.0 * t**2

            velocity_model = DummyModel()
            solver = ODESolver(velocity_model=velocity_model)
            x_init = torch.tensor([0.0, 0.0])
            step_size = 0.001
            time_grid = torch.tensor([0.0, 1.0])

            result = solver.sample(x_init=x_init, step_size=step_size, time_grid=time_grid)

        Args:
            x_init (Tensor): initial conditions (e.g., source samples :math:`X_0 \sim p`). Shape: [batch_size, ...].
            step_size (Optional[float]): The step size. Must be None for adaptive step solvers.
            method (str): A method supported by torchdiffeq. Defaults to "euler". Other commonly used solvers are "dopri5", "midpoint" and "heun3". For a complete list, see torchdiffeq.
            atol (float): Absolute tolerance, used for adaptive step solvers.
            rtol (float): Relative tolerance, used for adaptive step solvers.
            time_grid (Tensor): The process is solved in the interval [min(time_grid, max(time_grid)] and if step_size is None then time discretization is set by the time grid. May specify a descending time_grid to solve in the reverse direction. Defaults to torch.tensor([0.0, 1.0]).
            return_intermediates (bool, optional): If True then return intermediate time steps according to time_grid. Defaults to False.
            enable_grad (bool, optional): Whether to compute gradients during sampling. Defaults to False.
            **model_extras: Additional input for the model.

        Returns:
            Union[Tensor, Sequence[Tensor]]: The last timestep when return_intermediates=False, otherwise all values specified in time_grid.
        """

        time_grid = time_grid.to(x_init.device)

        def ode_func(t, x):
            return velocity_model(x=x, t=t, **model_extras)

        ode_opts = {"step_size": step_size} if step_size is not None else {}

        with torch.set_grad_enabled(enable_grad):
            # Approximate ODE solution with numerical ODE solver
            sol = odeint(
                ode_func,
                x_init,
                time_grid,
                method=method,
                options=ode_opts,
                atol=atol,
                rtol=rtol,
            )

        if return_intermediates:
            return sol
        else:
            return sol[-1]


    def compute_likelihood(
        self,
        velocity_model: Callable,
        x_1: Tensor,
        log_p0: Callable[[Tensor], Tensor],
        step_size: Optional[float],
        method: str = "euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Tensor = torch.tensor([1.0, 0.0]),
        return_intermediates: bool = False,
        exact_divergence: bool = False,
        enable_grad: bool = False,
        **model_extras,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Sequence[Tensor], Tensor]]:
        r"""Solve for log likelihood given a target sample at :math:`t=0`.

        Works similarly to sample, but solves the ODE in reverse to compute the log-likelihood. The velocity model must be differentiable with respect to x.
        The function assumes log_p0 is the log probability of the source distribution at :math:`t=0`.

        Args:
            x_1 (Tensor): target sample (e.g., samples :math:`X_1 \sim p_1`).
            log_p0 (Callable[[Tensor], Tensor]): Log probability function of the source distribution.
            step_size (Optional[float]): The step size. Must be None for adaptive step solvers.
            method (str): A method supported by torchdiffeq. Defaults to "euler". Other commonly used solvers are "dopri5", "midpoint" and "heun3". For a complete list, see torchdiffeq.
            atol (float): Absolute tolerance, used for adaptive step solvers.
            rtol (float): Relative tolerance, used for adaptive step solvers.
            time_grid (Tensor): If step_size is None then time discretization is set by the time grid. Must start at 1.0 and end at 0.0, otherwise the likelihood computation is not valid. Defaults to torch.tensor([1.0, 0.0]).
            return_intermediates (bool, optional): If True then return intermediate time steps according to time_grid. Otherwise only return the final sample. Defaults to False.
            exact_divergence (bool): Whether to compute the exact divergence or use the Hutchinson estimator.
            enable_grad (bool, optional): Whether to compute gradients during sampling. Defaults to False.
            **model_extras: Additional input for the model.

        Returns:
            Union[Tuple[Tensor, Tensor], Tuple[Sequence[Tensor], Tensor]]: Samples at time_grid and log likelihood values of given x_1.
        """
        assert (
            time_grid[0] == 1.0 and time_grid[-1] == 0.0
        ), f"Time grid must start at 1.0 and end at 0.0. Got {time_grid}"

        # Fix the random projection for the Hutchinson divergence estimator
        if not exact_divergence:
            z = (torch.randn_like(x_1).to(x_1.device) < 0) * 2.0 - 1.0

        def ode_func(x, t):
            return velocity_model(x=x, t=t, **model_extras)

        def dynamics_func(t, states):
            xt = states[0]
            with torch.set_grad_enabled(True):
                xt.requires_grad_()
                ut = ode_func(xt, t)

                if exact_divergence:
                    # Compute exact divergence
                    div = 0
                    for i in range(ut.flatten(1).shape[1]):
                        div += gradient(ut[:, i], xt, create_graph=True)[:, i]
                else:
                    # Compute Hutchinson divergence estimator E[z^T D_x(ut) z]
                    ut_dot_z = torch.einsum(
                        "ij,ij->i", ut.flatten(start_dim=1), z.flatten(start_dim=1)
                    )
                    grad_ut_dot_z = gradient(ut_dot_z, xt)
                    div = torch.einsum(
                        "ij,ij->i",
                        grad_ut_dot_z.flatten(start_dim=1),
                        z.flatten(start_dim=1),
                    )

            return ut.detach(), div.detach()

        y_init = (x_1, torch.zeros(x_1.shape[0], device=x_1.device))
        ode_opts = {"step_size": step_size} if step_size is not None else {}

        with torch.set_grad_enabled(enable_grad):
            sol, log_det = odeint(
                dynamics_func,
                y_init,
                time_grid,
                method=method,
                options=ode_opts,
                atol=atol,
                rtol=rtol,
            )

        x_source = sol[-1]
        source_log_p = log_p0(x_source)

        if return_intermediates:
            return sol, source_log_p + log_det[-1]
        else:
            return sol[-1], source_log_p + log_det[-1]