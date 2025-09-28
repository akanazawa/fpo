# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

from torch import Tensor

from dataclasses import dataclass, field

from puffer_phc.flow_matching_utils.scheduler import Scheduler, CondOTScheduler

def expand_tensor_like(input_tensor: Tensor, expand_to: Tensor) -> Tensor:
    """`input_tensor` is a 1d vector of length equal to the batch size of `expand_to`,
    expand `input_tensor` to have the same shape as `expand_to` along all remaining dimensions.

    Args:
        input_tensor (Tensor): (batch_size,).
        expand_to (Tensor): (batch_size, ...).

    Returns:
        Tensor: (batch_size, ...).
    """
    assert input_tensor.ndim == 1, "Input tensor must be a 1d vector."
    assert (
        input_tensor.shape[0] == expand_to.shape[0]
    ), f"The first (batch_size) dimension must match. Got shape {input_tensor.shape} and {expand_to.shape}."

    dim_diff = expand_to.ndim - input_tensor.ndim

    t_expanded = input_tensor.clone()
    t_expanded = t_expanded.reshape(-1, *([1] * dim_diff))

    return t_expanded.expand_as(expand_to)


@dataclass
class PathSample:
    r"""Represents a sample of a conditional-flow generated probability path.

    Attributes:
        x_1 (Tensor): the target sample :math:`X_1`.
        x_0 (Tensor): the source sample :math:`X_0`.
        t (Tensor): the time sample :math:`t`.
        x_t (Tensor): samples :math:`X_t \sim p_t(X_t)`, shape (batch_size, ...).
        dx_t (Tensor): conditional target :math:`\frac{\partial X}{\partial t}`, shape: (batch_size, ...).

    """

    x_1: Tensor = field(metadata={"help": "target samples X_1 (batch_size, ...)."})
    x_0: Tensor = field(metadata={"help": "source samples X_0 (batch_size, ...)."})
    t: Tensor = field(metadata={"help": "time samples t (batch_size, ...)."})
    x_t: Tensor = field(
        metadata={"help": "samples x_t ~ p_t(X_t), shape (batch_size, ...)."}
    )
    dx_t: Tensor = field(
        metadata={"help": "conditional target dX_t, shape: (batch_size, ...)."}
    )


class ProbPath(ABC):
    r"""Abstract class, representing a probability path.

    A probability path transforms the distribution :math:`p(X_0)` into :math:`p(X_1)` over :math:`t=0\rightarrow 1`.

    The ``ProbPath`` class is designed to support model training in the flow matching framework. It supports two key functionalities: (1) sampling the conditional probability path and (2) conversion between various training objectives.
    Here is a high-level example

    .. code-block:: python

        # Instantiate a probability path
        my_path = ProbPath(...)

        for x_0, x_1 in dataset:
            # Sets t to a random value in [0,1]
            t = torch.rand()

            # Samples the conditional path X_t ~ p_t(X_t|X_0,X_1)
            path_sample = my_path.sample(x_0=x_0, x_1=x_1, t=t)

            # Optimizes the model. The loss function varies, depending on model and path.
            loss(path_sample, my_model(x_t, t)).backward()

    """

    @abstractmethod
    def sample(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> PathSample:
        r"""Sample from an abstract probability path:

        | given :math:`(X_0,X_1) \sim \pi(X_0,X_1)`.
        | returns :math:`X_0, X_1, X_t \sim p_t(X_t)`, and a conditional target :math:`Y`, all objects are under ``PathSample``.

        Args:
            x_0 (Tensor): source data point, shape (batch_size, ...).
            x_1 (Tensor): target data point, shape (batch_size, ...).
            t (Tensor): times in [0,1], shape (batch_size).

        Returns:
            PathSample: a conditional sample.
        """

    def assert_sample_shape(self, x_0: Tensor, x_1: Tensor, t: Tensor):
        assert (
            t.ndim == 1
        ), f"The time vector t must have shape [batch_size]. Got {t.shape}."
        assert (
            t.shape[0] == x_0.shape[0] == x_1.shape[0]
        ), f"Time t dimension must match the batch size [{x_1.shape[0]}]. Got {t.shape}"



class AffineProbPath(ProbPath):
    r"""The ``AffineProbPath`` class represents a specific type of probability path where the transformation between distributions is affine.
    An affine transformation can be represented as:

    .. math::

        X_t = \alpha_t X_1 + \sigma_t X_0,

    where :math:`X_t` is the transformed data point at time `t`. :math:`X_0` and :math:`X_1` are the source and target data points, respectively. :math:`\alpha_t` and :math:`\sigma_t` are the parameters of the affine transformation at time `t`.

    The scheduler is responsible for providing the time-dependent parameters :math:`\alpha_t` and :math:`\sigma_t`, as well as their derivatives, which define the affine transformation at any given time `t`.

    Using ``AffineProbPath`` in the flow matching framework:

    .. code-block:: python

        # Instantiates a probability path
        my_path = AffineProbPath(...)
        mse_loss = torch.nn.MSELoss()

        for x_1 in dataset:
            # Sets x_0 to random noise
            x_0 = torch.randn()

            # Sets t to a random value in [0,1]
            t = torch.rand()

            # Samples the conditional path X_t ~ p_t(X_t|X_0,X_1)
            path_sample = my_path.sample(x_0=x_0, x_1=x_1, t=t)

            # Computes the MSE loss w.r.t. the velocity
            loss = mse_loss(path_sample.dx_t, my_model(x_t, t))
            loss.backward()

    Args:
        scheduler (Scheduler): An instance of a scheduler that provides the parameters :math:`\alpha_t`, :math:`\sigma_t`, and their derivatives over time.

    """

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler

    def sample(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> PathSample:
        r"""Sample from the affine probability path:

        | given :math:`(X_0,X_1) \sim \pi(X_0,X_1)` and a scheduler :math:`(\alpha_t,\sigma_t)`.
        | return :math:`X_0, X_1, X_t = \alpha_t X_1 + \sigma_t X_0`, and the conditional velocity at :math:`X_t, \dot{X}_t = \dot{\alpha}_t X_1 + \dot{\sigma}_t X_0`.

        Args:
            x_0 (Tensor): source data point, shape (batch_size, ...).
            x_1 (Tensor): target data point, shape (batch_size, ...).
            t (Tensor): times in [0,1], shape (batch_size).

        Returns:
            PathSample: a conditional sample at :math:`X_t \sim p_t`.
        """
        self.assert_sample_shape(x_0=x_0, x_1=x_1, t=t)

        scheduler_output = self.scheduler(t)

        alpha_t = expand_tensor_like(
            input_tensor=scheduler_output.alpha_t, expand_to=x_1
        )
        sigma_t = expand_tensor_like(
            input_tensor=scheduler_output.sigma_t, expand_to=x_1
        )
        d_alpha_t = expand_tensor_like(
            input_tensor=scheduler_output.d_alpha_t, expand_to=x_1
        )
        d_sigma_t = expand_tensor_like(
            input_tensor=scheduler_output.d_sigma_t, expand_to=x_1
        )

        # construct xt ~ p_t(x|x1).
        x_t = sigma_t * x_0 + alpha_t * x_1
        dx_t = d_sigma_t * x_0 + d_alpha_t * x_1

        return PathSample(x_t=x_t, dx_t=dx_t, x_1=x_1, x_0=x_0, t=t)

    def target_to_velocity(self, x_1: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        r"""Convert from x_1 representation to velocity.

        | given :math:`X_1`.
        | return :math:`\dot{X}_t`.

        Args:
            x_1 (Tensor): target data point.
            x_t (Tensor): path sample at time t.
            t (Tensor): time in [0,1].

        Returns:
            Tensor: velocity.
        """
        scheduler_output = self.scheduler(t)

        alpha_t = scheduler_output.alpha_t
        d_alpha_t = scheduler_output.d_alpha_t
        sigma_t = scheduler_output.sigma_t
        d_sigma_t = scheduler_output.d_sigma_t

        a_t = d_sigma_t / sigma_t
        b_t = (d_alpha_t * sigma_t - d_sigma_t * alpha_t) / sigma_t

        return a_t * x_t + b_t * x_1

    def epsilon_to_velocity(self, epsilon: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        r"""Convert from epsilon representation to velocity.

        | given :math:`\epsilon`.
        | return :math:`\dot{X}_t`.

        Args:
            epsilon (Tensor): noise in the path sample.
            x_t (Tensor): path sample at time t.
            t (Tensor): time in [0,1].

        Returns:
            Tensor: velocity.
        """
        scheduler_output = self.scheduler(t)

        alpha_t = scheduler_output.alpha_t
        d_alpha_t = scheduler_output.d_alpha_t
        sigma_t = scheduler_output.sigma_t
        d_sigma_t = scheduler_output.d_sigma_t

        a_t = d_alpha_t / alpha_t
        b_t = (d_sigma_t * alpha_t - d_alpha_t * sigma_t) / alpha_t

        return a_t * x_t + b_t * epsilon

    def velocity_to_target(self, velocity: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        r"""Convert from velocity to x_1 representation.

        | given :math:`\dot{X}_t`.
        | return :math:`X_1`.

        Args:
            velocity (Tensor): velocity at the path sample.
            x_t (Tensor): path sample at time t.
            t (Tensor): time in [0,1].

        Returns:
            Tensor: target data point.
        """
        scheduler_output = self.scheduler(t)

        alpha_t = scheduler_output.alpha_t
        d_alpha_t = scheduler_output.d_alpha_t
        sigma_t = scheduler_output.sigma_t
        d_sigma_t = scheduler_output.d_sigma_t

        a_t = -d_sigma_t / (d_alpha_t * sigma_t - d_sigma_t * alpha_t)
        b_t = sigma_t / (d_alpha_t * sigma_t - d_sigma_t * alpha_t)

        return a_t * x_t + b_t * velocity

    def epsilon_to_target(self, epsilon: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        r"""Convert from epsilon representation to x_1 representation.

        | given :math:`\epsilon`.
        | return :math:`X_1`.

        Args:
            epsilon (Tensor): noise in the path sample.
            x_t (Tensor): path sample at time t.
            t (Tensor): time in [0,1].

        Returns:
            Tensor: target data point.
        """
        scheduler_output = self.scheduler(t)

        alpha_t = scheduler_output.alpha_t
        sigma_t = scheduler_output.sigma_t

        a_t = 1 / alpha_t
        b_t = -sigma_t / alpha_t

        return a_t * x_t + b_t * epsilon

    def velocity_to_epsilon(self, velocity: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        r"""Convert from velocity to noise representation.

        | given :math:`\dot{X}_t`.
        | return :math:`\epsilon`.

        Args:
            velocity (Tensor): velocity at the path sample.
            x_t (Tensor): path sample at time t.
            t (Tensor): time in [0,1].

        Returns:
            Tensor: noise in the path sample.
        """
        scheduler_output = self.scheduler(t)

        alpha_t = scheduler_output.alpha_t
        d_alpha_t = scheduler_output.d_alpha_t
        sigma_t = scheduler_output.sigma_t
        d_sigma_t = scheduler_output.d_sigma_t

        a_t = -d_alpha_t / (d_sigma_t * alpha_t - d_alpha_t * sigma_t)
        b_t = alpha_t / (d_sigma_t * alpha_t - d_alpha_t * sigma_t)

        return a_t * x_t + b_t * velocity

    def target_to_epsilon(self, x_1: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        r"""Convert from x_1 representation to velocity.

        | given :math:`X_1`.
        | return :math:`\epsilon`.

        Args:
            x_1 (Tensor): target data point.
            x_t (Tensor): path sample at time t.
            t (Tensor): time in [0,1].

        Returns:
            Tensor: noise in the path sample.
        """
        scheduler_output = self.scheduler(t)

        alpha_t = scheduler_output.alpha_t
        sigma_t = scheduler_output.sigma_t

        a_t = 1 / sigma_t
        b_t = -alpha_t / sigma_t

        return a_t * x_t + b_t * x_1


class CondOTProbPath(AffineProbPath):
    r"""The ``CondOTProbPath`` class represents a conditional optimal transport probability path.

    This class is a specialized version of the ``AffineProbPath`` that uses a conditional optimal transport scheduler to determine the parameters of the affine transformation.

    The parameters :math:`\alpha_t` and :math:`\sigma_t` for the conditional optimal transport path are defined as:

    .. math::

        \alpha_t = t \quad \text{and} \quad \sigma_t = 1 - t.
    """

    def __init__(self):
        self.scheduler = CondOTScheduler()