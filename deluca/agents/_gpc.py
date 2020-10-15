"""deluca.agents._gpc"""
from numbers import Real
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad
from jax import jit

from deluca.agents._lqr import LQR
from deluca.agents.core import Agent


def quad_loss(x: jnp.ndarray, u: jnp.ndarray) -> Real:
    """
    Quadratic loss.

    Args:
        x (jnp.ndarray):
        u (jnp.ndarray):

    Returns:
        Real
    """
    return jnp.sum(x.T @ x + u.T @ u)


class GPC(Agent):
    def __init__(
        self,
        T: int,
        A: jnp.ndarray,
        B: jnp.ndarray,
        Q: jnp.ndarray = None,
        R: jnp.ndarray = None,
        K: jnp.ndarray = None,
        start_time: int = 0,
        cost_fn: Callable[[jnp.ndarray, jnp.ndarray], Real] = None,
        H: int = 3,
        HH: int = 2,
        lr_scale: Real = 0.0001,
        lr_scale_decay: Real = 1.0,
        decay: bool = True,
    ) -> None:
        """
        Description: Initialize the dynamics of the model.

        Args:
            A (jnp.ndarray): system dynamics
            B (jnp.ndarray): system dynamics
            Q (jnp.ndarray): cost matrices (i.e. cost = x^TQx + u^TRu)
            R (jnp.ndarray): cost matrices (i.e. cost = x^TQx + u^TRu)
            K (jnp.ndarray): Starting policy (optional). Defaults to LQR gain.
            start_time (int):
            cost_fn (Callable[[jnp.ndarray, jnp.ndarray], Real]):
            H (postive int): history of the controller
            HH (positive int): history of the system
            lr_scale (Real):
            lr_scale_decay (Real):
            decay (Real):
        """

        cost_fn = cost_fn or quad_loss

        d_state, d_action = B.shape  # State & Action Dimensions

        self.A, self.B = A, B  # System Dynamics

        self.T = T - start_time
        self.t = 0  # Time Counter (for decaying learning rate)

        self.H, self.HH = H, HH

        def fixed_lr(_):
            """Fixed learning rate"""
            return lr_scale

        def decay_lr(t):
            """Decay learning rate"""
            lr_scale_decay / (1 + t)

        self.lr_fn = decay_lr if decay else fixed_lr

        # Model Parameters
        # initial linear policy / perturbation contributions / bias
        self.K = K if K is not None else LQR(self.A, self.B, Q, R).K

        self.M = jnp.zeros((H, d_action, d_state))

        # Past H + HH noises ordered increasing in time
        self.noise_history = jnp.zeros((H + HH, d_state, 1))

        # past state and past action
        self.state, self.action = jnp.zeros((d_state, 1)), jnp.zeros((d_action, 1))

        def last_h_noises():
            """Get noise history"""
            return jax.lax.dynamic_slice_in_dim(self.noise_history, -H, H)

        self.last_h_noises = last_h_noises

        def policy_loss(M, w):
            """Surrogate cost function"""

            def action(state, h):
                """Action function"""
                return -self.K @ state + jnp.tensordot(
                    M, jax.lax.dynamic_slice_in_dim(w, h, H), axes=([0, 2], [0, 1])
                )

            def evolve(state, h):
                """Evolve function"""
                return self.A @ state + self.B @ action(state, h) + w[h + H], None

            final_state, _ = jax.lax.scan(evolve, np.zeros((d_state, 1)), np.arange(H - 1))
            return cost_fn(final_state, action(final_state, HH - 1))

        self.policy_loss = policy_loss
        self.grad = jit(grad(policy_loss, (0, 1)))

    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Description: Return the action based on current state and internal parameters.

        Args:
            state (jnp.ndarray): current state

        Returns:
           jnp.ndarray: action to take
        """
        state = state.reshape(self.state.shape)

        action = self.get_action(state)
        self.update(state)
        return action.squeeze()

    def update(self, state: jnp.ndarray) -> None:
        """
        Description: update agent internal state.

        Args:
            state (jnp.ndarray):

        Returns:
            None
        """
        self._update_params()
        self._update_noise(state)
        self._update_history(state)

        self.t = self.t + 1

    def get_action(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Description: get action from state.

        Args:
            state (jnp.ndarray):

        Returns:
            jnp.ndarray
        """
        return -self.K @ state + jnp.tensordot(self.M, self.last_h_noises(), axes=([0, 2], [0, 1]))

    def _update_params(self) -> None:
        """Update parameters"""
        delta_M, delta_bias = self.grad(self.M, self.noise_history)

        lr = self.lr_fn(self.t)
        self.M -= lr * delta_M

    def _update_noise(self, state: jnp.ndarray) -> None:
        """Update noise"""
        noise = state - self.A @ self.state - self.B @ self.action
        self.noise_history = jax.ops.index_update(self.noise_history, 0, noise)
        self.noise_history = jnp.roll(self.noise_history, -1, axis=0)

    def _update_history(self, state) -> None:
        """Update history"""
        self.state = state
        self.action = -self.K @ state + jnp.tensordot(
            self.M, self.last_h_noises(), axes=([0, 2], [0, 1])
        )
