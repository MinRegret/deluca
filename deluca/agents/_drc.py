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
from deluca.utils import Random

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


class DRC(Agent):
    def __init__(
        self,
        A: jnp.ndarray,
        B: jnp.ndarray,
        C: jnp.ndarray = None,
        K: jnp.ndarray = None,
        cost_fn: Callable[[jnp.ndarray, jnp.ndarray], Real] = None,
        m: int = 10,
        h: int = 50,
        lr_scale: Real = 0.01,
        decay: bool = False,
        seed: int = 0
    ) -> None:
        """
        Description: Initialize the dynamics of the model.

        Args:
            A (jnp.ndarray): system dynamics
            B (jnp.ndarray): system dynamics
            C (jnp.ndarray): system dynamics
            Q (jnp.ndarray): cost matrices (i.e. cost = x^TQx + u^TRu)
            R (jnp.ndarray): cost matrices (i.e. cost = x^TQx + u^TRu)
            K (jnp.ndarray): Starting policy (optional). Defaults to LQR gain.
            start_time (int):
            cost_fn (Callable[[jnp.ndarray, jnp.ndarray], Real]):
            H (postive int): history of the controller
            HH (positive int): history of the system
            lr_scale (Real):
            decay (boolean):
            seed (int):
        """

        cost_fn = cost_fn or quad_loss

        self.random = Random(seed)

        d_state, d_action = B.shape  # State & Action Dimensions

        C = jnp.identity(d_state) if C is None else C

        d_obs = C.shape[0] # Observation Dimension

        self.t = 0  # Time Counter (for decaying learning rate)

        self.m, self.h = m, h

        self.lr_scale, self.decay = lr_scale, decay

        # Construct truncated markov operator G
        self.G = jnp.zeros((h, d_obs, d_action))
        A_power = jnp.identity(d_state)
        for i in range(h):
            self.G = jax.ops.index_update(self.G, i, C @ A_power @ B)
            A_power = A_power @ A

        # Model Parameters
        # initial linear policy / perturbation contributions / bias
        self.K = K if K is not None else jnp.zeros((d_action, d_obs))

        self.M = lr_scale * jax.random.normal(self.random.generate_key(), shape=(m, d_action, d_obs))

        # Past m nature y's such that y_nat[0] is the most recent
        self.y_nat = jnp.zeros((m, d_obs, 1))

        # Past h u's such that u[0] is the most recent
        self.us = jnp.zeros((h, d_action, 1))

        def policy_loss(M, G, y_nat, us):
            """Surrogate cost function"""

            def action(obs):
                """Action function"""
                return -self.K @ obs + jnp.tensordot(M, y_nat, axes=([0, 2], [0, 1]))

            final_state = y_nat[0] + jnp.tensordot(G, us, axes=([0, 2], [0, 1]))
            return cost_fn(final_state, action(final_state))

        self.policy_loss = policy_loss
        self.grad = jit(grad(policy_loss))

    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """
        Description: Return the action based on current state and internal parameters.

        Args:
            state (jnp.ndarray): current state

        Returns:
           jnp.ndarray: action to take
        """

        # update y_nat
        y_nat = obs - jnp.tensordot(self.G, self.us, axes=([0, 2], [0, 1]))
        self.y_nat = jnp.roll(self.y_nat, 1, axis=0)
        self.y_nat = jax.ops.index_update(self.y_nat, 0, y_nat)

        # get action
        action = -self.K @ obs + jnp.tensordot(self.M, self.y_nat, axes=([0, 2], [0, 1]))

        # update parameters
        delta_M = self.grad(self.M, self.G, self.y_nat, self.us)
        lr = self.lr_scale
        lr *= (1/ (self.t+1)) if self.decay else 1
        self.M -= lr * delta_M

        # update us
        self.us = jnp.roll(self.us, 1, axis=0)
        self.us = jax.ops.index_update(self.us, 0, action)

        self.t += 1

        return action