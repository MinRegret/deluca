import jax.numpy as jnp
from scipy.linalg import solve_discrete_are as dare

from deluca.agents.core import Agent


class LQR(Agent):
    def __init__(self, A, B, Q=None, R=None):
        """
        Description: Initialize the infinite-time horizon LQR.
        Args:
            A, B (float/numpy.ndarray): system dynamics
            Q, R (float/numpy.ndarray): cost matrices (i.e. cost = x^TQx + u^TRu)
        """

        state_size, action_size = B.shape

        Q = jnp.identity(state_size, dtype=jnp.float32) if Q is None else Q
        R = jnp.identity(action_size, dtype=jnp.float32) if R is None else R

        # solve the ricatti equation
        X = dare(A, B, Q, R)

        # compute LQR gain
        self.K = jnp.linalg.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    def __call__(self, state):
        """
        Description: Return the action based on current state and internal parameters.

        Args:
            state (float/numpy.ndarray): current state

        Returns:
            action (float/numpy.ndarray): action to take
        """
        return (-self.K @ state).squeeze()
