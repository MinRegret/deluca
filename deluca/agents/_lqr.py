"""deluca.agents._lqr"""
import jax.numpy as jnp
from scipy.linalg import solve_discrete_are as dare

from deluca.agents.core import Agent


class LQR(Agent):
    """
    LQR
    """

    def __init__(
        self, A: jnp.ndarray, B: jnp.ndarray, Q: jnp.ndarray = None, R: jnp.ndarray = None
    ) -> None:
        """
        Description: Initialize the infinite-time horizon LQR.
        Args:
            A (jnp.ndarray): system dynamics
            B (jnp.ndarray): system dynamics
            Q (jnp.ndarray): cost matrices (i.e. cost = x^TQx + u^TRu)
            R (jnp.ndarray): cost matrices (i.e. cost = x^TQx + u^TRu)

        Returns:
            None
        """

        state_size, action_size = B.shape

        if Q is None:
            Q = jnp.identity(state_size, dtype=jnp.float32)

        if R is None:
            R = jnp.identity(action_size, dtype=jnp.float32)

        # solve the ricatti equation
        X = dare(A, B, Q, R)

        # compute LQR gain
        self.K = jnp.linalg.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    def __call__(self, state) -> jnp.ndarray:
        """
        Description: Return the action based on current state and internal parameters.

        Args:
            state (float/numpy.ndarray): current state

        Returns:
           jnp.ndarray: action to take
        """
        return (-self.K @ state).squeeze()
