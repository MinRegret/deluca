import jax.numpy as jnp

from deluca.agents.core import Agent


class Zero(Agent):
    def __init__(self, action_size):
        """
        """
        self.action_size = action_size

    def __call__(self, state):
        """
        Description: Return the action based on current state and internal parameters.

        Args:
            state (float/numpy.ndarray): current state

        Returns:
            action (float/numpy.ndarray): action to take
        """
        return jnp.zeros((self.action_size, 1))
