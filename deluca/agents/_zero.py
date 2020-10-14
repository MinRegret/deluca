"""deluca.agents._zero.py"""
from typing import Sequence

import jax.numpy as jnp

from deluca.agents.core import Agent


class Zero(Agent):
    def __init__(self, shape: Sequence[int]):
        """"""
        if not isinstance(shape, Sequence) or not all(isinstance(x, int) for x in shape):
            self.throw(ValueError, "Shape must be a list or tuple of ints")

        # Enforce (N, 1) for 1-dimensional tensors
        if len(shape) == 1:
            shape = (shape[0], 1)

        self.shape = tuple(shape)
        self.action = jnp.zeros(shape)

    def __call__(self, state):
        """
        Description: Return the action based on current state and internal parameters.

        Args:
            state (float/numpy.ndarray): current state

        Returns:
            action (float/numpy.ndarray): action to take
        """
        return self.action
