import jax
import jax.numpy as jnp

from deluca.core import JaxObject


class Random(JaxObject):
    def __init__(self, seed=0) -> None:
        """Initialize instance"""
        self.set_key(seed)

    def set_key(self, seed=None) -> None:
        self.key = jax.random.PRNGKey(seed)

    def get_key(self) -> jnp.ndarray:
        return self.key

    def generate_key(self) -> jnp.ndarray:
        """Generates random subkey"""
        self.key, subkey = jax.random.split(self.key)
        return subkey
