import gym
import jax
import jax.numpy as jnp
import numpy as np

from deluca.envs.core import Env
from deluca.utils import Random


@jax.jit
def angle_normalize(x):
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi


def default_reward_fn(x, u):
    return -(np.sum(angle_normalize(x[0]) ** 2 + 0.1 * x[1] ** 2 + 0.001 * (u ** 2)))


class Pendulum(Env):
    max_speed = 8.0
    max_torque = 2.0  # gym 2.
    high = np.array([1.0, 1.0, max_speed])

    action_space = gym.spaces.Box(low=-max_torque, high=max_torque, shape=(1,), dtype=np.float32)
    observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)

    def __init__(self, reward_fn=None, seed=0):
        self.reward_fn = reward_fn or default_reward_fn
        self.dt = 0.05
        self.viewer = None

        self.state_size = 2
        self.action_size = 1

        self.n, self.m = 2, 1
        self.angle_normalize = angle_normalize

        self.random = Random(seed)

        self.reset()

    @jax.jit
    def dynamics(self, state, action):
        th, thdot = state
        g = 10.0
        m = 1.0
        ell = 1.0
        dt = self.dt

        # Do not limit the control signals
        action = jnp.clip(action, -self.max_torque, self.max_torque)

        newthdot = (
            thdot + (-3 * g / (2 * ell) * jnp.sin(th + jnp.pi) + 3.0 / (m * ell ** 2) * action) * dt
        )
        newth = th + newthdot * dt
        newthdot = jnp.clip(newthdot, -self.max_speed, self.max_speed)

        return jnp.array([newth, newthdot])

    def reset(self):
        th = jax.random.uniform(self.random.generate_key(), minval=-jnp.pi, maxval=jnp.pi)
        thdot = jax.random.uniform(self.random.generate_key(), minval=-1.0, maxval=1.0)

        self.state = jnp.array([th, thdot])
