"""
@author: Olivier Sigaud
A merge between two sources:
* Adaptation of the MountainCar Environment from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0), adapted by  'Tom Schaul, tom@idsia.ch'
and then modified by Arnaud de Broissia
* the OpenAI/gym MountainCar environment
itself from
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""
import jax
import jax.numpy as jnp
import numpy as np
from gym import spaces

from deluca.envs.core import Env
from deluca.utils import Random


class MountainCar(Env):
    def __init__(self, goal_velocity=0, seed=0):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.goal_velocity = goal_velocity
        self.power = 0.0015
        self.random = Random(seed)

        self.low_state = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high_state = np.array([self.max_position, self.max_speed], dtype=np.float32)

        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )

        self.reset()

    @jax.jit
    def dynamics(self, state, action):
        position = state[0]
        velocity = state[1]

        force = jnp.minimum(jnp.maximum(action, self.min_action), self.max_action)

        velocity += force * self.power - 0.0025 * jnp.cos(3 * position)
        velocity = jnp.clip(velocity, -self.max_speed, self.max_speed)

        position += velocity
        position = jnp.clip(position, self.min_position, self.max_position)

        reset_velocity = (position == self.min_position) & (velocity < 0)
        velocity = jax.lax.cond(reset_velocity == 1, lambda x: 0.0, lambda x: x, velocity)
        # if (position == self.min_position and velocity < 0): velocity = 0

        return jnp.array([position, velocity])

    def step(self, action):

        self.state = self.dynamics(self.state, action)
        position = self.state[0]
        velocity = self.state[1]

        # Convert a possible numpy bool to a Python bool.
        done = (position >= self.goal_position) & (velocity >= self.goal_velocity)

        reward = 100.0 * done
        reward -= jnp.power(action, 2) * 0.1

        return self.state, reward, done, {}

    def reset(self):
        self.state = jnp.array(
            [jax.random.uniform(self.random.generate_key(), minval=-0.6, maxval=0.4), 0]
        )
        return self.state

    def _height(self, xs):
        return jnp.sin(3 * xs) * 0.45 + 0.55
