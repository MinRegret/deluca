import math

import gym
import jax
import jax.numpy as jnp
import numpy as np

from deluca.envs.core import Env
from deluca.utils import Random


class CartPole(Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num	Observation               Min             Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                -24 deg         24 deg
        3	Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200. (not really - only in make, not in
                                            the actual CartPoleEnv class)
        Solved Requirements:
        Considered solved when the average reward is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    def __init__(self, reward_fn=None, seed=0):

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        self.random = Random(seed)

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = jnp.array([0, 1])
        # TODO: no longer use gym.spaces
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        self.state_size, self.action_size = 4, 1
        self.observation_size = self.state_size

        self.reset()

    @jax.jit
    def dynamics(self, state, action):

        x, x_dot, theta, theta_dot = state

        force = jax.lax.cond(action == 1, lambda x: x, lambda x: -x, self.force_mag)

        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        return jnp.array([x, x_dot, theta, theta_dot])

    def reset(self):
        self.state = jax.random.uniform(
            self.random.get_key(), shape=(4,), minval=-0.05, maxval=0.05
        )
        return self.state

    def step(self, action):

        self.state = self.dynamics(self.state, action)
        x, x_dot, theta, theta_dot = self.state

        done = jax.lax.cond(
            (jnp.abs(x) > jnp.abs(self.x_threshold))
            + (jnp.abs(theta) > jnp.abs(self.theta_threshold_radians)),
            lambda done: True,
            lambda done: False,
            None,
        )

        reward = 1 - done

        return self.state, reward, done, {}
