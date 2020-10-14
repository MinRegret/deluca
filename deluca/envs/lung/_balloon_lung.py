import jax
import jax.numpy as jnp

from deluca.envs.core import Env
from deluca.envs.lung import BreathWaveform


def PropValve(x):
    y = 3.0 * x
    flow_new = 1.0 * (jnp.tanh(0.03 * (y - 130)) + 1.0)
    flow_new = jnp.clip(flow_new, 0.0, 1.72)
    return flow_new


def Solenoid(x):
    return x > 0


class BalloonLung(Env):
    """
    Lung simulator based on the two-balloon experiment

    Source: https://en.wikipedia.org/wiki/Two-balloon_experiment

    TODO:
    - time / dt
    - dynamics
    - waveform
    - phase
    """

    def __init__(
        self,
        leak=False,
        peep_valve=5.0,
        PC=40.0,
        P0=0.0,
        C=10.0,
        R=15.0,
        dt=0.03,
        waveform=None,
        reward_fn=None,
    ):
        # dynamics hyperparameters
        self.min_volume = 1.5
        self.C = C
        self.R = R
        self.PC = PC
        self.P0 = 0.0
        self.leak = leak
        self.peep_valve = peep_valve
        self.time = 0.0
        self.dt = dt

        self.waveform = waveform or BreathWaveform()

        self.r0 = (3.0 * self.min_volume / (4.0 * jnp.pi)) ** (1.0 / 3.0)

        # reset states
        self.reset()

    def reset(self):
        self.target = self.waveform.at(self.time)
        self.volume, self.pressure = self.dynamics((self.min_volume, -1.0), (0.0, 0.0))
        return self.observation

    @property
    def observation(self):
        return {
            "measured": self.pressure,
            "target": self.target,
            "dt": self.dt,
            "phase": self.waveform.phase(self.time),
        }

    def dynamics(self, state, action):
        """
        state: (volume, pressure)
        action: (u_in, u_out)
        """
        volume, pressure = state
        u_in, u_out = action

        flow = jnp.clip(PropValve(u_in) * self.R, 0.0, 2.0)
        flow -= jax.lax.cond(
            pressure > self.peep_valve,
            lambda x: jnp.clip(Solenoid(u_out), 0.0, 2.0) * 0.05 * pressure,
            lambda x: 0.0,
            flow,
        )

        volume += flow * self.dt
        volume += jax.lax.cond(
            self.leak,
            lambda x: (self.dt / (5.0 + self.dt) * (self.min_volume - volume)),
            lambda x: 0.0,
            0.0,
        )

        r = (3.0 * volume / (4.0 * jnp.pi)) ** (1.0 / 3.0)
        pressure = self.P0 + self.PC * (1.0 - (self.r0 / r) ** 6.0) / (self.r0 ** 2.0 * r)
        # pressure = flow * self.R + volume / self.C + self.peep_valve

        return volume, pressure

    def step(self, action):
        self.target = self.waveform.at(self.time)
        reward = -jnp.abs(self.target - self.pressure)
        self.volume, self.pressure = self.dynamics((self.volume, self.pressure), action)

        self.time += self.dt

        return self.observation, reward, False, {}
