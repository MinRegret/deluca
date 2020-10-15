import jax.numpy as jnp

from deluca.envs.core import Env
from deluca.envs.lung import BreathWaveform


class LearnedLung(Env):
    def __init__(
        self,
        min_volume=1.5,
        R_lung=10,
        C_lung=6,
        delay=25,
        inertia=0.995,
        control_gain=0.02,
        dt=0.03,
        waveform=None,
        reward_fn=None,
    ):
        self.waveform = waveform or BreathWaveform()
        self.r0 = (3 * self.min_volume / (4 * jnp.pi)) ** (1 / 3)
        self.reset()

    def reset(self):
        self.volume = self.min_volume
        self.pipe_pressure = 0
        self.in_history = jnp.zeros(self.delay)
        self.out_history = jnp.zeros(self.delay)
        self.T = 0

        self.state = (self.volume, self.pipe_pressure)
        return self.observation

    @property
    def observation(self):
        return {
            "measured": self.state["pressure"],
            "target": self.target,
            "dt": self.dt,
            "phase": self.waveform.phase(self.time),
        }

    def dynamics(self, state, action):
        """
        state: (volume, pressure)
        action: (u_in, u_out)
        """
        flow = self.state["pressure"] / self.R_lung
        volume = self.state["volume"] + flow * self.dt
        volume = jnp.maximum(volume, self.min_volume)

        r = (3.0 * self.volume / (4.0 * jnp.pi)) ** (1.0 / 3.0)
        lung_pressure = self.C_lung * (1 - (self.r0 / r) ** 6) / (self.r0 ** 2 * r)

        if self.T < self.delay:
            pipe_impulse = 0
            peep = 0
        else:
            pipe_impulse = self.control_gain * self.in_history[0]
            peep = self.out_history[0]

        pipe_pressure = self.inertia * state["pipe_pressure"] + pipe_impulse
        pressure = jnp.maximum(0, pipe_pressure - lung_pressure)

        if peep:
            pipe_pressure *= 0.995

        return volume, pressure, pipe_pressure

    def step(self, action):
        u_in, u_out = action
        u_in = max(0, u_in)

        self.in_history = jnp.roll(self.in_history, shift=1)
        self.in_history = self.in_history.at[0].set(u_in)
        self.out_history = jnp.roll(self.out_history, shift=1)
        self.out_history = self.out_history.at[0].set(u_out)

        self.target = self.waveform.at(self.time)
        reward = -jnp.abs(self.target - self.state["pressure"])

        self.state = self.dynamics(self.state, action)
        self.T += 1

        return self.observation, reward, False, {}
