from deluca.agents.core import Agent

DEFAULT_K = [3.0, 4.0, 0.0]


class PID(Agent):
    def __init__(self, K=None, RC=0.5, **kwargs):
        # controller coeffs
        self.K_P, self.K_I, self.K_D = K or DEFAULT_K

        # controller states
        self.P, self.I, self.D = 0.0, 0.0, 0.0

        self.RC = RC

    def act(self, state):
        err = state["target"] - state["measured"]

        dt = state["dt"]

        decay = dt / (dt + self.RC)

        self.I += decay * (err - self.I)
        self.D += decay * (err - self.P - self.D)
        self.P = err

        return self.K_P * self.P + self.K_I * self.I + self.K_D * self.D
