import jax.numpy as jnp
from jax.numpy.linalg import inv

from deluca.agents.core import Agent


class Hinf(Agent):
    def __init__(self, A, B, T, Q=None, R=None):
        """
        H-infinity controller (approximately).  Solves a lagrangian min-max
        dynamic program similiar to H-infinity control rescales disturbance to
        have norm 1.
        """
        self.t = 0
        d_x, d_u = B.shape
        Q = jnp.identity(d_x, dtype=jnp.float32) if Q is None else Q
        R = jnp.identity(d_u, dtype=jnp.float32) if R is None else R

        self.K, self.W = solve_hinf(A, B, Q, R, T)

    def act(self, state):
        action = -self.K[self.t] @ state
        self.t += 1
        return action

    def __str__(self):
        return "Hinf"


def solve_hinf(A, B, Q, R, T, gamma_range=(0.1, 10.8)):
    gamma_low, gamma_high = gamma_range
    n, m = B.shape
    M = [jnp.zeros(shape=(n, n))] * (T + 1)
    K = [jnp.zeros(shape=(m, n))] * T
    W = [jnp.zeros(shape=(n, 1))] * T

    while gamma_high - gamma_low > 0.01:
        K = [jnp.zeros(shape=(m, n))] * T
        W = [jnp.zeros(shape=(n, 1))] * T

        gamma = 0.5 * (gamma_high + gamma_low)
        M[T] = Q
        for t in range(T - 1, -1, -1):
            Lambda = jnp.eye(n) + (B @ inv(R) @ B.T - gamma ** (-2) * jnp.eye(n)) @ M[t + 1]
            M[t] = Q + A.T @ M[t + 1] @ inv(Lambda) @ A

            K[t] = -inv(R) @ B.T @ M[t + 1] @ inv(Lambda) @ A
            W[t] = (gamma ** (-2)) * M[t + 1] @ inv(Lambda) @ A
            if not psd_test(M[t], gamma):
                gamma_low = gamma
                break
        if gamma_low != gamma:
            gamma_high = gamma
    return K, W


def psd_test(P, gamma):
    return jnp.all(jnp.linalg.eigvals(P) < gamma ** 2 + 1e-5)
