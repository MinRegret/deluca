import jax
import jax.numpy as jnp

from deluca.agents.core import Agent
from deluca.utils import Random


# generic deep controller
class Deep(Agent):
    def __init__(
        self, env, learning_rate=0.001, gamma=0.99, max_episode_length=500, seed=0, **kwargs
    ):
        # Create gym and seed numpy
        self.env = env
        self.max_episode_length = max_episode_length
        self.lr = learning_rate
        self.gamma = gamma

        self.random = Random(seed)

        self.reset()

    def reset(self):
        # Init weight
        self.W = jax.random.uniform(
            self.random.generate_key(),
            shape=(self.env.observation_size, len(self.env.action_space)),
            minval=0,
            maxval=1,
        )

        # Keep stats for final print of graph
        self.episode_rewards = []

        self.current_episode_length = 0
        self.current_episode_reward = 0
        self.episode_rewards = jnp.zeros(self.max_episode_length)
        self.episode_grads = jnp.zeros((self.max_episode_length, self.W.shape[0], self.W.shape[1]))

    # Our policy that maps state to action parameterized by w
    def policy(self, state, w):
        z = jnp.dot(state, w)
        exp = jnp.exp(z)
        return exp / jnp.sum(exp)

    # Vectorized softmax Jacobian
    def softmax_grad(self, softmax):
        s = softmax.reshape(-1, 1)
        return jnp.diagflat(s) - jnp.dot(s, s.T)

    def __call__(self, observation):
        self.observation = observation
        self.probs = self.policy(observation, self.W)
        self.action = jax.random.choice(
            self.random.generate_key(), a=self.env.action_space, p=self.probs
        )
        return self.action

    def feed(self, reward):
        # Compute gradient and save with reward in memory for our weight updates
        dsoftmax = self.softmax_grad(self.probs)[self.action, :]
        dlog = dsoftmax / self.probs[self.action]
        grad = self.observation.reshape(-1, 1) @ dlog.reshape(1, -1)

        self.episode_rewards = jax.ops.index_update(
            self.episode_rewards, self.current_episode_length, reward
        )
        self.episode_grads = jax.ops.index_update(
            self.episode_grads, self.current_episode_length, grad
        )
        self.current_episode_length += 1
        return self

    def update(self):
        # Weight update
        for i in range(self.current_episode_length):
            # Loop through everything that happend in the episode and update
            # towards the log policy gradient times **FUTURE** reward
            self.W += self.lr * self.episode_grads[i] + jnp.sum(
                jnp.array(
                    [
                        r * (self.gamma ** r)
                        for r in self.episode_rewards[i : self.current_episode_length]
                    ]
                )
            )
        # reset episode length
        self.current_episode_length = 0
