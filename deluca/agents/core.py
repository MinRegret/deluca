import inspect
import logging

from deluca.core import JaxObject

AgentRegistry = {}

logger = logging.getLogger(__name__)

# TODO
# - Support hierarchical agents


def make_agent(cls, *args, **kwargs):
    if cls not in AgentRegistry:
        raise ValueError(f"Agent `{cls}` not found.")
    return AgentRegistry[cls](*args, **kwargs)


class Agent(JaxObject):
    @classmethod
    def __init_subclass__(cls, *args, **kwargs):
        """For avoiding a decorator for each subclass"""
        super().__init_subclass__(*args, **kwargs)
        if cls.__name__ not in AgentRegistry and not inspect.isabstract(cls):
            AgentRegistry[cls.__name__] = cls

    @property
    def name(self):
        return self.__class__.__name__

    def __call__(self, observation):
        raise NotImplementedError()

    def reset(self):
        logger.warn(f"`reset` for agent `{self.name}` not implemented")

    def feed(self, reward):
        logger.warn(f"`feed` for agent `{self.name}` not implemented")
