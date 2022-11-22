from functools import partial
from .multiagentenv import MultiAgentEnv
from .starcraft2 import StarCraft2Env
from .turn import TurnEnv
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["turn"] = partial(env_fn, env=TurnEnv)
