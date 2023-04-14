from functools import partial
#from .starcraft import StarCraft2Env
from smac.env import StarCraft2Env
#from .multiagentenv import MultiAgentEnv
from smac.env import MultiAgentEnv
from smac.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

try:
    from smac.env import StarCraft2CustomEnv
except Exception as e:
    print(e)

import sys
import os


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {
    "sc2": partial(env_fn, env=StarCraft2Env),
}
#REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["sc2wrapped"] = partial(env_fn, env=StarCraftCapabilityEnvWrapper)
try:
    REGISTRY["sc2custom"] = partial(env_fn, env=StarCraft2CustomEnv)
except Exception as e:
    print(e)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))


