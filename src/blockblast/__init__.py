from .block_blast_env import BlockBlastEnv
from .block_blast_3p_env import BlockBlast3PEnv

# to register the environments with Gymnasium
from gymnasium.envs.registration import register

register(
    id="BlockBlast-v0",
    entry_point="blockblast:BlockBlastEnv",
)

register(
    id="BlockBlast3P-v0",
    entry_point="blockblast:BlockBlast3PEnv",
)