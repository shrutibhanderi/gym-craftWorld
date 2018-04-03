from gym.envs.registration import register

register(
    id='craftWorld-v0',
    entry_point='gym_craftWorld.envs:craftWorldEnv',
)
register(
    id='craftWorld-extrahard-v0',
    entry_point='gym_craftWorld.envs:craftWorldExtraHardEnv',
)
