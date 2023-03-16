from gym.envs.registration import register

register(
    id='TurtleBot-v0',
    entry_point='TurtleBot_v0.envs:TurtleBotV0Env',
)

register(
    id='TurtleBot-v1',
    entry_point='TurtleBot_v0.envs:TurtleBotV1Env',
)

register(
    id='TurtleBot-v2',
    entry_point='TurtleBot_v0.envs:TurtleBotV2Env',
)