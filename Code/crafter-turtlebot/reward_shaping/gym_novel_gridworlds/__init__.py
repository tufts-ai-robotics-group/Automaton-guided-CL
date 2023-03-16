from gym.envs.registration import register
# import gym_novel_gridworlds.constant
# import gym_novel_gridworlds.wrappers
# import gym_novel_gridworlds.novelty_wrappers
# import gym_novel_gridworlds.observation_wrappers

register(
    id='NovelGridworld-v0',
    entry_point='gym_novel_gridworlds.envs:NovelGridworldV0Env',
)
