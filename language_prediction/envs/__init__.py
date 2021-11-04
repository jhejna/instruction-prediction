from .babyai_wrappers import FullyObsLanguageWrapper, LanguageWrapper

# Register the mazebase env
from gym.envs.registration import register
register(
    id='mazebase-v0',
    max_episode_steps=100,
    entry_point='language_prediction.envs.mazebase:MazebaseGame',
)