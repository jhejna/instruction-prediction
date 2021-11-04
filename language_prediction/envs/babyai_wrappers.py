import numpy as np
import gym

from gym import spaces
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX, IDX_TO_OBJECT, IDX_TO_COLOR, STATE_TO_IDX


WORD_TO_IDX = {'PAD': 0, 'END' : 1, 'END_MISSION': 2, 'END_SUBGOAL': 3, 'the': 4, 'to': 5, 
               'a': 6, 'put': 7, 'next': 8, 'ball': 9, 'key': 10, 'box': 11, 'pick': 12, 'up': 13,
               'green': 14, 'yellow': 15, 'purple': 16, 'blue': 17, 'red': 18, 'grey': 19, 'door': 20,
               'and': 21, 'open': 22, 'go': 23, 'object': 24, 'you': 25, 'on': 26, 'your': 27,
               'after': 28, 'then': 29, 'right': 30, 'left': 31, 'behind': 32, 'in': 33, 'front': 34,
               'of': 35, 'move': 36, 'drop': 37, 'close': 38}

class LanguageWrapper(gym.Wrapper):

    def __init__(self, env, max_len=-1, pad=False):
        super().__init__(env)
        self.max_len = max_len
        self.pad = pad
        self.parsed_mission = None
        mission_sp_dim = 1 if self.max_len < 0 else self.max_len
        self.observation_space = spaces.Dict(
            image=spaces.Box(low=0, high=147, shape=(3, 7, 7), dtype='uint8'),
            mission=spaces.Box(low=0, high=len(WORD_TO_IDX), shape=(mission_sp_dim,), dtype='long'),
        )

    @staticmethod
    def parse_text(text):
        text = text.lower().strip().replace(',', '')
        words = text.split(' ')
        tokens = [WORD_TO_IDX[word] for word in words]
        return tokens

    @staticmethod
    def convert_mission_to_data(mission, max_len=-1, pad=True):
        tokens = LanguageWrapper.parse_text(mission)
        tokens.append(WORD_TO_IDX['END_MISSION'])
        if max_len > 0:
            if len(tokens) > max_len:
                return None
            while len(tokens) < max_len and pad:
                tokens.append(WORD_TO_IDX['PAD'])
        return np.array(tokens, dtype=np.long)

    @staticmethod
    def convert_subgoals_to_data(subgoals, max_len=-1, pad=True):
        tokens = []
        for subgoal in subgoals:
            tokens.extend(LanguageWrapper.parse_text(subgoal))
            tokens.append(WORD_TO_IDX['END_SUBGOAL'])
        tokens.append(WORD_TO_IDX['END']) # NOTE: Changed to keeping the end subgoal token. This should help with subgoal pred level task. 
        if max_len > 0:
            if len(tokens) > max_len:
                return None
            while len(tokens) < max_len and pad:
                tokens.append(WORD_TO_IDX['PAD'])
        return np.array(tokens, dtype=np.long)

    def observation(self, obs):
        img = np.transpose(obs['image'], (2, 0, 1)) # Transpose the observaiton data
        return dict(image=img, mission=self.parsed_mission)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        success = True if reward > 0 and done else False
        info['is_success'] = success
        return self.observation(observation), reward, done, info

    def reset(self, **kwargs):
        self.parsed_mission = None
        while self.parsed_mission is None:
            observation = self.env.reset(**kwargs)
            self.parsed_mission = self.convert_mission_to_data(observation['mission'], max_len=self.max_len, pad=self.pad)
        return self.observation(observation)