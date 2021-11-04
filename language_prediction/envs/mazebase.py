import numpy as np
import gym
from gym import spaces
from torchtext import vocab as vocabtorch
import yaml
import os
import json

import mazebasev2.lib.mazebase.games as games
from mazebasev2.lib.mazebase.games import featurizers
import mazebasev2

def get_summed_embedding(phrase, glove, embed_size):

    phrase = phrase.split(' ')
    phrase_vector = np.zeros((embed_size), dtype=np.float32)

    if isinstance(glove, dict):
        for p in phrase:
            phrase_vector += glove[p.lower()]
    else:
        for p in phrase:
            phrase_vector += glove.vectors[glove.stoi[p.lower()]].data.cpu().numpy()

    return phrase_vector

def get_inventory_embedding(inventory, glove, embed_size):

    inventory_embedding = np.zeros((embed_size), dtype=np.float32)

    first = True
    for item in inventory:
        
        if inventory[item] > 0: # added this to match code in train_hiearchy.py from ask your humans.
            if first:
                inventory_embedding = get_summed_embedding(item, glove, embed_size)
                first = False
            else:
                inventory_embedding = inventory_embedding + get_summed_embedding(item, glove, embed_size)

    return inventory_embedding


# input: batched mazebase grid 
# output: 
def get_grid_embedding(batch_grid, glove, embed_size):
    goal_embedding_array = np.zeros((5, 5, embed_size), dtype=np.float32)

    for x in range(5):
        for y in range(5):

            for index, item in enumerate(batch_grid[x][y]):
                if item == "ResourceFont" or item == "CraftingContainer" or item == "CraftingItem":
                    goal_embedding_array[x][y] = get_summed_embedding(batch_grid[x][y][index+1], glove, embed_size)
            
    return goal_embedding_array

def get_goal_embedding(goal, glove, embed_size):

    #currently all crafts are 2 word phrases
    # goal in the format of "Make Diamond Boots (Diamond Boots=1)" --> just extract diamond boots part

    goal_embedding = np.zeros((embed_size), dtype=np.float32)

    goal = goal.split(' ')

    if isinstance(glove, dict):
        item1_vec = glove[goal[1].lower()]
        item2_vec = glove[goal[2].lower()]
    else:
        item1_vec = glove.vectors[glove.stoi[goal[1].lower()]].data.cpu().numpy()
        item2_vec = glove.vectors[glove.stoi[goal[2].lower()]].data.cpu().numpy()

    goal_embedding = item1_vec+item2_vec

    return goal_embedding

def one_hot_grid(grid, glove, embed_size):

    grid_embedding_array = np.zeros((5, 5, 7), dtype=np.float32)

    for x in range(5):
        for y in range(5):

            for index, item in enumerate(grid[x][y]):
                if item == 'Corner':
                    grid_embedding_array[x][y][0] = 1
                elif item == 'Agent':
                    grid_embedding_array[x][y][1] = 1
                elif 'Door' in item:
                    grid_embedding_array[x][y][2] = 1
                elif item == 'Key':
                    grid_embedding_array[x][y][3] = 1
                elif item == 'Switch':
                    grid_embedding_array[x][y][4] = 1
                elif item == 'Block':
                    grid_embedding_array[x][y][5] = 1

                # Now make this seperate to mark closed equally!!! Was BUG in ask your humans code!
                if 'closed' in item: # door closed
                    grid_embedding_array[x][y][6] = 1

    return grid_embedding_array

def get_action_name(action):

    if action == 1:
        return 'up'
    elif action == 2:
        return 'down'
    elif action == 3:
        return 'left'
    elif action == 4:
        return 'right'
    elif action == 5:
        return  'toggle_switch'
    elif action == 6:
        return 'grab'
    elif action == 7:
        return 'mine'
    elif action  == 0:
        return 'craft'
    elif action == 8:
        return 'stop'

def one_hot_action(action):

        if action == 'up':
            return np.array([1])
        elif action == 'down':
            return np.array([2])
        elif action == 'left':
            return np.array([3])
        elif action == 'right':
            return np.array([4])
        elif action == 'toggle_switch':
            return np.array([5])
        elif action == 'grab':
            return np.array([6])
        elif action == 'mine':
            return np.array([7])
        elif action == 'craft':
            return np.array([0])
        elif action == 'stop':
            return np.array([8])

CONFIGS = {'minimum_viable_rl',
                        'length12task',
                        'length123task',
                        'length12345task',
                        'length12345task_missing',
                        'length35task',
                        'length2task',
                        'length1task',
                        'length3task',
                        'length45uncommon',
                        'length45common',
                        'unseen_tasks',
                        'unseen_full',
                        'seen_full',
                        'unseen_2only',
                        'unseen_3only',
                        'unseen_5only'}

class MazebaseGame(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config='length1task'):

        self.embed_size = 300
        if self.embed_size == 50:
            glove = vocabtorch.GloVe(name='6B', dim=50)
        else:
            glove = vocabtorch.GloVe(name='840B', dim=300)

        vocab = ['Gold', 'Ore', 'Vein', 'Key', "Pickaxe", "Iron", "Diamond", "Boots", "Station", "Brick", "Stairs", "Station", "Factory", "Cobblestone", "Stash", "Sword", "Ingot", "Coal", "Leggins", "Leggings", "Leather", "Rabbit", "Hide", "Chestplate", "Helmet", "Wood", "Plank", "Door", "Tree", "Wooden", "Axe", "Stick", "Stone"]

        self.glove = {}

        for word in vocab:
            try:
                self.glove[word.lower()] = glove.vectors[glove.stoi[word.lower()]].data.cpu().numpy()
            except:
                print(word, "not found")

        # Add a handler for this.
        assert config in CONFIGS, "Did not provide a valid config."
        mazebase_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        mazebase_dir = os.path.join(mazebase_dir, "mazebase/mazebasev2")
        yaml_file =  os.path.join(mazebase_dir, "options/knowledge_planner/" + config + '.yaml')

        with open(yaml_file, 'r') as handle:
            options = yaml.load(handle)

        # Get sub opts
        method_opt = options['method']
        env_opt = options['env']
        log_opt = options['logs'] 

        # Set up the mazebase environment
        knowledge_root = env_opt['knowledge_root']
        world_knowledge_file = os.path.join(mazebase_dir, knowledge_root, env_opt['world_knowledge']['train'])
        with open(world_knowledge_file) as f:
          world_knowledge = json.load(f)

        # Make the world
        map_size = (env_opt['state_rep']['w'], env_opt['state_rep']['w'], env_opt['state_rep']['h'], env_opt['state_rep']['h'])
        self.all_games = [games.BasicKnowledgeGame(world_knowledge=world_knowledge, proposed_knowledge=[], options=env_opt, load_items=None, map_size=map_size)]

        self.reset()
        
        self.action_space = gym.spaces.Discrete(9) # Changed the action space to 9.
        obs_spaces = {} # change state space to dict to match everything else!
        for k, v in self.state.items():
            obs_spaces[k] = spaces.Box(low=0, high=1, shape=v.shape)
        self.observation_space = spaces.Dict(obs_spaces)

    def step(self, action):
        self.count = self.count + 1
        action = get_action_name(action)
        
        #execute action
        self.game.act(action)

        #print(self.game.game.inventory)

        ## more rewards -- if has pickaxe, and nearby the item to mine.

        hasPickaxe = False
        for item in self.game.game.inventory:
            if item == 'Pickaxe':
                hasPickaxe = True

        #calculate reward
        if self.game.is_over():
            self.reward = 5
            self.done = True
            self.count = 0
            is_success = True
        else:
            self.reward = 0
            self.done = False
            is_success = False

        #extra info
        self.add = {}
        #self.add['episode'] = self.game.game.inventory
        self.add['episode'] = {'l':self.count,'r':self.reward}
        self.add['is_success'] = is_success


        # get observation
        try:
            config = self.game.observe()
        except Exception:
            return self.state, self.reward, True, self.add
        grid_obs, side_info = config['observation']
        
        inventory = self.game.game.inventory
        goal = self.game.game.goal

        obs = (grid_obs, inventory, goal)

        state, inventory, goal = obs
        
        states_embedding = get_grid_embedding(state, self.glove, self.embed_size)
        states_onehot = one_hot_grid(state, self.glove, self.embed_size)
        goal = get_goal_embedding(goal, self.glove, self.embed_size)
        inventory = get_inventory_embedding(inventory, self.glove, self.embed_size)

        counts = np.array([self.game.game.count])
        # Note: removed counts from state

        self.state = {'grid_embedding' : states_embedding,
                      'grid_onehot': states_onehot,
                      'goal': goal,
                      'inventory': inventory}

        return self.state, self.reward, self.done, self.add

    def reset(self):

        self.count = 0

        # Game wrapper
        self.game = games.MazeGame(
          self.all_games,
          featurizer=featurizers.GridFeaturizer()
        )

        #get observation
        try:
            config = self.game.observe()
        except Exception:
            return self.reset()
        
        grid_obs, side_info = config['observation']
        
        inventory = self.game.game.inventory
        goal = self.game.game.goal

        print(goal)

        obs = (grid_obs, inventory, goal)

        state, inventory, goal = obs
        states_embedding = get_grid_embedding(state, self.glove, self.embed_size)
        states_onehot = one_hot_grid(state, self.glove, self.embed_size)
        goal = get_goal_embedding(goal, self.glove, self.embed_size)
        inventory = get_inventory_embedding(inventory, self.glove, self.embed_size)
        
        # counts = np.array([self.game.game.count])
        self.state = {'grid_embedding' : states_embedding,
                      'grid_onehot': states_onehot,
                      'goal': goal,
                      'inventory': inventory}

        return self.state
