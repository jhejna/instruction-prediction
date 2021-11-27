import os
import yaml
import pprint
import importlib
import subprocess
import copy

import babyai # Add the envs to registry
import gym
import language_prediction

class Config(object):

    def __init__(self):
        # Define the necesary structure for a complete training configuration
        self.parsed = False
        self.config = dict()

        # Algorithm Args
        self.config['alg'] = None
        self.config['alg_kwargs'] = {}
        self.config['train_kwargs'] = {}
        self.config['seed'] = None # Does nothing right now.

        # network Args
        self.config['network'] = None
        self.config['network_kwargs'] = {}

        # Environment Args
        self.config['env'] = None
        self.config['env_kwargs'] = {}
        self.config['wrapper'] = None
        self.config['wrapper_kwargs'] = {}
        self.config['time_limit'] = None

    def parse(self):
        self.parsed = True
        self.parse_helper(self.config)

    def parse_helper(self, d):
        for k, v in d.items():
            if isinstance(v, list) and len(v) > 1 and v[0] == "import":
                # parse the value to an import
                d[k] = getattr(importlib.import_module(v[1]), v[2])
            elif isinstance(v, dict):
                self.parse_helper(v)
        
    def update(self, d):
        self.config.update(d)
    
    def save(self, path):
        if self.parsed:
            print("[CONFIG ERROR] Attempting to saved parsed config. Must save before parsing to classes. ")
            return
        if os.path.isdir(path):
            path = os.path.join(path, "config.yaml")
        with open(path, 'w') as f:
            yaml.dump(self.config, f)

    @classmethod
    def load(cls, path):
        if os.path.isdir(path):
            path = os.path.join(path, "config.yaml")
        with open(path, 'r') as f:
            data = yaml.load(f, Loader=yaml.Loader)
        config = cls()
        config.update(data)
        return config

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value

    def __contains__(self, key):
        return self.config.__contains__(key)

    def __str__(self):
        return pprint.pformat(self.config, indent=4)

    def copy(self):
        assert not self.parsed, "Cannot copy a parsed config"
        config = type(self)()
        config.config = copy.deepcopy(self.config)
        return config

def get_env(config):
    assert isinstance(config, Config)
    try:
        env =  vars(language_prediction.envs)[config['env']](**config['env_kwargs'])
    except:
        env = gym.make(config['env'], **config['env_kwargs'])
    if config['wrapper']:
        env = vars(language_prediction.envs)[config['wrapper']](env, **config['wrapper_kwargs'])
    # Note: env is currently not seeded.
    if not config['time_limit'] is None:
        env = gym.wrappers.TimeLimit(env, config['time_limit'])
    return env

def get_eval_env(config):
    assert isinstance(config, Config)
    try:
        env =  vars(language_prediction.envs)[config['eval_env']](**config['eval_env_kwargs'])
    except:
        env = gym.make(config['eval_env'], **config['eval_env_kwargs'])
    if config['wrapper']:
        env = vars(language_prediction.envs)[config['wrapper']](env, **config['wrapper_kwargs'])
    # Note: env is currently not seeded.
    if not config['time_limit'] is None:
        env = gym.wrappers.TimeLimit(env, config['time_limit'])
    return env

def get_alg_class(config):
    assert isinstance(config, Config)
    alg_name = config['alg']
    alg = vars(language_prediction.algs)[alg_name]
    return alg

def get_network_class(config):
    assert isinstance(config, Config)
    return vars(language_prediction.networks)[config['network']]

def train(config, path, device="auto"):
    # Create the save path and save the config
    print("[language_prediction] Training agent with config:")
    print(config)
    os.makedirs(path, exist_ok=False)
    config.save(path)

    process = subprocess.Popen(['git', 'rev-parse', 'HEAD'], shell=False, stdout=subprocess.PIPE)
    git_head_hash = process.communicate()[0].strip()
    with open(os.path.join(path, 'git_hash.txt'), 'wb') as f:
        f.write(git_head_hash)

    config.parse() # Parse the config

    env = get_env(config) if config['env'] else None
    eval_env = get_eval_env(config) if 'eval_env' in config and config['eval_env'] else None
    alg_class = get_alg_class(config)
    network_class = get_network_class(config)

    algo = alg_class(env, network_class, network_kwargs=config['network_kwargs'], device=device, eval_env=eval_env, **config['alg_kwargs'],)
    algo.train(path, **config['train_kwargs'])

    return algo

def load(config, model_path, device="auto", strict=True):
    env = get_env(config)
    alg_class = get_alg_class(config)
    network_class = get_network_class(config)
    algo = alg_class(env, network_class, network_kwargs=config['network_kwargs'], device=device, **config['alg_kwargs'],)
    algo.load(model_path, strict=strict)
    return algo, env
