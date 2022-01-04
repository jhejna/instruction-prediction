import json
import os
import itertools
import copy
from language_prediction.utils.trainer import Config # This is the offending import that pulls in pytorch.
import tempfile
import argparse
import copy

import subprocess

STORAGE_ROOT = '..' # The parent directory
ENV_SETUP_SCRIPT = "setup_shell.sh"
TMP_DIR = os.path.join(STORAGE_ROOT, "tmp")
SLURM_LOG_DEFAULT = os.path.join(STORAGE_ROOT, "slurm_logs")

SLURM_ARGS = {
    "partition" : {"type": str, "required": True},
    "time" : {"type": str, "default": "48:00:00"},
    "nodes": {"type": int, "default": 1},
    "ntasks-per-node": {"type": int, "default": 1},
    "cpus": {"type": int, "required": True},
    "gpus": {"type": str, "required": True},
    "mem": {"type": str, "required": True},
    "output": {"type" : str, "default": SLURM_LOG_DEFAULT},
    "error": {"type" : str, "default": SLURM_LOG_DEFAULT},
    "job-name" : {"type": str, "required": True},
    "exclude" : {"type": str, "required": False, "default": None}
}

SLURM_NAME_OVERRIDES = {
    "gpus" : "gres",
    "cpus": "cpus-per-task"
}

def get_slurm_parser():
    parser = argparse.ArgumentParser()
    # SLURM Arguments
    for k, v in SLURM_ARGS.items():
        parser.add_argument("--" + k, **v)
    return parser

def write_slurm_header(f, args):
    # Make a copy of the args to prevent corruption
    args = copy.deepcopy(args)
    # Modify everything in the name space to later write it all at once
    for key in SLURM_ARGS.keys():
        assert key.replace('-', '_') in args, "Key " + key + " not found."
    
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    if not os.path.isdir(args.error):
        os.makedirs(args.error)
    
    args.output = os.path.join(args.output, args.job_name + "_%A.out")
    args.error = os.path.join(args.error, args.job_name + "_%A.err")
    args.gpus = "gpu:" + str(args.gpus)

    NL = '\n'
    f.write("#!/bin/bash" + NL)
    f.write(NL)
    for arg_name in SLURM_ARGS.keys():
        arg_value = vars(args)[arg_name.replace('-', '_')]
        if arg_name in SLURM_NAME_OVERRIDES:
            arg_name = SLURM_NAME_OVERRIDES[arg_name]
        if not arg_value is None:
            f.write("#SBATCH --" + arg_name + "=" + str(arg_value) + NL)
    f.write(NL)
    f.write('echo "SLURM_JOBID = "$SLURM_JOBID' + NL)
    f.write('echo "SLURM_JOB_NODELIST = "$SLURM_JOB_NODELIST' + NL)
    f.write('echo "SLURM_JOB_NODELIST = "$SLURM_JOB_NODELIST' + NL)
    f.write('echo "SLURM_NNODES = "$SLURM_NNODES' + NL)
    f.write('echo "SLURMTMPDIR = "$SLURMTMPDIR' + NL)
    f.write('echo "working directory = "$SLURM_SUBMIT_DIR' + NL)
    f.write(NL)
    f.write(". " + ENV_SETUP_SCRIPT)
    f.write(NL)

'''
Below is the experiment sweeper
'''

FOLDER_KEYS = []

class Experiment(dict):

    def __init__(self, base=None, name=None):
        super().__init__()
        self._name = name
        self.base_config = Config.load(base)

    @property
    def name(self):
        return self._name

    @classmethod
    def load(cls, path):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, 'r') as fp:
            data = json.load(fp)
        # Run formatting checks
        assert 'base' in data, "Did not supply a base config"
        base_config = data['base']
        del data['base'] # Remove the base configuration

        for k, v in data.items():
            assert isinstance(k, str)
            assert isinstance(v, list)
        experiment = cls(base=base_config, name=name)
        experiment.update(data)
        return experiment

    def get_variants(self):
        variants = itertools.product(*[val for val in self.values()])
        variants = [{key:variant[i] for i, key in enumerate(self.keys())} for variant in variants]
        return variants

    def generate_configs_and_names(self):
        variants = self.get_variants()
        configs_and_names = []
        for i, variant in enumerate(variants):
            config = self.base_config.copy()
            name = ""
            remove_trailing_underscore = False
            for k, v in variant.items():
                config_path = k.split('.')
                config_dict = config
                while len(config_path) > 1:
                    if not config_path[0] in config_dict:
                        raise ValueError("Experiment specified key not in config: " + str(k))
                    config_dict = config_dict[config_path[0]]
                    config_path.pop(0)
                if not config_path[0] in config_dict:
                        raise ValueError("Experiment specified key not in config: " + str(k))
                config_dict[config_path[0]] = v
                
                if k in FOLDER_KEYS:
                    name = os.path.join(v, name)
                elif len(self[k]) > 1:
                    # Add it to the path name if it is different for each run.
                    if isinstance(v, str):
                        str_val = v
                    elif isinstance(v, int) or isinstance(v, float) or isinstance(v, bool) or v is None:
                        str_val = str(v)
                    elif isinstance(v, list):
                        str_val = v[-1]
                        assert isinstance(str_val, str)
                    else:
                        raise ValueError("Could not convert config value to str.")

                    name += str(config_path[0]) + '-' + str_val + '_'
                    remove_trailing_underscore = True

            if remove_trailing_underscore:
                name = name[:-1]
            name = os.path.join(self.name, name)    
            if not os.path.exists(TMP_DIR):
                os.mkdir(TMP_DIR)
            _, config_path = tempfile.mkstemp(text=True, prefix='config', suffix='.json', dir=TMP_DIR)
            print("Variant", i+1)
            print(config)
            config.save(config_path)
            configs_and_names.append((config_path, name))
        
        return configs_and_names

