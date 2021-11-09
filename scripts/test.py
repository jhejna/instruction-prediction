
import argparse
import os
import torch
from language_prediction.utils.trainer import Config, load
from language_prediction.utils.evaluate import eval_policy

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-c", type=str, default=None)
    parser.add_argument("--device", "-d", type=str, default="auto")
    parser.add_argument("--best", action='store_true', default=False)
    parser.add_argument("--num-ep", type=int, default=20)
    parser.add_argument("--override", action='append', default=[], nargs=2, metavar=('key', 'value'), help="Override configuration")
    parser.add_argument("--eval-mode", action='store_true', default=False)
    args = parser.parse_args()

    paths = []
    for root, dirs, files in os.walk(args.path, topdown=False):
        for name in files:
            if name == "config.yaml":
                paths.append(root)
    paths = sorted(paths)

    rewards, lengths, stds, successes = [], [], [], []
    for path in paths:

        config = Config.load(os.path.join(path, "config.yaml"))
        for key, value in args.override:
            # Progress down the config path (seperated by '.') until we reach the final value to override.
            config_path = key.split('.')
            config_dict = config
            while len(config_path) > 1:
                config_dict = config_dict[config_path[0]]
                config_path.pop(0)
            config_dict[config_path[0]] = value

        print(config) # Print the modified config.
        config.parse() # Parse the config

        model_path = os.path.join(path, "best_model.pt")
        if not args.best and os.path.exists(os.path.join(path, "final_model.pt")):
            model_path = os.path.join(path, "final_model.pt")
        
        # Set all of the seeds. This is important so evals of different policies are consistent.
        import random
        seed = int(1e9)+10000
        random.seed(seed) # Make sure the seed is higher than those used in evaluation ep
        import numpy as np
        np.random.seed(seed)

        model, env = load(config, model_path, device=args.device, strict=False) # Do an un-strict load.

        if args.eval_mode:
            model.network.eval()

        with torch.no_grad(): # run the eval without gradients.
            metrics = eval_policy(env, model, args.num_ep, seed=seed)
        rewards.append(metrics['reward'])
        stds.append(metrics['stddev'])
        lengths.append(metrics['length'])
        successes.append(metrics['success_rate'])
        
    print("Ran", args.num_ep, "episodes")

    for path, reward, std, length, success in zip(paths, rewards, stds, lengths, successes):
        print("Path:", path)
        print("Success rate:", success)

