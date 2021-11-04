import os
import argparse
import subprocess

from language_prediction.datasets.datasets import BabyAITrajectoryDataset

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, required=True, help="name of the env.")
parser.add_argument("--dataset-type", type=str, nargs='+', required=True, help="The type of the dataset")
parser.add_argument("--path", type=str, required=True, help="The path to save the data")
parser.add_argument("--episodes", type=int, default=1000, help="The number of episodes")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--max-mission-len", type=int, default=-1, help="Max length of mission in tokens")
parser.add_argument("--jobs", type=int, default=1)
parser.add_argument("--valid-episodes", type=int, default=500, help="Number of validation episodes")
parser.add_argument("--skip", default=1, type=int, required=False, help="Skip for contrastive.")

args = parser.parse_args()

entry_point = os.path.join(os.path.dirname(os.path.dirname(__file__)), "language_prediction/datasets/create_babyai_dataset.py")
demos_per_job = args.episodes // args.jobs
shard_paths = [args.path + '_shard' + str(i) for i in range(args.jobs)]
processes = []

for i in range(args.jobs):
    cmd_str = [
        'python', entry_point,
        '--env', args.env,
        '--dataset-type', *args.dataset_type,
        '--path', shard_paths[i],
        '--episodes', demos_per_job,
        '--seed', args.seed + i * demos_per_job,
        '--max-mission-len', args.max_mission_len,
        '--skip', args.skip
    ]
    cmd_str = list(map(str, cmd_str))
    proc = subprocess.Popen(cmd_str)
    processes.append(proc)

exit_codes = [p.wait() for p in processes]

# prune the shard paths
for dtype in args.dataset_type:
    # Test both file extension
    found_shard_paths = [path + "_" + dtype + ".pkl" for path in shard_paths]
    found_shard_paths = [path for path in found_shard_paths if os.path.exists(path)]
    datasets = [BabyAITrajectoryDataset.load(path) for path in found_shard_paths]
    if len(datasets) == 0:
        raise Exception("There were no found datasets.")
    elif len(datasets) == 1:
        merged_dataset = datasets[0]
    else:
        merged_dataset = BabyAITrajectoryDataset.merge(datasets)
    merged_dataset.save(args.path + "_" + dtype)

    # Clean up the shards
    print("Cleaning up the shards.")
    [os.remove(p) for p in found_shard_paths]

if args.valid_episodes > 0:
    print("Creating the validation dataset")
    cmd_str = [
            'python', entry_point,
            '--env', args.env,
            '--dataset-type', *args.dataset_type,
            '--path', args.path + "_valid",
            '--episodes', args.valid_episodes,
            '--seed', int(1e9),
            '--max-mission-len', args.max_mission_len,
            '--skip', args.skip
        ]
    cmd_str = list(map(str, cmd_str))
    proc = subprocess.Popen(cmd_str)
    proc.wait()

print("Done.")
