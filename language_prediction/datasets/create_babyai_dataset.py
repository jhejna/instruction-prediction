import argparse
from language_prediction.datasets.babyai_demos import create_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, required=True, help="name of the env.")
parser.add_argument("--dataset-type", type=str, nargs='+', required=True, help="The type of the dataset")
parser.add_argument("--path", type=str, required=True, help="The path to save the data")
parser.add_argument("--episodes", type=int, default=1000, help="The number of episodes")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--max-mission-len", type=int, default=-1, help="Max length of mission in tokens")
parser.add_argument("--skip", default=1, type=int, required=False, help="Skip for contrastive.")

args = parser.parse_args()

create_dataset(args.path, args.dataset_type, 
                env_name=args.env, 
                seed=args.seed,
                n_episodes=args.episodes,
                max_mission_len=args.max_mission_len,
                skip=args.skip)
