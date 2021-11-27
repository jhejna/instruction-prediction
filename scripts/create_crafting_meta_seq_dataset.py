import ast
import json
import os
import numpy as np
import pickle
import random
import argparse

from spellchecker import SpellChecker

'''
This script was sent to me by the authors of https://arxiv.org/abs/2011.00517
Then, I modified it with a number of bug fixes.
'''

parser = argparse.ArgumentParser()
parser.add_argument("--input-path", "-p", type=str, required=True, help="Path to the .json dataset file.")
parser.add_argument("--output-path", "-o", type=str, required=True, help="Path to the data.")
args = parser.parse_args()

PATH_TO_FULL_DATA = args.input_path 

## Input: json file of game traces
## Output: train/test sets for board states, inventories, and actions

def read_dataset(save_path):

    spell = SpellChecker()

    with open(PATH_TO_FULL_DATA) as f:
        dataset = json.load(f)

    processed_dataset = {}
    print("**dataset loading**")

    # Used for saving the complete vocab.
    all_instructions = []

    #read in all traces
    num_completed = 0
    num_games_removed = 0
    
    for trace in dataset:

        # not sure why this one is problematic
        if trace == "AP9GRAU77J5G.619158":
            continue
        
        game = dataset[trace]
        game = str(game)
        game = ast.literal_eval(game)

        for indiv_game in game:
            
            ep_states = []
            ep_inventories = []
            ep_actions = []
            ep_instructions = []
            
            for i in range(len(indiv_game)): # loop through the game
                temp = indiv_game[i]
                ## need to do this because when the json was saved, it is the resulting state, so need the previous state
                if isinstance(temp, dict):
                    if i > 0:
                        ep_actions.append(temp['action'])
                    ep_states.append(temp['observation'][0])
                    ep_inventories.append(temp['inventory'])
                    ep_goal = temp['goal']

                if isinstance(temp, str):
                    # do spelling correction for each word:
                    instruction = [spell.correction(word) for word in temp.lower().split(' ')]
                    all_instructions.append(instruction)
                    ep_instructions.append(instruction)

            if len(ep_states) < 3:
                # Remove it
                num_games_removed += 1
                print("REMOVED GAME", trace)
                continue # Do not append it
            
            # Flatten the instructions
            ep_instructions = [item for sublist in ep_instructions for item in sublist]
            assert len(ep_states) - 1== len(ep_inventories) - 1 == len(ep_actions)

            ep = {
                "state": ep_states[:-1],
                "inventory": ep_inventories[:-1],
                "action": ep_actions,
                "instruction": ep_instructions
            }

            # Add the episode
            if not ep_goal in processed_dataset:
                processed_dataset[ep_goal] = []
            processed_dataset[ep_goal].append(ep)

            num_completed += 1
            if num_completed % 250 == 0:
                print("Completed", num_completed, "Removed", num_games_removed)

    if not save_path.endswith(".pkl"):
        save_path += ".pkl"
    print("**Saving Dataset**")
    with open(save_path, 'wb') as f:
        pickle.dump(processed_dataset, f)
    print("**Saving Vocabulary**")
    save_path = save_path[:-4]  # Remove the extension
    save_path += "_all_instr.pkl" # Add the vocab path back in.
    with open(save_path, 'wb') as f:
        pickle.dump(all_instructions, f)

read_dataset(args.output_path)
