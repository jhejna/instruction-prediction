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

SUBGOAL_DIVIDE_TOKEN = "<subgoal>"

## Input: json file of game traces
## Output: train/test sets for board states, inventories, and actions

def read_dataset(save_path, all_future_instrs=False):

    spell = SpellChecker()

    with open(PATH_TO_FULL_DATA) as f:
        dataset = json.load(f)

    print("**dataset loading**")

    processed_dataset = {}

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

            temp_compile = []
            current_instruction = None
            
            for i in range(len(indiv_game)):
                temp = indiv_game[i]
                ## need to do this because when the json was saved, it is the resulting state, so need the previous state
                if isinstance(temp, dict):
                    if i > 0:
                        temp_compile.append(temp['action'])
                        temp_compile.append(current_instruction)
                    grid = temp['observation'][0]
                    temp_compile.append([grid,temp['inventory'],temp['goal']])
                if isinstance(temp, str):

                    # do spelling correction for each word:
                    updated_temp = [spell.correction(word) for word in temp.lower().split(' ')]
                    
                    all_instructions.append(updated_temp)

                    current_instruction = updated_temp

            temp_compile.append("stop")
            temp_compile.append(current_instruction)

            ep_states = []
            ep_inventories = []
            ep_actions = []
            ep_instructions = []
            ep_goal = None

            for i in range(0, len(temp_compile), 3):
                ep_states.append(temp_compile[i][0])
                ep_inventories.append(temp_compile[i][1])
                ep_goal = temp_compile[i][2]
                ep_actions.append(temp_compile[i+1])
                ep_instructions.append(temp_compile[i+2])

            if any([instr is None for instr in ep_instructions]) or ep_goal is None:
                num_games_removed += 1
                print("REMOVED GAME", trace)
                continue # Do not append it

            # Now add to the dataset
            if not ep_goal in processed_dataset:
                processed_dataset[ep_goal] = {
                    "state": [],
                    "inventory": [],
                    "action": [],
                    "instruction": []
                }
            
            processed_dataset[ep_goal]["state"].extend(ep_states)
            processed_dataset[ep_goal]["inventory"].extend(ep_inventories)
            processed_dataset[ep_goal]["action"].extend(ep_actions)
            processed_dataset[ep_goal]["instruction"].extend(ep_instructions)

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