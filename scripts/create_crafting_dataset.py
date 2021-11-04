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

PATH_TO_FULL_DATA = args.path 

SUBGOAL_DIVIDE_TOKEN = "<subgoal>"

## Input: json file of game traces
## Output: train/test sets for board states, inventories, and actions

def read_dataset(save_path, all_future_instrs=False):

    spell = SpellChecker()

    with open(PATH_TO_FULL_DATA) as f:
        dataset = json.load(f)

    print("**dataset loading**")

    states = []
    inventories = []
    actions = []
    goals = []
    instructions = []

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
            ep_goals = []
            ep_actions = []
            ep_instructions = []

            for i in range(0, len(temp_compile), 3):
                ep_states.append(temp_compile[i][0])
                ep_inventories.append(temp_compile[i][1])
                ep_goals.append(temp_compile[i][2])
                ep_actions.append(temp_compile[i+1])
                ep_instructions.append(temp_compile[i+2])

            if any([instr is None for instr in ep_instructions]):
                num_games_removed += 1
                print("REMOVED GAME", trace)
                continue # Do not append it

            if all_future_instrs:
                new_ep_instrs = []
                # We need to modify instructions to be future
                for i in range(len(ep_instructions)):
                    seen_instr = set()
                    new_instr = []

                    for j in range(i, len(ep_instructions)):
                        # Now try to add all the other instrs
                        cur_instr = tuple(ep_instructions[j])
                        if cur_instr in seen_instr:
                            pass
                        else:
                            seen_instr.add(cur_instr)
                            new_instr.extend(ep_instructions[j].copy())
                            new_instr.append(SUBGOAL_DIVIDE_TOKEN)
                    new_ep_instrs.append(new_instr[:-1])
                    
                ep_instructions = new_ep_instrs
            
            states.append(ep_states)
            inventories.append(ep_inventories)
            goals.append(ep_goals)
            actions.append(ep_actions)
            instructions.append(ep_instructions)

            num_completed += 1
            if num_completed % 250 == 0:
                print("Completed", num_completed, "Removed", num_games_removed)

    print("Dataset Size", len(states))

    # Shuffle at the demonstration level.
    temp = list(zip(states, inventories, goals, actions, instructions))
    random.Random(1).shuffle(temp)
    states, inventories, goals, actions, instructions = zip(*temp)

    states = [item for sublist in states for item in sublist]
    inventories = [item for sublist in inventories for item in sublist]
    goals = [item for sublist in goals for item in sublist]
    actions = [item for sublist in actions for item in sublist]
    instructions = [item for sublist in instructions for item in sublist]

    print(len(states), len(inventories), len(actions), len(goals), len(instructions))
    assert len(states) == len(inventories) == len(actions) == len(goals) == len(instructions)

    states = np.array(states)
    inventories = np.array(inventories)
    actions = np.array(actions)
    goals = np.array(goals)
    instructions = np.array(instructions)

    print("**dataset loaded**")

    # TODO: Split this into train and valid.

    # Now we have finished creating the dataset. Clean it up and save it in the same format as the original authors do.
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    train_save_path = os.path.join(save_path, "dataset_")
    if train_save_path.endswith('/'):
        train_save_path = train_save_path[:-1]
    
    with open(train_save_path + 'states', 'wb') as f:
        pickle.dump(states, f)

    with open(train_save_path + 'inventories', 'wb') as f:
        pickle.dump(inventories, f)

    with open(train_save_path + 'actions', 'wb') as f:
        pickle.dump(actions, f)

    with open(train_save_path + 'goals', 'wb') as f:
        pickle.dump(goals, f)

    with open(train_save_path + 'instructions', 'wb') as f:
        pickle.dump(instructions, f)

    with open(train_save_path + 'all_instructions', 'wb') as f:
        pickle.dump(all_instructions, f)

    # Now save the validation dataset of 15%. Note that this will overlap with the train set, but for the train dataset we usually only take the first %.
    num_valid_pts = int(0.15*len(states))
    if save_path.endswith('/'):
        save_path = save_path[:-1]
    valid_save_path = save_path + "_valid"
    if not os.path.exists(valid_save_path):
        os.makedirs(valid_save_path)
    valid_save_path = os.path.join(save_path, "dataset_")
    if valid_save_path.endswith('/'):
        valid_save_path = valid_save_path[:-1]
    
    with open(valid_save_path + 'states', 'wb') as f:
        pickle.dump(states[-num_valid_pts:], f)

    with open(valid_save_path + 'inventories', 'wb') as f:
        pickle.dump(inventories[-num_valid_pts:], f)

    with open(valid_save_path + 'actions', 'wb') as f:
        pickle.dump(actions[-num_valid_pts:], f)

    with open(valid_save_path + 'goals', 'wb') as f:
        pickle.dump(goals[-num_valid_pts:], f)

    with open(valid_save_path + 'instructions', 'wb') as f:
        pickle.dump(instructions[-num_valid_pts:], f)

    with open(valid_save_path + 'all_instructions', 'wb') as f:
        pickle.dump(all_instructions, f)

read_dataset(args.output_path, all_future_instrs=False)
