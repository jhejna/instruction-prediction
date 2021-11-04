import babyai
import gym
from babyai.bot import GoNextToSubgoal, PickupSubgoal, DropSubgoal, OpenSubgoal, LanguageObj
from babyai.utils.agent import FullyObsBotAgent, BotAgent

import numpy as np
import time
import logging
from collections import defaultdict
import itertools

from language_prediction.envs.babyai_wrappers import WORD_TO_IDX, LanguageWrapper
from .datasets import BabyAITrajectoryDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO', format="%(asctime)s: %(levelname)s: %(message)s")

def merge_ids(*args):
    final_ids = []
    for arg in args:
        if isinstance(arg, int):
            final_ids.append(arg)
        elif isinstance(args, tuple):
            final_ids.extend(arg)
    return tuple(final_ids)

def process_subgoals(subgoals, aggressive_mask=False):
    seen_ids = set()
    used_subgoals = []
    time_to_id = []
    for subgoal in subgoals:
        time_to_id.append(subgoal.id)

    for subgoal in reversed(subgoals):
        if not subgoal.id in seen_ids:
            used_subgoals.append(subgoal)
            seen_ids.add(subgoal.id)
    subgoals = list(reversed(used_subgoals))
    combine_inds = []
    combine_instrs = []
    invalid_inds = set()

    # Remove doubles
    for i in range(len(subgoals) - 1):
        if subgoals[i].subgoal_type is GoNextToSubgoal and \
            not subgoals[i].obj_type is None and \
            subgoals[i+1].subgoal_type is PickupSubgoal and \
            not i in invalid_inds and \
            not i+1 in invalid_inds and \
            subgoals[i].pos == subgoals[i+1].pos:
            # print("COMBINING DOUBLE", subgoals[i].text, subgoals[i].id, subgoals[i+1].text, subgoals[i+1].id)

            # Removing duplicate
            combine_inds.append([i, i+1])
            text = "pick up the " + subgoals[i+1].obj_color + " " + subgoals[i+1].obj_type
            combine_instrs.append(LanguageObj(text=text, obj_type=None, 
                                         obj_color=subgoals[i+1].obj_color, subgoal_type=PickupSubgoal, 
                                         id=merge_ids(subgoals[i].id, subgoals[i+1].id), pos=subgoals[i+1].pos))
            invalid_inds.add(i)
            invalid_inds.add(i+1)

    # remove triples
    for i in range(len(subgoals) - 2):
        if subgoals[i].subgoal_type is PickupSubgoal and \
            subgoals[i+1].subgoal_type is GoNextToSubgoal and \
            subgoals[i+2].subgoal_type is DropSubgoal and \
            not i in invalid_inds and \
            not i+1 in invalid_inds and \
            not i+2 in invalid_inds and \
            subgoals[i].obj_color == subgoals[i+2].obj_color and \
            subgoals[i].obj_type == subgoals[i+2].obj_type:
            # Moving an object out of the way.
            # Compute the distance traveled:
            # print("COMBINING TRIPLE", subgoals[i].text, subgoals[i].id, subgoals[i+1].text, subgoals[i+1].id, subgoals[i+2].text, subgoals[i+2].id)

            dist_traveled = 0
            dist_traveled += abs(subgoals[i].pos[0] - subgoals[i+1].pos[0]) + abs(subgoals[i].pos[1] - subgoals[i+1].pos[1])
            dist_traveled += abs(subgoals[i+1].pos[0] - subgoals[i+2].pos[0]) + abs(subgoals[i+1].pos[1] - subgoals[i+2].pos[1])
            if dist_traveled <= 5:
                invalid_inds.add(i)
                invalid_inds.add(i+1)
                invalid_inds.add(i+2)
                combine_inds.append([i, i+1, i+2])
                text = "move the " + subgoals[i].obj_color + " " + subgoals[i].obj_type
                combine_instrs.append(
                    LanguageObj(text=text, obj_type=None, 
                                         obj_color=None, subgoal_type=None, 
                                         id=merge_ids(subgoals[i].id, subgoals[i+1].id, subgoals[i+2].id), pos=None)
                )

    for inds, instr in reversed(sorted(zip(combine_inds, combine_instrs), key=lambda x: x[0][0])):
        for ind in reversed(sorted(inds)):
            del subgoals[ind]
        subgoals.insert(ind, instr)

    # Now clear the subgoals that are blank
    for i in reversed(range(len(subgoals))):
        if subgoals[i].subgoal_type is GoNextToSubgoal and subgoals[i].obj_type is None:
            del subgoals[i]

    # Combine any remaining pickup-drop instruction combinations
    for i in reversed(range(len(subgoals) - 1)):
        
        if subgoals[i].subgoal_type is PickupSubgoal and \
            subgoals[i+1].subgoal_type is DropSubgoal:
            dist_traveled = abs(subgoals[i].pos[0] - subgoals[i+1].pos[0]) + abs(subgoals[i].pos[1] - subgoals[i+1].pos[1])
            if dist_traveled <= 5:
                text, obj_type, obj_color, subgoal_type = subgoals[i].text, subgoals[i].obj_type, subgoals[i].obj_color, subgoals[i].subgoal_type
                subgoals[i] = LanguageObj(text=text, obj_type=obj_type, obj_color=obj_color, subgoal_type=subgoal_type,
                                             id=merge_ids(subgoals[i].id, subgoals[i+1].id), pos=subgoals[i].pos)
                del subgoals[i+1] # delete the drop instr, will implicitly work if we pick up another obj
    
    # Combine any remaining gotodoor and open door instructions
    inds_to_del = []
    for i in reversed(range(len(subgoals) - 1)):
        if subgoals[i].subgoal_type is GoNextToSubgoal and \
                subgoals[i+1].subgoal_type is OpenSubgoal and \
                subgoals[i].pos == subgoals[i+1].pos: 
            # We can remove the the GoNextToSubgoal
            inds_to_del.append(i)
    for ind in inds_to_del:
        del subgoals[ind] # Should delete in reverse order

    # figure out the best way to sort the subgoals
    cur_subgoal_index = 0
    seen = defaultdict(lambda: False)
    new_subgoal_index = []
    for t, subgoal_id in enumerate(time_to_id):
        new_subgoal_index.append(cur_subgoal_index)
        seen[subgoal_id] = True
        cur_subgoal = subgoals[cur_subgoal_index]
        if isinstance(cur_subgoal.id, int) and seen[cur_subgoal.id]:
            cur_subgoal_index += 1
        elif isinstance(cur_subgoal.id, tuple) and all(seen[_id] for _id in cur_subgoal.id):
            cur_subgoal_index += 1
        cur_subgoal_index = min(cur_subgoal_index, len(subgoals) - 1)
    
    # Grab and the convert the text
    print(len(new_subgoal_index) / len(subgoals))
    subgoals_per_timestep = []
    for ind in new_subgoal_index:
        text = [subgoal.text for subgoal in subgoals[ind:]]
        s_t = FullyObsLanguageWrapper.convert_subgoals_to_data(text, max_len=-1, pad=False)
        subgoals_per_timestep.append(s_t)

    # Get the total length of the text for all the subgoals. This is in the first subgoal for timestep
    # Need to construct a tensor of shape (T, S) where T is target seq and S is src seq.
    # Basically, we need need to block cross-attention from everywhere where the instruction is being executed.
    mask = np.ones((len(subgoals_per_timestep[0]), len(new_subgoal_index)), dtype=np.bool)
    # For each target, can attend to all timesteps where it hasn't been completed.
    for i, ind in enumerate(new_subgoal_index):
        before_size = sum([len(sg) for sg in subgoals[:ind + (1 if aggressive_mask else 0)]])
        mask[:before_size, i] = False
    if aggressive_mask:
        mask[:, 0] = True

    return subgoals_per_timestep, mask

def generate_demos(env_name, seed, n_episodes, max_mission_len=None, log_interval=500, **kwargs):

    env = gym.make(env_name)
    agent = BotAgent(env)
    env = LanguageWrapper(env, max_len=max_mission_len, pad=False)

    demos = []

    checkpoint_time = time.time()
    just_crashed = False

    while True:
        if len(demos) == n_episodes:
            break
        done = False
        if just_crashed:
            logger.info("reset the environment to find a mission that the bot can solve")
            env.reset()
        else:
            env.seed(seed + len(demos)) # deterministically seed the demos
        
        obs = env.reset()
        agent.on_reset()

        # Construct the parts that will contain the data
        subgoals = []
        actions = []
        images = []
        mission = obs['mission']
        inventorys = []

        try:
            while not done:
                agent_action = agent.act(obs)
                action = agent_action['action']
                subgoals.append(agent_action['subgoal'].as_language())
                # env.render()
                new_obs, reward, done, _ = env.step(action)
                agent.analyze_feedback(reward, done)

                actions.append(action)
                images.append(obs['image'])
                if 'inventory' in obs:
                    inventorys.append(obs['inventory'])
                obs = new_obs

            # Append the final obs
            images.append(obs['image'])
            if 'inventory' in obs:
                inventorys.append(obs['inventory'])

            if reward > 0:
                images = np.array(images, dtype=np.uint8)
                if len(inventorys) > 0:
                    inventorys = np.array(inventorys, dtype=np.uint8)
                else:
                    inventorys = None
                actions = np.array(actions, dtype=np.int32)
                assert len(subgoals) == len(actions)
                subgoals, mask = process_subgoals(subgoals)
                    
                demos.append((mission, images, inventorys, actions, subgoals, mask))
                just_crashed = False

            else:
                just_crashed = True
                logger.info("mission failed")

        except (Exception, AssertionError) as e:
            print("EXCEPTION!", e)
            just_crashed = True
            logger.exception("error while generating demo #{}".format(len(demos)))
            continue

        if len(demos) and len(demos) % log_interval == 0:
            now = time.time()
            demos_per_second = log_interval / (now - checkpoint_time)
            to_go = (n_episodes - len(demos)) / demos_per_second
            logger.info("demo #{}, {:.3f} demos per second, {:.3f} seconds to go".format(
                len(demos) - 1, demos_per_second, to_go))
            checkpoint_time = now

    return demos

def create_traj_dataset(demos, seed, max_mission_len=None, **kwargs):
    missions, imgs, ac, subgoals, masks = [], [], [], [], []
    for demo in demos:
        # The image sequence is shape (L, C, H, W)
        imgs.append(demo[1][:-1]) # All but the last image
        ac.append(demo[3])

        # Pad the mission out to the entire length.
        mission_pad = np.zeros(max_mission_len, dtype=np.int32)
        mission_pad[:len(demo[0])] = demo[0]
        missions.append(mission_pad)

        # We need to add a start generation token to the subgoals
        subgoal = np.concatenate((np.array([WORD_TO_IDX['END_MISSION']], dtype=np.int32), demo[4][0]), axis=0)
        mask = np.concatenate((demo[5][0:1], demo[5]), axis=0) # Re-concatenate the first row mask for the subgoal.
        # Construct the subgoals, this can be done without padding.
        subgoals.append(subgoal)
        masks.append(mask)
    
    return BabyAITrajectoryDataset(imgs, missions, subgoals, ac, masks=masks)

def create_traj_contrastive_dataset(demos, seed, max_mission_len=None, skip=1):
    missions, imgs, ac, subgoals, masks, next_imgs = [], [], [], [], [], []
    for demo in demos:
        if len(demo[1]) < skip+1:
            continue

        # The image sequence is shape (L, C, H, W)
        imgs.append(demo[1][:-1]) # all but the last image
        ac.append(demo[3])

        next_image = demo[1][skip:]
        if next_image.shape[0] < imgs[-1].shape[0]: # imgs[-1] is what we just appended
            # pad out with the last frame
            pad = np.expand_dims(next_image[-1], 0).repeat(imgs[-1].shape[0] - next_image.shape[0], axis=0)
            next_image = np.concatenate((next_image, pad), axis=0)
        assert next_image.shape[0] == imgs[-1].shape[0]
        next_imgs.append(next_image) # Done with dataset creation. Now we need to add the skip factor.

        # Pad the mission out to the entire length.
        mission_pad = np.zeros(max_mission_len, dtype=np.int32)
        mission_pad[:len(demo[0])] = demo[0]
        missions.append(mission_pad)

        # We need to add a start generation token to the subgoals
        subgoal = np.concatenate((np.array([WORD_TO_IDX['END_MISSION']], dtype=np.int32), demo[4][0]), axis=0)
        mask = np.concatenate((demo[5][0:1], demo[5]), axis=0) # Re-concatenate the first row mask for the subgoal.
        # Construct the subgoals, this can be done without padding.
        subgoals.append(subgoal)
        masks.append(mask)
    
    return BabyAITrajectoryDataset(imgs, missions, subgoals, ac, masks=masks, next_images=next_imgs)

def create_dataset(path, dataset_type, env_name, seed, n_episodes, **kwargs):
    demos = generate_demos(env_name, seed, n_episodes, **kwargs)
    if not isinstance(dataset_type, list):
        dataset_type = [dataset_type]
    for dtype in dataset_type:
        dataset_fn = {
            "traj" : create_traj_dataset,
            "traj_contrastive": create_traj_contrastive_dataset
        }[dtype]

        dataset = dataset_fn(demos, seed, **kwargs)
        dataset.save(path + "_" + dtype)
