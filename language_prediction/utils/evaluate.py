import torch
import numpy as np

def update_history(history, new_obs):
    if history is None:
        if isinstance(new_obs, dict):
            history = {k: np.expand_dims(v, axis=0) if isinstance(v, np.ndarray) else [v] for k,v in new_obs.items()}
        else:
            history = np.expand_dims(new_obs, axis=0)
        return history
    
    if isinstance(new_obs, dict):
        for k in history.keys():
            if isinstance(history[k], np.ndarray):
                history[k] = np.concatenate((history[k], np.expand_dims(new_obs[k], axis=0)), axis=0)
            else:
                history[k].append(new_obs[k])
    else:
        history = np.concatenate((history, np.expand_dims(new_obs, axis=0)), axis=0)
    
    return history

def eval_policy(env, model, num_ep, seed=None, mask=None):
    ep_rewards = []
    ep_lengths = []
    num_successes = 0
    for i in range(num_ep):
        history = None
        done = False
        ep_reward = 0
        ep_length = 0
        if not seed is None:
            env.seed(seed + i)
        obs = env.reset()
        history = update_history(history, obs)
        while not done:
            with torch.no_grad():
                action = model.predict(obs, deterministic=True, history=history)
            obs, reward, done, info = env.step(action)
            
            history = update_history(history, obs)
            ep_reward += reward
            ep_length += 1
            if 'is_success' in info and info['is_success']:
                num_successes += 1
       
        ep_rewards.append(ep_reward)
        ep_lengths.append(ep_length)

    return np.mean(ep_rewards), np.std(ep_rewards), np.mean(ep_lengths), num_successes / num_ep