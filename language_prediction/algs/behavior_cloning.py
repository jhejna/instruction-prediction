import os
import time
import torch
import numpy as np

from language_prediction.utils.logger import Logger
from language_prediction.utils.evaluate import eval_policy
from language_prediction.utils.utils import to_tensor, to_device, unsqueeze

import language_prediction
from functools import partial
from collections import defaultdict

def compute_curl_loss(anchor, target, proj_mat, actions):
    if len(anchor.shape) == 3:
        # Flatten everything and do it all as one big batch in the sequence case.
        B, S, D = anchor.shape
        anchor = anchor.reshape(B*S, D)
        target = target.reshape(B*S, D)
        actions = actions.reshape(B*S,)
    
    Wz = torch.matmul(proj_mat, target.T)
    logits = torch.matmul(anchor, Wz)
    logits = logits - torch.max(logits, 1)[0][:, None]
    with torch.no_grad():
        labels = torch.arange(logits.shape[0], dtype=torch.long, device=logits.device)
        labels[actions == -100] = -100
    return torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100)

def ema_parameters(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)

def log_from_dict(logger, loss_lists, prefix):
    keys_to_remove = []
    for loss_name, loss_value in loss_lists.items():
        if isinstance(loss_value, list) and len(loss_value) > 0:
            logger.record(prefix + "/" + loss_name, np.mean(loss_value))
            keys_to_remove.append(loss_name)
        else:
            logger.record(prefix + "/" + loss_name, loss_value)
            keys_to_remove.append(loss_name)
    for key in keys_to_remove:
        del loss_lists[key]

class BehaviorCloning(object):

    def __init__(self, env, network_class, network_kwargs={}, device="cpu",
                 optim_cls=torch.optim.Adam, eval_env=None,
                 dataset=None,
                 validation_dataset=None,
                 optim_kwargs={
                     'lr': 0.0001
                 },
                 batch_size=64,
                 grad_norm=None,
                 lang_coeff=1.0,
                 unsup_coeff=1.0,
                 action_coeff=1.0,
                 unsup_ema_tau=0.005,
                 unsup_ema_update_freq=2,
                 checkpoint=None,
                 dataset_fraction=1,
                 ):
        
        self.env = env
        self.eval_env = eval_env
        num_actions = env.action_space.n # Only support discrete action spaces.
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.dataset = dataset
        self.validation_dataset = validation_dataset
        self.grad_norm = grad_norm
        self.action_coeff = action_coeff
        self.lang_coeff = lang_coeff
        self.unsup_coeff = unsup_coeff
        self.unsup_ema_update_freq = unsup_ema_update_freq
        self.unsup_ema_tau = unsup_ema_tau
        self.dataset_fraction = dataset_fraction

        network_args = [num_actions]
        if isinstance(self.env.unwrapped, language_prediction.envs.mazebase.MazebaseGame):
            # Now we have to create the vocab here.
            import pickle
            with open(self.dataset +'all_instructions', 'rb') as f:
                all_instructions = pickle.load(f)
            from language_prediction.datasets import crafting_dataset
            self.vocab, vocab_weights = crafting_dataset.build_vocabulary(all_instructions)
            network_args.append(self.vocab)
            network_args.append(vocab_weights)

        # Create the network and optimizer.
        self.network = network_class(*network_args, **network_kwargs).to(self.device)
        self.optim = optim_cls(self.network.parameters(), **optim_kwargs)

        if checkpoint:
            self.load(checkpoint, strict=True)
        
        self.action_criterion = torch.nn.CrossEntropyLoss(ignore_index=network.action_pad_idx)
        self.instruction_criterion = torch.nn.CrossEntropyLoss(ignore_index=network.lang_pad_idx)

        if self.unsup_coeff > 0.0: 
            # create the target networks because we are in the unsupervised case.
            self.ema_network = network_class(*network_args, **network_kwargs).to(self.device)
            for param in self.ema_network.parameters():
                param.requires_grad = False
        
    def predict(self, obs, deterministic=True, history=None):
        obs = unsqueeze(to_device(to_tensor(obs), self.device), 0)
        if history is not None:
            history = unsqueeze(to_device(to_tensor(history), self.device), 0)

        if hasattr(self.network, "predict"):
            # Support custom predict methods if a network has it
            return self.network.predict(obs, deterministic=deterministic, history=history)
        else:
            logits, _, _ = self.network(obs) # Remove any aux loss
            logits = logits[0] # remove the batch dim
            if deterministic:
                return torch.argmax(logits).item()
            else:
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.categorical.Categorical(probs)
                return dist.sample().item()
    
    def save(self, path, extension):
        save_dict = {"network" : self.network.state_dict(), "optim": self.optim.state_dict()}
        torch.save(save_dict, os.path.join(path, extension + ".pt"))

    def load(self, checkpoint, strict=True):
        print("[Language Prediction' Loading Checkpoint:", checkpoint)
        checkpoint = torch.load(checkpoint, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'], strict=strict)
        if strict:
            # Only load the optimizer state dict if we are being strict.
            self.optim.load_state_dict(checkpoint['optim'])

    def _compute_loss(self, obs, actions):
        metrics = {}
        obs = to_device(obs, self.device)
        actions = to_device(actions, self.device).long()

        # Remove language inputs for a speedup if we don't need it.
        instruction_inputs = obs['label'][:, :-1] if 'label' in obs and self.lang_coeff > 0 else None
        instruction_labels = obs['label'][:, 1:] if 'label' in obs and self.lang_coeff > 0 else None

        action_logits, lang_logits, anchor_logits = self.network(obs, instructions=instruction_inputs, is_target=False)
        
        if self.unsup_coeff > 0: # Note that we run unsupervised losses first to avoid action reshaping issues.
            with torch.no_grad():
                _, _, target_logits = self.ema_network(obs, labels=None, is_target=True)
            unsup_loss = compute_curl_loss(anchor_logits, target_logits, self.network.unsup_proj, actions)
        else:
            unsup_loss = 0

        if self.action_coeff > 0:    
            if len(actions.shape) > 1: # Reshape the actions if we are using sequences
                action_logits = action_logits.reshape(-1, action_logits.size(-1))
                actions = actions.reshape(-1)
            action_loss = self.action_criterion(action_logits, actions)
            metrics['action_loss'] = action_loss.item()
            with torch.no_grad(): # Also compute the accuracy
                action_pred = torch.argmax(action_logits, dim=-1)
                accuracy = (action_pred == actions).sum().item() / actions.shape[0]
                metrics['accuracy'] = accuracy
        else:
            action_loss = 0

        if self.lang_coeff > 0:
            if len(instruction_labels.shape) > 1:
                lang_logits = lang_logits.reshape(-1, lang_logits.size(-1))
                instruction_labels = instruction_labels.reshape(-1)
            lang_loss = self.instruction_criterion(lang_logits, instruction_labels)
            metrics['lang_loss'] = lang_loss.item()
        else:
            lang_loss = 0

        total_loss = self.action_coeff * action_loss + \
                     self.lang_coeff * lang_loss + \
                     self.unsup_coeff * unsup_loss
                     
        metrics['total_loss'] = total_loss.item()
        return total_loss, metrics
        

    def train(self, path, total_steps, log_freq=100, eval_freq=5000, eval_ep=0, validation_metric="loss", use_eval_mode=True):        
        logger = Logger(path=path)

        print("[Lang RL] Training a model with tunable parameters", sum(p.numel() for p in self.network.parameters() if p.requires_grad))
        # Setup for the different ind

        if isinstance(self.env.unwrapped, None):
            from language_prediction.datasets.babayai_dataset import BabyAITrajectoryDataset, traj_collate_fn
            collate_fn = traj_collate_fn
            dataset = BabyAITrajectoryDataset.load(self.dataset, fraction=self.dataset_fraction)
            if not self.validation_dataset is None:
                validation_dataset = BabyAITrajectoryDataset.load(self.validation_dataset)
        elif isinstance(self.env.unwrapped, language_prediction.envs.mazebase.MazebaseGame):
            from language_prediction.datasets import crafting_dataset
            skip = 1 if self.unsup_coeff > 0.0 else -1
            dataset = crafting_dataset.CraftingDataset(self.dataset, self.vocab, dataset_fraction=self.dataset_fraction, skip=skip) # Must have created the vocab. Note that it was already given to the agent.
            if not self.validation_dataset is None:
                validation_dataset = crafting_dataset.CraftingDataset(self.validation_dataset, self.vocab, dataset_fraction=1.0, skip=skip)
            collate_fn = partial(crafting_dataset.collate_fn, vocab_size=len(self.vocab))
        else:
            raise ValueError("Unknown environment type passed in.")

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
        if not self.validation_dataset is None:
            validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

        step = 0
        num_epochs = 0
        start_time = time.time()
        loss_lists = defaultdict(list)
        best_validation_metric = float('inf')
        start_time = time.time()
        self.network.train()

        while step < total_steps:

            for obs, actions in dataloader:

                self.optim.zero_grad()
                loss, metrics = self._compute_loss(obs, actions)
                loss.backward()
                if self.grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_norm)
                self.optim.step()
                for metric_name, metric_value in metrics.items():
                    loss_lists[metric_name].append(metrics)
                step += 1

                if self.unsup_coeff > 0.0 and step % self.unsup_ema_update_freq == 0:
                    ema_parameters(self.network, self.ema_network, self.unsup_ema_tau)

                # Run the train logging
                if step % log_freq == 0:
                    current_time = time.time()
                    log_from_dict(logger, loss_lists, "train")
                    logger.record("time/epochs", num_epochs)
                    logger.record("time/steps_per_seceonds", log_freq / (time.time() - start_time))
                    start_time = current_time
                    logger.dump(step=step)
                
                # Run validation logging
                if step % eval_freq == 0:
                    if use_eval_mode:
                        self.network.eval()

                    if eval_ep > 0 and not self.env is None:
                        with torch.no_grad():
                            mean_reward, std_reward, mean_length, success_rate = eval_policy(self.env, self, eval_ep)
                        logger.record("env/mean_reward", mean_reward)
                        logger.record("env/reward_std", std_reward)
                        logger.record("env/mean_length", mean_length)
                        logger.record("env/success_rate", success_rate)
                        non_empty_dump = True

                    if eval_ep > 0 and not self.eval_env is None:
                        with torch.no_grad():
                            mean_reward, std_reward, mean_length, success_rate = eval_policy(self.eval_env, self, eval_ep)
                        logger.record("eval_env/mean_reward", mean_reward)
                        logger.record("eval_env/reward_std", std_reward)
                        logger.record("eval_env/mean_length", mean_length)
                        logger.record("eval_env/success_rate", success_rate)
                        non_empty_dump = True
                    
                    if self.validation_dataset:
                        validation_loss_lists = defaultdict(list)
                        with torch.no_grad():
                            for valid_obs, valid_ac in validation_dataloader:
                                _, metrics = self._compute_loss(valid_obs, valid_ac)
                                for metric_name, metric_value in metrics.items():
                                    validation_loss_lists[metric_name].append(metrics)

                        current_validation_metric = np.mean(validation_loss_lists[validation_metric])
                        if current_validation_metric < best_validation_metric:
                            self.save(path, "best_model")
                        log_from_dict(logger, validation_loss_lists, "valid")
                    
                    # Every eval period also save the "final model"
                    logger.dump(step=step)
                    self.save(path, "final_model")

                if step >= total_steps:
                    break
            
            num_epochs += 1

        print("Finished.")
        self.save(path, "final_model")
