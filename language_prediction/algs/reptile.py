import os
import time
import torch
import random
import numpy as np

from language_prediction.utils.logger import Logger
from language_prediction.utils.evaluate import eval_policy
from language_prediction.utils.utils import to_tensor, to_device, unsqueeze

# Env imports for type checking
from language_prediction.envs.mazebase import MazebaseGame
from gym_minigrid.minigrid import MiniGridEnv

import language_prediction
from functools import partial
from collections import defaultdict


def sample_loss_coeffs():
    """Randomly sample alpha balancing coefficient for re-scaling 
    the behavior cloning and language (instruction) reconstrution losses.
    """
    alpha = random.random()
    return alpha, (1-alpha)


class Reptile(object):

    def __init__(self, env, network_class, network_kwargs={}, device="cpu",
                 optim_cls=torch.optim.Adam, eval_env=None,
                 meta_optim_cls=torch.optim.SGD,
                 dataset=None,
                 validation_dataset=None,
                 optim_kwargs={
                     'lr': 0.0001
                 },
                 meta_optim_kwargs={
                     'lr' : 0.01
                 },
                 batch_size=4,
                 lang_coeff=1.0,
                 action_coeff=1.0,
                 random_coeff=False,
                 checkpoint=None,
                 dataset_fraction=1,
                 num_support=12,
                 num_query=1,
                 meta_batch_size=1,
                 inner_iters=3,
                 ):
        
        self.env = env
        self.eval_env = eval_env
        assert eval_env is None, "Eval Env not supported for meta learning"
        num_actions = env.action_space.n # Only support discrete action spaces.
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.num_support = num_support
        self.num_query = num_query
        self.inner_iters = inner_iters
        assert self.num_support % self.batch_size == 0, "Batch size for the fast updates must divide the query size"
        assert self.inner_iters * self.batch_size == self.num_support, "The number of iters ties the batch size must be the support size."
        self.dataset = dataset
        self.validation_dataset = validation_dataset
        self.action_coeff = action_coeff
        self.lang_coeff = lang_coeff
        self.random_coeff = random_coeff
        self.dataset_fraction = dataset_fraction
        self.meta_batch_size = meta_batch_size

        network_args = [num_actions]
        if isinstance(self.env.unwrapped, MazebaseGame):
            # Now we have to create the vocab here.
            import pickle
            with open(self.dataset + '_all_instr.pkl', 'rb') as f:
                all_instructions = pickle.load(f)
            from language_prediction.datasets import crafting_dataset
            self.vocab, vocab_weights = crafting_dataset.build_vocabulary(all_instructions)
            network_args.append(self.vocab)
            network_args.append(vocab_weights)

        # Create the network and optimizer.
        self.network = network_class(*network_args, **network_kwargs).to(self.device)
        self.meta_network = network_class(*network_args, **network_kwargs).to(self.device)
        # Sync the networks.
        self.network.load_state_dict(self.meta_network.state_dict())

        self.optim = optim_cls(self.network.parameters(), **optim_kwargs)
        self.meta_optim = optim_cls(self.meta_network.parameters(), **optim_kwargs)

        if checkpoint:
            self.load(checkpoint, strict=True)
        
        self.action_criterion = torch.nn.CrossEntropyLoss(ignore_index=self.network.action_pad_idx)
        self.instruction_criterion = torch.nn.CrossEntropyLoss(ignore_index=self.network.lang_pad_idx)

    def predict(self, obs, deterministic=True, history=None):
        raise NotImplementedError
    
    def save(self, path, extension):
        save_dict = {"network" : self.meta_network.state_dict(), "optim": self.meta_optim.state_dict()}
        torch.save(save_dict, os.path.join(path, extension + ".pt"))

    def load(self, checkpoint, strict=True):
        print("[Language Prediction' Loading Checkpoint:", checkpoint)
        checkpoint = torch.load(checkpoint, map_location=self.device)
        self.meta_network.load_state_dict(checkpoint['network'], strict=strict)
        if strict:
            # Only load the optimizer state dict if we are being strict.
            self.meta_optim.load_state_dict(checkpoint['optim'])

    def _compute_loss(self, batch, action_coeff=1.0, lang_coeff=0.0):
        if self.random_coeff or action_coeff is None or lang_coeff is None:
            action_coeff, lang_coeff = sample_loss_coeffs()

        metrics = {}
        batch = to_device(batch, self.device)
        actions = batch['action'].long()
        if 'instruction' in batch:
            instruction_labels = batch['instruction'][:, 1:].long()
            batch['instruction'] = batch['instruction'][:, :-1].long()

        # Remove language inputs for a speedup if we don't need it.
        action_logits, lang_logits = self.network(batch)
        
        if action_coeff > 0:    
            if len(actions.shape) > 1: # Reshape the actions if we are using sequences
                action_logits = action_logits.reshape(-1, action_logits.size(-1))
                actions = actions.reshape(-1)
            action_loss = self.action_criterion(action_logits, actions)
            metrics['action_loss'] = action_loss.item()
            with torch.no_grad(): # Also compute the accuracy
                action_pred = torch.argmax(action_logits, dim=-1)
                mask = actions != self.network.action_pad_idx
                accuracy = ((action_pred == actions) * mask).sum().item() / mask.sum().item()
                metrics['action_accuracy'] = accuracy
        else:
            action_loss = 0

        if lang_coeff > 0:
            if len(instruction_labels.shape) > 1:
                lang_logits = lang_logits.reshape(-1, lang_logits.size(-1))
                instruction_labels = instruction_labels.reshape(-1)
            lang_loss = self.instruction_criterion(lang_logits, instruction_labels)
            metrics['lang_loss'] = lang_loss.item()
            with torch.no_grad():
                lang_pred = torch.argmax(lang_logits, dim=-1)
                mask = lang_pred != self.network.lang_pad_idx
                accuracy = ((lang_pred == instruction_labels) * mask).sum().item() / mask.sum().item()
                metrics['lang_accuracy'] = accuracy
        else:
            lang_loss = 0

        total_loss = action_coeff * action_loss + \
                     lang_coeff * lang_loss

        metrics['total_loss'] = total_loss.item()
        return total_loss, metrics

    def _adapt(self, support_set):
        # Adapt the fast network parameters based on the support set
        for i in range(self.inner_iters):
            current_batch = {k: v[i*self.batch_size:(i+1)*self.batch_size] for k, v in support_set.items()}
            self.optim.zero_grad()
            loss, metrics = self._compute_loss(current_batch, action_coeff=self.action_coeff, lang_coeff=self.lang_coeff)
            loss.backward()
            self.optim.step()
        return metrics

    def _training_step(self, meta_batch, loss_lists):
        """Take a single training step update with Reptile.
        """
        weights = []
        for support_set, _ in meta_batch:
            # update the fast weights
            metrics = self._adapt(support_set)
            # Record most recent metrics for the sample task.
            for metric_name, metric_value in metrics.items():
                loss_lists[metric_name].append(metrics[metric_name])
            weights.append(self.network.state_dict())       
            self.network.load_state_dict(self.meta_network.state_dict())

        if len(weights) == 1:
            weights = weights[0]
        else:
            #  We have a meta batch size larger than one and need to average the weights
            with torch.no_grad():
                avg_weights = {}
                for k in weights[0]:
                    avg_weights[k] = sum([weight[k] for weight in weights]) / len(weights)
            weights = avg_weights
        self.network.load_state_dict(weights)

        # Now set the gradients of the meta optimizer to be the weight differences
        self.meta_optim.zero_grad()
        for meta_param, param in zip(self.meta_network.parameters(), self.network.parameters()):
            if meta_param.grad is None:
                meta_param.grad = torch.autograd.Variable(torch.zeros(meta_param.size())).to(self.device)
            meta_param.grad.data.zero_() # perhaps unnecesary
            meta_param.grad.data.add_(meta_param.data - param.data) # Phi - W is the gradient
        
        self.meta_optim.step()

    def train(self, path, total_steps, log_freq=10, eval_freq=100, eval_ep=0, validation_metric="adapt/action_loss", use_eval_mode=True, workers=4):        
        assert eval_ep == 0, "Eval episodes not supported for meta learning"
        
        logger = Logger(path=path)

        print("[Lang IL] Training a model with tunable parameters", sum(p.numel() for p in self.network.parameters() if p.requires_grad))
        # Setup for the different ind

        if isinstance(self.env.unwrapped, MazebaseGame) and isinstance(self.network, language_prediction.networks.CraftingDT):
            from language_prediction.datasets import meta_crafting_dataset
            dataset = meta_crafting_dataset.SeqMetaCraftingDataset.load(self.dataset, self.vocab, num_support=self.num_support, num_query=self.num_query, dataset_fraction=self.dataset_fraction) # Must have created the vocab. Note that it was already given to the agent.
            if not self.validation_dataset is None:
                validation_dataset = meta_crafting_dataset.SeqMetaCraftingDataset.load(self.validation_dataset, self.vocab, num_support=self.num_support, num_query=self.num_query, dataset_fraction=1.0)
            collate_fn = meta_crafting_dataset.collate_fn
        elif isinstance(self.env.unwrapped, MazebaseGame):
            from language_prediction.datasets import meta_crafting_dataset
            dataset = meta_crafting_dataset.MetaCraftingDataset.load(self.dataset, self.vocab, num_support=self.num_support, num_query=self.num_query, dataset_fraction=self.dataset_fraction) # Must have created the vocab. Note that it was already given to the agent.
            if not self.validation_dataset is None:
                validation_dataset = meta_crafting_dataset.MetaCraftingDataset.load(self.validation_dataset, self.vocab, num_support=self.num_support, num_query=self.num_query, dataset_fraction=1.0)
            collate_fn = meta_crafting_dataset.collate_fn
        else:
            raise ValueError("Unknown environment type passed in.")

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.meta_batch_size, shuffle=True, num_workers=workers, pin_memory=True, collate_fn=collate_fn)
        if not self.validation_dataset is None:
            validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

        step = 0
        num_epochs = 0
        start_time = time.time()
        loss_lists = defaultdict(list)
        best_validation_metric = float('inf')
        start_time = time.time()
        self.network.train()

        while step < total_steps:
            # Note that the meta batch size is usually one.
            for meta_batch in dataloader:
                
                # Take a single reptile training step
                self._training_step(meta_batch, loss_lists)
                step += 1

                # Now sync the weights of the two networks.
                self.network.load_state_dict(self.meta_network.state_dict())

                # Run the train logging
                if step % log_freq == 0:
                    current_time = time.time()
                    logger.log_from_dict(loss_lists, "train")
                    logger.record("time/epochs", num_epochs)
                    logger.record("time/steps_per_seceonds", log_freq / (time.time() - start_time))
                    start_time = current_time
                    logger.dump(step=step)
                
                # Run validation logging
                if step % eval_freq == 0:
                    
                    if self.validation_dataset:
                        validation_loss_lists = defaultdict(list)
                        for ((support_set, query_set),) in validation_dataloader:
                            # Adapt the weights
                            adapt_metrics = self._adapt(support_set)
                            if use_eval_mode:
                                self.network.eval()
                            with torch.no_grad():
                                _, query_metrics = self._compute_loss(query_set, action_coeff=self.action_coeff, lang_coeff=self.lang_coeff)
                            self.network.train()
                            for metric_name, metric_value in adapt_metrics.items():
                                validation_loss_lists["adapt/" + metric_name].append(metrics[metric_name])
                            for metric_name, metric_value in query_metrics.items():
                                validation_loss_lists["query/" + metric_name].append(metrics[metric_name])
                            
                            # Now sync the weights of the two networks.
                            self.network.load_state_dict(self.meta_network.state_dict())

                        current_validation_metric = np.mean(validation_loss_lists[validation_metric])
                        
                        if current_validation_metric < best_validation_metric:
                            self.save(path, "best_model")
                        logger.log_from_dict(validation_loss_lists, "valid")

                    # Sync the networks back
                    self.network.load_state_dict(self.meta_network.state_dict())
                    print("Finisehd eval!")

                    logger.dump(step=step)
                    # Every eval period also save the "final model"
                    self.save(path, "final_model")

                if step >= total_steps:
                    break
            
            num_epochs += 1

        print("Finished.")
        self.save(path, "final_model")
