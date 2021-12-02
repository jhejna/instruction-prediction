import numpy as np
import torch
import torchtext.vocab as vocabtorch
from torch.utils.data.dataset import Dataset
import pickle
import random
from torch.nn.utils.rnn import pad_sequence

from language_prediction.envs.mazebase import get_grid_embedding, one_hot_grid, one_hot_action, one_hot_action, get_goal_embedding, get_inventory_embedding

class SeqMetaCraftingDataset(Dataset):

    def __init__(self, dataset, vocab, tasks=[], num_support=3, num_query=1, dataset_fraction=1):
        '''
        The `dataset` argument should have the following structure:
        {
            "goal":
                [{
                    "state": [ list of states]
                    "action": [list of actions]
                    "inventory" : [list of inventories]
                    "instruction" : [ the instructions concatenate together]
                    },
                 {
                    "state": [ list of states]
                    "action": [list of actions]
                    "inventory" : [list of inventories]
                    "instruction" : [ the instructions concatenate together]
                    },
                 }
                ],

            ...
        }
        '''
        super().__init__()
        assert dataset_fraction == 1, "Smaller Dataset sizes not supportes"
        self.dataset = dataset
        for goal in self.dataset.keys():
            if goal not in tasks:
                del self.dataset[goal]

        self.vocab = vocab
        self.num_support = num_support
        self.num_query = num_query

        # Preprocess the data
        embed_size = 300
        glove = vocabtorch.GloVe(name='840B', dim=300)

        # Check that we can actually sample with the given numbers for the dataset
        too_short = []
        for task in self.dataset.keys():
            if len(self.dataset[task]) < self.num_support + self.num_query:
                print("TASK", task, "too short, removing with size", len(self.dataset[task]))
                too_short.append(task)
        for task in too_short:
            del self.dataset[task]

        # Preprocess the dataset.
        # This is done on load to conserve storage space.
        for task in self.dataset.keys():                
            for ep_idx in range(len(self.dataset[task])):
                # Run preprocessing over the transitions
                current_ep = self.dataset[task][ep_idx]
                # States
                self.dataset[task][ep_idx]["grid_embedding"] = np.array(
                        [get_grid_embedding(state, glove, embed_size) for state in current_ep['state']], dtype=np.float32)
                self.dataset[task][ep_idx]["grid_onehot"] = np.array(
                        [one_hot_grid(state, glove, embed_size) for state in current_ep['state']], dtype=np.float32)
                del self.dataset[task][ep_idx]['state'] # conserve memory usage because we split it into two names
                # Inventory
                self.dataset[task][ep_idx]['inventory'] = np.array(
                        [get_inventory_embedding(inventory, glove, embed_size) for inventory in current_ep['inventory']], dtype=np.float32)
                # Actions
                self.dataset[task][ep_idx]['action'] = np.array(
                        [one_hot_action(action) for action in current_ep['action']], dtype=np.int32
                )
                # Check that all the shapes line up
                assert len(current_ep['action']) == len(current_ep['grid_embedding']) == len(current_ep['grid_onehot']) == len(current_ep['inventory'])
                del current_ep # Clean up any remaining references
        
        self.idx_to_key = {i: k for i, k in enumerate(self.dataset.keys())}

    @classmethod
    def save(cls, dataset, vocab, path):
        # Run checks on the dataset structure
        assert isinstance(dataset, dict)
        assert all([isinstance(k, str) for k in dataset.keys()])
        assert len(dataset.keys()) > 0
        first_task = dataset[dataset.keys()[0]]
        assert isinstance(first_task, list)
        assert "state" in first_task[0]
        assert "action" in first_task[0]
        assert "inventory" in first_task[0]
        assert "instruction" in first_task[0]

        if not path.endswith(".pkl"):
            path += ".pkl"
        with open(path, 'wb') as f:
            pickle.dump(dataset, f)        
        
    @classmethod
    def load(cls, path, vocab, **kwargs):
        if not path.endswith('.pkl'):
            path += '.pkl'           
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
        return cls(dataset, vocab, **kwargs)

    def _create_batch(self, task, ep_idxs):
        key_to_pad = {
            "grid_embedding": 0,
            "grid_onehot": 0,
            "inventory": 0,
            "action": -100,
            "instruction": len(self.vocab) - 1
        }
        batch = {}        
        for key in key_to_pad.keys():
            if key == "instruction":
                instructions = []
                for ep_idx in ep_idxs:
                    instruction = []
                    instruction.append(self.vocab('<start>'))
                    instruction.extend([self.vocab(token) for token in self.dataset[task][ep_idx][key]])
                    instruction.append(self.vocab('<end>'))
                    instructions.append(torch.Tensor(instruction))
                batch[key] = pad_sequence(
                    instructions,
                    batch_first=True,
                    padding_value=key_to_pad[key]
                )
            else:
                batch[key] = pad_sequence(
                    [torch.from_numpy(self.dataset[task][ep_idx][key]) for ep_idx in ep_idxs],
                    batch_first=True,
                    padding_value=key_to_pad[key]
                )
        return batch

    def __getitem__(self, index):
        '''
        Returns a tuple of a given task: (support, query). The structure is as follows
        '''
        task = self.idx_to_key[index]
        # sample a query set and a batch set
        idxs = np.random.choice(len(self.dataset[task]), size=self.num_query+self.num_support)
        query_set = self._create_batch(task, idxs[:self.num_query])
        support_set = self._create_batch(task, idxs[self.num_query:])
        return support_set, query_set

    def __len__(self):
        return len(self.dataset)

class MetaCraftingDataset(Dataset):

    def __init__(self, dataset, vocab, num_support=3, num_query=1, dataset_fraction=1):
        '''
        The `dataset` argument should have the following structure:
        {
            "goal":
                {
                    "state": [ list of states]
                    "action": [list of actions]
                    "inventory" : [list of inventories]
                    "instruction" : [ the instructions concatenate together]
                },
            ...
        }
        '''
        super().__init__()
        assert dataset_fraction == 1, "Smaller Dataset sizes not supportes"
        self.dataset = dataset
        self.vocab = vocab
        self.num_support = num_support
        self.num_query = num_query

        # Preprocess the data
        embed_size = 300
        glove = vocabtorch.GloVe(name='840B', dim=300)

        # Check that we can actually sample with the given numbers for the dataset
        too_short = []
        for task in self.dataset.keys():
            if len(self.dataset[task]['state']) < self.num_support + self.num_query:
                print("TASK", task, "too short, removing with size", len(self.dataset[task]))
                too_short.append(task)
        for task in too_short:
            del self.dataset[task]
        
        # Preprocess the dataset.
        # This is done on load to conserve storage space.
        for task in self.dataset.keys():
            # Run preprocessing over the transitions
            current_task = self.dataset[task]
            # States
            self.dataset[task]["grid_embedding"] = np.array(
                    [get_grid_embedding(state, glove, embed_size) for state in current_task['state']], dtype=np.float32)
            self.dataset[task]["grid_onehot"] = np.array(
                    [one_hot_grid(state, glove, embed_size) for state in current_task['state']], dtype=np.float32)
            del self.dataset[task]['state'] # conserve memory usage because we split it into two names
            # Inventory
            self.dataset[task]['inventory'] = np.array(
                    [get_inventory_embedding(inventory, glove, embed_size) for inventory in current_task['inventory']], dtype=np.float32)
            # Actions
            self.dataset[task]['action'] = np.array(
                    [one_hot_action(action) for action in current_task['action']], dtype=np.int32
            )
            # Check that all the shapes line up
            assert len(current_task['action']) == len(current_task['grid_embedding']) == len(current_task['grid_onehot']) == len(current_task['inventory'])
        
        self.idx_to_key = {i: k for i, k in enumerate(self.dataset.keys())}
    
    @classmethod
    def save(cls, dataset, vocab, path):
        # Run checks on the dataset structure
        assert isinstance(dataset, dict)
        assert all([isinstance(k, str) for k in dataset.keys()])
        assert len(dataset.keys()) > 0
        first_task = dataset[dataset.keys()[0]]
        assert isinstance(first_task, dict)
        assert "state" in first_task
        assert "action" in first_task
        assert "inventory" in first_task
        assert "instruction" in first_task

        if not path.endswith(".pkl"):
            path += ".pkl"
        with open(path, 'wb') as f:
            pickle.dump(dataset, f)        
        
    @classmethod
    def load(cls, path, vocab, **kwargs):   
        if not path.endswith('.pkl'):
            path += '.pkl'     
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
        return cls(dataset, vocab, **kwargs)

    def _create_batch(self, task, idxs):
        key_to_pad = {
            "grid_embedding": 0,
            "grid_onehot": 0,
            "inventory": 0,
            "action": -100,
            "instruction": len(self.vocab) - 1
        }
        batch = {}        
        for key in key_to_pad.keys():
            if key != "instruction":
                batch[key] = torch.stack([torch.from_numpy(self.dataset[task][key][idx]) for idx in idxs], dim=0)
            else:
                instructions = []
                for idx in idxs:
                    instruction = []
                    instruction.append(self.vocab('<start>'))
                    instruction.extend([self.vocab(token) for token in self.dataset[task][key][idx]])
                    instruction.append(self.vocab('<end>'))
                    instructions.append(torch.Tensor(instruction))
                batch[key] = pad_sequence(
                    instructions,
                    batch_first=True,
                    padding_value=key_to_pad[key]
                )
        return batch

    def __getitem__(self, index):
        '''
        Returns a tuple of a given task: (support, query). The structure is as follows
        '''
        task = self.idx_to_key[index]
        # sample a query set and a batch set
        idxs = np.random.choice(len(self.dataset[task]['action']), size=self.num_query+self.num_support)
        query_set = self._create_batch(task, idxs[:self.num_query])
        support_set = self._create_batch(task, idxs[self.num_query:])
        return support_set, query_set

    def __len__(self):
        return len(self.dataset)
    
def collate_fn(data):
    # Identity collate as we already handle conversions and batching in the dataset.
    return data