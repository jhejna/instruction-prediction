import numpy as np
import torch
import torchtext.vocab as vocabtorch
from torch.utils.data.dataset import Dataset
import pickle
import random
from torch.nn.utils.rnn import pad_sequence

from language_prediction.envs.mazebase import get_grid_embedding, one_hot_grid, one_hot_action, one_hot_action, get_goal_embedding, get_inventory_embedding

class MetaCraftingDataset(Dataset):

    def __init__(self, dataset, vocab, num_support=3, num_query=1, dataset_fraction=1):
        '''
        The `dataset` argument should have the following structure:
        {
            "goal":
                [{
                    "states": [ list of states]
                    "actions": [list of actions]
                    "inventories" : [list of inventories]
                    "instructions" : [ the instructions concatenate together]
                    },
                 {
                    "states": [ list of states]
                    "actions": [list of actions]
                    "inventories" : [list of inventories]
                    "instructions" : [ the instructions concatenate together]
                    },
                 }
                ],

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
        for task in self.dataset.keys():
            assert len(task) >= self.num_support + self.num_query
        
        # Preprocess the dataset.
        # This is done on load to conserve storage space.
        for task in self.dataset.keys():
            for ep_idx in range(len(self.dataset[task])):
                # Run preprocessing over the transitions
                current_ep = self.dataset[task][ep_idx]
                # States
                self.dataset[task][ep_idx]["grid_embedding"] = np.array(
                        [get_grid_embedding(state, glove, embed_size) for state in current_ep['states']], dtype=np.float32)
                self.dataset[task][ep_idx]["grid_onehot"] = np.array(
                        [one_hot_grid(state, glove, embed_size) for state in current_ep['states']], dtype=np.float32)
                del self.dataset[task][ep_idx]['states'] # conserve memory usage because we split it into two names
                # Inventory
                self.dataset['inventory'] = np.array(
                        [get_inventory_embedding(inventory, glove, embed_size) for inventory in current_ep['inventory']], dtype=np.float32)
                # Actions
                self.dataset[task][ep_idx]['actions'] = np.array(
                        [one_hot_action(action) for action in current_ep['actions']], dtype=np.int32
                )
                # Check that all the shapes line up
                assert len(current_ep['actions']) == len(current_ep['grid_embedding']) == len(current_ep['grid_onehot']) == len(current_ep['inventory'])
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
        assert "states" in first_task[0]
        assert "actions" in first_task[0]
        assert "inventories" in first_task[0]
        assert "instructions" in first_task[0]

        if not path.endswith(".pkl"):
            path += ".pkl"
        with open(path, 'wb') as f:
            pickle.dump(dataset, f)        
        
    @classmethod
    def load(cls, path, vocab):        
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
        return cls(dataset, vocab)

    def _create_batch(self, task, ep_idxs):
        key_to_pad = {
            "grid_embedding": 0,
            "grid_onehot": 0,
            "inventory": 0,
            "actions": -100,
            "instructions": len(self.vocab) - 1
        }
        batch = {}        
        for key in key_to_pad.keys():
            batch[key] = pad_sequence(
                [torch.from_numpy(self.dataset[task][ep_idx]) for ep_idx in ep_idxs],
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

def collate_fn(data):
    # Identity collate as we already handle conversions and batching in the dataset.
    return data