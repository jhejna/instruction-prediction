import numpy as np
import torch
import torchtext.vocab as vocabtorch
from torch.utils.data.dataset import Dataset
import pickle
import random

from language_prediction.envs.mazebase import get_grid_embedding, one_hot_grid, one_hot_action, one_hot_action, get_goal_embedding, get_inventory_embedding

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocabulary(train_instructions):

    freqs = {}
    embed_dim = 300

    if embed_dim == 300:
        glove = vocabtorch.GloVe(name='840B', dim=300) #maybe switch this out!
    elif embed_dim == 50:
        glove = vocabtorch.GloVe(name='6B', dim=50)

    instruction_size = []
    instructs = {}

    for instruction in train_instructions:

        instruction_joined = " ".join(instruction)
        instruction_size.append(len(instruction))
        inst_lower = instruction_joined.lower()

        if inst_lower in instructs:
            instructs[inst_lower] = instructs[inst_lower] + 1
        else:
            instructs[inst_lower] = 1

        for word in instruction:
            try:
                vec = glove.vectors[glove.stoi[word]]
                if word in freqs:
                    freqs[word] = freqs[word] + 1
                else:
                    freqs[word] = 1
            except:
                if 'UNK' in freqs:
                    freqs['UNK'] = freqs['UNK'] + 1
                else:
                    freqs['UNK'] = 1

    vocab_size = 0
    for i, key in enumerate(freqs):
        if 'UNK' != key and freqs[key] >= 10:
            vocab_size = vocab_size + 1

    vocab_weights = np.random.uniform(-0.01, 0.01, (vocab_size + 5, embed_dim)).astype("float32")
    vocab = Vocabulary()

    count = 0
    for i, key in enumerate(freqs):

        if 'UNK' == key:
            vec_string = '0.22418134 -0.28881392 0.13854356 0.00365387 -0.12870757 0.10243822 0.061626635 0.07318011 -0.061350107 -1.3477012 0.42037755 -0.063593924 -0.09683349 0.18086134 0.23704372 0.014126852 0.170096 -1.1491593 0.31497982 0.06622181 0.024687296 0.076693475 0.13851812 0.021302193 -0.06640582 -0.010336159 0.13523154 -0.042144544 -0.11938788 0.006948221 0.13333307 -0.18276379 0.052385733 0.008943111 -0.23957317 0.08500333 -0.006894406 0.0015864656 0.063391194 0.19177166 -0.13113557 -0.11295479 -0.14276934 0.03413971 -0.034278486 -0.051366422 0.18891625 -0.16673574 -0.057783455 0.036823478 0.08078679 0.022949161 0.033298038 0.011784158 0.05643189 -0.042776518 0.011959623 0.011552498 -0.0007971594 0.11300405 -0.031369694 -0.0061559738 -0.009043574 -0.415336 -0.18870236 0.13708843 0.005911723 -0.113035575 -0.030096142 -0.23908928 -0.05354085 -0.044904727 -0.20228513 0.0065645403 -0.09578946 -0.07391877 -0.06487607 0.111740574 -0.048649278 -0.16565254 -0.052037314 -0.078968436 0.13684988 0.0757494 -0.006275573 0.28693774 0.52017444 -0.0877165 -0.33010918 -0.1359622 0.114895485 -0.09744406 0.06269521 0.12118575 -0.08026362 0.35256687 -0.060017522 -0.04889904 -0.06828978 0.088740796 0.003964443 -0.0766291 0.1263925 0.07809314 -0.023164088 -0.5680669 -0.037892066 -0.1350967 -0.11351585 -0.111434504 -0.0905027 0.25174105 -0.14841858 0.034635577 -0.07334565 0.06320108 -0.038343467 -0.05413284 0.042197507 -0.090380974 -0.070528865 -0.009174437 0.009069661 0.1405178 0.02958134 -0.036431845 -0.08625681 0.042951006 0.08230793 0.0903314 -0.12279937 -0.013899368 0.048119213 0.08678239 -0.14450377 -0.04424887 0.018319942 0.015026873 -0.100526 0.06021201 0.74059093 -0.0016333034 -0.24960588 -0.023739101 0.016396184 0.11928964 0.13950661 -0.031624354 -0.01645025 0.14079992 -0.0002824564 -0.08052984 -0.0021310581 -0.025350995 0.086938225 0.14308536 0.17146006 -0.13943303 0.048792403 0.09274929 -0.053167373 0.031103406 0.012354865 0.21057427 0.32618305 0.18015954 -0.15881181 0.15322933 -0.22558987 -0.04200665 0.0084689725 0.038156632 0.15188617 0.13274793 0.113756925 -0.095273495 -0.049490947 -0.10265804 -0.27064866 -0.034567792 -0.018810693 -0.0010360252 0.10340131 0.13883452 0.21131058 -0.01981019 0.1833468 -0.10751636 -0.03128868 0.02518242 0.23232952 0.042052146 0.11731903 -0.15506615 0.0063580726 -0.15429358 0.1511722 0.12745973 0.2576985 -0.25486213 -0.0709463 0.17983761 0.054027 -0.09884228 -0.24595179 -0.093028545 -0.028203879 0.094398156 0.09233813 0.029291354 0.13110267 0.15682974 -0.016919162 0.23927948 -0.1343307 -0.22422817 0.14634751 -0.064993896 0.4703685 -0.027190214 0.06224946 -0.091360025 0.21490277 -0.19562101 -0.10032754 -0.09056772 -0.06203493 -0.18876675 -0.10963594 -0.27734384 0.12616494 -0.02217992 -0.16058226 -0.080475815 0.026953284 0.110732645 0.014894041 0.09416802 0.14299914 -0.1594008 -0.066080004 -0.007995227 -0.11668856 -0.13081996 -0.09237365 0.14741232 0.09180138 0.081735 0.3211204 -0.0036552632 -0.047030564 -0.02311798 0.048961394 0.08669574 -0.06766279 -0.50028914 -0.048515294 0.14144728 -0.032994404 -0.11954345 -0.14929578 -0.2388355 -0.019883996 -0.15917352 -0.052084364 0.2801028 -0.0029121689 -0.054581646 -0.47385484 0.17112483 -0.12066923 -0.042173345 0.1395337 0.26115036 0.012869649 0.009291686 -0.0026459037 -0.075331464 0.017840583 -0.26869613 -0.21820338 -0.17084768 -0.1022808 -0.055290595 0.13513643 0.12362477 -0.10980586 0.13980341 -0.20233242 0.08813751 0.3849736 -0.10653763 -0.06199595 0.028849555 0.03230154 0.023856193 0.069950655 0.19310954 -0.077677034 -0.144811'
            average_glove_vector = np.array(vec_string.split(" "))

        else:
            if freqs[key] >= 10:
                vocab_weights[count] = glove.vectors[glove.stoi[key]]
                count = count + 1
                vocab.add_word(key)

    # Add the four tokens from earlier.
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    vocab.add_word('<subgoal>')
    vocab.add_word('<pad>') # Pad is the last word in the vocab so its index is equal to vocab_size - 1. This was done to ensure easy loss calculation
    
    vocab_weights[vocab.word2idx['<pad>'], :] = 0.0 # Give it zero weight
    
    print("Total vocabulary size: {}".format(len(vocab)))

    return vocab, vocab_weights
    
def load_vocabulary(load_name):

    with open(load_name+'_vocab', 'rb') as f:
        vocab = pickle.load(f)

    weights = np.load(load_name+"_vocab_weights.npy")

    return weights, vocab


class CraftingDataset(Dataset):

    def __init__(self, path, vocab, dataset_fraction=1, skip=-1):
        self.dataset_name = path
        self.skip = skip

        with open(self.dataset_name +'states', 'rb') as f:
            self.train_states = pickle.load(f)
        with open(self.dataset_name +'inventories', 'rb') as f:
            self.train_inventories = pickle.load(f)
        with open(self.dataset_name +'actions', 'rb') as f:
            self.train_actions = pickle.load(f)
        with open(self.dataset_name +'goals', 'rb') as f:
            self.train_goals = pickle.load(f)
        with open(self.dataset_name +'instructions', 'rb') as f:
            self.train_instructions = pickle.load(f)

        if dataset_fraction < 1:
            # Take only the first part of the dataset.
            num_data_pts = int(len(self.train_states) * dataset_fraction)
            self.train_states = self.train_states[:num_data_pts]
            self.train_inventories = self.train_inventories[:num_data_pts]
            self.train_actions = self.train_actions[:num_data_pts]
            self.train_goals = self.train_goals[:num_data_pts]
            self.train_instructions = self.train_instructions[:num_data_pts]

        self.vocab = vocab

        # Preprocess the data
        embed_size = 300
        glove = vocabtorch.GloVe(name='840B', dim=300)

        self.train_states_embedding = [get_grid_embedding(state, glove, embed_size) for state in self.train_states]
        print("embedding loaded")
        self.train_states_onehot = [one_hot_grid(state, glove, embed_size) for state in self.train_states]
        print("one hot loaded")
        del self.train_states
        self.train_actions_onehot = [one_hot_action(action) for action in self.train_actions]
        print("actions loaded")
        del self.train_actions
        self.train_goals_embedding = [get_goal_embedding(goal, glove, embed_size) for goal in self.train_goals]
        print("goals loaded")
        del self.train_goals
        self.train_inventory_embedding = [get_inventory_embedding(inventory, glove, embed_size) for inventory in self.train_inventories]
        print("done loading dataset")
        del self.train_inventories

    def _get_obs(self, index):
        states_embedding = torch.Tensor(self.train_states_embedding[index])
        states_onehot = torch.Tensor(self.train_states_onehot[index])
        goal = torch.Tensor(self.train_goals_embedding[index])
        inventory = torch.Tensor(self.train_inventory_embedding[index])
        
        return {'grid_embedding' : states_embedding, 'grid_onehot': states_onehot,
                'goal': goal, 'inventory': inventory}

    def __getitem__(self, index):

        obs = self._get_obs(index)
        action = torch.Tensor(self.train_actions_onehot[index])
        if self.skip > 0:
            # We have the skip
            next_index = min(index + self.skip, len(self) - 1)
            next_obs_dict = {"next_" + k: v for k,v in self._get_obs(next_index).items()}
            obs.update(next_obs_dict) # Update it with next

        try:
            temp_instruction = self.train_instructions[index]
            instruction = []
            instruction.append(self.vocab('<start>'))
            instruction.extend([self.vocab(token) for token in temp_instruction])
            instruction.append(self.vocab('<end>'))
            target = torch.Tensor(instruction)
        except:
            # If we error, return the next indices.
            return self.__getitem__(index + 10)
        
        obs['subgoal'] = target
        return obs, action
        
    def __len__(self):
        return len(self.train_states_embedding)

def collate_fn(data, vocab_size=217):
    data.sort(key=lambda x: len(x[0]['subgoal']), reverse=True)
    # The data has now been sorted by length
    obs = {}
    data_keys = [k for k in data[0][0].keys() if k != 'subgoal'] # Everything but subgoal gets stacked.
    for k in data_keys:
        obs[k] = torch.stack([data_pt[0][k] for data_pt in data], 0)
    
    # Now we need to do the different language attributes, which don't have a fixed length.
    lengths = [len(data_pt[0]['subgoal']) for data_pt in data]
    subgoals = (vocab_size-1)*torch.ones(len(data), max(lengths), dtype=torch.long) # NOTE: Vocab length is hardcoded here!!!!

    for i, data_pt in enumerate(data):
        end = lengths[i]
        subgoals[i, :end] = data_pt[0]['subgoal'][:end] # grab just the end.
    
    obs['label'] = subgoals
    obs['decode_lengths'] = [length - 1 for length in lengths]
    # Finally, do the actions
    actions = torch.cat([data_pt[1] for data_pt in data], 0)
    return (obs, actions) # Should be set! Wohoo
