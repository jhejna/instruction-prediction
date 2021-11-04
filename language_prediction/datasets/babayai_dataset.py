import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle

class BabyAITrajectoryDataset(Dataset):

    def __init__(self, images, missions, subgoals, actions, masks=None, next_images=None):
        self.images = images
        self.missions = missions
        self.subgoals = subgoals
        self.actions = actions

        if not masks is None:
            self.masks = masks
        else:
            self.masks = None

        if not next_images is None:
            self.next_images = next_images
        else:
            self.next_images = None

    def __getitem__(self, index):
        img = torch.from_numpy(self.images[index])
        mission = torch.from_numpy(self.missions[index])
        subgoal = torch.from_numpy(self.subgoals[index])
        action = torch.from_numpy(self.actions[index])
        
        if not self.next_images is None:
            next_img = torch.from_numpy(self.next_images[index])
        else:
            next_img = None

        if not self.masks is None:
            mask = torch.from_numpy(self.masks[index])
        else:
            mask = None

        # Trim so we don't run out of block size.
        max_length = 450
        if subgoal.shape[0] > max_length:
            subgoal = subgoal[:max_length]
            mask = mask[:max_length, :]
        if img.shape[0] > max_length:
            img = img[:max_length]
            action = action[:max_length]
            if not self.masks is None:
                mask = mask[:, :max_length]
            if not self.next_images is None:
                next_img = next_img[:max_length]

        return img, mission, mask, subgoal, action, next_img

    def __len__(self):
        return len(self.images)

    def save(self, path):
        dataset = {
            'images' : self.images,
            'missions' : self.missions,
            'subgoals': self.subgoals,
            'actions' : self.actions,
            'masks': self.masks,
            'next_images': self.next_images
        }
        with open(path + ".pkl", 'wb') as f:
            pickle.dump(dataset, f)
        
    @classmethod
    def load(cls, path, fraction=1):
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
        if fraction < 1.0:
            # Load only a fraction of the dataset
            num_data_pts = int(fraction*len(dataset['images']))
            dataset = {k:v[:num_data_pts] for k,v in dataset.items() if not v is None}
        if not 'next_images' in dataset:
            dataset['next_images'] = None # short hack
        if not 'masks' in dataset:
            dataset['masks'] = None
        return cls(**dataset)

    @classmethod
    def merge(cls, datasets):
        images, missions, subgoals, masks, actions, next_images = [], [], [], [], [], []
        for dataset in datasets:
            images.extend(dataset.images)
            missions.extend(dataset.missions)
            subgoals.extend(dataset.subgoals)
            actions.extend(dataset.actions)
            if not dataset.masks is None:
                masks.extend(dataset.masks)
            if not dataset.next_images is None:
                next_images.extend(dataset.next_images)
        if len(masks) == 0:
            masks = None
        if len(next_images) == 0:
            next_images = None
        return cls(images, missions, subgoals, actions, masks=masks, next_images=next_images)

def traj_collate_fn(batch):
    images, missions, masks, subgoals, actions, next_images = zip(*batch)
    # Fixed length items
    missions = torch.stack(missions, dim=0)
    # Determine the size of the batch via pad_sequence
    images = pad_sequence(images, batch_first=True, padding_value=0)
    actions = pad_sequence(actions, batch_first=True, padding_value=-100)
    subgoals = pad_sequence(subgoals, batch_first=True, padding_value=0).long()

    obs = {'image': images, 'mission': missions, 'label': subgoals, }

    if not next_images[0] is None:
        next_images = pad_sequence(next_images, batch_first=True, padding_value=0)
        obs['next_image'] = next_images

    if not masks[0] is None:
        # This will need to be done in a for loop to expand the mask.
        B, S, T = images.shape[0], images.shape[1], subgoals.shape[1]
        mask_tensor = torch.zeros(B, T, S, dtype=torch.bool)
        for i, mask in enumerate(masks):
            t, s = mask.shape[0], mask.shape[1]
            mask_tensor[i, :t, :s] = mask
        obs['mask'] = mask_tensor

    return obs, actions
