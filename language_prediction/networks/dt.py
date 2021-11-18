import torch
import torch.nn as nn
from torch.nn import functional as F
import random

import math
import numpy as np

from language_prediction.envs.babyai_wrappers import WORD_TO_IDX

''' 
The general idea of this is the following:
1. Have a causual decision transformer for outputting actions at each step.
2. Feed the top layers to a decision transformer
'''

class Attention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_head, n_embd, attn_pdrop, resid_pdrop, block_size, causal=False):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.causal = causal
        if self.causal:
            self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))
        else:
            self.mask = None
        self.n_head = n_head

    def forward(self, q, k, v, mask=None):
        B, T, C = q.size()
        _, S, C = k.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(q).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = self.key(k).view(B, S, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, S, hs)
        v = self.value(v).view(B, S, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, S, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, S) -> (B, nh, T, S)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if not self.mask is None:
            att = att.masked_fill(self.mask[:,:,:T,:S] == 0, float('-inf'))
        if not mask is None: # Additional mask if provided in user input
            att = att.masked_fill(mask.unsqueeze(1) == 0, float('-inf')) # TODO: complete
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, S) x (B, nh, S, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class EncoderBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_head, n_embd, attn_pdrop, resid_pdrop, block_size, mlp_ratio):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = Attention(n_head, n_embd, attn_pdrop, resid_pdrop, block_size, causal=True)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_ratio * n_embd),
            nn.GELU(),
            nn.Linear(mlp_ratio * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, mask=None):
        ln_x = self.ln1(x)
        x = x + self.attn(ln_x, ln_x, ln_x, mask=mask)
        x = x + self.mlp(self.ln2(x))
        return x

class DecoderBlock(nn.Module):

    def __init__(self, n_head, n_embd, attn_pdrop, resid_pdrop, block_size, mlp_ratio):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.self_attn = Attention(n_head, n_embd, attn_pdrop, resid_pdrop, block_size, causal=True)
        self.cross_attn = Attention(n_head, n_embd, attn_pdrop, resid_pdrop, block_size, causal=False)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_ratio * n_embd),
            nn.GELU(),
            nn.Linear(mlp_ratio * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, trg, src, cross_attn_mask=None):
        ln_trg = self.ln1(trg)
        x = trg + self.self_attn(ln_trg, ln_trg, ln_trg, mask=None) # No mask, but it is causal attention
        x = x + self.cross_attn(self.ln2(x), src, src, mask=cross_attn_mask)
        x = x + self.mlp(self.ln3(x))
        return x

class GPTEncoder(nn.Module):

    def __init__(self, n_head=2, n_embd=128, attn_pdrop=0.1, resid_pdrop=0.1, block_size=384, mlp_ratio=2, n_layer=4, embd_pdrop=0.1):
        super().__init__()
        self.n_embd = n_embd
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.blocks = nn.Sequential(*[EncoderBlock(n_head, n_embd, attn_pdrop, resid_pdrop, block_size, mlp_ratio) 
                                        for _ in range(n_layer)])
        self.drop = nn.Dropout(embd_pdrop)

    def forward(self, x):
        position_embeddings = self.pos_emb[:, :x.shape[1], :]
        x = self.drop(x + position_embeddings)
        x = self.blocks(x)
        return x

class Seq2SeqDecoder(nn.Module):

    def __init__(self, n_head=2, n_embd=128, attn_pdrop=0.1, resid_pdrop=0.1, block_size=384, mlp_ratio=2, n_layer=2, embd_pdrop=0.1):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.drop = nn.Dropout(embd_pdrop)
        self.blocks = nn.ModuleList([DecoderBlock(n_head, n_embd, attn_pdrop, resid_pdrop, block_size, mlp_ratio) 
                                        for _ in range(n_layer)])

    def forward(self, targets, source, mask=None):
        position_embeddings = self.pos_emb[:, :targets.shape[1], :]
        targets = self.drop(targets + position_embeddings)
        for layer in self.blocks:
                targets = layer(targets, source, cross_attn_mask=mask)
        return targets

class BabyAIFeatureExtractor(nn.Module):
    '''
    Feature extractor from BabyAI.

    Note that this is designed to work on sequence data.
    '''

    def __init__(self, dim):
        super().__init__()
        self.cnn = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.Conv2d(64, dim, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(dim),
                        nn.ReLU(),
            )
    def forward(self, x):
        '''
        x: Float tensor of shape (B, S, c, h, w)
        output: Float tensor of shape (B, S, c)
        '''
        B, S, c, h, w = x.shape
        x = x.view(B*S, c, h, w) # Flatten
        x = self.cnn(x)
        _, c, h, w = x.shape # Get the shape after
        x = x.view(B*S, c, h*w) # Flatten spatial dims
        x = torch.max(x, dim=-1)[0] # Run Spatial MaxPooling
        x = x.view(B, S, c)
        return x

class DT(nn.Module):
    '''
    Decision Transformer with extra modeling components for language and unsupervised learning.
    '''

    def __init__(self, num_actions, 
                       n_head=2, n_embd=128, attn_pdrop=0.1, resid_pdrop=0.1, embd_pdrop=0.1, block_size=384, mlp_ratio=2, # Attention Params
                       vocab_size=40, n_layer=4, n_dec_layer=1, # Enc/Dec Params
                       unsup_dim=128, use_mask=True): # Unsup params
        super().__init__()

        self.cnn = BabyAIFeatureExtractor(n_embd)
        self.text_emb = nn.Embedding(vocab_size, n_embd, padding_idx=0) # TODO: import padding idx
        self.encoder = GPTEncoder(n_head=n_head, n_embd=n_embd, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, embd_pdrop=embd_pdrop, 
                                  block_size=block_size, mlp_ratio=mlp_ratio, n_layer=n_layer)
        self.action_head = nn.Linear(n_embd, num_actions)

        # Decoder
        if n_dec_layer > 0:
            self.decoder = Seq2SeqDecoder(n_head=n_head, n_embd=n_embd, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, embd_pdrop=embd_pdrop, 
                                          block_size=block_size, mlp_ratio=mlp_ratio, n_layer=n_dec_layer)
            self.vocab_head = nn.Linear(n_embd, vocab_size, bias=False)
        else:
            self.decoder = None
        self.use_mask = use_mask

        # Init params before creating unsup head, which is initialized with pytorch default
        self.apply(self._init_weights)

        # Unsupervised Head
        if unsup_dim > 0:
            self._unsup_proj = nn.Parameter(torch.rand(unsup_dim, unsup_dim))
            self.unsup_head = nn.Linear(n_embd, unsup_dim)
            self.unsup_mlp = nn.Sequential(nn.Linear(unsup_dim, 2*unsup_dim), nn.ReLU(), nn.Linear(2*unsup_dim, unsup_dim)) # Forward MLP for ATC.
        else:
            self._unsup_proj = None
            self.unsup_head = None
            self.unsup_mlp = None
        
    @property
    def unsup_proj(self):
        # Property return for use in the training alg.
        return self._unsup_proj

    @property
    def action_pad_idx(self):
        return -100
    
    @property
    def lang_pad_idx(self):
        return WORD_TO_IDX['PAD']

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, obs, instructions=None, is_target=False):
        '''
        obs: observation, expected to be dict with keys: image, mission, and optionally mask.
        labels: this contains the langauge instructions.
        is_target: whether or not this is the target network. Used to swap image with next_image
        '''
        img = obs['next_image'] if is_target else obs['image']
        mission = obs['mission']
        mask = obs['mask'] if 'mask' in obs else None

        img = self.cnn(img.float())
        mission = self.text_emb(mission)
        x = torch.cat((mission, img), dim=1)
        x = self.encoder(x)
        action_logits = self.action_head(x[:, mission.shape[1]:, :])

        if instructions is not None and self.decoder is not None:
            # We run language prediction, but first we need to append to the mask according to the mission length.
            if self.use_mask:
                B, T = instructions.shape
                mission_mask = torch.ones(B, T, mission.shape[1], device=mission.device)
                mask = torch.cat((mission_mask, mask), dim=2)
            else:
                mask = None
            target = self.text_emb(instructions)
            lang_logits = self.vocab_head(self.decoder(target, x, mask=mask))
        else:
            lang_logits = None

        if self.unsup_head is not None:
            # Run unsupervised prediction
            unsup_logits = self.unsup_head(x[:, mission.shape[1]:, :]) # Must remove the mission latents
            if not is_target:
                unsup_logits = self.unsup_mlp(unsup_logits) # Forward through the projection MLP if non-target as in ATC.
        else:
            unsup_logits = None

        return action_logits, lang_logits, unsup_logits

    def predict(self, obs, deterministic=True, history=None):
        assert not history is None
        assert history['image'].shape[0] == 1, "Only a batch size of 1 is currently supported"
        # Run the model on the observation
        # We only want to send in the mission once. Can use the current timestep one.
        combined_obs = {'image': history['image'], 'mission': obs['mission']}
        action_logits, _, _ = self(combined_obs, instructions=None, is_target=False)
        # We only care about the last timestep action logits
        action_logits = action_logits[0, -1, :]
        action = torch.argmax(action_logits).item()
        return action # return the predicted action.


class DTForward(nn.Module):
    '''
    Decision Transformer with extra modeling components for language and unsupervised learning.
    '''

    def __init__(self, num_actions, 
                       n_head=2, n_embd=128, attn_pdrop=0.1, resid_pdrop=0.1, embd_pdrop=0.1, block_size=384, mlp_ratio=2, # Attention Params
                       vocab_size=40, n_layer=4, n_dec_layer=1, # Enc/Dec Params
                       unsup_dim=128, reuse_action_head=False): # Unsup params
        super().__init__()

        self.cnn = BabyAIFeatureExtractor(n_embd)
        self.text_emb = nn.Embedding(vocab_size, n_embd, padding_idx=0) # TODO: import padding idx
        self.encoder = GPTEncoder(n_head=n_head, n_embd=n_embd, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, embd_pdrop=embd_pdrop, 
                                  block_size=block_size, mlp_ratio=mlp_ratio, n_layer=n_layer)
        self.action_head = nn.Linear(n_embd, num_actions)
        self.num_actions = num_actions

        # Decoder
        if n_dec_layer > 0:
            self.action_emb = nn.Embedding(num_actions + 1, n_embd, padding_idx=num_actions)
            self.decoder = Seq2SeqDecoder(n_head=n_head, n_embd=n_embd, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, embd_pdrop=embd_pdrop, 
                                          block_size=block_size, mlp_ratio=mlp_ratio, n_layer=n_dec_layer)
            if reuse_action_head:
                self.decoder_action_head = self.action_head
            else:
                self.decoder_action_head = nn.Linear(n_embd, num_actions)
        else:
            self.decoder = None

        # Init params before creating unsup head, which is initialized with pytorch default
        self.apply(self._init_weights)

        # Unsupervised Head
        if unsup_dim > 0:
            self._unsup_proj = nn.Parameter(torch.rand(unsup_dim, unsup_dim))
            self.unsup_head = nn.Linear(n_embd, unsup_dim)
            self.unsup_mlp = nn.Sequential(nn.Linear(unsup_dim, 2*unsup_dim), nn.ReLU(), nn.Linear(2*unsup_dim, unsup_dim)) # Forward MLP for ATC.
        else:
            self._unsup_proj = None
            self.unsup_head = None
            self.unsup_mlp = None
        
    @property
    def unsup_proj(self):
        # Property return for use in the training alg.
        return self._unsup_proj

    @property
    def action_pad_idx(self):
        return -100
    
    @property
    def lang_pad_idx(self):
        return WORD_TO_IDX['PAD']

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, obs, forward_inputs=None, is_target=False):
        '''
        obs: observation, expected to be dict with keys: image, mission, and optionally mask.
        labels: this contains the langauge instructions.
        is_target: whether or not this is the target network. Used to swap image with next_image
        '''
        img = obs['next_image'] if is_target else obs['image']
        mission = obs['mission']
        mask = obs['mask'] if 'mask' in obs else None

        img = self.cnn(img.float())
        mission = self.text_emb(mission)
        x = torch.cat((mission, img), dim=1)
        x = self.encoder(x)
        action_logits = self.action_head(x[:, mission.shape[1]:, :])

        if forward_inputs is not None and self.decoder is not None:
            # We run language prediction, but first we need to append to the mask according to the mission length.
            B, S, _ = img.shape
            T = forward_inputs.shape[1]
            skip = int(S/T)
            with torch.no_grad():
                mask = torch.zeros(B, T, S, device=mission.device, dtype=torch.bool)
                for i in range(T):
                    mask[:, i, 1+skip*i] = True
                mission_mask = torch.ones(B, T, mission.shape[1], device=mission.device, dtype=torch.bool)
                mask = torch.cat((mission_mask, mask), dim=2)
            forward_inputs = forward_inputs.clone()
            forward_inputs[forward_inputs == -100] = self.num_actions
            target = self.action_emb(forward_inputs)
            forward_logits = self.decoder_action_head(self.decoder(target, x, mask=mask))
        else:
            forward_logits = None

        if self.unsup_head is not None:
            # Run unsupervised prediction
            unsup_logits = self.unsup_head(x[:, mission.shape[1]:, :]) # Must remove the mission latents
            if not is_target:
                unsup_logits = self.unsup_mlp(unsup_logits) # Forward through the projection MLP if non-target as in ATC.
        else:
            unsup_logits = None

        return action_logits, forward_logits, unsup_logits

    def predict(self, obs, deterministic=True, history=None):
        assert not history is None
        assert history['image'].shape[0] == 1, "Only a batch size of 1 is currently supported"
        # Run the model on the observation
        # We only want to send in the mission once. Can use the current timestep one.
        combined_obs = {'image': history['image'], 'mission': obs['mission']}
        action_logits, _, _ = self(combined_obs, instructions=None, is_target=False)
        # We only care about the last timestep action logits
        action_logits = action_logits[0, -1, :]
        action = torch.argmax(action_logits).item()
        return action # return the predicted action.