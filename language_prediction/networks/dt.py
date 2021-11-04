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

    def __init__(self, dim):
        self.cnn = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.Conv2d(64, dim, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(dim),
                        nn.ReLU(),
            )
    def forward(self, x):
        x = self.cnn(x)
        B, S, c, h, w = x.shape # Get the shape after
        x = x.view(B*S, c, h*w) # Flatten spatial dims
        x = torch.max(x, dim=-1)[0] # Run Spatial MaxPooling
        return x

class DT(nn.Module):
    '''
    Decision Transformer with extra modeling components for language and unsupervised learning.
    '''

    def __init__(self, num_actions, 
                       n_head=2, n_embd=128, attn_pdrop=0.1, resid_pdrop=0.1, embd_pdrop=0.1, block_size=384, mlp_ratio=2, # Attention Params
                       vocab_size=40, n_layer=4, n_decoder_layer=1, # Enc/Dec Params
                       unsup_dim=128, use_mask=True): # Unsup params
        super().__init__()

        self.cnn = BabyAIFeatureExtractor(n_embd)
        self.text_emb = nn.Embedding(vocab_size, n_embd, padding_idx=0) # TODO: import padding idx
        self.encoder = GPTEncoder(n_head=n_head, n_embd=n_embd, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, embd_pdrop=embd_pdrop, 
                                  block_size=block_size, mlp_ratio=mlp_ratio, n_layer=n_layer)
        self.action_head = nn.Linear(n_embd, num_actions)

        # Decoder
        if n_decoder_layer > 0:
            self.decoder = Seq2SeqDecoder(n_head=n_head, n_embd=n_embd, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, embd_pdrop=embd_pdrop, 
                                          block_size=block_size, mlp_ratio=mlp_ratio, n_layer=n_decoder_layer)
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
    def get_action_pad_idx(self):
        return -100
    
    @property
    def get_lang_pad_idx(self):
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
        mission = self.tok_emb(mission)
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
            lang_logits = self.decoder(target, x, mask=mask)
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
        action_logits, _, _ = self(combined_obs, skip_size=-1)
        # We only care about the last timestep action logits
        action_logits = action_logits[0, -1, :]
        action = torch.argmax(action_logits).item()
        return action # return the predicted action.




class DT(nn.Module):

    def __init__(self, num_actions, n_head=2, n_embd=128, attn_pdrop=0.1, resid_pdrop=0.1, 
                       block_size=384, mlp_ratio=2, n_layer=4,
                       vocab_size=40, embd_pdrop=0.1):
        super().__init__()
        # Encode each trajectory into
        self.cnn = nn.Sequential(
                                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(),
                                nn.Conv2d(64, n_embd, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(n_embd),
                                nn.ReLU(),
                                )
        self.n_embd = n_embd

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.drop = nn.Dropout(embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[EncoderBlock(n_head, n_embd, attn_pdrop, resid_pdrop, block_size, mlp_ratio) 
                                        for _ in range(n_layer)])
                                        
        # self.action_head = nn.Sequential(nn.Linear(n_embd, 128), nn.ReLU(), nn.Linear(128, num_actions))
        self.action_head = nn.Linear(n_embd, num_actions)

        self.block_size = block_size
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, obs, labels=None, skip_size=-1):
        if skip_size > 0:
            assert not 'mission' in obs, "Using a skip and the mission was provided!"
        
        B, S, c, h, w = obs['image'].shape
        img = obs['image'].view(B*S, c, h, w) # Reshape it
        img = self.cnn(img.float())
        img = torch.max(img.view(B*S, self.n_embd, h*w), dim=-1)[0]
        img = img.view(B, S, self.n_embd) # Now we have the image sequence
        
        if skip_size > 0:
            position_embeddings = self.pos_emb[:, skip_size:skip_size+img.shape[1], :]
            x = self.drop(img + position_embeddings)
            x = self.blocks(x)
            # cannot return actions if we are skipping the mission
            actions = None
        else:
            mission = self.tok_emb(obs['mission'])
            # Now concatenate everything together
            x = torch.cat((mission, img), dim=1)
            # Need to be careful with blocksize
            position_embeddings = self.pos_emb[:, :x.shape[1], :]
            x = self.drop(x + position_embeddings)
            x = self.blocks(x)
            actions = self.action_head(x[:, obs['mission'].shape[1]:, :]) # This gives the length
        
        return actions, x # This gives the latents.

    def predict(self, obs, deterministic=True, history=None):
        assert not history is None
        assert history['image'].shape[0] == 1, "Only a batch size of 1 is currently supported"
        # Run the model on the observation
        # We only want to send in the mission once. Can use the current timestep one.
        combined_obs = {'image': history['image'], 'mission': obs['mission']}
        action_logits, _ = self(combined_obs, skip_size=-1)
        # We only care about the last timestep action logits
        action_logits = action_logits[0, -1, :]
        action = torch.argmax(action_logits).item()
        return action # return the predicted action.

class Seq2SeqDT(nn.Module):

    def __init__(self, num_actions, n_head=2, n_embd=128, attn_pdrop=0.1, resid_pdrop=0.1, 
                       block_size=384, mlp_ratio=2, n_enc_layer=4, n_dec_layer=1, embd_pdrop=0.1, vocab_size=40, 
                       use_mask=True, unsup_dim=128, unsup_proj='mat'):
        super().__init__()
        self.dt = DT(num_actions, n_head=n_head, n_embd=n_embd, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop,
                    block_size=block_size, mlp_ratio=mlp_ratio, n_layer=n_enc_layer, 
                    vocab_size=vocab_size, embd_pdrop=embd_pdrop)
        
        self.pos_emb_trg = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.drop = nn.Dropout(embd_pdrop)
        self.decoder = nn.ModuleList([DecoderBlock(n_head, n_embd, attn_pdrop, resid_pdrop, block_size, mlp_ratio) 
                                        for _ in range(n_dec_layer)])
        self.use_mask = use_mask
        self.vocab_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.decoder.apply(self.dt._init_weights) # apply the proper initialization for the transformer.

        # For unsupervised
        self.unsup_head = nn.Linear(n_embd, unsup_dim)
        if unsup_proj == 'mat':
            self._unsup_proj = nn.Parameter(torch.rand(unsup_dim, unsup_dim))
        else:
            self._unsup_proj = None
        self.unsup_mlp = nn.Sequential(nn.Linear(unsup_dim, 2*unsup_dim), nn.ReLU(), nn.Linear(2*unsup_dim, unsup_dim))

    def forward(self, obs, labels=None, is_target=False):
        if is_target:
            obs = {k: v for k, v in obs.items() if k != 'image'}
            obs['image'] = obs['next_image'] # Change the image to the next image.
        actions, latents = self.dt(obs)

        # Now pass the latents into the decoder
        if not labels is None:
            tgt = self.dt.tok_emb(labels[:, :-1]) # Ignore the last dim
            position_embeddings = self.pos_emb_trg[:, :tgt.shape[1], :]
            tgt = self.drop(tgt + position_embeddings)
            # Modify the mask to support the size of the img sequence
            B, T, _ = tgt.shape
            if self.use_mask:
                mission_mask = torch.ones(B, T, obs['mission'].shape[1], device=obs['mission'].device)
                mask = torch.cat((mission_mask, obs['mask'][:, :-1]), dim=2)
            else:
                mask = None
            # Everyone can attend to all the mission positions. Padding oopsies
            for layer in self.decoder:
                tgt = layer(tgt, latents, cross_attn_mask=mask)
            lang_logits = self.vocab_head(tgt)
            lang_aux = F.cross_entropy(lang_logits.reshape(-1, lang_logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=0)
        else:
            lang_aux = None
        # Now run the unsupervised prediction
        unsup_logits = self.unsup_head(latents[:, obs['mission'].shape[1]:, :]) # Must remove the mission latents
        if not is_target:
            unsup_logits = self.unsup_mlp(unsup_logits) # Forward through the projection MLP
        return actions, lang_aux, unsup_logits

    @property
    def unsup_proj(self):
        return self._unsup_proj

    def predict(self, obs, deterministic=True, history=None, **kwargs):
        return self.dt.predict(obs, deterministic=deterministic, history=history)

    def generate_instr(self, obs, history=None, mask=None):
        device = torch.device("cpu")
        if isinstance(obs, dict):
            obs = {k: torch.from_numpy(v).to(device).unsqueeze(0) if isinstance(v, np.ndarray) else v for k,v in obs.items()}
            if not history is None:
                history = {k: torch.from_numpy(v).to(device).unsqueeze(0) if isinstance(v, np.ndarray) else v for k,v in history.items()}
        else:
            obs = torch.from_numpy(obs).to(device).unsqueeze(0)
            if not history is None:
                history = torch.from_numpy(history).to(device).unsqueeze(0)
       

        combined_obs = {'image': history['image'], 'mission': obs['mission']}

        max_plan_tokens = 100
        from language_prediction.envs.babyai_wrappers import WORD_TO_IDX
        plan = WORD_TO_IDX['END_MISSION'] * torch.ones(1, 1, dtype=torch.long, device=obs['mission'].device)
        _, language_latents = self.dt(combined_obs, skip_size=-1)
        
        if not mask is None:
            mask = torch.from_numpy(mask).to(device).unsqueeze(0)
            pad_mask = torch.ones(1, max_plan_tokens, language_latents.shape[1])
            pad_mask[0, :mask.shape[1], :mask.shape[2]] = mask[:, :max_plan_tokens, :language_latents.shape[1]]
            mask = pad_mask
        # The mask is shape T, S


        for i in range(1, max_plan_tokens):
            tgt = self.dt.tok_emb(plan)
            position_embeddings = self.pos_emb_trg[:, :tgt.shape[1], :]
            tgt = self.drop(tgt + position_embeddings)
            for layer in self.decoder:
                mask = None if mask is None else mask[:, :tgt.shape[1]]
                tgt = layer(tgt, language_latents, cross_attn_mask=mask) # can see everything during inference
            lang_logits = self.vocab_head(tgt)
            lang_logits = lang_logits[:, -1, :] # Get the logits at the last timestep
            _, ix = torch.topk(lang_logits, k=1, dim=-1)
            plan = torch.cat((plan, ix), dim=1)
            # Check to see if the index was something we should break on.
            if ix.item() == WORD_TO_IDX['END']:
                break # we can break

        plan = plan.detach().cpu().numpy()[0]
        print(plan)
        IDX_TO_WORD = {v:k for k,v in WORD_TO_IDX.items()}
        print([IDX_TO_WORD[idx] for idx in plan])
