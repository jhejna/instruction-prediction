import torch
from torch import nn
from torch import functionl as F

from .dt import GPTEncoder, Seq2SeqDecoder


class CraftingCNNStateEncoder(nn.Module):

    def __init__(self, embed_dim, output_dim, downsample=-1, inv_dim=48, hidden_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.onehot_fc = nn.Linear(7, embed_dim)
        if downsample > 0:
            self.downsample = nn.Linear(embed_dim, downsample)
            self.inventory = nn.Sequential(nn.ReLU(), nn.Linear(downsample, inv_dim))
        else:
            self.downsample = None
            self.inventory = nn.Linear(embed_dim, inv_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(embed_dim + inv_dim, hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
        )
        
    def forward(self, grid_embedding, grid_onehot, inventory):
        B, S = grid_embedding.shape[:2]
        onehot = self.onehot_fc(grid_onehot).view(B, S, 5, 5, self.embed_dim)
        state = grid_embedding + one_hot
        inventory = inventory.view(B, S, 1, 1, self.embed_dim)
        if self.downsample is not None:
            state = self.downsample(state)
            inventory = self.downsample(inventory)
        inventory = self.inventory(inventory) 
        # combine the two streams
        inventory = inventory.expand_as(state)
        x = torch.cat((state, inventory), dim=-1).view(B*S, 5, 5, -1)
        x = x.permute(0, 3, 1, 2)
        x = self.cnn(x)
        _, c, h, w = x.shape # Get the shape after
        x = x.view(B*S, c, h*w) # Flatten spatial dims
        x = torch.max(x, dim=-1)[0] # Run Spatial MaxPooling
        x = x.view(B, S, c)
        return x

        
class CraftingDT(nn.Module):
    
    def __init__(self, num_actions, vocab, embed_weights
                       n_head=2, n_embd=128, attn_pdrop=0.1, resid_pdrop=0.1, embd_pdrop=0.1, block_size=384, mlp_ratio=2, # Attention Params
                       vocab_size=40, n_layer=4, n_dec_layer=1, # Enc/Dec Params
                       unsup_dim=128, use_mask=True): # Unsup params
        super().__init__()

        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        embed_dim = 300
        # Setup the embeddings
        self.embedding = nn.Embedding(self.vocab_size, self.glove_embed_dim, self.vocab_size - 1)
        self.embedding.load_state_dict({'weight': torch.from_numpy(embed_weights)})
        self.embedding.weight.requires_grad = False

        self.cnn = CraftingCNNStateEncoder(embed_dim, n_embd, **extractor_kwargs)
        self.encoder = GPTEncoder(n_head=n_head, n_embd=n_embd, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, embd_pdrop=embd_pdrop, 
                                  block_size=block_size, mlp_ratio=mlp_ratio, n_layer=n_layer)
        self.action_head = nn.Linear(n_embd, num_actions)

        # Decoder
        if n_dec_layer > 0:
            self.decoder = Seq2SeqDecoder(n_head=n_head, n_embd=n_embd, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, embd_pdrop=embd_pdrop, 
                                          block_size=block_size, mlp_ratio=mlp_ratio, n_layer=n_dec_layer)
            self.vocab_head = nn.Linear(n_embd, self.vocab_size - 1, bias=False)
        else:
            self.decoder = None
        self.use_mask = use_mask

        # Init params before creating unsup head, which is initialized with pytorch default
        self.apply(self._init_weights)

    @property
    def action_pad_idx(self):
        return self.vocab_size - 1 # A consequence of the Crafting Dataset implementation.
    
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

    def forward(self, obs):
        grid_embedding = obs['grid_embedding']
        grid_onehot = obs['grid_onehot']
        inventory = obs['inventory']
        x = self.cnn(grid_embedding, grid_onehot, inventory)
        x = self.encoder(x)
        action_logits = self.action_head(x)
        if self.decoder is not None:
            target = self.embedding(obs['instructions'])
            lang_logits = self.vocab_head(self.decoder(target, x, mask=None))
        else:
            lang_logits = None
        return action_logits, lang_logits

