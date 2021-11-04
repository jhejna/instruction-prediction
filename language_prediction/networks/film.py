import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# From https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

# Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871
class FiLM(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=imm_channels,
            kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(imm_channels)
        self.conv2 = nn.Conv2d(
            in_channels=imm_channels, out_channels=out_features,
            kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        weight = self.weight(y).unsqueeze(2).unsqueeze(3)
        bias = self.bias(y).unsqueeze(2).unsqueeze(3)
        out = x * weight + bias
        return F.relu(self.bn2(out))

class BabyAIModel(nn.Module):

    def __init__(self, num_actions, instr_dim=128, image_dim=128, memory_dim=128,
                        bidirectional=False, res=True):
        super().__init__()

        self.instr_dim = instr_dim
        self.image_dim = image_dim
        self.memory_dim = memory_dim
        self.res = res
        self.bidirectional = bidirectional
        
        self.image_conv = nn.Sequential(*[
            nn.Conv2d(3, out_channels=128, kernel_size=(2, 2), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        ])

        self.word_embedding = nn.Embedding(40, self.instr_dim)
        if bidirectional:
            gru_dim = self.instr_dim // 2
        else:
            gru_dim = self.instr_dim
        self.instr_rnn = nn.GRU(self.instr_dim, gru_dim, batch_first=True, bidirectional=bidirectional)
        
        num_module = 2
        self.controllers = []
        for ni in range(num_module):
            mod = FiLM(
                in_features=self.instr_dim,
                out_features=128 if ni < num_module-1 else self.image_dim,
                in_channels=128, imm_channels=128)
            self.controllers.append(mod)
            self.add_module('FiLM_' + str(ni), mod)
        
        self.memory = nn.LSTM(self.image_dim, self.memory_dim, batch_first=True)

        self.head = nn.Sequential(nn.Linear(self.image_dim, 64),
                                  nn.Tanh(),
                                  nn.Linear(64, num_actions))
        
        self.apply(initialize_parameters)

    def forward(self, obs, instructions=None, is_target=False):
        text = obs['mission'].long()
        lengths = (text != 0).sum(1).long()
        rnn_output, rnn_states = self.instr_rnn(self.word_embedding(text))
        if self.bidirectional:
            text = rnn_states
            text = text.transpose(0, 1).contiguous() # See GRU docs for better explanation
            text = text.view(text.shape[0], -1)
        else:
            text = rnn_output[range(len(lengths)), lengths-1, :] # Should be shape (B, C)
        # Need to reshape the text to make its shape better.

        B, S, c, h, w = obs['image'].shape
        x = obs['image'].view(B*S, c, h, w) # Reshape it
        x = self.image_conv(x.float())
        # Reshape the text so it works with the new dimensions
        text = text.unsqueeze(1).expand(-1, S, -1).reshape(B*S, -1)

        for controller in self.controllers:
            out = controller(x, text)
            if self.res:
                out += x
            x = out
        _, c, h, w = x.shape
        # Run pooling then ReLU, this is how it was done in their code.
        x = x.view(B*S, c, h*w).max(dim=-1)[0]
        x = F.relu(x)
        # Now reshape back to channels
        x = x.view(B, S, c)
        x, _ = self.memory(x)

        return self.head(x), None, None

    def predict(self, obs, deterministic=True, history=None):
        assert not history is None
        assert history['image'].shape[0] == 1, "Only a batch size of 1 is currently supported"
        # Run the model on the observation
        # We only want to send in the mission once. Can use the current timestep one.
        combined_obs = {'image': history['image'], 'mission': obs['mission']}
        action_logits, _, _ = self(combined_obs)
        # We only care about the last timestep action logits
        action_logits = action_logits[0, -1, :]
        action = torch.argmax(action_logits).item()
        return action # return the predicted action.
