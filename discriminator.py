# Discriminator Network
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from torch import nn
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from torch.nn.modules.activation import Sigmoid
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2seq_encoders import (LstmSeq2SeqEncoder,PytorchTransformer)
from tqdm import tqdm
from tokenizer import *
from full_model import *
from train import *
from generator import *


class RecurrentDiscriminator(nn.Module):

    def __init__(self, hidden_size, vocab_size, start_token, bidirectional=True):

        super().__init__()
        self.start_token = start_token
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.rnn = LstmSeq2SeqEncoder(hidden_size, hidden_size, num_layers=1, bidirectional=bidirectional)
        if bidirectional:
            hidden_size = hidden_size * 2

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, _ = x.size()
        starts = torch.full(
            size=(batch_size, 1), fill_value=self.start_token, device=x.device).long()
        x = torch.cat([starts, x], dim=1)
        mask = x > 0
        emb = self.embedding(x)
        x = self.rnn(emb, mask)
        out = self.fc(x).squeeze(-1)
        return {'out': out[:, 1:], 'mask': mask.float()[:, 1:]}