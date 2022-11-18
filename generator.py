# Generator Network
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
from discriminator import *
from train import *

class Generator(nn.Module):

    def __init__(self, latent_dim, vocab_size, start_token, end_token):

        super().__init__()

        self.vocab_size = vocab_size
        self.start_token = start_token
        self.end_token = end_token

        self.embedding_layer = nn.Embedding(self.vocab_size, latent_dim)

        self.project = FeedForward(
            input_dim=latent_dim,
            num_layers=2,
            hidden_dims=[latent_dim * 2, latent_dim * 2],
            activations=[nn.ReLU(), nn.ELU(alpha=0.1)],
            dropout=[0.1, 0.1]
        )

        self.rnn = nn.LSTMCell(latent_dim, latent_dim)

        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, vocab_size - 1)
        )

    def forward(self, z, max_len=20):

        batch_size = z.shape[0]

        starts = torch.full(
            size=(batch_size,), fill_value=self.start_token, device=z.device).long()
        emb = self.embedding_layer(starts)
        x = []
        log_probabilities = []
        entropies = []
        h, c = self.project(z).chunk(2, dim=1)

        for i in range(max_len):
            h, c = self.rnn(emb, (h, c))
            logits = self.output_layer(h)
            dist = Categorical(logits=logits)
            sample = dist.sample()
            x.append(sample)
            log_probabilities.append(dist.log_prob(sample))
            entropies.append(dist.entropy())
            emb = self.embedding_layer(sample)

        x = torch.stack(x, dim=1)
        log_probabilities = torch.stack(log_probabilities, dim=1)
        entropies = torch.stack(entropies, dim=1)

        end_pos = (x == self.end_token).float().argmax(dim=1).cpu()
        seq_lengths = end_pos + 1
        seq_lengths.masked_fill_(seq_lengths == 1, max_len)
        _x = []
        _log_probabilities = []
        _entropies = []
        for x_i, logp, ent, length in zip(x, log_probabilities, entropies, seq_lengths):
            _x.append(x_i[:length])
            _log_probabilities.append(logp[:length])
            _entropies.append(ent[:length].mean())

        x = torch.nn.utils.rnn.pad_sequence(
            _x, batch_first=True, padding_value=-1)

        x = x + 1

        return {'x': x, 'log_probabilities': _log_probabilities, 'entropies': _entropies}