# Tokenizer

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
from train import *
from full_model import *
from discriminator import *
from generator import *

class Tokenizer(object):

    def __init__(self, data):
        unique_char = list(set(''.join(data))) + ['<eos>'] + ['<sos>']
        self.mapping = {'<pad>': 0}
        for i, c in enumerate(unique_char, start=1):
            self.mapping[c] = i

        self.inv_mapping = {v: k for k, v in self.mapping.items()}
        self.start_token = self.mapping['<sos>']
        self.end_token = self.mapping['<eos>']
        self.vocab_size = len(self.mapping.keys())

    def encode_smile(self, mol, add_eos=True):
        out = [self.mapping[i] for i in mol]

        if add_eos:
            out = out + [self.end_token]
        return torch.LongTensor(out)

    def batch_tokenize(self, batch):

        out = map(lambda x: self.encode_smile(x), batch)
        return torch.nn.utils.rnn.pad_sequence(list(out), batch_first=True)