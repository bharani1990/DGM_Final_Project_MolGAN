
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
from generator import *

data = []
with open('qm9.csv', "r") as f:
    for line in f.readlines()[1:]:
        data.append(line.split(",")[1])

cuda = 'cuda:0'
device = torch.device(cuda if torch.cuda.is_available() else "cpu")
gan_mol = MolGen(data, hidden_dim=64, lr=1e-3, device=device)

loader = gan_mol.create_dataloader(data, batch_size=64, shuffle=True, num_workers=10)
gan_mol.train_n_steps(loader, max_step=5000, evaluate_every=1000)