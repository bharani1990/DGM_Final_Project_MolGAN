# FULL MODEL
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
from train import *
from discriminator import *
from generator import *


class MolGen(nn.Module):

    def __init__(self, data, hidden_dim=128, lr=1e-3, device='cpu'):
        super().__init__()

        self.device = device
        self.hidden_dim = hidden_dim
        self.tokenizer = Tokenizer(data)

        self.generator = Generator(
            latent_dim=hidden_dim,
            vocab_size=self.tokenizer.vocab_size - 1,
            start_token=self.tokenizer.start_token - 1,  # no need token
            end_token=self.tokenizer.end_token - 1,
        ).to(device)

        self.discriminator = RecurrentDiscriminator(
            hidden_size=hidden_dim,
            vocab_size=self.tokenizer.vocab_size,
            start_token=self.tokenizer.start_token,
            bidirectional=True
        ).to(device)

        self.generator_optim = torch.optim.Adam(
            self.generator.parameters(), lr=lr)

        self.discriminator_optim = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr)

        self.b = 0.

    def sample_latent(self, batch_size):

        return torch.randn(batch_size, self.hidden_dim).to(self.device)

    def discriminator_loss(self, x, y):

        y_pred, mask = self.discriminator(x).values()
        loss = F.binary_cross_entropy(
            y_pred, y, reduction='none') * mask
        loss = loss.sum() / mask.sum()

        return loss

    def train_step(self, x):

        batch_size, len_real = x.size()
        x_real = x.to(self.device)
        y_real = torch.ones(batch_size, len_real).to(self.device)

        z = self.sample_latent(batch_size)
        generator_outputs = self.generator.forward(z, max_len=20)
        x_gen, log_probs, entropies = generator_outputs.values()

        _, len_gen = x_gen.size()
        y_gen = torch.zeros(batch_size, len_gen).to(self.device)

        # D Train
        self.discriminator_optim.zero_grad()
        fake_loss = self.discriminator_loss(x_gen, y_gen)
        real_loss = self.discriminator_loss(x_real, y_real)
        discr_loss = 0.5 * (real_loss + fake_loss)
        discr_loss.backward()
        clip_grad_value_(self.discriminator.parameters(), 0.1)
        self.discriminator_optim.step()

        # G Train
        self.generator_optim.zero_grad()
        y_pred, y_pred_mask = self.discriminator(x_gen).values()
        R = (2 * y_pred - 1)
        lengths = y_pred_mask.sum(1).long()
        list_rewards = [rw[:ln] for rw, ln in zip(R, lengths)]
        generator_loss = []
        for reward, log_p in zip(list_rewards, log_probs):
            reward_baseline = reward - self.b

            generator_loss.append((- reward_baseline * log_p).sum())

        generator_loss = torch.stack(generator_loss).mean() - \
                         sum(entropies) * 0.01 / batch_size

        with torch.no_grad():
            mean_reward = (R * y_pred_mask).sum() / y_pred_mask.sum()
            self.b = 0.9 * self.b + (1 - 0.9) * mean_reward
        generator_loss.backward()
        clip_grad_value_(self.generator.parameters(), 0.1)
        self.generator_optim.step()

        return {'loss_disc': discr_loss.item(), 'mean_reward': mean_reward}

    def create_dataloader(self, data, batch_size=128, shuffle=True, num_workers=5):

        return DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.tokenizer.batch_tokenize,
            num_workers=num_workers
        )

    def train_n_steps(self, train_loader, max_step=5000, evaluate_every=1000):

        iter_loader = iter(train_loader)
        for step in tqdm(range(max_step)):
            try:
                batch = next(iter_loader)
            except:
                iter_loader = iter(train_loader)
                batch = next(iter_loader)

            self.train_step(batch)
            if step % evaluate_every == 0:
                self.eval()
                score = self.evaluate_n(evaluate_every)
                self.train()
                print(f'valid = {score: .2f}')

    def get_mapped(self, seq):
        return ''.join([self.tokenizer.inv_mapping[i] for i in seq])

    @torch.no_grad()
    def generate_n(self, n):

        z = torch.randn((n, self.hidden_dim)).to(self.device)
        x = self.generator(z)['x'].cpu()
        lenghts = (x > 0).sum(1)
        return [self.get_mapped(x[:l - 1].numpy()) for x, l in zip(x, lenghts)]

    def evaluate_n(self, n):

        pack = self.generate_n(n)
        valid = np.array([Chem.MolFromSmiles(k) is not None for k in pack])

        return valid.mean()