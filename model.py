import torch
from torch import nn
import math
import torch.nn.functional as F
from train import train_loop, TrainingDataset, TrainingDataloader
from utils import Config, set_seed

class AttentionHead(nn.Module):

    def __init__(self, d_k, d_model, seq_length):
        super().__init__()

        self.d_k = d_k
        self.seq_length = seq_length
        self.wq = nn.Linear(in_features=d_model, out_features=d_k)
        self.wk = nn.Linear(in_features=d_model, out_features=d_k)
        self.wv = nn.Linear(in_features=d_model, out_features=d_k)

    def forward(self, x):
        # B (batch_size), T (seq_length), C (d_model)
        B, T, C = x.shape

        # (batch_size, seq_length, d_model) -> (batch_size, seq_length, d_k)
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # (batch_size, seq_length, d_k) * (batch_size, d_k, seq_length)
        # -> (batch_size, seq_length, seq_length)
        qkt = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)

        # apply mask (seq_length*seq_length) now
        mask = torch.tril((torch.ones(self.seq_length, self.seq_length) == 1)) == False
        qkt = qkt.masked_fill(mask, -torch.inf)

        sm = F.softmax(qkt, dim=-1)

        # (batch_size, seq_length, seq_length) * (batch_size, seq_length, d_model)
        # -> (batch_size, seq_length, d_model)
        att = sm @ v

        return att

class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, seq_length):
        super().__init__()

        self.h = h
        self.d_model = d_model
        self.seq_length = seq_length
        self.d_k = int(d_model // h)

        self.mheads = nn.ModuleList([AttentionHead(self.d_k, d_model, seq_length) for i in range(h)])
        self.wo = nn.Linear(in_features=d_model, out_features=d_model)

    def forward(self, x):
        yh_cat = self.mheads[0](x)
        for i in range(1, self.h):
            yhi = self.mheads[i](x)
            yh_cat = torch.cat((yh_cat, yhi), -1)

        y = self.wo(yh_cat)

        return y

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=4 * d_model),
            nn.GELU(),
            nn.Linear(in_features=4 * d_model, out_features=d_model),
        )

    def forward(self, x):
        return (self.ff(x))


class TransformerBlock(nn.Module):
    def __init__(self, h, d_model, seq_length):
        super().__init__()

        self.mha = MultiHeadAttention(h, d_model, seq_length)
        self.ff = FeedForward(d_model)
        self.ln1 = nn.LayerNorm([d_model]);
        self.ln2 = nn.LayerNorm([d_model]);

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.cfg = config
        self.inp_emb = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        # each (integer) position in the seq_length tokens will get its own embedding
        self.pos_emb = nn.Embedding(self.cfg.seq_length, self.cfg.d_model)

        self.blist = nn.ModuleList([TransformerBlock(
            self.cfg.h,
            self.cfg.d_model,
            self.cfg.seq_length)
            for _ in range(self.cfg.N)])
        self.tblocks = nn.Sequential(*self.blist)

        self.ln_out = nn.LayerNorm([self.cfg.d_model]);
        self.lin_out = nn.Linear(in_features=self.cfg.d_model,
                                 out_features=self.cfg.vocab_size,
                                 bias=False)

    def forward(self, x, target=None):
        # (batch_size, seq_length) -> (batch_size, seq_length, d_model)
        x = self.inp_emb(x)
        # (batch_size, seq_length, d_model) + (seq_length, d_model) ->
        # (batch_size, seq_length, d_model)
        x = x + self.pos_emb(torch.arange(self.cfg.seq_length))
        # go through the N transformer blocks
        # (batch_size, seq_length, d_model) -> (batch_size, seq_length, d_model)
        x = self.tblocks(x)
        # normalise
        # (batch_size, seq_length, d_model) -> (batch_size, seq_length, d_model)
        x = self.ln_out(x)
        # (batch_size, seq_length, d_model) -> (batch_size, seq_length, vocab_size)
        x = self.lin_out(x)

        loss = None
        if target is not None:
            # x -> (batch_size, seq_length, vocab_size) -> (batch_size*seq_length, vocab_size)
            # target -> (batch_size*seq_length)
            loss = F.cross_entropy(x.reshape(-1,self.cfg.vocab_size), target.reshape(-1,))

        return x, loss

    def generate(self, x, max_new_tokens):
        for _ in range(max_new_tokens):
            # x is (batch_size, seq_length) array of indices in the current context
            # (batch_size, seq_length) -> (batch_size, seq_length, vocab_size)
            y = self(x)
            # get the last element of (seq_length) tokens
            y_last = y[:, -1:, :]
            # use softmax to convert into probs
            sm = F.softmax(y_last, dim=-1)
            # sample from a vector of integers [0,vocab_size] using that probability distribution
            # y_next -> (batch_size, 1)
            x_next = torch.multinomial(sm.view(-1, self.cfg.vocab_size), num_samples=1, replacement=True)
            # append a one-hot vector of vocab_size to the end of x
            # output share to be (batch_size, seq_length)
            x = torch.cat((x, x_next), -1)[:, 1:]

        return x

set_seed(seed=None)
cfg = Config(model_name='gpt2')
training_data = TrainingDataset(cfg)
train_dataloader = TrainingDataloader(training_data, batch_size=10)

model = Transformer(cfg)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_loop(cfg, train_dataloader, model, optimizer)

