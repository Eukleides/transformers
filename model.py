import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from transformers import GPT2LMHeadModel

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config['d_model']
        self.n_head = config['n_head']
        self.c_attn = nn.Linear(config['d_model'], 3*config['d_model'], bias=config['bias'])
        self.c_proj = nn.Linear(config['d_model'], config['d_model'], bias=config['bias'])
        self.attn_dropout = nn.Dropout(config['dropout'])
        self.resid_dropout = nn.Dropout(config['dropout'])

        # create a lower triangular matrix (context_size, context_size) with 0s above the diagonal
        self.bias = torch.tril(torch.ones(config['context_size'], config['context_size'])).view(1, 1, config['context_size'], config['context_size']).to(config['device_type'])

    def forward(self, x):
        B, T, C = x.size();

        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, n_head, T, C//n_head)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, n_head, T, C//n_head)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, n_head, T, C//n_head)

        # this step mixes the q(uery) tokens with the k(ey) tokens per head
        # (B, n_head, T, C//n_head) @ (B, n_head, C//n_head, T) = (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # set everything above the diagonal to -inf
        # (B, n_head, T, T) -> (B, n_head, T, T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # apply softmax ( exp(xi)/Sum(exp(xi)) ) , ie sets probabilities of 0 above the diagonal
        # (B, n_head, T, T) -> (B, n_head, T, T)
        # for each (B, n_head) element of size (T,T), each row element is the dot product of query tokens and key tokens, and the sum of each row is 1.0
        att = F.softmax(att, dim=-1)
        # dropout - only used during training
        att = self.attn_dropout(att)
        # (B, n_head, T, T) @ (B, n_head, T, C//n_head) -> (B, n_head, T, C//n_head)
        y = att @ v
        # (B, n_head, T, C//n_head) -> (B, T, n_head, C//n_head)
        y = y.transpose(1, 2).contiguous()
        # (B, T, n_head, C // n_head) -> (B, T, C)
        y = y.view(B, T, C)
        # (B, T, C) * (C, C) -> (B, T, C)
        y = self.c_proj(y)
        # dropout - only used during training
        y = self.resid_dropout(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config['d_model'], 4*config['d_model'], bias=config['bias'])
        self.c_proj = nn.Linear(4*config['d_model'], config['d_model'], bias=config['bias'])
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        # x -> (B, T, C)
        # (B, T, C) -> (B, T, 4*C)
        x = self.c_fc(x)
        # (B, T, 4*C) -> (B, T, 4*C)
        x = self.gelu(x)
        # (B, T, 4*C) -> (B, T, C)
        x = self.c_proj(x)
        # (B, T, C) -> (B, T, C)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config['d_model'], bias=config['bias'])
        self.ln_2 = nn.LayerNorm(config['d_model'], bias=config['bias'])
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        # x -> (B, T, C)
        # ln_1 normalises the C dimension to (mean 0 , std 1)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config['vocab_size'], config['d_model']),
            wpe = nn.Embedding(config['context_size'], config['d_model']),
            drop = nn.Dropout(config['dropout']),
            h = nn.ModuleList([Block(config) for _ in range(config['n_layer'])]),
            ln_f = nn.LayerNorm(config['d_model'], bias=config['bias'])
        ))

        self.lm_head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)

    @classmethod
    def from_pretrained(cls, model_name):
        assert(model_name.upper() == 'GPT2', 'Unrecognised model name')
        # model_hf_orig = GPT2LMHeadModel.from_pretrained('gpt2')
        # torch.save(model_hf_orig, 'assets\\gpt2.pt')
        # load manually saved gpt2 model
        model_hf = torch.load('assets\\gpt2.pt')

        config = dict(vocab_size = 50257, context_size=1024, d_model=768, n_layer=12, n_head=12, dropout=0., bias=True)
        config['device_type'] = 'cuda'

        model = Transformer(config)

        hf_keys = model_hf.state_dict()
        model_keys = model.state_dict()

        for key, value in hf_keys.items():
            copy_filter = ['wte', 'wpe', 'ln_f', 'lm_head', '.ln_1', '.ln_2', 'attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']
            transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
            for item in copy_filter:
                if item in key:
                    assert key in model_keys, 'key not found in transformer model'
                    is_transpose = False
                    for titem in transposed:
                        if titem in key:
                            assert model_keys[key].shape == value.t().shape, 'Cannot copy incompatible key'
                            print(f'copying {key} ## {value.t().shape}')
                            with torch.no_grad():
                                model_keys[key].copy_(value.t())
                            is_transpose = True

                    if not is_transpose:
                        print(f'copying {key} ## {value.shape}')
                        assert model_keys[key].shape == value.shape, 'Cannot copy incompatible key'
                        with torch.no_grad():
                            model_keys[key].copy_(value)

        return model

    def forward(self, idx, targets=None):
        # idx is of (batch, token) size
        b,t = idx.size()
        device = idx.device

        # (b,t) ->  (b,t,c)  i.e. each token gets a token embedding vector
        tok_emb = self.transformer.wte(idx) # (b, t, c)

        pos = torch.arange(0, t, dtype=torch.long, device=device)
        # t -> (t,c) i.e. each token position gets a position embedding vector
        pos_emb = self.transformer.wpe(pos) # (t, c)
        # apply dropout (only used during training, not inference)
        # (b,t,c) + (t,c) -> (b,t,c)
        idx = self.transformer.drop(tok_emb+pos_emb)

        # apply N decoder blocks
        for block in self.transformer.h:
            idx = block(idx)

        # apply LayerNorm (b,t,c) -> (b,t,c)
        idx = self.transformer.ln_f(idx)

        if targets is None:
            # no targets means we are doing inference not training and are only interested to get the next token
            # apply Linear to the last token only: (b,t[-1],c) -> (b,1,vocab_size)
            logits = self.lm_head(idx[:,[-1],:])
        else:
            # if we are given some desired targets also calculate the loss
            # (b,t,c) -> (b,t,vocab_size)
            logits = self.lm_head(idx)
            # (b,t,vocab_size) -> (b*t, vocab_size)
            logits_view = logits.view(-1, logits.size(-1))
            loss = F.cross_entropy(logits_view, targets.view(-1), ignore_index=-1)


        # logits = (b,1,vocab_size)
        return logits

    def generate(self, x, max_new_tokens=50, temperature=0.1):

        for _ in range(max_new_tokens):
            # x -> (B, T, vocab_size)
            logits = self.forward(x)
            logits = logits / temperature
            sm = F.softmax(logits, dim=-1).view(-1)
            y = torch.multinomial(sm, 1).view(1,-1)
            x = torch.cat((x,y), 1)

        return x


###################################################################################################
# set random seed for reproducible results
seed = 1337

from contextlib import nullcontext
device_type = 'cuda'
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

enc = tiktoken.get_encoding("gpt2")
model = Transformer.from_pretrained('gpt2')

prompt = 'Today i plan to code'
x = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device_type).view(1,-1)

model.eval()
model.to(device_type)

with torch.no_grad():
    with ctx:
        y = model.forward(x, torch.randn((1*5, 50257)))
        y = model.generate(x, max_new_tokens=20, temperature=0.1).tolist()[0]
        print(enc.decode(y))
        print('------')