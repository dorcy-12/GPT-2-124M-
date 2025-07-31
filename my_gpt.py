# %% [code] {"execution":{"iopub.status.busy":"2025-07-23T16:19:18.549977Z","iopub.execute_input":"2025-07-23T16:19:18.550230Z","iopub.status.idle":"2025-07-23T16:19:24.741160Z","shell.execute_reply.started":"2025-07-23T16:19:18.550210Z","shell.execute_reply":"2025-07-23T16:19:24.740428Z"},"jupyter":{"outputs_hidden":false}}

import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0  # same embd_size per head
        self.attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Flag for decreasing layer variance. (With addition of the results of each block variance builds up.)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(
            1, 1, config.block_size, config.block_size
        ))
        # T, 3 * C

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.attn(x)  # B,T, 3 * C
        q, k, v = qkv.split(self.n_embd, dim=2)  # each B, T, C

        # hs = C//n_head

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # B, n_head, T, hs
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # B, n_head, T, hs
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # B, n_head, T, hs

        # att = (q @ k.transpose(-2,-1)) * (1 / math.sqrt(k.size(-1))) # B,n_head, T, T
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # B, n_head, T, hs
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # B, T, C

        # mix it up
        y = self.c_proj(y)

        return y

# %% [code] {"execution":{"iopub.status.busy":"2025-07-23T13:16:58.716835Z","iopub.execute_input":"2025-07-23T13:16:58.717196Z","iopub.status.idle":"2025-07-23T13:16:58.721896Z","shell.execute_reply.started":"2025-07-23T13:16:58.717171Z","shell.execute_reply":"2025-07-23T13:16:58.721163Z"},"jupyter":{"outputs_hidden":false}}
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()  # did not use the tanh approximation as there is no need anymore
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # Flag for decreasing layer variance

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


# %% [code] {"execution":{"iopub.status.busy":"2025-07-23T13:16:58.722627Z","iopub.execute_input":"2025-07-23T13:16:58.722961Z","iopub.status.idle":"2025-07-23T13:16:58.741778Z","shell.execute_reply.started":"2025-07-23T13:16:58.722940Z","shell.execute_reply":"2025-07-23T13:16:58.741238Z"},"jupyter":{"outputs_hidden":false}}
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# %% [code] {"execution":{"iopub.status.busy":"2025-07-23T13:16:58.743298Z","iopub.execute_input":"2025-07-23T13:16:58.743549Z","iopub.status.idle":"2025-07-23T13:16:58.762494Z","shell.execute_reply.started":"2025-07-23T13:16:58.743533Z","shell.execute_reply":"2025-07-23T13:16:58.761951Z"},"jupyter":{"outputs_hidden":false}}
@dataclass  # creates the init for gpt config class
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            pte=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd)
        ))
        self.ln_layer = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing. Since basically they do the same thing and this has been successful in improving performance.
        # saves number of parameters too. like 30%. since they ahve the most amount of weights [50257, 768]
        self.transformer.wte.weight = self.ln_layer.weight

        # a torch.nn.Module function that applies a function to all modules recursively
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5  # we multiply by two since there are two residual additions per block
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.size()
        assert T <= self.config.block_size

        pos = torch.arange(0, T, dtype=torch.long, device=x.device)

        tok_embd = self.transformer.wte(x)
        pos_embd = self.transformer.pte(pos)
        x = tok_embd + pos_embd

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.ln_layer(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # or logits.view(-1, C)
            targets = targets.view(B * T)  # or targets.view(-1)

            # cross entropy expects the Channel dim to be second
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device):
        # get all parameters requiring gradients
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # create Optim_groups. Any parameters with dim >= 2 (Embeddings, Linear) will be decayed
        # other 1D params (Biases, LayerNorms ) will not be decayed

        decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for pn, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]

        # count number of decayed and non-decayed params
        num_decayed = sum(p.numel() for p in decay_params)
        num_non_decayed = sum(p.numel() for p in no_decay_params)
        print(f"number of decay params: {num_decayed}")
        print(f"number of non-decay params: {num_non_decayed}")

        # Use fusion optimization for the AdamW Optim (if available)
        fused_avail = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_avail and 'cuda' in device
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"fused_avail: {fused_avail} and using fused AdamW: {use_fused}")
        return optimizer

