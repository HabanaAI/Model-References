# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company.
# Extracted from: https://github.com/EleutherAI/gpt-neox
import torch

try:
    from habana_frameworks.torch.hpex.kernels import RotaryPosEmbeddingHelperV1
except ImportError:
    print("failed to import RotaryPosEmbeddingHelperV1")


class RotaryEmbedding(torch.nn.Module):

    def __init__(self, dim, base=10000, precision=torch.half):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.precision = precision

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()
            # [sx, 1 (b * np), hn]
            self.cos_cached = emb.cos()[:, None, :]
            self.sin_cached = emb.sin()[:, None, :]
            if self.precision == torch.bfloat16:
                self.cos_cached = self.cos_cached.bfloat16()
                self.sin_cached = self.sin_cached.bfloat16()
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


class RotaryEmbeddingV2(torch.nn.Module):

    def __init__(self, dim, base=10000, precision=torch.half):
        super().__init__()
        self.dim = dim
        self.base = base
        self.precision = precision
        self.max_seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        shape = (dim // 2) + (dim % 2)
        self.register_buffer('inv_freq', torch.empty(shape)) # for ckpt backward compatiblity with RotaryEmbedding

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = seq_len
            inv_freq = 1. / (self.base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=torch.float32)
            freqs = torch.einsum('i,j->ij', t, inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # [sx, 1 (b * np), hn]
            self.cos_cached = emb.cos()[:, None, :]
            self.sin_cached = emb.sin()[:, None, :]
            self.cos_cached = self.cos_cached.to(self.precision)
            self.sin_cached = self.sin_cached.to(self.precision)
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


# rotary pos emb helpers:

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions


# @torch.jit.script # commented due to jit issues on GPU while using HPU related stuff
def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    if q.device.type == "hpu":
        return RotaryPosEmbeddingHelperV1.apply(q, cos, sin, offset), RotaryPosEmbeddingHelperV1.apply(k, cos, sin, offset)
    cos, sin = cos[offset:q.shape[0] + offset, ...], sin[offset:q.shape[0] + offset, ...]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def apply_rotary_pos_emb_torch(q, k, cos, sin, offset: int = 0):  # jitting fails with bf16
    if q.device.type == "hpu":
        return RotaryPosEmbeddingHelperV1.apply(q, cos, sin, offset), RotaryPosEmbeddingHelperV1.apply(k, cos, sin, offset)
    cos, sin = cos[offset:q.shape[0] + offset, ...], sin[offset:q.shape[0] + offset, ...]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
