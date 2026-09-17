"""Standalone decoder-only MoE with causal attention and top-k SwiGLU experts.

This module defines task forward passes and router losses; SALAAD auxiliary
states live in solver.py. Experts run sequentially for readability and validation;
the current implementation has no specialized sparse or grouped GEMM kernel.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight.to(x.dtype)


def route(logits, topk):
    """Normalize selected top-k weights; compute balance/z-loss from all experts' logits."""
    logits = logits.float()
    scores, indices = logits.topk(topk, dim=-1)
    weights = scores.softmax(-1)
    # Balance loss compares actual selection fractions with mean routing
    # probabilities; assignment counts themselves are not differentiable.
    probabilities = logits.softmax(-1)
    counts = torch.bincount(indices.reshape(-1), minlength=logits.shape[-1]).float()
    fractions = counts / indices.numel()
    balance = logits.shape[-1] * (fractions.detach() * probabilities.mean(0)).sum()
    z_loss = logits.logsumexp(-1).square().mean()
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1).mean()
    return (
        weights,
        indices,
        balance,
        z_loss,
        {"counts": counts.detach(), "entropy": entropy.detach()},
    )


class ExpertMLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        k, d, f = c["num_experts"], c["hidden_size"], c["expert_ffn_hidden_size"]
        # Each projection stacks all experts into one parameter tensor,
        # matching one SALAAD group.
        self.gate = nn.Parameter(torch.empty(k, f, d))
        self.up = nn.Parameter(torch.empty(k, f, d))
        self.down = nn.Parameter(torch.empty(k, d, f))

    def forward(self, x, weights, indices):
        # Forward passes use the full dense weights X. Expert slices with no
        # routed tokens have zero task gradients but still receive SALAAD constraints.
        result = torch.zeros_like(x)
        for expert in range(self.gate.shape[0]):
            token, slot = torch.where(indices == expert)
            if token.numel() == 0:
                continue
            h = x.index_select(0, token)
            # SwiGLU = SiLU(gate(x)) * up(x), followed by the down projection
            # back to the model's hidden size.
            hidden = F.silu(F.linear(h, self.gate[expert])) * F.linear(h, self.up[expert])
            output = F.linear(hidden, self.down[expert])
            output = output * weights[token, slot, None].to(output.dtype)
            # Each token selects multiple experts; sum their weighted outputs.
            result.index_add_(0, token, output.to(result.dtype))
        return result


class MoE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.router = nn.Linear(c["hidden_size"], c["num_experts"], bias=False)
        self.experts = ExpertMLP(c)
        self.topk = c["router_topk"]

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = F.linear(flat.float(), self.router.weight.float())
            weights, indices, balance, z_loss, stats = route(logits, self.topk)
        return self.experts(flat, weights, indices).view_as(x), balance, z_loss, stats


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c["hidden_size"]
        self.heads = c["num_attention_heads"]
        self.head_dim = d // self.heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.register_buffer(
            "inv_freq",
            1.0 / (c["rotary_base"] ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)),
            persistent=False,
        )

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, self.head_dim).unbind(2)
        q, k, v = (a.transpose(1, 2) for a in (q, k, v))
        phase = torch.outer(
            torch.arange(n, device=x.device, dtype=torch.float32), self.inv_freq.float()
        )
        phase = torch.cat((phase, phase), dim=-1)[None, None]
        cos, sin = phase.cos().to(q.dtype), phase.sin().to(q.dtype)

        def rotate(a):
            left, right = a.chunk(2, dim=-1)
            return a * cos + torch.cat((-right, left), dim=-1) * sin

        y = F.scaled_dot_product_attention(rotate(q), rotate(k), v, dropout_p=0.0, is_causal=True)
        return self.out(y.transpose(1, 2).contiguous().view(b, n, d))


class Block(nn.Module):
    def __init__(self, c, recompute):
        super().__init__()
        self.attention_norm = RMSNorm(c["hidden_size"], c["norm_epsilon"])
        self.attention = Attention(c)
        self.moe_norm = RMSNorm(c["hidden_size"], c["norm_epsilon"])
        self.moe = MoE(c)
        self.recompute = recompute

    def forward(self, x):
        h = self.attention_norm(x)
        a = (
            checkpoint(self.attention, h, use_reentrant=False)
            if self.recompute and self.training
            else self.attention(h)
        )
        x = x + a
        y, balance, z_loss, stats = self.moe(self.moe_norm(x))
        return x + y, balance, z_loss, stats


@dataclass
class LMOutput:
    logits: torch.Tensor
    lm_loss: torch.Tensor | None
    balance_loss: torch.Tensor
    z_loss: torch.Tensor
    router_stats: list

    def task_loss(self, config):
        # Vanilla and SALAAD use identical task losses. Trainer adds the
        # structural penalty directly to gradients after DP averaging, so it
        # is not included in this loss.
        t = config["training"]
        return (
            self.lm_loss
            + t["load_balancing_coefficient"] * self.balance_loss
            + t["router_z_loss_coefficient"] * self.z_loss
        )


class MoELanguageModel(nn.Module):
    def __init__(self, config, initialize=True):
        super().__init__()
        self.config = config
        c = config["model"]
        self.embedding = nn.Embedding(c["padded_vocab_size"], c["hidden_size"])
        self.layers = nn.ModuleList(
            [
                Block(c, config["training"]["activation_recompute"] == "selective_attention")
                for _ in range(c["num_layers"])
            ]
        )
        self.norm = RMSNorm(c["hidden_size"], c["norm_epsilon"])
        self.lm_head = nn.Linear(c["hidden_size"], c["padded_vocab_size"], bias=False)
        if initialize:
            self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        c = self.config["model"]
        for name, p in self.named_parameters():
            if p.ndim == 1:
                p.fill_(1)
            else:
                std = c["init_std"]
                if name.endswith("attention.out.weight") or name.endswith("experts.down"):
                    std /= math.sqrt(2 * c["num_layers"])
                p.normal_(std=std)

    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)
        balance, z_loss, stats = (
            x.new_zeros((), dtype=torch.float32),
            x.new_zeros((), dtype=torch.float32),
            [],
        )
        for layer in self.layers:
            x, b, z, detail = layer(x)
            balance, z_loss = balance + b, z_loss + z
            stats.append(detail)
        logits = self.lm_head(self.norm(x))
        # The reader already returns tokens[:-1] and tokens[1:]; do not shift again.
        loss = (
            None
            if labels is None
            else F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))
        )
        return LMOutput(logits, loss, balance, z_loss, stats)
