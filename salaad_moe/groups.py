"""Logical [expert, output, input] views, including FP32 master gradients."""
from __future__ import annotations

import torch


def master(parameter):
    result = getattr(parameter, "main_param", parameter)
    if result.dtype != torch.float32:
        raise TypeError("SALAAD requires FP32 masters; bind groups after optimizer construction")
    return result


class StackedGroup:
    def __init__(self, name, parameter):
        self.name, self.parameter = name, parameter

    def weight(self):
        return master(self.parameter).detach()

    def add_gradient(self, delta):
        p = master(self.parameter)
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        p.grad.add_(delta)


class SequentialFusedGroup:
    """Megatron SequentialMLP: FC1=[gate; up], FC2=down, both output-first."""

    def __init__(self, name, parameters, projection):
        self.name, self.parameters, self.projection = name, parameters, projection

    def view(self, tensor):
        if self.projection == "down":
            return tensor
        if tensor.shape[0] % 2:
            raise ValueError("Fused SwiGLU FC1 must have an even output dimension")
        return tensor.chunk(2, dim=0)[0 if self.projection == "gate" else 1]

    def weight(self):
        return torch.stack([self.view(master(p).detach()) for p in self.parameters])

    def add_gradient(self, delta):
        for p, update in zip(self.parameters, delta.unbind(0)):
            p = master(p)
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            self.view(p.grad).add_(update)


class GroupedFusedGroup:
    """Pinned GroupedMLP storage uses expert-major views, then x @ weight."""

    def __init__(self, name, parameter, projection, experts, hidden):
        self.name, self.parameter, self.projection = name, parameter, projection
        self.experts, self.hidden = experts, hidden

    def view(self, tensor):
        if self.projection == "down":
            return tensor.view(self.experts, -1, self.hidden).transpose(1, 2)
        fused = tensor.view(self.experts, self.hidden, -1)
        return fused.chunk(2, dim=-1)[0 if self.projection == "gate" else 1].transpose(1, 2)

    def weight(self):
        return self.view(master(self.parameter).detach())

    def add_gradient(self, delta):
        p = master(self.parameter)
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        self.view(p.grad).add_(delta)


def native_groups(model, projections=("gate", "up", "down")):
    return [
        StackedGroup(f"layers.{i}.moe.experts.{p}", getattr(layer.moe.experts, p))
        for i, layer in enumerate(model.layers)
        for p in ("gate", "up", "down")
        if p in projections
    ]


def megatron_groups(model_chunks, config):
    """Reject unknown/fused TE layouts rather than guessing a weight mapping."""
    if len(model_chunks) != 1:
        raise ValueError("The adapter requires PP=1 and a single model chunk")
    model = model_chunks[0]
    while hasattr(model, "module"):
        model = model.module
    groups = []
    m, s = config["model"], config["salaad"]
    for i, layer in enumerate(model.decoder.layers):
        experts = layer.mlp.experts
        if getattr(layer.mlp, "shared_experts", None) is not None:
            raise ValueError("Shared experts are not part of this architecture")
        for projection in ("gate", "up", "down"):
            if projection not in s["projections"]:
                continue
            name = f"layers.{i}.moe.experts.{projection}"
            if hasattr(experts, "local_experts"):
                linear = "linear_fc2" if projection == "down" else "linear_fc1"
                parameters = [getattr(e, linear).weight for e in experts.local_experts]
                group = SequentialFusedGroup(name, parameters, projection)
            elif hasattr(experts, "weight1") and hasattr(experts, "weight2"):
                parameter = experts.weight2 if projection == "down" else experts.weight1
                group = GroupedFusedGroup(
                    name, parameter, projection, m["num_experts"], m["hidden_size"]
                )
            else:
                raise NotImplementedError(
                    "TEGroupedMLP is unsupported; select SequentialMLP or GroupedMLP"
                )
            f, d = m["expert_ffn_hidden_size"], m["hidden_size"]
            expected = (
                (m["num_experts"], d, f) if projection == "down" else (m["num_experts"], f, d)
            )
            if tuple(group.weight().shape) != expected:
                raise ValueError(f"{name}: expected {expected}, got {tuple(group.weight().shape)}")
            groups.append(group)
    if len(groups) != m["num_layers"] * len(s["projections"]):
        raise ValueError("The configured model must have an MoE in every layer")
    return groups
