"""
VelocityAI Mixture-of-Experts (MoE) Architecture Module.
Implements dynamic conditional computation: only Top-K experts are active per token.
"""

from typing import List, Tuple
import numpy as np
import velocityai as vai
from velocityai.nn.module import Module
from velocityai.nn.linear import Linear
from velocityai.nn.transformer import SwiGLUMLP


class MoERouter(Module):
    """
    Top-K Gating Router for Mixture-of-Experts.
    Given input hidden states x (dim), predicts routing logits over num_experts,
    and returns top-k expert indices and their normalized softmax weights.
    """
    def __init__(self, dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.gate = Linear(dim, num_experts, bias=False, init_weights=True)

    def forward(self, x: vai.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            x: (batch, dim) or (batch, seq, dim)
        Returns:
            top_k_indices: (batch, top_k) int array of active expert IDs
            top_k_weights: (batch, top_k) float array of softmax routing probabilities
        """
        logits = self.gate(x)
        logits_np = logits.numpy()

        exp_logits = np.exp(logits_np - np.max(logits_np, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        top_k_indices = np.argsort(-probs, axis=-1)[..., :self.top_k]
        top_k_weights = np.take_along_axis(probs, top_k_indices, axis=-1)

        top_k_weights = top_k_weights / np.sum(top_k_weights, axis=-1, keepdims=True)
        return top_k_indices, top_k_weights


class MoEBlock(Module):
    """
    Sparse Mixture-of-Experts Transformer Feed-Forward Layer.
    Only top_k experts evaluate their weights per token; remaining experts stay inactive.
    """
    def __init__(
        self,
        dim: int,
        intermediate_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        init_weights: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = MoERouter(dim=dim, num_experts=num_experts, top_k=top_k)

        self.experts: List[SwiGLUMLP] = []
        for i in range(num_experts):
            expert = SwiGLUMLP(dim=dim, intermediate_size=intermediate_size, init_weights=init_weights)
            self.experts.append(expert)
            self._modules[f"expert_{i}"] = expert

    def forward(self, x: vai.Tensor) -> vai.Tensor:
        """
        Conditional forward pass: Only the selected top_k experts compute for input x.
        """
        top_indices, top_weights = self.router(x)

        if x.ndim == 2 and x.shape[0] == 1:
            indices = top_indices[0]
            weights = top_weights[0]

            out_accum = None
            for idx, w in zip(indices, weights):
                expert_out = self.experts[int(idx)](x) * float(w)
                if out_accum is None:
                    out_accum = expert_out
                else:
                    out_accum = out_accum + expert_out
            return out_accum

        indices = top_indices.reshape(-1, self.top_k)
        weights = top_weights.reshape(-1, self.top_k)
        out_accum = None
        for k in range(self.top_k):
            idx = int(indices[0, k])
            w = float(weights[0, k])
            expert_out = self.experts[idx](x) * w
            if out_accum is None:
                out_accum = expert_out
            else:
                out_accum = out_accum + expert_out

        return out_accum
