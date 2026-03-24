"""Scoring network modules for comparator and unit-aware retrieval."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn


class TwoLayerFFN(nn.Module):
    """Simple two-layer feed-forward network with ReLU activation."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        """Build Linear(input->hidden)->ReLU->Linear(hidden->output)."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two-layer feed-forward projection."""
        device_type = x.device.type
        ctx = torch.autocast(device_type=device_type, enabled=False) if device_type in {"cuda", "cpu"} else nullcontext()
        with ctx:
            return self.net(x.to(dtype=torch.float32)).to(dtype=torch.float32)


class ComparatorPredictor(nn.Module):
    """Predicts comparator probabilities over {<, =, >}."""

    def __init__(self) -> None:
        """Initialize comparator prediction network."""
        super().__init__()
        self.predictor = TwoLayerFFN(768, 128, 3)

    def forward(self, y_a: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities over comparator classes, shape (batch, 3)."""
        logits = self.predictor(y_a).to(dtype=torch.float32)
        return torch.softmax(logits, dim=-1)


class UnitCompatibilityScorer(nn.Module):
    """Scores query-document unit compatibility using attention-softmax."""

    def __init__(self) -> None:
        """Initialize shared unit projection network N_u."""
        super().__init__()
        self.N_u = TwoLayerFFN(768, 128, 768)

    def forward(self, y_a: torch.Tensor, y_b_set: torch.Tensor) -> torch.Tensor:
        """Compute softmax attention over document-side unit embeddings, shape (N_docs,)."""
        if y_a.ndim != 1:
            raise ValueError("y_a must have shape (J,).")
        if y_b_set.ndim != 2:
            raise ValueError("y_b_set must have shape (N_docs, J).")

        u_a = F.normalize(self.N_u(y_a).reshape(-1), p=2, dim=0, eps=1e-6)
        u_b_set = F.normalize(self.N_u(y_b_set), p=2, dim=-1, eps=1e-6)
        scores = torch.matmul(u_b_set, u_a).clamp(min=-1.0, max=1.0)
        return torch.softmax(scores, dim=0)


class ComparatorPairScorer(nn.Module):
    """Dot-product scorer with operator-specific query and document networks."""

    VALID_OPS = {"lt", "eq", "gt"}

    def __init__(self, op: str) -> None:
        """Initialize operator-specific query/document projection networks."""
        super().__init__()
        if op not in self.VALID_OPS:
            raise ValueError(f"op must be one of {self.VALID_OPS}, got {op!r}.")
        self.op = op
        self.N_op_q = TwoLayerFFN(768, 128, 768)
        self.N_op_d = TwoLayerFFN(768, 128, 768)

    def precompute_doc_side(self, y_b: torch.Tensor) -> torch.Tensor:
        """Precompute document-side representations for ANN indexing."""
        return F.normalize(self.N_op_d(y_b), p=2, dim=-1, eps=1e-6)

    def forward(self, y_a: torch.Tensor, y_b: torch.Tensor) -> torch.Tensor:
        """Return scalar dot product N_op_q(y_a) · N_op_d(y_b)."""
        q_vec = F.normalize(self.N_op_q(y_a).reshape(-1), p=2, dim=0, eps=1e-6)
        d_vec = F.normalize(self.N_op_d(y_b).reshape(-1), p=2, dim=0, eps=1e-6)
        return torch.dot(q_vec, d_vec).clamp(min=-1.0, max=1.0)


class RegularizationLoss:
    """Regularization term enforcing comparator consistency."""

    def compute(
        self,
        N_eq_score: torch.Tensor | float,
        N_lt_score: torch.Tensor | float,
        N_gt_score: torch.Tensor | float,
    ) -> torch.Tensor:
        """Compute L_reg = |1 - N_eq| * exp(-|N_lt - N_gt|^2)."""
        n_eq = N_eq_score if isinstance(N_eq_score, torch.Tensor) else torch.tensor(float(N_eq_score))
        n_lt = N_lt_score if isinstance(N_lt_score, torch.Tensor) else torch.tensor(float(N_lt_score))
        n_gt = N_gt_score if isinstance(N_gt_score, torch.Tensor) else torch.tensor(float(N_gt_score))

        return torch.abs(1.0 - n_eq) * torch.exp(-(torch.abs(n_lt - n_gt) ** 2))


if __name__ == "__main__":
    torch.manual_seed(11)

    predictor = ComparatorPredictor()
    query_batch = torch.randn(4, 768, dtype=torch.float32)
    probs = predictor(query_batch)
    print("ComparatorPredictor output:", probs.shape)

    unit_scorer = UnitCompatibilityScorer()
    query = torch.randn(768, dtype=torch.float32)
    docs = torch.randn(6, 768, dtype=torch.float32)
    attn = unit_scorer(query, docs)
    print("UnitCompatibilityScorer output:", attn.shape)

    pair_scorer = ComparatorPairScorer("lt")
    a = torch.randn(768, dtype=torch.float32)
    b = torch.randn(768, dtype=torch.float32)
    print("ComparatorPairScorer scalar:", pair_scorer(a, b).shape)

    reg = RegularizationLoss().compute(
        N_eq_score=torch.tensor(0.8),
        N_lt_score=torch.tensor(0.3),
        N_gt_score=torch.tensor(0.1),
    )
    print("RegularizationLoss scalar:", reg.shape)
