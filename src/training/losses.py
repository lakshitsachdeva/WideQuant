"""Loss functions for WideQuant training."""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import Tensor, nn


class InfoNCELoss(nn.Module):
    """Retrieval contrastive loss L_retr."""

    def __init__(self, temperature: float = 0.02) -> None:
        """Initialize InfoNCE temperature."""
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")
        self.temperature = float(temperature)

    def forward(self, pos_scores: Tensor, neg_scores: Tensor) -> Tensor:
        """Compute mean InfoNCE over a batch."""
        if pos_scores.ndim != 1:
            raise ValueError("pos_scores must have shape (batch,).")
        if neg_scores.ndim != 2:
            raise ValueError("neg_scores must have shape (batch, n_neg).")
        if neg_scores.shape[0] != pos_scores.shape[0]:
            raise ValueError("Batch size mismatch between pos_scores and neg_scores.")

        pos_scaled = pos_scores.to(dtype=torch.float32) / self.temperature
        neg_scaled = neg_scores.to(dtype=torch.float32) / self.temperature
        logits = torch.cat([pos_scaled.unsqueeze(1), neg_scaled], dim=1)
        loss = -(pos_scaled - torch.logsumexp(logits, dim=1))
        return loss.mean()


class QuantityReconstructionLoss(nn.Module):
    """Quantity reconstruction loss L_quant."""

    def __init__(self) -> None:
        """Initialize component criteria."""
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred_exponent_logits: Tensor,
        true_exponent: Tensor,
        pred_mantissa: Tensor,
        true_mantissa: Tensor,
        pred_unit_logits: Tensor,
        true_unit: Tensor,
    ) -> Tensor:
        """Compute L_exp + L_man + L_unit."""
        l_exp = self.ce(pred_exponent_logits.to(dtype=torch.float32), true_exponent.to(dtype=torch.long))
        l_man = self.mse(pred_mantissa.to(dtype=torch.float32), true_mantissa.to(dtype=torch.float32))
        l_unit = self.ce(pred_unit_logits.to(dtype=torch.float32), true_unit.to(dtype=torch.long))
        return l_exp + l_man + l_unit


class RegularizationLoss(nn.Module):
    """Regularization loss L_reg for comparator consistency."""

    def __init__(self) -> None:
        """Initialize regularization module."""
        super().__init__()

    def forward(self, N_eq_scores: Tensor, N_lt_scores: Tensor, N_gt_scores: Tensor) -> Tensor:
        """Compute mean of |1 - N_eq| * exp(-|N_lt - N_gt|^2)."""
        n_eq = N_eq_scores.to(dtype=torch.float32)
        n_lt = N_lt_scores.to(dtype=torch.float32)
        n_gt = N_gt_scores.to(dtype=torch.float32)
        reg = torch.abs(1.0 - n_eq) * torch.exp(-(torch.abs(n_lt - n_gt) ** 2))
        return reg.mean()


class ComparisonSupervisionLoss(nn.Module):
    """Comparator supervision loss L_comp."""

    def __init__(self, eps: float = 1e-8) -> None:
        """Initialize numeric stability epsilon."""
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        N_lt_scores: Tensor,
        N_eq_scores: Tensor,
        N_gt_scores: Tensor,
        true_relations: Tensor,
    ) -> Tensor:
        """Compute mean -log probability of the true comparator class."""
        stacked = torch.stack(
            [
                N_lt_scores.to(dtype=torch.float32),
                N_eq_scores.to(dtype=torch.float32),
                N_gt_scores.to(dtype=torch.float32),
            ],
            dim=-1,
        )
        probs = torch.clamp(stacked, min=self.eps, max=1.0)
        selected = probs.gather(dim=-1, index=true_relations.to(dtype=torch.long).unsqueeze(-1)).squeeze(-1)
        return -torch.log(torch.clamp(selected, min=self.eps)).mean()


class ArithRetrievalLoss(nn.Module):
    """Arithmetic candidate loss L_arith."""

    def __init__(self, eps: float = 1e-8) -> None:
        """Initialize numeric stability epsilon."""
        super().__init__()
        self.eps = float(eps)

    def forward(self, resolved_candidate_scores: Tensor, is_satisfying_mask: Tensor) -> Tensor:
        """Compute mean -log(rel_arith) for satisfying candidates only."""
        scores = resolved_candidate_scores.to(dtype=torch.float32)
        mask = is_satisfying_mask.to(dtype=torch.bool)
        if scores.ndim != 1:
            raise ValueError("resolved_candidate_scores must have shape (N_candidates,).")
        if mask.ndim != 1 or mask.shape[0] != scores.shape[0]:
            raise ValueError("is_satisfying_mask must have shape (N_candidates,).")

        selected = scores[mask]
        if selected.numel() == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=scores.device)
        return -torch.log(torch.clamp(selected, min=self.eps)).mean()


class TotalLoss(nn.Module):
    """Combined training objective."""

    def __init__(self, lambda_arith: float = 0.5) -> None:
        """Initialize all sub-losses and arithmetic weight."""
        super().__init__()
        self.lambda_arith = float(lambda_arith)
        self.retr_loss = InfoNCELoss()
        self.quant_loss = QuantityReconstructionLoss()
        self.reg_loss = RegularizationLoss()
        self.comp_loss = ComparisonSupervisionLoss()
        self.arith_loss = ArithRetrievalLoss()

    @staticmethod
    def _find_device(model_outputs: Dict[str, Any]) -> torch.device:
        """Pick a device from available tensor values."""
        for value in model_outputs.values():
            if isinstance(value, Tensor):
                return value.device
        return torch.device("cpu")

    def forward(self, model_outputs: dict) -> dict:
        """Compute total and individual losses from model outputs."""
        device = self._find_device(model_outputs)
        zero = torch.tensor(0.0, dtype=torch.float32, device=device)

        l_retr = zero
        if "pos_scores" in model_outputs and "neg_scores" in model_outputs:
            l_retr = self.retr_loss(model_outputs["pos_scores"], model_outputs["neg_scores"])

        l_quant = zero
        quant_keys = {
            "pred_exponent_logits",
            "true_exponent",
            "pred_mantissa",
            "true_mantissa",
            "pred_unit_logits",
            "true_unit",
        }
        if quant_keys.issubset(model_outputs.keys()):
            l_quant = self.quant_loss(
                model_outputs["pred_exponent_logits"],
                model_outputs["true_exponent"],
                model_outputs["pred_mantissa"],
                model_outputs["true_mantissa"],
                model_outputs["pred_unit_logits"],
                model_outputs["true_unit"],
            )

        l_reg = zero
        if {"N_eq_scores", "N_lt_scores", "N_gt_scores"}.issubset(model_outputs.keys()):
            l_reg = self.reg_loss(
                model_outputs["N_eq_scores"],
                model_outputs["N_lt_scores"],
                model_outputs["N_gt_scores"],
            )

        l_comp = zero
        if {"N_lt_scores", "N_eq_scores", "N_gt_scores", "true_relations"}.issubset(model_outputs.keys()):
            l_comp = self.comp_loss(
                model_outputs["N_lt_scores"],
                model_outputs["N_eq_scores"],
                model_outputs["N_gt_scores"],
                model_outputs["true_relations"],
            )

        l_arith = zero
        if {"resolved_candidate_scores", "is_satisfying_mask"}.issubset(model_outputs.keys()):
            l_arith = self.arith_loss(
                model_outputs["resolved_candidate_scores"],
                model_outputs["is_satisfying_mask"],
            )

        total = l_retr + l_quant + l_reg + l_comp + self.lambda_arith * l_arith
        return {
            "total": total,
            "L_retr": l_retr,
            "L_quant": l_quant,
            "L_reg": l_reg,
            "L_comp": l_comp,
            "L_arith": l_arith,
        }


if __name__ == "__main__":
    torch.manual_seed(3)

    losses = TotalLoss(lambda_arith=0.5)
    batch = 4
    n_neg = 3
    n_candidates = 6
    num_units = 500

    model_outputs = {
        "pos_scores": torch.rand(batch),
        "neg_scores": torch.rand(batch, n_neg),
        "pred_exponent_logits": torch.randn(batch, 41),
        "true_exponent": torch.randint(0, 41, (batch,)),
        "pred_mantissa": torch.rand(batch),
        "true_mantissa": torch.rand(batch),
        "pred_unit_logits": torch.randn(batch, num_units),
        "true_unit": torch.randint(0, num_units, (batch,)),
        "N_lt_scores": torch.rand(batch),
        "N_eq_scores": torch.rand(batch),
        "N_gt_scores": torch.rand(batch),
        "true_relations": torch.randint(0, 3, (batch,)),
        "resolved_candidate_scores": torch.rand(n_candidates),
        "is_satisfying_mask": torch.tensor([True, False, True, False, True, False]),
    }
    out = losses(model_outputs)
    print({k: float(v.detach().cpu()) for k, v in out.items()})
