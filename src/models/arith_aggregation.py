"""Arithmetic Aggregation Network (AAN) for WideQuant."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TYPE_TO_INDEX = {"TYPE_A": 0, "TYPE_B": 1, "TYPE_C": 2}


class ArithAggregationNetwork(nn.Module):
    """Aggregate sub-quantity embeddings into one resolved quantity embedding."""

    def __init__(
        self,
        hidden_dim: int = 768,
        n_heads: int = 4,
        n_layers: int = 2,
        n_types: int = 3,
    ) -> None:
        """Initialize arithmetic-type embedding, transformer encoder, and output head."""
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.n_types = int(n_types)

        self.arith_type_embedding = nn.Embedding(self.n_types, self.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.n_heads,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, sub_quantity_embeddings: Tensor, arith_type: int | Tensor) -> Tensor:
        """Aggregate a set of sub-quantity embeddings into one resolved embedding."""
        if sub_quantity_embeddings.ndim != 2 or sub_quantity_embeddings.shape[1] != self.hidden_dim:
            raise ValueError(
                f"Expected sub_quantity_embeddings with shape (k, {self.hidden_dim}), "
                f"got {tuple(sub_quantity_embeddings.shape)}"
            )
        if sub_quantity_embeddings.shape[0] < 1:
            raise ValueError("sub_quantity_embeddings must contain at least one sub-quantity embedding.")

        device = sub_quantity_embeddings.device
        if isinstance(arith_type, Tensor):
            arith_type_tensor = arith_type.to(device=device, dtype=torch.long).view(1)
        else:
            arith_type_tensor = torch.tensor([int(arith_type)], dtype=torch.long, device=device)

        type_emb = self.arith_type_embedding(arith_type_tensor)  # (1, hidden_dim)
        x = torch.cat([type_emb, sub_quantity_embeddings], dim=0)  # (k+1, hidden_dim)
        encoded = self.transformer_encoder(x.unsqueeze(1)).squeeze(1)  # (k+1, hidden_dim)
        pooled = encoded.mean(dim=0)  # (hidden_dim,)
        return self.output_projection(pooled)  # (hidden_dim,)


class AANTrainingObjective:
    """Supervise AAN outputs against atomic-document quantity embeddings."""

    def __init__(self) -> None:
        """Initialize the mean-squared-error objective."""
        self.loss_fn = nn.MSELoss()

    def compute_loss(self, aan_output: Tensor, atomic_bert_output: Tensor) -> Tensor:
        """Return MSE between AAN output and the detached atomic target embedding."""
        return self.loss_fn(aan_output, atomic_bert_output.detach())


def _stack_quantity_outputs(encoded: dict[str, Any]) -> Tensor:
    """Stack quantity outputs from a model encoding dictionary."""
    quantity_outputs = list(encoded.get("quantity_outputs", {}).values())
    if not quantity_outputs:
        raise ValueError("Expected at least one quantity output in encoded representation.")
    return torch.stack(quantity_outputs, dim=0)


def _resolve_eval_pair(model: Any, batch: Any) -> tuple[Tensor, Tensor, int]:
    """Resolve one verification pair from a batch into sub-embeddings, atomic embedding, and type index."""
    if isinstance(batch, dict) and "sub_quantity_embeddings" in batch and "atomic_bert_output" in batch:
        sub_quantity_embeddings = batch["sub_quantity_embeddings"]
        atomic_bert_output = batch["atomic_bert_output"]
        arith_type = batch.get("arith_type", 0)
        if isinstance(arith_type, str):
            arith_type = TYPE_TO_INDEX[arith_type]
        return sub_quantity_embeddings, atomic_bert_output, int(arith_type)

    if isinstance(batch, dict) and "decomposed_doc_batch" in batch and "atomic_doc_batch" in batch:
        if model is None or not hasattr(model, "encode_document"):
            raise ValueError(
                "verify_aan_quality received raw document batches, but the model does not expose encode_document()."
            )

        decomposed_batch = batch["decomposed_doc_batch"]
        atomic_batch = batch["atomic_doc_batch"]
        arith_type = batch.get("arith_type", 0)
        if isinstance(arith_type, str):
            arith_type = TYPE_TO_INDEX[arith_type]
        elif isinstance(arith_type, Tensor):
            arith_type = int(arith_type.item())
        else:
            arith_type = int(arith_type)

        decomposed_enc = model.encode_document(**decomposed_batch)
        atomic_enc = model.encode_document(**atomic_batch)
        sub_quantity_embeddings = _stack_quantity_outputs(decomposed_enc)
        atomic_bert_output = _stack_quantity_outputs(atomic_enc)[0]
        return sub_quantity_embeddings, atomic_bert_output, arith_type

    raise ValueError(
        "Unsupported batch format for verify_aan_quality(). "
        "Expected either direct tensors (`sub_quantity_embeddings`, `atomic_bert_output`) "
        "or raw batches (`decomposed_doc_batch`, `atomic_doc_batch`)."
    )


def verify_aan_quality(
    model: Any,
    aan: ArithAggregationNetwork,
    data_loader: Any,
    threshold: float = 0.80,
) -> dict[str, float]:
    """Measure cosine similarity between AAN outputs and atomic targets over up to 100 pairs."""
    cosine_values: list[float] = []
    aan_device = next(aan.parameters()).device

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= 100:
                break
            sub_quantity_embeddings, atomic_bert_output, arith_type = _resolve_eval_pair(model, batch)
            sub_quantity_embeddings = sub_quantity_embeddings.to(device=aan_device, dtype=torch.float32)
            atomic_bert_output = atomic_bert_output.to(device=aan_device, dtype=torch.float32)
            aan_output = aan(sub_quantity_embeddings, arith_type=arith_type)
            cosine = F.cosine_similarity(aan_output.unsqueeze(0), atomic_bert_output.unsqueeze(0)).item()
            cosine_values.append(float(cosine))

    if not cosine_values:
        raise ValueError("verify_aan_quality() received no usable evaluation pairs.")

    mean_cosine = float(sum(cosine_values) / len(cosine_values))
    variance = float(sum((value - mean_cosine) ** 2 for value in cosine_values) / len(cosine_values))
    std_cosine = float(math.sqrt(variance))
    print(f"AAN cosine similarity: mean={mean_cosine:.3f} std={std_cosine:.3f}")
    if mean_cosine > threshold:
        print("AAN QUALITY: PASS")
    else:
        print(f"AAN QUALITY: FAIL - mean sim={mean_cosine:.3f}, need {threshold:.3f}")
    return {"mean_cosine": mean_cosine, "std_cosine": std_cosine}


if __name__ == "__main__":
    aan = ArithAggregationNetwork()
    sub_embs = torch.randn(3, 768)
    out = aan(sub_embs, arith_type=0)
    assert out.shape == (768,)
    assert out.requires_grad
    print("AAN smoke test: PASS")
