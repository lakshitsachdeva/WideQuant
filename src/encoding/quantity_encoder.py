"""Quantity encoding modules for WideQuant."""

from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import torch
from torch import nn

try:
    from src.encoding.cqe_wrapper import QuantitySpan
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from cqe_wrapper import QuantitySpan


def gaussian_mantissa_encoding(
    m: float,
    J_m: int = 691,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Encode a scalar mantissa with Gaussian basis prototypes."""
    if J_m < 2:
        raise ValueError("J_m must be >= 2.")
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")

    prototypes = torch.linspace(-10.0, 10.0, J_m, dtype=torch.float32)
    encoded = torch.exp(-((float(m) - prototypes) ** 2) / (sigma**2))
    return encoded.to(dtype=torch.float32)


class ExponentEmbedding(nn.Module):
    """Learned embedding table for clipped decimal exponents."""

    def __init__(self, num_classes: int = 41, J_e: int = 77) -> None:
        """Initialize exponent embedding matrix E in R^{num_classes x J_e}."""
        super().__init__()
        self.num_classes = num_classes
        self.J_e = J_e
        self.embedding = nn.Embedding(num_classes, J_e)

    @property
    def weight(self) -> torch.Tensor:
        """Expose the underlying embedding table for verification and optimization checks."""
        return self.embedding.weight

    def forward(self, exponent: int) -> torch.Tensor:
        """Embed exponent after clipping to [-20, 20] and indexing with (e + 20)."""
        clipped = max(-20, min(20, int(exponent)))
        index = clipped + 20
        idx_tensor = torch.tensor(index, dtype=torch.long, device=self.embedding.weight.device)
        return self.embedding(idx_tensor).to(dtype=torch.float32)


class QuantityInjector(nn.Module):
    """Inject quantity-aware [num] embeddings into BERT token embeddings."""

    def __init__(
        self,
        base_bert_embeddings: nn.Embedding,
        num_token_id: int,
        J: int = 768,
        J_m: int = 691,
        J_e: int = 77,
    ) -> None:
        """Initialize injector with base token embedding table and quantity encoders."""
        super().__init__()
        self.base_bert_embeddings = base_bert_embeddings
        self.num_token_id = int(num_token_id)
        self.J = int(J)
        self.J_m = int(J_m)
        self.J_e = int(J_e)
        self.exponent_embedding = ExponentEmbedding(num_classes=41, J_e=J_e)

        if self.J_m + self.J_e != self.J:
            raise ValueError("J_m + J_e must equal J.")
        if self.base_bert_embeddings.embedding_dim != self.J:
            raise ValueError("base_bert_embeddings.embedding_dim must equal J.")

    def forward(self, input_ids: torch.Tensor, quantity_spans: List[QuantitySpan]) -> torch.Tensor:
        """Replace [num] token embeddings with quantity-aware vectors."""
        token_embeddings = self.base_bert_embeddings(input_ids).to(dtype=torch.float32)
        num_positions = (input_ids == self.num_token_id).nonzero(as_tuple=False)

        if len(quantity_spans) != num_positions.shape[0]:
            raise ValueError(
                f"Found {num_positions.shape[0]} [num] tokens but received {len(quantity_spans)} quantity spans."
            )

        num_id_tensor = torch.tensor(self.num_token_id, device=input_ids.device, dtype=torch.long)
        base_num_embedding = self.base_bert_embeddings(num_id_tensor).to(dtype=torch.float32)

        for span, position in zip(quantity_spans, num_positions):
            batch_idx = int(position[0].item())
            token_idx = int(position[1].item())

            exp_vec = self.exponent_embedding(int(span.exponent)).to(
                device=token_embeddings.device,
                dtype=torch.float32,
            )
            man_vec = gaussian_mantissa_encoding(
                m=float(span.mantissa),
                J_m=self.J_m,
                sigma=1.0,
            ).to(device=token_embeddings.device, dtype=torch.float32)

            quantity_vec = torch.cat([exp_vec, man_vec], dim=0)
            if quantity_vec.numel() != self.J:
                raise RuntimeError("Quantity vector dimensionality mismatch.")

            v_num = base_num_embedding + quantity_vec
            token_embeddings[batch_idx, token_idx] = v_num

        return token_embeddings


if __name__ == "__main__":
    torch.manual_seed(7)

    # Mock modules for verification.
    mock_base_embeddings = nn.Embedding(30522, 768)
    _mock_exponent_embedding = ExponentEmbedding(num_classes=41, J_e=77)
    _mock_injector = QuantityInjector(
        base_bert_embeddings=mock_base_embeddings,
        num_token_id=30521,
        J=768,
        J_m=691,
        J_e=77,
    )

    mantissas = [0.0, 2.5, 5.0, 7.5, 10.0]
    J_m = 691
    sigma = 1.0
    prototypes = torch.linspace(-10.0, 10.0, J_m, dtype=torch.float32)

    plt.figure(figsize=(10, 6))
    for m in mantissas:
        vector = gaussian_mantissa_encoding(m=m, J_m=J_m, sigma=sigma)
        plt.plot(prototypes.tolist(), vector.tolist(), label=f"m={m}")

        peak_index = int(torch.argmax(vector).item())
        expected_index = int(torch.argmin(torch.abs(prototypes - m)).item())
        result = "PASS" if peak_index == expected_index else "FAIL"
        print(
            f"m={m:>4.1f} | peak_mu={prototypes[peak_index].item():>7.4f} | "
            f"expected_mu={prototypes[expected_index].item():>7.4f} | {result}"
        )

    plt.title("Gaussian Mantissa Encoding Verification")
    plt.xlabel("Prototype position (mu_j)")
    plt.ylabel("Encoding value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("verify_mantissa.png", dpi=200)
    plt.close()
