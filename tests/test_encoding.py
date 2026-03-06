"""Pytest coverage for quantity encoding components."""

from __future__ import annotations

from typing import List

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.encoding.cqe_wrapper import QuantitySpan
from src.encoding.quantity_encoder import (
    ExponentEmbedding,
    QuantityInjector,
    gaussian_mantissa_encoding,
)


def test_mantissa_peak_position() -> None:
    """Peak index should match the closest prototype position for each mantissa."""
    values = [-10.0, -5.0, 0.0, 2.56, 5.0, 10.0]
    J_m = 691
    prototypes = torch.linspace(-10.0, 10.0, J_m, dtype=torch.float32)

    for m in values:
        encoded = gaussian_mantissa_encoding(m, J_m=J_m)
        peak_index = int(torch.argmax(encoded).item())
        closest_index = int(torch.argmin(torch.abs(prototypes - m)).item())
        assert peak_index == closest_index


def test_mantissa_dimension() -> None:
    """Mantissa encoder output should always be 691-dimensional by default."""
    encoded = gaussian_mantissa_encoding(2.56)
    assert encoded.shape == (691,)
    assert encoded.dtype == torch.float32


def test_exponent_clipping() -> None:
    """ExponentEmbedding should clip out-of-range exponents to [-20, 20]."""
    module = ExponentEmbedding(num_classes=41, J_e=77)

    high = module(25)
    clipped_high = module(20)
    low = module(-25)
    clipped_low = module(-20)

    assert high.shape == (77,)
    assert low.shape == (77,)
    assert torch.allclose(high, clipped_high)
    assert torch.allclose(low, clipped_low)


def test_exponent_dimension() -> None:
    """Exponent embedding output vector should match J_e=77."""
    module = ExponentEmbedding(num_classes=41, J_e=77)
    out = module(3)
    assert out.shape == (77,)
    assert out.dtype == torch.float32


def test_concat_dimension() -> None:
    """Concatenating exponent + mantissa encodings should produce 768 dims."""
    exp_module = ExponentEmbedding(num_classes=41, J_e=77)
    exponent_vec = exp_module(2)
    mantissa_vec = gaussian_mantissa_encoding(2.56, J_m=691)
    concatenated = torch.cat([exponent_vec, mantissa_vec], dim=0)
    assert concatenated.shape == (768,)


def test_injector_gradients() -> None:
    """QuantityInjector should backpropagate and produce exponent embedding gradients."""
    torch.manual_seed(13)
    vocab_size = 100
    hidden_dim = 768
    num_token_id = 99

    base_embeddings = nn.Embedding(vocab_size, hidden_dim)
    injector = QuantityInjector(
        base_bert_embeddings=base_embeddings,
        num_token_id=num_token_id,
        J=768,
        J_m=691,
        J_e=77,
    )

    input_ids = torch.tensor(
        [
            [1, num_token_id, 5, 6],
            [7, 8, num_token_id, 9],
        ],
        dtype=torch.long,
    )

    spans: List[QuantitySpan] = [
        QuantitySpan(
            text="256",
            mantissa=2.56,
            exponent=2,
            unit="GB",
            concept="storage",
            start_char=0,
            end_char=3,
        ),
        QuantitySpan(
            text="1",
            mantissa=1.0,
            exponent=0,
            unit="TB",
            concept="storage",
            start_char=0,
            end_char=1,
        ),
    ]

    output = injector(input_ids=input_ids, quantity_spans=spans)
    loss = output.sum()
    loss.backward()

    grad = injector.exponent_embedding.embedding.weight.grad
    assert grad is not None
    assert torch.any(grad != 0)
