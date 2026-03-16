"""Verify the full [num] token injection pipeline before training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.encoding.cqe_wrapper import QuantitySpan, replace_with_num_tokens_regex, setup_tokenizer
from src.models.deepquant import DeepQuant


def _load_config(config_path: str) -> dict[str, Any]:
    """Load YAML config from disk."""
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_verification(config: dict[str, Any] | None = None) -> None:
    """Run all verification steps and raise on the first failure."""
    active_config = config or _load_config(str(ROOT / "configs" / "default.yaml"))

    print("Step 1 - Tokenizer setup")
    tokenizer, num_token_id = setup_tokenizer(
        model_name=str(active_config.get("model", {}).get("encoder", "bert-base-uncased")),
        local_files_only=bool(active_config.get("model", {}).get("local_files_only", False)),
        cache_dir=active_config.get("model", {}).get("cache_dir"),
    )
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"[num] token id: {num_token_id}")
    assert num_token_id != 100, "FAIL: [num] is [UNK]"
    print("Step 1 PASS: tokenizer correctly adds [num] special token")

    print("Step 2 - String replacement")
    test_cases = [
        ("revenue over 5 billion dollars", True),
        ("laptop with 256 GB storage", True),
        ("the company reported strong growth", False),
        ("P/E ratio of 15.3", True),
        ("energy: 1046 kJ per serving", True),
    ]
    for text, should_have_num in test_cases:
        modified = replace_with_num_tokens_regex(text)
        has_num = "[num]" in modified
        status = "PASS" if has_num == should_have_num else "FAIL"
        print(f"  {status}: '{text[:40]}' -> '{modified[:40]}'")
        assert has_num == should_have_num, (
            f"Step 2 failed for text={text!r}: expected should_have_num={should_have_num}, "
            f"got modified={modified!r}"
        )

    print("Step 3 - Tokenization survives [num]")
    for text, _ in test_cases:
        modified = replace_with_num_tokens_regex(text)
        if "[num]" not in modified:
            continue
        ids = tokenizer(modified)["input_ids"]
        assert num_token_id in ids, (
            "FAIL: [num] in string but not in token ids!\n"
            f"  text: {modified}\n"
            f"  ids: {ids}\n"
            f"  num_token_id: {num_token_id}"
        )
        pos = ids.index(num_token_id)
        print(f"  PASS: [num] at position {pos} in '{modified[:40]}'")

    print("Step 4 - Injection vector shape")
    model = DeepQuant(active_config)
    model.zero_grad(set_to_none=True)
    sample_span = QuantitySpan(
        text="256",
        mantissa=2.56,
        exponent=2,
        unit="GB",
        concept="storage",
        start_char=12,
        end_char=15,
    )
    text = "laptop with [num] GB storage"
    ids = tokenizer(text, return_tensors="pt")["input_ids"]
    embs = model.quantity_injector(ids, [sample_span])
    hidden_dim = int(active_config.get("model", {}).get("hidden_dim", 768))
    assert embs.shape == (1, ids.shape[1], hidden_dim), (
        f"FAIL: expected embedding shape (1, {ids.shape[1]}, {hidden_dim}), got {tuple(embs.shape)}"
    )
    num_pos = (ids[0] == num_token_id).nonzero(as_tuple=False)[0].item()
    print(f"  PASS: injection vector shape {tuple(embs.shape)}, [num] at position {num_pos}")

    print("Step 5 - Gradient flow")
    embs.sum().backward()
    assert model.quantity_injector.exponent_embedding.weight.grad is not None, (
        "FAIL: exponent embedding gradients are None after backward()"
    )
    print("  PASS: gradients flow through [num] injection")

    print("\n=== VERIFICATION COMPLETE ===")
    print("All 5 steps passed. [num] injection pipeline is working.")
    print("Safe to run training.")


def parse_args() -> argparse.Namespace:
    """Parse verifier CLI arguments."""
    parser = argparse.ArgumentParser(description="Verify WideQuant [num] token injection")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for [num] pipeline verification."""
    args = parse_args()
    try:
        config = _load_config(args.config)
        run_verification(config)
    except Exception as exc:
        print(f"VERIFICATION FAILED: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
