"""Inspect one training batch to diagnose span metadata and numeric losses."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.encoding.cqe_wrapper import QuantitySpan
from src.models.deepquant import DeepQuant
from src.training.trainer import DeepQuantTrainer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Diagnose one WideQuant training batch")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_dir", type=str, default="data/finquant")
    parser.add_argument("--max_length", type=int, default=256)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows from disk."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class RetrievalJsonlDataset(Dataset):
    """Minimal retrieval dataset wrapper for one-batch diagnostics."""

    def __init__(self, triples: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        """Store triples and tokenizer."""
        self.triples = triples
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.triples)

    def _encode_text(self, text: str, spans: list[dict[str, Any]]) -> dict[str, Any]:
        """Tokenize text and attach quantity spans."""
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "quantity_spans": list(spans),
            "text": text,
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one sample for inspection."""
        item = self.triples[idx]
        return {
            "query_batch": self._encode_text(str(item["query_text"]), list(item.get("query_spans", []))),
            "doc_pos_batch": self._encode_text(str(item["pos_doc_text"]), list(item.get("pos_doc_spans", []))),
            "raw_row": item,
        }


def _collate_identity(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep samples unchanged for batch size 1."""
    return batch[0]


def _count_num_tokens(input_ids: Tensor, num_token_id: int) -> int:
    """Count [num] token occurrences."""
    return int((input_ids.reshape(-1) == int(num_token_id)).sum().item())


def _placeholder_spans_from_input_ids(input_ids: Tensor, num_token_id: int) -> list[QuantitySpan]:
    """Create synthetic spans from [num] token positions in tokenized input."""
    flat_ids = input_ids.reshape(-1)
    positions = (flat_ids == int(num_token_id)).nonzero(as_tuple=False).flatten().tolist()
    return [
        QuantitySpan(
            text="[num]",
            mantissa=1.0,
            exponent=0,
            unit="UNK",
            concept="UNK",
            start_char=int(pos),
            end_char=int(pos) + 1,
        )
        for pos in positions
    ]


def _print_nop_samples(loss_inputs: dict[str, Any]) -> None:
    """Print up to three comparator-score triplets."""
    n_lt = loss_inputs["N_lt_scores"].detach().cpu()
    n_eq = loss_inputs["N_eq_scores"].detach().cpu()
    n_gt = loss_inputs["N_gt_scores"].detach().cpu()
    sample_count = min(3, int(n_lt.numel()))
    print("N_op samples:")
    for idx in range(sample_count):
        print(
            f"  sample {idx + 1}: "
            f"N_lt={float(n_lt[idx]):.6f}, "
            f"N_eq={float(n_eq[idx]):.6f}, "
            f"N_gt={float(n_gt[idx]):.6f}"
        )


def _run_one_pass(
    trainer: DeepQuantTrainer,
    query_batch: dict[str, Any],
    doc_pos_batch: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Run one forward pass and build loss inputs."""
    model_outputs = trainer.model(query_batch, doc_pos_batch, None)
    loss_inputs = trainer._build_loss_inputs(model_outputs)
    loss_dict = trainer.loss_fn(loss_inputs)
    return loss_inputs, loss_dict


def main() -> None:
    """Run one-batch diagnostics over a retrieval JSONL dataset."""
    args = parse_args()
    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing training split: {train_path}")

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    triples = _load_jsonl(train_path)
    if not triples:
        raise ValueError(f"No rows found in {train_path}")

    model = DeepQuant(config)
    dataset = RetrievalJsonlDataset(triples, tokenizer=model.tokenizer, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_collate_identity)
    trainer = DeepQuantTrainer(
        model=model,
        config=config,
        train_dataloader=dataloader,
        dev_dataloader=dataloader,
    )

    batch = next(iter(dataloader))
    query_batch = trainer._to_device(batch["query_batch"])
    doc_pos_batch = trainer._to_device(batch["doc_pos_batch"])

    query_spans = list(query_batch.get("quantity_spans", []))
    pos_doc_spans = list(doc_pos_batch.get("quantity_spans", []))
    query_num_tokens = _count_num_tokens(query_batch["input_ids"], model.num_token_id)
    pos_num_tokens = _count_num_tokens(doc_pos_batch["input_ids"], model.num_token_id)

    print("=== BATCH DIAGNOSTIC ===")
    print("Number of examples in batch: 1")
    print(f"Examples with non-empty query_spans: {int(bool(query_spans))}")
    print(f"Examples with non-empty pos_doc_spans: {int(bool(pos_doc_spans))}")
    print(f"Number of [num] tokens in query input_ids: {query_num_tokens}")
    print(f"Number of [num] tokens in pos_doc input_ids: {pos_num_tokens}")

    loss_inputs, loss_dict = _run_one_pass(trainer, query_batch, doc_pos_batch)
    print(f"Raw L_comp value: {float(loss_dict['L_comp'].detach().cpu()):.6f}")
    print(f"Raw L_reg value: {float(loss_dict['L_reg'].detach().cpu()):.6f}")
    _print_nop_samples(loss_inputs)

    if not query_spans and not pos_doc_spans:
        print("ROOT CAUSE: All quantity spans are empty.")
        print("L_comp requires quantity pairs. With no spans, it computes nothing meaningful.")
        print("Fix: either use CQE to extract spans, or use synthetic quantity data")
        print("that has real [num] tokens with associated span metadata.")
    elif (query_num_tokens > 0 and not query_spans) or (pos_num_tokens > 0 and not pos_doc_spans):
        print("PARTIAL FIX NEEDED: [num] tokens exist but span metadata missing.")
        print("The injector fires but L_comp has no trustworthy span metadata to supervise.")
        print("Fix: reconstruct span metadata from [num] token positions.")

    fixed_query_batch = copy.deepcopy(query_batch)
    fixed_doc_pos_batch = copy.deepcopy(doc_pos_batch)

    if not fixed_query_batch.get("quantity_spans"):
        fixed_query_batch["quantity_spans"] = _placeholder_spans_from_input_ids(
            fixed_query_batch["input_ids"],
            model.num_token_id,
        )
    if not fixed_doc_pos_batch.get("quantity_spans"):
        fixed_doc_pos_batch["quantity_spans"] = _placeholder_spans_from_input_ids(
            fixed_doc_pos_batch["input_ids"],
            model.num_token_id,
        )

    print("\n=== AFTER PLACEHOLDER SPAN RECONSTRUCTION ===")
    print(f"Reconstructed query_spans: {len(fixed_query_batch.get('quantity_spans', []))}")
    print(f"Reconstructed pos_doc_spans: {len(fixed_doc_pos_batch.get('quantity_spans', []))}")

    fixed_loss_inputs, fixed_loss_dict = _run_one_pass(trainer, fixed_query_batch, fixed_doc_pos_batch)
    print(f"New L_comp value: {float(fixed_loss_dict['L_comp'].detach().cpu()):.6f}")
    print(f"New L_reg value: {float(fixed_loss_dict['L_reg'].detach().cpu()):.6f}")
    _print_nop_samples(fixed_loss_inputs)


if __name__ == "__main__":
    main()
