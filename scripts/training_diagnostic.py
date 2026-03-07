"""Run a 3-epoch real-data diagnostic before full DeepQuant training."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.deepquant import DeepQuant
from src.training.trainer import DeepQuantTrainer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run WideQuant 3-epoch training diagnostic")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_dir", type=str, default="data/finquant")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--plot_path", type=str, default="training_diagnostic.png")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set deterministic seeds."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file into a list of dict rows."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class FinQuantRetrievalDataset(Dataset):
    """Dataset over retrieval triples with multi-hard-negative payloads."""

    def __init__(
        self,
        triples: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int = 256,
        with_negatives: bool = True,
        num_hard_negatives: int = 7,
    ) -> None:
        """Store triples and tokenizer."""
        self.triples = triples
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.with_negatives = with_negatives
        self.num_hard_negatives = int(num_hard_negatives)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.triples)

    def _encode_text(self, text: str, spans: list[dict[str, Any]]) -> dict[str, Any]:
        """Tokenize processed text and carry quantity spans."""
        if spans and "[num]" not in text:
            raise AssertionError("Quantity spans present but [num] missing in text before tokenization.")
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
            "quantity_spans": spans,
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one query-positive sample with multiple hard negatives."""
        item = self.triples[idx]
        query_text = str(item.get("query_text", ""))
        pos_doc_text = str(item.get("pos_doc_text", ""))
        query_spans = list(item.get("query_spans", []))
        pos_doc_spans = list(item.get("pos_doc_spans", []))

        sample: dict[str, Any] = {
            "query_batch": self._encode_text(query_text, query_spans),
            "doc_pos_batch": self._encode_text(pos_doc_text, pos_doc_spans),
        }

        if self.with_negatives:
            neg_texts = [str(x) for x in item.get("neg_doc_texts", [])]
            neg_spans = list(item.get("neg_doc_spans", []))
            neg_pairs = list(zip(neg_texts, neg_spans))
            if not neg_pairs:
                neg_pairs = [(pos_doc_text, pos_doc_spans)]
            if len(neg_pairs) < self.num_hard_negatives:
                neg_pairs = neg_pairs + neg_pairs[: self.num_hard_negatives - len(neg_pairs)]
            neg_pairs = neg_pairs[: self.num_hard_negatives]
            sample["doc_neg_batch"] = [
                self._encode_text(neg_text, list(neg_span_list))
                for neg_text, neg_span_list in neg_pairs
            ]

        return sample


def _collate_identity(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep sample-level dictionaries unchanged for batch size 1."""
    return batch[0]


def plot_loss_curves(losses_by_epoch: dict[str, list[float]], output_path: Path) -> None:
    """Plot per-loss curves across epochs."""
    epochs = [1, 2, 3]
    plt.figure(figsize=(8, 5))
    for key in ["L_retr", "L_quant", "L_reg", "L_comp"]:
        values = losses_by_epoch.get(key, [0.0, 0.0, 0.0])
        plt.plot(epochs, values, marker="o", label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Diagnostic Loss Curves (3 Epochs)")
    plt.xticks(epochs)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    """Run full-data 3-epoch diagnostic."""
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.jsonl"
    dev_path = data_dir / "dev.jsonl"
    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError(
            f"Missing dataset files at {data_dir}. "
            "Expected train.jsonl and dev.jsonl generated by finquant_loader."
        )

    train_rows = _load_jsonl(train_path)
    dev_rows = _load_jsonl(dev_path)
    print(f"Train dataset size: {len(train_rows)}")
    print(f"Dev dataset size: {len(dev_rows)}")

    if len(train_rows) < 1000:
        raise ValueError(f"Training dataset too small: {len(train_rows)} (expected >= 1000)")
    if len(dev_rows) < 100:
        raise ValueError(f"Dev dataset too small: {len(dev_rows)} (expected >= 100)")

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config.setdefault("training", {})
    config["training"]["warmup_ratio"] = 0.10
    config["training"]["log_every_steps"] = 50

    model = DeepQuant(config)
    train_dataset = FinQuantRetrievalDataset(
        triples=train_rows,
        tokenizer=model.tokenizer,
        max_length=args.max_length,
        with_negatives=True,
        num_hard_negatives=7,
    )
    dev_dataset = FinQuantRetrievalDataset(
        triples=dev_rows,
        tokenizer=model.tokenizer,
        max_length=args.max_length,
        with_negatives=False,
        num_hard_negatives=7,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=_collate_identity)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False, collate_fn=_collate_identity)

    trainer = DeepQuantTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        dev_dataloader=dev_dataloader,
    )

    steps_per_epoch = len(train_dataloader)
    opt_steps_per_epoch = int(math.ceil(steps_per_epoch / float(max(trainer.gradient_accumulation_steps, 1))))
    total_optimizer_steps = max(1, opt_steps_per_epoch * 3)
    warmup_steps = int(total_optimizer_steps * trainer.warmup_ratio)
    trainer.bert_scheduler = trainer._build_linear_warmup_scheduler(
        trainer.bert_optimizer,
        total_steps=total_optimizer_steps,
        warmup_steps=warmup_steps,
    )
    trainer.head_scheduler = trainer._build_linear_warmup_scheduler(
        trainer.head_optimizer,
        total_steps=total_optimizer_steps,
        warmup_steps=warmup_steps,
    )
    print(f"Scheduler: total_steps={total_optimizer_steps}, warmup_steps={warmup_steps}")

    losses_by_epoch: dict[str, list[float]] = {"L_retr": [], "L_quant": [], "L_reg": [], "L_comp": []}
    warnings: list[str] = []
    criticals: list[str] = []
    dev_mrr_epoch3 = 0.0

    for epoch in range(1, 4):
        train_losses = trainer.train_epoch(epoch)
        dev_metrics = trainer.evaluate(dev_dataloader)

        l_retr = float(train_losses.get("L_retr", 0.0))
        l_quant = float(train_losses.get("L_quant", 0.0))
        l_reg = float(train_losses.get("L_reg", 0.0))
        l_comp = float(train_losses.get("L_comp", 0.0))
        losses_by_epoch["L_retr"].append(l_retr)
        losses_by_epoch["L_quant"].append(l_quant)
        losses_by_epoch["L_reg"].append(l_reg)
        losses_by_epoch["L_comp"].append(l_comp)

        print(
            f"Epoch {epoch} | "
            f"L_retr: {l_retr:.4f} | L_quant: {l_quant:.4f} | "
            f"L_reg: {l_reg:.4f} | L_comp: {l_comp:.4f}"
        )

        if epoch == 1 and l_retr <= 0.1:
            criticals.append(
                f"CRITICAL: L_retr at epoch 1 is {l_retr:.6f} (expected > 0.1) — InfoNCE may not be working"
            )

        for loss_name, loss_value in [("L_retr", l_retr), ("L_quant", l_quant), ("L_reg", l_reg), ("L_comp", l_comp)]:
            if f"{loss_value:.6f}" == "0.000000":
                criticals.append(f"CRITICAL: {loss_name} is exactly 0.000000 at epoch {epoch} — loss is dead")

        if epoch > 1:
            prev = losses_by_epoch["L_retr"][epoch - 2]
            if l_retr >= prev:
                warnings.append(
                    f"WARN: L_retr did not decrease from epoch {epoch - 1} to {epoch} ({prev:.6f} -> {l_retr:.6f})"
                )

        if epoch == 3:
            dev_mrr_epoch3 = float(dev_metrics["MRR@10"])

    print(f"Epoch 3 Dev MRR@10: {dev_mrr_epoch3:.4f}")
    if dev_mrr_epoch3 > 0.50:
        print("On track — run full 8 epochs")
    if dev_mrr_epoch3 < 0.30:
        print("UNDERPERFORMING — diagnose before full run")

    plot_path = Path(args.plot_path)
    plot_loss_curves(losses_by_epoch, output_path=plot_path)
    print(f"Saved loss plot: {plot_path}")

    if warnings:
        print("Warnings:")
        for msg in warnings:
            print(msg)
    if criticals:
        print("Critical findings:")
        for msg in criticals:
            print(msg)
        trainer.writer.flush()
        trainer.writer.close()
        raise SystemExit(1)

    trainer.writer.flush()
    trainer.writer.close()


if __name__ == "__main__":
    main()
