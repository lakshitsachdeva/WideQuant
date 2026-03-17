"""Train DeepQuant on retrieval triples built from FinQuant or MS MARCO."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, List

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.finquant_loader import build_and_save_splits as build_finquant_splits
from src.data.finquant_loader import verify_hard_negatives as verify_finquant_hard_negatives
from src.data.msmarco_loader import build_and_save_splits as build_msmarco_splits
from src.data.msmarco_loader import verify_hard_negatives as verify_msmarco_hard_negatives
from src.models.deepquant import DeepQuant
from src.training.trainer import DeepQuantTrainer
from scripts.verify_num_injection import run_verification


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train DeepQuant on retrieval triples")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_dir", type=str, default="data/finquant")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_dev_examples", type=int, default=None)
    parser.add_argument("--rebuild_data", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--build_max_examples", type=int, default=None)
    parser.add_argument("--build_skip_cqe", action="store_true")
    parser.add_argument("--streaming_data", action="store_true")
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set deterministic seeds."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows from disk."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _resolve_dataset_backend(data_dir: Path) -> tuple[str, Any, Any]:
    """Resolve dataset-specific builder and verifier from the output directory."""
    data_dir_text = str(data_dir).lower()
    if "combined" in data_dir_text:
        return "combined", None, verify_msmarco_hard_negatives
    if "msmarco" in data_dir_text:
        return "msmarco", build_msmarco_splits, verify_msmarco_hard_negatives
    return "finquant", build_finquant_splits, verify_finquant_hard_negatives


def _ensure_dataset_ready(
    data_dir: Path,
    seed: int,
    rebuild_data: bool,
    n_negatives: int,
    build_max_examples: int | None,
    build_skip_cqe: bool,
    streaming_data: bool,
    hf_cache_dir: str | None,
) -> None:
    """Build retrieval triples if missing or incompatible with required negative count."""
    dataset_name, dataset_builder, _ = _resolve_dataset_backend(data_dir)
    train_path = data_dir / "train.jsonl"
    dev_path = data_dir / "dev.jsonl"
    test_path = data_dir / "test.jsonl"

    need_build = rebuild_data or not (train_path.exists() and dev_path.exists() and test_path.exists())
    if not need_build and train_path.exists():
        sample_rows = _load_jsonl(train_path)[:1]
        if sample_rows:
            negs = sample_rows[0].get("neg_doc_texts", [])
            if len(negs) < n_negatives:
                need_build = True

    if need_build:
        if dataset_builder is None:
            raise ValueError(
                f"Dataset directory {data_dir} is missing train/dev/test JSONL files. "
                "Build the combined dataset first with "
                "src/data/synthetic_quantity_triples.py."
            )
        print(
            f"Building {dataset_name} retrieval triples at {data_dir} "
            f"with {n_negatives} hard negatives/query ..."
        )
        counts = dataset_builder(
            output_dir=str(data_dir),
            seed=seed,
            n_negatives=n_negatives,
            max_examples=build_max_examples,
            skip_cqe=build_skip_cqe,
            streaming=streaming_data,
            cache_dir=hf_cache_dir,
        )
        print(f"Built splits: train={counts['train']} dev={counts['dev']} test={counts['test']}")
    else:
        print(f"Using existing retrieval triples at {data_dir}")


class FinQuantRetrievalDataset(Dataset):
    """Dataset over retrieval triples with multi-hard-negative payloads."""

    def __init__(
        self,
        triples: List[dict[str, Any]],
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
            raise AssertionError("Quantity spans present but [num] missing in text; replacement must happen pre-tokenization.")
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
        query_spans = item.get("query_spans", [])
        pos_doc_spans = item.get("pos_doc_spans", [])

        sample: dict[str, Any] = {
            "query_batch": self._encode_text(query_text, query_spans),
            "doc_pos_batch": self._encode_text(pos_doc_text, pos_doc_spans),
        }

        if self.with_negatives:
            neg_texts = list(item.get("neg_doc_texts", []))
            neg_spans = list(item.get("neg_doc_spans", []))
            neg_pairs = list(zip(neg_texts, neg_spans))

            if not neg_pairs:
                neg_pairs = [(pos_doc_text, pos_doc_spans)]
            if len(neg_pairs) < self.num_hard_negatives:
                neg_pairs = neg_pairs + neg_pairs[: self.num_hard_negatives - len(neg_pairs)]
            neg_pairs = neg_pairs[: self.num_hard_negatives]

            sample["doc_neg_batch"] = [
                self._encode_text(str(neg_text), list(neg_span_list))
                for neg_text, neg_span_list in neg_pairs
            ]

        return sample


def _collate_identity(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep per-sample dicts unchanged for batch size 1."""
    return batch[0]


def _trim_examples(rows: list[dict[str, Any]], max_examples: int | None) -> list[dict[str, Any]]:
    """Trim rows to max_examples if provided."""
    if max_examples is None:
        return rows
    return rows[: max(0, int(max_examples))]


def _evaluate_optional_dev_splits(
    trainer: DeepQuantTrainer,
    model: DeepQuant,
    data_dir: Path,
    max_length: int,
    num_hard_negatives: int,
) -> dict[str, dict[str, float]]:
    """Evaluate optional named dev splits such as dev_msmarco/dev_synthetic if they exist."""
    split_files = {
        "MS MARCO dev": data_dir / "dev_msmarco.jsonl",
        "Synthetic dev": data_dir / "dev_synthetic.jsonl",
    }
    results: dict[str, dict[str, float]] = {}

    ckpt_path = trainer.ckpt_dir / "best_deepquant.pt"
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=trainer.device)
        model.load_state_dict(state["model_state_dict"])

    for label, path in split_files.items():
        if not path.exists():
            continue
        rows = _load_jsonl(path)
        if not rows:
            continue
        dataset = FinQuantRetrievalDataset(
            triples=rows,
            tokenizer=model.tokenizer,
            max_length=max_length,
            with_negatives=False,
            num_hard_negatives=num_hard_negatives,
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_collate_identity)
        metrics = trainer.evaluate(dataloader)
        results[label] = metrics
    return results


def main() -> None:
    """Run full DeepQuant training with scheduler and gate checks."""
    args = parse_args()
    set_seed(args.seed)

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config.setdefault("model", {})
    config["model"].setdefault("local_files_only", False)
    if args.hf_cache_dir is not None:
        config["model"]["cache_dir"] = args.hf_cache_dir

    config.setdefault("training", {})
    config["training"]["warmup_ratio"] = 0.10
    config["training"]["log_every_steps"] = 50
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)
    else:
        config["training"]["epochs"] = int(config["training"].get("epochs", 8))

    run_verification(config)

    data_dir = Path(args.data_dir)
    n_hard_negatives = 7
    _ensure_dataset_ready(
        data_dir=data_dir,
        seed=args.seed,
        rebuild_data=args.rebuild_data,
        n_negatives=n_hard_negatives,
        build_max_examples=args.build_max_examples,
        build_skip_cqe=args.build_skip_cqe,
        streaming_data=args.streaming_data,
        hf_cache_dir=args.hf_cache_dir,
    )

    train_rows = _load_jsonl(data_dir / "train.jsonl")
    dev_rows = _load_jsonl(data_dir / "dev.jsonl")

    if args.dry_run:
        train_rows = _trim_examples(train_rows, 32)
        dev_rows = _trim_examples(dev_rows, 32)

    train_rows = _trim_examples(train_rows, args.max_train_examples)
    dev_rows = _trim_examples(dev_rows, args.max_dev_examples)

    print(f"Train triples: {len(train_rows)} | Dev triples: {len(dev_rows)}")

    model = DeepQuant(config)
    train_dataset = FinQuantRetrievalDataset(
        triples=train_rows,
        tokenizer=model.tokenizer,
        max_length=args.max_length,
        with_negatives=True,
        num_hard_negatives=n_hard_negatives,
    )
    dev_dataset = FinQuantRetrievalDataset(
        triples=dev_rows,
        tokenizer=model.tokenizer,
        max_length=args.max_length,
        with_negatives=False,
        num_hard_negatives=n_hard_negatives,
    )

    if not args.dry_run:
        if len(train_dataset) < 1000:
            raise ValueError(
                f"Training set too small: {len(train_dataset)} examples. "
                f"Run finquant_loader.py first to build full dataset. "
                f"Expected > 5000 train triples."
            )
        if len(dev_dataset) < 100:
            raise ValueError(
                f"Dev set too small: {len(dev_dataset)} examples. "
                f"MRR@10 on < 100 examples is statistically meaningless."
            )

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=_collate_identity)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False, collate_fn=_collate_identity)

    trainer = DeepQuantTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        dev_dataloader=dev_dataloader,
    )
    trainer.ckpt_dir = Path(args.checkpoint_dir)
    trainer.ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("DRY RUN — not a real training run, gate not evaluated")
        train_losses = trainer.train_epoch(epoch=1)
        dev_metrics = trainer.evaluate(dev_dataloader)
        print(
            f"Dry run epoch 1 | Train Loss: {train_losses.get('total', 0.0):.4f} | "
            f"Dev MRR@10: {dev_metrics['MRR@10']:.4f}"
        )
        trainer.writer.flush()
        trainer.writer.close()
        return

    dataset_name, _, hard_negative_verifier = _resolve_dataset_backend(data_dir)
    print(f"Training dataset backend: {dataset_name}")
    bm25_mrr = hard_negative_verifier(dev_rows, sample_n=min(100, len(dev_rows)), seed=args.seed)
    results = trainer.train(n_epochs=int(config["training"]["epochs"]))
    best_mrr10 = float(results["best_mrr10"])
    print(f"Final best MRR@10: {best_mrr10:.4f}")
    print(f"Gate evaluated on {len(dev_dataset)} dev examples")
    if len(dev_dataset) < 100:
        print("WARNING: Gate passed on tiny dev set — result is not meaningful")

    if len(dev_dataset) >= 100 and best_mrr10 >= 0.70:
        print("PHASE 2 COMPLETE. PROCEED TO PHASE 3")
    else:
        loss_curves = results.get("loss_curves", {})
        print("Loss curves by epoch:")
        for key in ["L_retr", "L_quant", "L_reg", "L_comp", "L_arith", "total"]:
            curve = loss_curves.get(key, [])
            print(f"{key}: {curve}")
        print("DIAGNOSE BEFORE SCALING")

    loss_curves = results.get("loss_curves", {})
    l_retr_curve = loss_curves.get("L_retr", [])
    l_quant_curve = loss_curves.get("L_quant", [])
    l_reg_curve = loss_curves.get("L_reg", [])
    l_comp_curve = loss_curves.get("L_comp", [])
    ckpt_path = trainer.ckpt_dir / "best_deepquant.pt"
    ckpt_size_mb = ckpt_path.stat().st_size / (1024.0 * 1024.0) if ckpt_path.exists() else 0.0

    print("=== DEEPQUANT TRAINING COMPLETE ===")
    print(f"Best Dev MRR@10: {best_mrr10:.4f}")
    print(f"Gate (>= 0.70): {'PASS' if best_mrr10 >= 0.70 else 'FAIL'}")
    print()
    print("Individual losses at epoch 8:")
    print(f"L_retr:  {l_retr_curve[-1] if l_retr_curve else 0.0:.4f}  (should be < 1.0)")
    print(f"L_quant: {l_quant_curve[-1] if l_quant_curve else 0.0:.4f}  (should be < 0.5)")
    print(f"L_reg:   {l_reg_curve[-1] if l_reg_curve else 0.0:.4f}  (should be < 0.1)")
    print(f"L_comp:  {l_comp_curve[-1] if l_comp_curve else 0.0:.4f}  (should be < 0.5)")
    print()
    print(f"[num] token verified in training batches: {'YES' if trainer._checked_first_num_batch else 'NO'}")
    print(f"Hard negative BM25 MRR: {bm25_mrr:.4f} (should be < 0.40)")
    print()
    print(f"Checkpoint saved: {ckpt_path}")
    print(f"File size: {ckpt_size_mb:.2f} MB")

    extra_dev_results = _evaluate_optional_dev_splits(
        trainer=trainer,
        model=model,
        data_dir=data_dir,
        max_length=args.max_length,
        num_hard_negatives=n_hard_negatives,
    )
    for label, metrics in extra_dev_results.items():
        print(
            f"{label} | MRR@10: {metrics['MRR@10']:.4f} | "
            f"NDCG@10: {metrics['NDCG@10']:.4f} | P@10: {metrics['P@10']:.4f} | "
            f"R@100: {metrics['R@100']:.4f}"
        )

    if best_mrr10 >= 0.70:
        print("PHASE 2 GATE CLEARED. PROCEED TO PHASE 3.")
    else:
        print("If FAIL: paste this output and diagnose before proceeding.")


if __name__ == "__main__":
    main()
