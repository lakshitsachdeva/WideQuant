"""Diagnostics for WideQuant/DeepQuant training quality and data pipeline."""

from __future__ import annotations

import argparse
import importlib.util
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable, List

import torch
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover
    BM25Okapi = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.deepquant import DeepQuant
from src.training.trainer import DeepQuantTrainer

NUMBER_PATTERN = re.compile(r"(?<!\w)(?:\d+\.?\d*|\.\d+)(?!\w)")
LOSS_TAGS = ["L_retr", "L_quant", "L_reg", "L_comp"]


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Diagnose DeepQuant training behavior")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_deepquant.pt")
    parser.add_argument("--max_train_examples", type=int, default=256)
    parser.add_argument("--max_dev_examples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set reproducibility seeds."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pick_text(record: dict, keys: list[str]) -> str:
    """Pick first valid string field from the record."""
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _replace_numbers_with_num_token(text: str) -> str:
    """Replace numeric literals with [num] token."""
    return NUMBER_PATTERN.sub("[num]", text)


def download_finquant_dataset() -> Any:
    """Download FinQuant-like dataset from known working sources."""
    candidates = [
        ("chancefocus/flare-finqa", None),
        ("ChanceFocus/flare-finqa", None),
        ("dreamerdeo/finqa", None),
        ("ibm/finqa", None),
    ]
    for dataset_name, subset in candidates:
        try:
            dataset = load_dataset(dataset_name) if subset is None else load_dataset(dataset_name, subset)
            print(f"Loaded dataset: {dataset_name}")
            return dataset
        except Exception:
            continue

    raise RuntimeError(
        "Failed to download FinQuant dataset from known sources "
        "(tried chancefocus/flare-finqa, ChanceFocus/flare-finqa, dreamerdeo/finqa, ibm/finqa)."
    )


def build_pairs(split: Iterable[dict], max_examples: int) -> list[dict]:
    """Build query-document pairs from a split."""
    pairs: list[dict] = []
    for item in split:
        query = _pick_text(
            item,
            ["query", "question", "instruction", "input", "problem", "sentence1", "title"],
        )
        doc = _pick_text(
            item,
            ["document", "context", "text", "answer", "sentence2", "passage", "output"],
        )
        if query and doc:
            pairs.append(
                {
                    "query": _replace_numbers_with_num_token(query),
                    "doc": _replace_numbers_with_num_token(doc),
                }
            )
        if len(pairs) >= max_examples:
            break

    if not pairs:
        pairs = [
            {
                "query": "net income over [num] million",
                "doc": "The company reported net income of [num] million dollars.",
            },
            {
                "query": "debt to equity ratio [num]",
                "doc": "Debt-to-equity ratio was [num] in fiscal year [num].",
            },
            {
                "query": "revenue above [num]",
                "doc": "Revenue reached [num] in Q4.",
            },
        ]
    return pairs


class FinQuantPairDataset(Dataset):
    """Pair dataset with BM25 hard-negative assignment."""

    def __init__(self, pairs: List[dict], tokenizer: Any, max_length: int = 128, with_negatives: bool = True) -> None:
        """Initialize dataset and hard-negative mapping."""
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_negatives = with_negatives
        self.hard_negative_indices = self._build_hard_negative_indices() if with_negatives else []

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.pairs)

    def _build_hard_negative_indices(self) -> List[int]:
        """Build one hard negative index per query."""
        docs = [pair["doc"] for pair in self.pairs]
        tokenized_docs = [doc.lower().split() for doc in docs]
        negatives: list[int] = []

        if BM25Okapi is not None:
            bm25 = BM25Okapi(tokenized_docs)
            for i, pair in enumerate(self.pairs):
                scores = bm25.get_scores(pair["query"].lower().split())
                ranked = sorted(range(len(scores)), key=lambda idx: float(scores[idx]), reverse=True)
                neg = next((idx for idx in ranked if idx != i), (i + 1) % len(self.pairs))
                negatives.append(int(neg))
            return negatives

        for i, pair in enumerate(self.pairs):
            q_tokens = set(pair["query"].lower().split())
            ranked = sorted(
                range(len(docs)),
                key=lambda idx: len(q_tokens.intersection(set(docs[idx].lower().split()))),
                reverse=True,
            )
            neg = next((idx for idx in ranked if idx != i), (i + 1) % len(self.pairs))
            negatives.append(int(neg))
        return negatives

    def _encode_text(self, text: str) -> dict:
        """Tokenize one text into model input dict."""
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
            "quantity_spans": [],
        }

    def __getitem__(self, idx: int) -> dict:
        """Return one training sample (query, positive, optional negative)."""
        pair = self.pairs[idx]
        sample = {
            "query_batch": self._encode_text(pair["query"]),
            "doc_pos_batch": self._encode_text(pair["doc"]),
        }
        if self.with_negatives:
            neg_idx = self.hard_negative_indices[idx]
            sample["doc_neg_batch"] = self._encode_text(self.pairs[neg_idx]["doc"])
        return sample


def _collate_identity(batch: list[dict]) -> dict:
    """Pass-through collate for batch_size=1."""
    return batch[0]


def compute_bm25_mrr10(dev_pairs: list[dict]) -> float:
    """Compute BM25 MRR@10 on dev pairs."""
    docs = [pair["doc"] for pair in dev_pairs]
    tokenized_docs = [doc.lower().split() for doc in docs]
    queries = [pair["query"] for pair in dev_pairs]

    if BM25Okapi is not None:
        bm25 = BM25Okapi(tokenized_docs)
        rankings = []
        for query in queries:
            scores = bm25.get_scores(query.lower().split())
            ranked = sorted(range(len(scores)), key=lambda idx: float(scores[idx]), reverse=True)
            rankings.append(ranked)
    else:
        rankings = []
        for query in queries:
            q_tokens = set(query.lower().split())
            ranked = sorted(
                range(len(docs)),
                key=lambda idx: len(q_tokens.intersection(set(docs[idx].lower().split()))),
                reverse=True,
            )
            rankings.append(ranked)

    reciprocal_ranks = []
    for q_idx, ranked_doc_ids in enumerate(rankings):
        rr = 0.0
        for rank, doc_id in enumerate(ranked_doc_ids[:10], start=1):
            if doc_id == q_idx:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
    return float(sum(reciprocal_ranks) / max(len(reciprocal_ranks), 1))


def parse_epoch_losses_from_logs(runs_dir: Path) -> dict[str, dict[int, float]]:
    """Parse train loss scalars from tensorboard event logs if available."""
    tag_to_epoch_values: dict[str, dict[int, float]] = {tag: {} for tag in LOSS_TAGS}
    if not runs_dir.exists():
        return tag_to_epoch_values

    try:
        spec = importlib.util.find_spec("tensorboard.backend.event_processing.event_accumulator")
    except ModuleNotFoundError:
        spec = None
    if spec is None:
        return tag_to_epoch_values

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # type: ignore

    event_files = sorted(runs_dir.rglob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    if not event_files:
        return tag_to_epoch_values

    latest = event_files[-1]
    accumulator = EventAccumulator(str(latest))
    accumulator.Reload()

    available = set(accumulator.Tags().get("scalars", []))
    for tag in LOSS_TAGS:
        full_tag = f"train/{tag}"
        if full_tag not in available:
            continue
        for event in accumulator.Scalars(full_tag):
            tag_to_epoch_values[tag][int(event.step)] = float(event.value)

    return tag_to_epoch_values


def print_epoch_loss_logs(epoch_losses: dict[str, dict[int, float]]) -> bool:
    """Print epoch-1 and epoch-8 loss logs; return pass status."""
    print("=== Loss Logs (Epoch 1 and Epoch 8) ===")
    ok = True
    for tag in LOSS_TAGS:
        e1 = epoch_losses[tag].get(1)
        e8 = epoch_losses[tag].get(8)
        e1_str = "N/A" if e1 is None else f"{e1:.6f}"
        e8_str = "N/A" if e8 is None else f"{e8:.6f}"
        print(f"{tag}: epoch1={e1_str} | epoch8={e8_str}")
        if e1 is None or e8 is None:
            ok = False
    return ok


def main() -> None:
    """Run all requested diagnostics and print a summary."""
    args = parse_args()
    set_seed(args.seed)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # 1) Load checkpoint and model.
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    config = checkpoint.get("config")
    if config is None:
        with open(args.config, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    config.setdefault("model", {})
    config["model"].setdefault("local_files_only", True)

    model = DeepQuant(config)
    load_info = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Missing keys: {len(load_info.missing_keys)} | Unexpected keys: {len(load_info.unexpected_keys)}")

    dataset = download_finquant_dataset()
    train_split = dataset["train"] if "train" in dataset else next(iter(dataset.values()))
    if "validation" in dataset:
        dev_split = dataset["validation"]
    elif "valid" in dataset:
        dev_split = dataset["valid"]
    else:
        dev_split = dataset.get("test", train_split)

    train_pairs = build_pairs(train_split, max_examples=args.max_train_examples)
    dev_pairs = build_pairs(dev_split, max_examples=args.max_dev_examples)

    if model.tokenizer is None:
        raise RuntimeError("Tokenizer unavailable in loaded model; cannot run diagnostics.")

    train_dataset = FinQuantPairDataset(train_pairs, tokenizer=model.tokenizer, with_negatives=True)
    dev_dataset = FinQuantPairDataset(dev_pairs, tokenizer=model.tokenizer, with_negatives=False)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=_collate_identity)
    dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False, collate_fn=_collate_identity)

    trainer = DeepQuantTrainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        dev_dataloader=dev_loader,
    )

    # Restore optional auxiliary states if present.
    if "exp_head_state_dict" in checkpoint:
        trainer.exp_head.load_state_dict(checkpoint["exp_head_state_dict"])
    if "man_head_state_dict" in checkpoint:
        trainer.man_head.load_state_dict(checkpoint["man_head_state_dict"])
    if "unit_head_state_dict" in checkpoint:
        trainer.unit_head.load_state_dict(checkpoint["unit_head_state_dict"])

    # 2) One-batch losses + epoch logs.
    print("\n=== One-Batch Loss Values ===")
    one_batch = next(iter(train_loader))
    query_batch, doc_pos_batch, doc_neg_batch = trainer._unpack_batch(one_batch)
    query_batch = trainer._to_device(query_batch)
    doc_pos_batch = trainer._to_device(doc_pos_batch)
    doc_neg_batch = trainer._to_device(doc_neg_batch) if doc_neg_batch is not None else None
    with torch.no_grad():
        model_outputs = trainer.model(query_batch, doc_pos_batch, doc_neg_batch)
        loss_inputs = trainer._build_loss_inputs(model_outputs)
        loss_values = trainer.loss_fn(loss_inputs)
    for tag in LOSS_TAGS:
        print(f"{tag}: {float(loss_values[tag].detach().cpu()):.6f}")

    epoch_losses = parse_epoch_losses_from_logs(Path("runs"))
    logs_check_pass = print_epoch_loss_logs(epoch_losses)

    # 3) [num] injection token check.
    print("\n=== [num] Injection Check ===")
    sample_text = "company revenue over 5 billion dollars"
    encoded_raw = model.tokenizer(sample_text, return_tensors="pt") if model.tokenizer is not None else None
    num_token_id = int(model.num_token_id)
    appears = False
    if encoded_raw is not None:
        appears = bool((encoded_raw["input_ids"] == num_token_id).any().item())
    print(f"num_token_id: {num_token_id}")
    print(f"[num] appears in input_ids: {appears}")
    if not appears:
        print(
            "CRITICAL: [num] token not appearing in input_ids — "
            "replacement is happening after tokenization, not before"
        )
    num_check_pass = appears

    # 4) BM25 dev-set check.
    print("\n=== BM25 Dev Check ===")
    bm25_mrr10 = compute_bm25_mrr10(dev_pairs)
    print(f"BM25 MRR@10: {bm25_mrr10:.4f}")
    if bm25_mrr10 > 0.50:
        print("WARNING: negatives too easy, model learning lexical matching not numeric reasoning")
    bm25_check_pass = bm25_mrr10 <= 0.50

    # 5) Dataset format check.
    print("\n=== Dataset Format Check (3 samples) ===")
    dataset_check_pass = True
    n_show = min(3, len(train_dataset))
    for idx in range(n_show):
        sample = train_dataset[idx]
        q_text = train_dataset.pairs[idx]["query"]
        p_text = train_dataset.pairs[idx]["doc"]
        neg_idx = train_dataset.hard_negative_indices[idx]
        n_text = train_dataset.pairs[neg_idx]["doc"]

        q_spans = sample["query_batch"].get("quantity_spans", [])
        p_spans = sample["doc_pos_batch"].get("quantity_spans", [])
        n_spans = sample["doc_neg_batch"].get("quantity_spans", [])

        q_has_num = "[num]" in q_text
        p_has_num = "[num]" in p_text
        n_has_num = "[num]" in n_text

        print(f"Sample {idx + 1}:")
        print(f"  query: {q_text}")
        print(f"  positive_doc: {p_text}")
        print(f"  negative_doc: {n_text}")
        print(
            f"  quantity_spans_extracted: "
            f"query={bool(q_spans)} pos={bool(p_spans)} neg={bool(n_spans)}"
        )
        print(
            f"  has_[num]_token: "
            f"query={q_has_num} pos={p_has_num} neg={n_has_num}"
        )

        if not (bool(q_spans) and bool(p_spans) and bool(n_spans) and q_has_num and p_has_num and n_has_num):
            dataset_check_pass = False

    # 6) Summary.
    print("\n=== Summary ===")
    summary = {
        "1) loss_logs_epoch1_epoch8": logs_check_pass,
        "2) num_injection_tokenization": num_check_pass,
        "3) bm25_negative_difficulty": bm25_check_pass,
        "4) dataset_triples_and_spans": dataset_check_pass,
    }
    for key, passed in summary.items():
        print(f"{key}: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
