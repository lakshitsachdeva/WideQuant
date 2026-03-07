"""Verify full FinQuant data pipeline before running full training."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover - optional dependency fallback
    BM25Okapi = None

from src.data.finquant_loader import build_and_save_splits
from src.encoding.cqe_wrapper import setup_tokenizer

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Verify WideQuant FinQuant data pipeline")
    parser.add_argument("--data_dir", type=str, default="data/finquant")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_negatives", type=int, default=7)
    return parser.parse_args()


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


def _tokenize_for_bm25(text: str) -> list[str]:
    """Simple lexical tokenizer for BM25."""
    return TOKEN_PATTERN.findall(text.lower())


def _lexical_rank_fallback(query_tokens: list[str], doc_tokens: list[list[str]]) -> list[int]:
    """Rank by lexical overlap if BM25 is unavailable."""
    query_set = set(query_tokens)
    scored: list[tuple[int, int]] = []
    for idx, tokens in enumerate(doc_tokens):
        overlap = len(query_set.intersection(tokens))
        scored.append((overlap, idx))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [idx for _, idx in scored]


def compute_bm25_mrr10(dev_rows: list[dict[str, Any]], sample_size: int, seed: int) -> float:
    """Compute BM25 MRR@10 on sampled dev triples."""
    if not dev_rows:
        return 0.0

    rng = random.Random(seed)
    rows = list(dev_rows)
    if len(rows) > sample_size:
        rows = rng.sample(rows, sample_size)

    reciprocal_ranks: list[float] = []
    for row in rows:
        query = str(row.get("query_text", ""))
        pos_doc = str(row.get("pos_doc_text", ""))
        neg_docs = [str(x) for x in row.get("neg_doc_texts", [])]
        corpus = [pos_doc, *neg_docs]
        tokenized_corpus = [_tokenize_for_bm25(text) for text in corpus]
        query_tokens = _tokenize_for_bm25(query)

        if BM25Okapi is not None:
            bm25 = BM25Okapi(tokenized_corpus)
            scores = bm25.get_scores(query_tokens)
            ranked = sorted(range(len(scores)), key=lambda idx: float(scores[idx]), reverse=True)
        else:
            ranked = _lexical_rank_fallback(query_tokens, tokenized_corpus)

        rr = 0.0
        for rank, doc_idx in enumerate(ranked[:10], start=1):
            if doc_idx == 0:
                rr = 1.0 / float(rank)
                break
        reciprocal_ranks.append(rr)

    return float(sum(reciprocal_ranks) / max(1, len(reciprocal_ranks)))


def _compact(text: str) -> str:
    """Normalize whitespace for readable console output."""
    return re.sub(r"\s+", " ", text).strip()


def _shorten(text: str, max_len: int = 120) -> str:
    """Trim text for side-by-side preview."""
    value = _compact(text)
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def print_random_triples(rows: list[dict[str, Any]], n: int, seed: int) -> None:
    """Print random triples (query, positive, first negative) side by side."""
    if not rows:
        print("No rows available for random sample preview.")
        return

    rng = random.Random(seed)
    samples = rows if len(rows) <= n else rng.sample(rows, n)
    print("\n=== Sample Triples (Query | Positive | First Negative) ===")
    print(f"{'QUERY':<52} | {'POSITIVE':<52} | {'NEGATIVE[0]':<52}")
    print("-" * 162)
    for row in samples:
        query = _shorten(str(row.get("query_text", "")), max_len=52)
        pos = _shorten(str(row.get("pos_doc_text", "")), max_len=52)
        neg_list = row.get("neg_doc_texts", [])
        neg = _shorten(str(neg_list[0] if neg_list else ""), max_len=52)
        print(f"{query:<52} | {pos:<52} | {neg:<52}")


def main() -> None:
    """Run full data conversion and verify expected quality checks."""
    args = parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    print("Running finquant_loader end-to-end on full flare-finqa dataset ...")
    try:
        counts = build_and_save_splits(
            output_dir=args.data_dir,
            seed=int(args.seed),
            n_negatives=int(args.n_negatives),
        )
    except Exception as exc:  # pragma: no cover - runtime integration
        print(f"PIPELINE BROKEN: failed to build full dataset: {exc}")
        raise SystemExit(1) from exc

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.jsonl"
    dev_path = data_dir / "dev.jsonl"
    test_path = data_dir / "test.jsonl"
    for path in [train_path, dev_path, test_path]:
        if not path.exists():
            errors.append(f"Missing expected output file: {path}")

    if errors:
        print("PIPELINE BROKEN")
        for reason in errors:
            print(f"- {reason}")
        raise SystemExit(1)

    train_rows = _load_jsonl(train_path)
    dev_rows = _load_jsonl(dev_path)

    train_total = len(train_rows)
    dev_total = len(dev_rows)
    query_num_pct = 100.0 * sum("[num]" in str(row.get("query_text", "")) for row in train_rows) / max(train_total, 1)
    pos_num_pct = 100.0 * sum("[num]" in str(row.get("pos_doc_text", "")) for row in train_rows) / max(train_total, 1)
    mean_query_spans = (
        sum(len(row.get("query_spans", [])) for row in train_rows) / max(train_total, 1)
    )

    print("\n=== Data Statistics ===")
    print(f"Total train triples: {train_total}")
    print(f"Total dev triples: {dev_total}")
    print(f"% query_text with [num]: {query_num_pct:.2f}%")
    print(f"% pos_doc_text with [num]: {pos_num_pct:.2f}%")
    print(f"Mean quantity spans per query: {mean_query_spans:.3f}")

    if train_total <= 5000:
        errors.append(f"Total train triples too low: {train_total} (expected > 5000)")
    if dev_total <= 500:
        errors.append(f"Total dev triples too low: {dev_total} (expected > 500)")
    if query_num_pct <= 40.0:
        errors.append(f"% query_text with [num] too low: {query_num_pct:.2f}% (expected > 40%)")
    if pos_num_pct <= 40.0:
        errors.append(f"% pos_doc_text with [num] too low: {pos_num_pct:.2f}% (expected > 40%)")
    if mean_query_spans <= 1.0:
        errors.append(
            f"Mean quantity spans per query too low: {mean_query_spans:.3f} (expected > 1.0)"
        )

    print_random_triples(train_rows, n=5, seed=int(args.seed))

    bm25_mrr10 = compute_bm25_mrr10(dev_rows, sample_size=200, seed=int(args.seed))
    print("\n=== BM25 Hardness Check ===")
    print(f"BM25 MRR@10 on 200 random dev triples: {bm25_mrr10:.4f}")
    if bm25_mrr10 > 0.50:
        msg = "WARNING: negatives too easy"
        warnings.append(msg)
        print(msg)
    if bm25_mrr10 < 0.15:
        msg = "WARNING: negatives may be too hard or dataset conversion is broken"
        warnings.append(msg)
        print(msg)
    if 0.15 <= bm25_mrr10 <= 0.50:
        print("BM25 hardness in target range: 0.15 - 0.50")

    print("\n=== Tokenizer [num] Check ===")
    try:
        _, num_id = setup_tokenizer(local_files_only=True)
    except Exception:
        _, num_id = setup_tokenizer(local_files_only=False)
    assert num_id != 100, "CRITICAL: [num] mapped to UNK — tokenizer not set up correctly"
    print(f"[num] token id: {num_id} — PASS")

    print("\n=== Final Summary ===")
    if errors:
        print("PIPELINE BROKEN")
        for reason in errors:
            print(f"- {reason}")
        raise SystemExit(1)

    print("READY FOR TRAINING")
    for warning in warnings:
        print(f"- {warning}")
    print(f"Saved dataset sizes: train={counts['train']} dev={counts['dev']} test={counts['test']}")


if __name__ == "__main__":
    main()
