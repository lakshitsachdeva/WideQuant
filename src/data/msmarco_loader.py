"""MS MARCO loader that converts triplets into WideQuant retrieval triples."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from datasets import Dataset, load_dataset
from tqdm import tqdm

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover - optional dependency fallback
    BM25Okapi = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.encoding.cqe_wrapper import QuantitySpan, no_numbers_in_text, reconstruct_spans_from_num_tokens, replace_with_num_tokens_regex

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", flags=re.IGNORECASE)


def _safe_text(value: Any) -> str:
    """Normalize arbitrary values into stripped strings."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize text for lightweight lexical indexing."""
    return TOKEN_PATTERN.findall(text.lower())


def _lexical_rank_fallback(query_tokens: list[str], doc_tokens: list[list[str]]) -> list[int]:
    """Rank documents by token overlap when BM25 is unavailable."""
    query_set = set(query_tokens)
    scored: list[tuple[int, int]] = []
    for idx, tokens in enumerate(doc_tokens):
        scored.append((len(query_set.intersection(tokens)), idx))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [idx for _, idx in scored]


def _save_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    """Write rows to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _span_to_json(span: QuantitySpan) -> dict[str, Any]:
    """Serialize a QuantitySpan for JSONL output."""
    return {
        "text": span.text,
        "mantissa": float(span.mantissa),
        "exponent": int(span.exponent),
        "unit": span.unit,
        "concept": span.concept,
        "start_char": int(span.start_char),
        "end_char": int(span.end_char),
    }


def _bm25_mrr_at_10(triples: Sequence[dict[str, Any]], sample_size: int = 100, seed: int = 42) -> float:
    """Compute BM25 MRR@10 over positive-vs-negative candidate sets."""
    if not triples:
        return 0.0
    rng = random.Random(seed)
    sampled = list(triples)
    if len(sampled) > sample_size:
        sampled = rng.sample(sampled, sample_size)

    reciprocal_ranks: list[float] = []
    for triple in sampled:
        query_tokens = _tokenize_for_bm25(str(triple["query_text"]))
        corpus = [str(triple["pos_doc_text"]), *[str(text) for text in triple["neg_doc_texts"]]]
        tokenized_corpus = [_tokenize_for_bm25(text) for text in corpus]
        if BM25Okapi is not None:
            bm25 = BM25Okapi(tokenized_corpus)
            scores = bm25.get_scores(query_tokens)
            ranked = sorted(range(len(scores)), key=lambda idx: float(scores[idx]), reverse=True)
        else:
            ranked = _lexical_rank_fallback(query_tokens, tokenized_corpus)

        rr = 0.0
        for rank, idx in enumerate(ranked[:10], start=1):
            if idx == 0:
                rr = 1.0 / float(rank)
                break
        reciprocal_ranks.append(rr)
    return float(sum(reciprocal_ranks) / max(len(reciprocal_ranks), 1))


def verify_hard_negatives(
    dev_triples: Sequence[dict[str, Any]],
    sample_n: int = 100,
    seed: int = 42,
    pass_threshold: float = 0.40,
) -> float:
    """Evaluate BM25 MRR@10 on positive-vs-negative ranking and print pass/fail."""
    bm25_mrr = _bm25_mrr_at_10(dev_triples, sample_size=sample_n, seed=seed)
    status = "PASS" if bm25_mrr < pass_threshold else "FAIL"
    print(f"Hard negatives BM25 MRR@10: {bm25_mrr:.4f} [{status}]")
    return bm25_mrr


def _extract_triplet_fields(example: dict[str, Any]) -> tuple[str, str, str]:
    """Extract query, positive, and negative strings from a dataset row."""
    query = _safe_text(example.get("query") or example.get("question"))
    positive = _safe_text(
        example.get("positive")
        or example.get("pos")
        or example.get("passage")
        or example.get("positive_passage")
    )
    negative = _safe_text(
        example.get("negative")
        or example.get("neg")
        or example.get("negative_passage")
    )

    if isinstance(example.get("positive"), list) and example["positive"]:
        positive = _safe_text(example["positive"][0])
    if isinstance(example.get("negative"), list) and example["negative"]:
        negative = _safe_text(example["negative"][0])

    if not query or not positive or not negative:
        raise ValueError("MS MARCO row is missing one of query/positive/negative text.")
    return query, positive, negative


def _load_triplet_dataset(max_examples: int | None = None, cache_dir: str | None = None) -> Dataset:
    """Load a manageable MS MARCO BM25 triplet slice from Hugging Face."""
    requested = int(max_examples) if max_examples is not None else 50000
    requested = max(1, requested)
    split = f"train[:{requested}]"
    print(
        "Loading MS MARCO triplets from sentence-transformers/msmarco-bm25",
        "subset=triplet",
        f"split={split}",
    )
    return load_dataset(
        "sentence-transformers/msmarco-bm25",
        "triplet",
        split=split,
        cache_dir=cache_dir,
    )


def _build_candidate_index(
    candidate_docs: Sequence[str],
    max_postings_per_token: int = 4000,
) -> tuple[list[list[str]], dict[str, list[int]]]:
    """Build tokenized corpus and capped inverted index for approximate BM25 mining."""
    tokenized_docs = [_tokenize_for_bm25(text) for text in candidate_docs]
    postings: defaultdict[str, list[int]] = defaultdict(list)
    for doc_idx, tokens in enumerate(tokenized_docs):
        for token in set(tokens):
            postings[token].append(doc_idx)

    filtered_postings: dict[str, list[int]] = {}
    for token, doc_ids in postings.items():
        if len(doc_ids) <= max_postings_per_token:
            filtered_postings[token] = doc_ids
    return tokenized_docs, filtered_postings


def _mine_bm25_negatives(
    prepared_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
    num_extra_negatives: int = 6,
    shortlist_size: int = 200,
) -> list[list[str]]:
    """Mine additional lexical hard negatives from the candidate corpus."""
    candidate_docs = [str(item["pos_doc_text"]) for item in candidate_rows]
    candidate_source_ids = [str(item["source_id"]) for item in candidate_rows]
    tokenized_docs, inverted_index = _build_candidate_index(candidate_docs)

    mined: list[list[str]] = []
    progress = tqdm(prepared_rows, desc="Mining MS MARCO BM25 negatives", unit="qry")
    for row in progress:
        query_text = str(row["query_text"])
        query_tokens = _tokenize_for_bm25(query_text)
        query_source_id = str(row["source_id"])
        positive_text = str(row["pos_doc_text"])
        original_negative = str(row["original_negative"])

        overlap_counts: Counter[int] = Counter()
        for token in query_tokens:
            for doc_idx in inverted_index.get(token, []):
                if candidate_source_ids[doc_idx] == query_source_id:
                    continue
                overlap_counts[doc_idx] += 1

        if overlap_counts:
            candidate_ids = [doc_idx for doc_idx, _ in overlap_counts.most_common(shortlist_size)]
        else:
            candidate_ids = []

        if BM25Okapi is not None and candidate_ids:
            local_docs = [tokenized_docs[doc_idx] for doc_idx in candidate_ids]
            local_bm25 = BM25Okapi(local_docs)
            local_scores = local_bm25.get_scores(query_tokens)
            ranked_ids = [
                candidate_ids[idx]
                for idx in sorted(range(len(candidate_ids)), key=lambda item: float(local_scores[item]), reverse=True)
            ]
        else:
            local_docs = [tokenized_docs[doc_idx] for doc_idx in candidate_ids]
            ranked_local = _lexical_rank_fallback(query_tokens, local_docs) if candidate_ids else []
            ranked_ids = [candidate_ids[idx] for idx in ranked_local]

        selected: list[str] = []
        selected_set: set[str] = {original_negative, positive_text}
        for doc_idx in ranked_ids:
            candidate_text = candidate_docs[doc_idx]
            if candidate_text in selected_set:
                continue
            selected.append(candidate_text)
            selected_set.add(candidate_text)
            if len(selected) >= num_extra_negatives:
                break

        if len(selected) < num_extra_negatives:
            pool = [doc for doc in candidate_docs if doc not in selected_set]
            random.shuffle(pool)
            selected.extend(pool[: num_extra_negatives - len(selected)])

        mined.append(selected[:num_extra_negatives])
    return mined


def _prepare_rows(dataset: Dataset) -> list[dict[str, Any]]:
    """Normalize triplets into processed retrieval rows."""
    prepared: list[dict[str, Any]] = []
    progress = tqdm(dataset, desc="Preparing MS MARCO triplets", unit="row")
    for idx, example in enumerate(progress):
        query, positive, negative = _extract_triplet_fields(dict(example))
        query_text = replace_with_num_tokens_regex(query)
        pos_doc_text = replace_with_num_tokens_regex(positive)
        neg_doc_text = replace_with_num_tokens_regex(negative)

        assert "[num]" in query_text or no_numbers_in_text(query), f"Failed [num] replacement for query: {query[:100]}"
        assert "[num]" in pos_doc_text or no_numbers_in_text(positive), f"Failed [num] replacement for positive: {positive[:100]}"
        assert "[num]" in neg_doc_text or no_numbers_in_text(negative), f"Failed [num] replacement for negative: {negative[:100]}"

        prepared.append(
            {
                "source_id": f"msmarco:{idx}",
                "query_text": query_text,
                "query_spans": [_span_to_json(span) for span in reconstruct_spans_from_num_tokens(query)],
                "pos_doc_text": pos_doc_text,
                "pos_doc_spans": [_span_to_json(span) for span in reconstruct_spans_from_num_tokens(positive)],
                "original_negative": neg_doc_text,
                "original_negative_spans": [_span_to_json(span) for span in reconstruct_spans_from_num_tokens(negative)],
            }
        )
    return prepared


def _split_rows(prepared_rows: Sequence[dict[str, Any]], seed: int = 42) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows into train/dev/test."""
    rows = list(prepared_rows)
    rng = random.Random(seed)
    rng.shuffle(rows)

    total = len(rows)
    if total >= 50000:
        train_n, dev_n, test_n = 40000, 5000, 5000
    else:
        train_n = max(1, int(0.8 * total))
        remaining = max(2, total - train_n)
        dev_n = max(1, remaining // 2)
        test_n = max(1, total - train_n - dev_n)
    train_rows = rows[:train_n]
    dev_rows = rows[train_n : train_n + dev_n]
    test_rows = rows[train_n + dev_n : train_n + dev_n + test_n]
    return train_rows, dev_rows, test_rows


def _materialize_triples(
    prepared_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach the original negative plus six mined BM25 negatives."""
    mined_negatives = _mine_bm25_negatives(prepared_rows, candidate_rows, num_extra_negatives=6)
    candidate_span_map = {
        str(item["pos_doc_text"]): list(item.get("pos_doc_spans", []))
        for item in candidate_rows
    }
    triples: list[dict[str, Any]] = []
    for row, extra_negatives in zip(prepared_rows, mined_negatives):
        negative_texts = [str(row["original_negative"]), *[str(text) for text in extra_negatives]]
        negative_spans = [list(row.get("original_negative_spans", []))]
        for text in extra_negatives:
            negative_spans.append(list(candidate_span_map.get(str(text), [])))
        triples.append(
            {
                "query_text": str(row["query_text"]),
                "query_spans": list(row["query_spans"]),
                "pos_doc_text": str(row["pos_doc_text"]),
                "pos_doc_spans": list(row["pos_doc_spans"]),
                "neg_doc_texts": negative_texts,
                "neg_doc_spans": negative_spans,
            }
        )
    return triples


def verify(triples: Sequence[dict[str, Any]], seed: int = 42) -> dict[str, float]:
    """Print dataset diagnostics and return them as a dictionary."""
    total = len(triples)
    if total == 0:
        print("Total triples: 0")
        return {
            "total_triples": 0.0,
            "pct_query_has_num": 0.0,
            "pct_pos_has_num": 0.0,
            "mean_query_spans": 0.0,
            "mean_pos_doc_spans": 0.0,
            "bm25_mrr10_dev_100": 0.0,
        }

    pct_query_has_num = 100.0 * sum("[num]" in row["query_text"] for row in triples) / float(total)
    pct_pos_has_num = 100.0 * sum("[num]" in row["pos_doc_text"] for row in triples) / float(total)
    mean_query_spans = sum(len(row.get("query_spans", [])) for row in triples) / float(total)
    mean_pos_doc_spans = sum(len(row.get("pos_doc_spans", [])) for row in triples) / float(total)
    bm25_mrr = _bm25_mrr_at_10(triples, sample_size=min(100, total), seed=seed)

    print(f"Total triples: {total}")
    print(f"% of queries containing [num]: {pct_query_has_num:.2f}")
    print(f"% of positive docs containing [num]: {pct_pos_has_num:.2f}")
    print(f"Mean query spans: {mean_query_spans:.2f}")
    print(f"Mean positive-doc spans: {mean_pos_doc_spans:.2f}")
    print(f"BM25 MRR@10 on dev hard negatives: {bm25_mrr:.4f}")

    sample_count = min(3, total)
    rng = random.Random(seed)
    sample_rows = rng.sample(list(triples), sample_count) if total >= sample_count else list(triples)
    for idx, row in enumerate(sample_rows, start=1):
        print(f"Sample triple {idx}")
        print(f"  Query: {str(row['query_text'])[:160]}")
        print(f"  Pos:   {str(row['pos_doc_text'])[:160]}")
        print(f"  Neg:   {str(row['neg_doc_texts'][0])[:160]}")

    return {
        "total_triples": float(total),
        "pct_query_has_num": float(pct_query_has_num),
        "pct_pos_has_num": float(pct_pos_has_num),
        "mean_query_spans": float(mean_query_spans),
        "mean_pos_doc_spans": float(mean_pos_doc_spans),
        "bm25_mrr10_dev_100": float(bm25_mrr),
    }


def build_and_save_splits(
    output_dir: str = "data/msmarco",
    seed: int = 42,
    n_negatives: int = 7,
    max_examples: int | None = None,
    skip_cqe: bool = True,
    streaming: bool = False,
    cache_dir: str | None = None,
) -> dict[str, int]:
    """Build MS MARCO retrieval triples in WideQuant format and save them."""
    del skip_cqe  # Baseline path always uses regex [num] replacement only.
    del streaming
    if int(n_negatives) != 7:
        print(f"Requested n_negatives={n_negatives}; MS MARCO loader currently emits exactly 7 negatives.")

    random.seed(seed)
    dataset = _load_triplet_dataset(max_examples=max_examples, cache_dir=cache_dir)
    prepared_rows = _prepare_rows(dataset)
    train_rows, dev_rows, test_rows = _split_rows(prepared_rows, seed=seed)
    global_pool = [*train_rows, *dev_rows, *test_rows]

    print(
        "MS MARCO split sizes:",
        f"train={len(train_rows)}",
        f"dev={len(dev_rows)}",
        f"test={len(test_rows)}",
    )

    train_triples = _materialize_triples(train_rows, train_rows)
    dev_triples = _materialize_triples(dev_rows, global_pool)
    test_triples = _materialize_triples(test_rows, global_pool)

    output_path = Path(output_dir)
    train_path = output_path / "train.jsonl"
    dev_path = output_path / "dev.jsonl"
    test_path = output_path / "test.jsonl"

    _save_jsonl(train_triples, train_path)
    _save_jsonl(dev_triples, dev_path)
    _save_jsonl(test_triples, test_path)

    print(f"Saved {len(train_triples)} triples to {train_path}")
    print(f"Saved {len(dev_triples)} triples to {dev_path}")
    print(f"Saved {len(test_triples)} triples to {test_path}")
    verify(dev_triples, seed=seed)
    verify_hard_negatives(dev_triples, sample_n=min(100, len(dev_triples)), seed=seed)

    return {
        "train": len(train_triples),
        "dev": len(dev_triples),
        "test": len(test_triples),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for MS MARCO dataset conversion."""
    parser = argparse.ArgumentParser(description="Build MS MARCO retrieval triples for DeepQuant")
    parser.add_argument("--output_dir", type=str, default="data/msmarco")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    counts = build_and_save_splits(
        output_dir=args.output_dir,
        seed=args.seed,
        max_examples=args.max_examples,
        cache_dir=args.cache_dir,
    )
    print(
        "Saved splits:",
        f"train={counts['train']}, dev={counts['dev']}, test={counts['test']}",
    )
