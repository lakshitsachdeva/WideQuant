"""FinQuant loader that converts FLARE-FinQA QA samples into retrieval triples."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from datasets import Dataset, load_dataset
import numpy as np
from tqdm import tqdm

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover - optional dependency fallback
    BM25Okapi = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.encoding.cqe_wrapper import (
    CQEWrapper,
    QuantitySpan,
    no_numbers_in_text,
    replace_with_num_tokens_regex,
)

QUESTION_PATTERN = re.compile(r"Question:\s*(.*?)\s*Answer:", flags=re.IGNORECASE | re.DOTALL)
CONTEXT_PATTERN = re.compile(r"Context:\s*(.*?)\s*Question:", flags=re.IGNORECASE | re.DOTALL)
NUMBER_PATTERN = re.compile(r"(?<!\w)(?:\d+\.?\d*|\.\d+)(?!\w)")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", flags=re.IGNORECASE)
FINANCIAL_CONCEPT_KEYWORDS = {
    "revenue",
    "sales",
    "profit",
    "earnings",
    "income",
    "ebit",
    "ebitda",
    "margin",
    "cash",
    "debt",
    "assets",
    "liabilities",
    "equity",
    "expense",
    "expenses",
    "operating",
    "net",
}


def _safe_text(value: Any) -> str:
    """Normalize unknown value to a stripped string."""
    if not isinstance(value, str):
        return ""
    return value.strip()


def _extract_question_and_context(example: dict[str, Any]) -> tuple[str, str]:
    """Extract retrieval query and positive passage from one FLARE-FinQA record."""
    prompt = _safe_text(example.get("query"))
    question = _safe_text(example.get("text")) or _safe_text(example.get("question"))
    context = _safe_text(example.get("context")) or _safe_text(example.get("passage"))

    if prompt:
        question_match = QUESTION_PATTERN.search(prompt)
        context_match = CONTEXT_PATTERN.search(prompt)
        if question_match:
            parsed_question = _safe_text(question_match.group(1))
            if parsed_question:
                question = parsed_question
        if context_match:
            parsed_context = _safe_text(context_match.group(1))
            if parsed_context:
                context = parsed_context

    if not question:
        question = prompt
    if not context:
        context = prompt
    return question, context


def _tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize text for lightweight lexical BM25 indexing."""
    return TOKEN_PATTERN.findall(text.lower())


def _span_to_json(span: QuantitySpan) -> dict[str, Any]:
    """Convert QuantitySpan dataclass to JSON-serializable dictionary."""
    return {
        "text": span.text,
        "mantissa": float(span.mantissa),
        "exponent": int(span.exponent),
        "unit": span.unit,
        "concept": span.concept,
        "start_char": int(span.start_char),
        "end_char": int(span.end_char),
    }


def _span_from_json(data: dict[str, Any]) -> QuantitySpan:
    """Convert JSON dictionary back to QuantitySpan dataclass."""
    return QuantitySpan(
        text=str(data.get("text", "")),
        mantissa=float(data.get("mantissa", 0.0)),
        exponent=int(data.get("exponent", 0)),
        unit=str(data.get("unit", "")),
        concept=str(data.get("concept", "")),
        start_char=int(data.get("start_char", 0)),
        end_char=int(data.get("end_char", 0)),
    )


def _hash_text(text: str) -> str:
    """Build stable cache key from raw text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_cache(cache_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load CQE span cache if present, else return an empty cache."""
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("rb") as handle:
            data = pickle.load(handle)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _save_cache(cache: dict[str, list[dict[str, Any]]], cache_path: Path) -> None:
    """Persist CQE span cache for future runs."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(cache, handle)


def _fast_replace_numbers(text: str) -> str:
    """Replace numeric substrings using a simple regex fast path."""
    return replace_with_num_tokens_regex(text)


def _extract_numbers_fallback(text: str) -> tuple[str, ...]:
    """Fallback numeric signature when CQE spans are unavailable."""
    numbers = NUMBER_PATTERN.findall(text)
    return tuple(sorted(set(numbers)))


def _numeric_signature(text: str, spans: Sequence[QuantitySpan]) -> tuple[str, ...]:
    """Create a compact numeric signature for filtering same-valued negatives."""
    if not spans:
        return _extract_numbers_fallback(text)

    values: list[str] = []
    for span in spans:
        scalar = float(span.mantissa * (10 ** span.exponent))
        unit = span.unit.lower().strip()
        values.append(f"{scalar:.8g}|{unit}")
    return tuple(sorted(set(values)))


def _make_cqe_wrapper() -> CQEWrapper:
    """Create a CQE wrapper and attempt install if package is missing."""
    try:
        return CQEWrapper()
    except ImportError:
        CQEWrapper.install_cqe()
        return CQEWrapper()


def _extract_spans(cqe: CQEWrapper, text: str) -> list[QuantitySpan]:
    """Extract quantity spans with CQE; tolerate parser failures per sample."""
    try:
        return cqe.extract(text)
    except Exception:
        return []


def _extract_spans_with_cache(
    cqe: CQEWrapper,
    text: str,
    span_cache: dict[str, list[dict[str, Any]]],
    cache_stats: dict[str, int],
) -> list[QuantitySpan]:
    """Extract spans with a disk-backed cache keyed by raw text hash."""
    key = _hash_text(text)
    cached = span_cache.get(key)
    if cached is not None:
        cache_stats["hits"] = cache_stats.get("hits", 0) + 1
        return [_span_from_json(item) for item in cached]

    cache_stats["misses"] = cache_stats.get("misses", 0) + 1
    spans = _extract_spans(cqe, text)
    span_cache[key] = [_span_to_json(span) for span in spans]
    return spans


def _limit_split(split: Dataset, max_examples: int | None) -> Dataset:
    """Return a capped view over the split for fast development cycles."""
    if max_examples is None:
        return split
    limit = max(0, min(int(max_examples), len(split)))
    return split.select(range(limit))


def _materialize_streaming_split(
    split: Iterable[dict[str, Any]],
    max_examples: int,
    split_name: str,
) -> Dataset:
    """Materialize a bounded number of streamed examples into an in-memory Dataset."""
    rows: list[dict[str, Any]] = []
    progress = tqdm(total=max_examples, desc=f"[{split_name}] Streaming raw examples", unit="ex")
    for example in split:
        rows.append(dict(example))
        progress.update(1)
        if len(rows) >= max_examples:
            break
    progress.close()
    return Dataset.from_list(rows)


def _lexical_rank_fallback(query_tokens: list[str], doc_tokens: list[list[str]]) -> list[int]:
    """Rank documents by lexical overlap if BM25 is unavailable."""
    query_set = set(query_tokens)
    scored = []
    for idx, tokens in enumerate(doc_tokens):
        score = len(query_set.intersection(tokens))
        scored.append((score, idx))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [idx for _, idx in scored]


def _extract_numeric_values(text: str) -> list[float]:
    """Extract numeric values from free text for overlap filtering."""
    cleaned = text.replace(",", "")
    values: list[float] = []
    for token in NUMBER_PATTERN.findall(cleaned):
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def _is_likely_year(value: float) -> bool:
    """Heuristic filter for standalone reporting years that should not dominate matching."""
    return float(value).is_integer() and 1800.0 <= abs(value) <= 2100.0


def _significant_numeric_values(values: Sequence[float], limit: int = 3) -> list[float]:
    """Keep the most meaningful numeric values and ignore likely years when possible."""
    filtered = [float(v) for v in values if not _is_likely_year(float(v))]
    if filtered:
        values = filtered
    ranked = sorted((float(v) for v in values), key=lambda val: abs(val), reverse=True)
    return ranked[: max(1, int(limit))]


def _largest_abs_numeric_value(values: Sequence[float]) -> float | None:
    """Return largest-magnitude numeric value in a sequence."""
    significant = _significant_numeric_values(values, limit=1)
    if not significant:
        return None
    return significant[0]


def _relative_difference(a: float, b: float) -> float:
    """Compute symmetric-style relative difference vs. reference b."""
    denom = max(abs(b), 1e-6)
    return abs(a - b) / denom


def _has_numeric_overlap(
    pos_values: Sequence[float],
    cand_values: Sequence[float],
    tolerance: float = 0.05,
) -> bool:
    """Check whether any candidate number appears in positive numbers within +/- tolerance."""
    pos_values = _significant_numeric_values(pos_values, limit=3)
    cand_values = _significant_numeric_values(cand_values, limit=3)
    if not pos_values or not cand_values:
        return False
    for cand in cand_values:
        for pos in pos_values:
            denom = max(abs(pos), 1e-6)
            if abs(cand - pos) / denom <= tolerance:
                return True
    return False


def _concept_terms(text: str) -> set[str]:
    """Extract financial concept keywords present in the text."""
    tokens = set(_tokenize_for_bm25(text))
    return tokens.intersection(FINANCIAL_CONCEPT_KEYWORDS)


def _shares_query_concept(query_text: str, doc_text: str) -> bool:
    """Require concept overlap when query explicitly mentions a tracked financial concept."""
    query_concepts = _concept_terms(query_text)
    if not query_concepts:
        return True
    doc_concepts = _concept_terms(doc_text)
    return bool(query_concepts.intersection(doc_concepts))


def _build_hard_negative_indices(
    query_texts: Sequence[str],
    pos_doc_texts: Sequence[str],
    n_negatives: int = 5,
) -> list[list[int]]:
    """Mine hard negatives with numeric-adversarial filters and concept matching."""
    doc_tokens = [_tokenize_for_bm25(text) for text in pos_doc_texts]
    doc_numeric_values = [_extract_numeric_values(text) for text in pos_doc_texts]
    doc_largest_values = [_largest_abs_numeric_value(values) for values in doc_numeric_values]
    bm25 = BM25Okapi(doc_tokens) if BM25Okapi is not None else None
    all_indices = list(range(len(pos_doc_texts)))
    hard_negative_indices: list[list[int]] = []
    bm25_window = min(200, max(len(pos_doc_texts) - 1, 1))

    for i, query_text in enumerate(query_texts):
        query_tokens = _tokenize_for_bm25(query_text)
        if bm25 is not None:
            scores = bm25.get_scores(query_tokens)
            top_count = min(bm25_window + 1, len(scores))
            candidate_order = np.argsort(-scores)[:top_count].tolist()
            ranked = [int(idx) for idx in candidate_order if int(idx) != i]
        else:
            ranked = _lexical_rank_fallback(query_tokens, doc_tokens)
            ranked = [int(idx) for idx in ranked if int(idx) != i][:bm25_window]

        threshold_target = n_negatives // 2
        bm25_target = n_negatives - threshold_target
        base_pool = ranked[:bm25_window]
        selected: list[int] = []
        selected_set: set[int] = set()

        pos_values = doc_numeric_values[i]
        pos_largest = doc_largest_values[i]

        # Bucket 1: concept-consistent threshold-violation negatives.
        threshold_candidates = base_pool + [idx for idx in ranked if idx not in base_pool]
        for cand_idx in threshold_candidates:
            if cand_idx in selected_set:
                continue
            if not _shares_query_concept(query_text, pos_doc_texts[cand_idx]):
                continue
            cand_largest = doc_largest_values[cand_idx]
            if pos_largest is None or cand_largest is None:
                continue
            if _relative_difference(cand_largest, pos_largest) < 0.20:
                continue
            if _has_numeric_overlap(pos_values, doc_numeric_values[cand_idx], tolerance=0.05):
                continue
            selected.append(int(cand_idx))
            selected_set.add(int(cand_idx))
            if len(selected) >= threshold_target:
                break

        # Bucket 2: BM25 negatives with low numeric overlap and high lexical pressure.
        bm25_candidates = list(base_pool)
        for cand_idx in bm25_candidates:
            if cand_idx in selected_set:
                continue
            if not _shares_query_concept(query_text, pos_doc_texts[cand_idx]):
                continue
            cand_values = doc_numeric_values[cand_idx]
            if _has_numeric_overlap(pos_values, cand_values, tolerance=0.05):
                continue
            selected.append(int(cand_idx))
            selected_set.add(int(cand_idx))
            if len(selected) >= threshold_target + bm25_target:
                break

        # Fallback pass over a bounded ranking window for speed.
        if len(selected) < n_negatives:
            fallback_candidates = ranked[:bm25_window]
            for cand_idx in fallback_candidates:
                if cand_idx in selected_set:
                    continue
                cand_values = doc_numeric_values[cand_idx]
                if _has_numeric_overlap(pos_values, cand_values, tolerance=0.05):
                    continue
                selected.append(int(cand_idx))
                selected_set.add(int(cand_idx))
                if len(selected) == n_negatives:
                    break

        # Final fallback to guarantee cardinality.
        if len(selected) < n_negatives:
            pool = [idx for idx in all_indices if idx != i and idx not in selected_set]
            random.shuffle(pool)
            selected.extend(pool[: n_negatives - len(selected)])

        while len(selected) < n_negatives and len(pos_doc_texts) > 1:
            next_idx = (i + len(selected) + 1) % len(pos_doc_texts)
            if next_idx != i and next_idx not in selected_set:
                selected.append(next_idx)
                selected_set.add(next_idx)

        hard_negative_indices.append(selected[:n_negatives])
    return hard_negative_indices


def convert_split_to_triples(
    split: Dataset,
    cqe: CQEWrapper | None,
    n_negatives: int = 5,
    skip_cqe: bool = False,
    span_cache: dict[str, list[dict[str, Any]]] | None = None,
    cache_stats: dict[str, int] | None = None,
    split_name: str = "split",
) -> list[dict[str, Any]]:
    """Convert one split of QA examples into retrieval triples with hard negatives."""
    prepared: list[dict[str, Any]] = []
    total_examples = len(split)
    span_cache = span_cache if span_cache is not None else {}
    cache_stats = cache_stats if cache_stats is not None else {"hits": 0, "misses": 0}

    start_time = time.time()
    eta_printed = False
    progress = tqdm(
        split,
        total=total_examples,
        unit="ex",
        desc=f"[{split_name}] Processing example 0/{total_examples}",
    )

    for idx, example in enumerate(progress, start=1):
        question, context = _extract_question_and_context(example)
        elapsed = time.time() - start_time
        progress.set_description(f"[{split_name}] Processing example {idx}/{total_examples}")
        progress.set_postfix_str(f"cache hits: {cache_stats.get('hits', 0)} | elapsed: {elapsed:.1f}s")

        if not eta_printed and idx == 100:
            per_example = elapsed / float(idx)
            eta_seconds = per_example * float(max(total_examples - idx, 0))
            tqdm.write(
                f"[{split_name}] Estimated remaining time (first 100 examples): {eta_seconds:.1f}s"
            )
            eta_printed = True

        if not question or not context:
            continue

        if skip_cqe:
            query_spans: list[QuantitySpan] = []
            pos_spans: list[QuantitySpan] = []
            query_text = _fast_replace_numbers(question)
            pos_doc_text = _fast_replace_numbers(context)
        else:
            assert cqe is not None
            query_spans = _extract_spans_with_cache(cqe, question, span_cache, cache_stats)
            pos_spans = _extract_spans_with_cache(cqe, context, span_cache, cache_stats)
            query_text = cqe.replace_with_num_tokens(question, query_spans)
            pos_doc_text = cqe.replace_with_num_tokens(context, pos_spans)

        assert "[num]" in query_text or no_numbers_in_text(question), (
            f"replace_with_num_tokens failed on query: {question[:100]}"
        )
        assert "[num]" in pos_doc_text or no_numbers_in_text(context), (
            f"replace_with_num_tokens failed on document: {context[:100]}"
        )

        if idx == total_examples and total_examples < 100:
            tqdm.write(f"[{split_name}] Completed small split in {elapsed:.1f}s")

        prepared.append(
            {
                "query_text": query_text,
                "query_spans": query_spans,
                "pos_doc_text": pos_doc_text,
                "pos_doc_spans": pos_spans,
            }
        )

    if not prepared:
        return []

    query_texts = [item["query_text"] for item in prepared]
    pos_doc_texts = [item["pos_doc_text"] for item in prepared]
    hard_neg_indices = _build_hard_negative_indices(
        query_texts,
        pos_doc_texts,
        n_negatives=n_negatives,
    )

    triples: list[dict[str, Any]] = []
    for i, item in enumerate(prepared):
        neg_ids = hard_neg_indices[i]
        neg_doc_texts = [prepared[idx]["pos_doc_text"] for idx in neg_ids]
        neg_doc_spans = [prepared[idx]["pos_doc_spans"] for idx in neg_ids]
        triples.append(
            {
                "query_text": item["query_text"],
                "query_spans": [_span_to_json(span) for span in item["query_spans"]],
                "pos_doc_text": item["pos_doc_text"],
                "pos_doc_spans": [_span_to_json(span) for span in item["pos_doc_spans"]],
                "neg_doc_texts": neg_doc_texts,
                "neg_doc_spans": [[_span_to_json(span) for span in spans] for spans in neg_doc_spans],
            }
        )
    return triples


def _bm25_mrr_at_10(triples: Sequence[dict[str, Any]], sample_size: int = 100, seed: int = 42) -> float:
    """Compute BM25 MRR@10 over sampled triples using pos+neg candidate sets."""
    if not triples:
        return 0.0
    rng = random.Random(seed)
    sampled = list(triples)
    if len(sampled) > sample_size:
        sampled = rng.sample(sampled, sample_size)

    reciprocal_ranks: list[float] = []
    for triple in sampled:
        query_tokens = _tokenize_for_bm25(triple["query_text"])
        corpus = [triple["pos_doc_text"], *triple["neg_doc_texts"]]
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
) -> float:
    """Evaluate BM25 MRR@10 on positive-vs-negatives ranking and print pass/fail."""
    bm25_mrr = _bm25_mrr_at_10(dev_triples, sample_size=sample_n, seed=seed)
    status = "PASS" if bm25_mrr < 0.30 else "FAIL"
    print(f"Hard negatives BM25 MRR@10: {bm25_mrr:.4f} [{status}]")
    return bm25_mrr


def verify(dev_triples: Sequence[dict[str, Any]], seed: int = 42) -> dict[str, float]:
    """Print conversion diagnostics for dev triples and return computed metrics."""
    total = len(dev_triples)
    if total == 0:
        print("Total triples: 0")
        print("% [num] in query_text: 0.00")
        print("% [num] in pos_doc_text: 0.00")
        print("Mean query spans: 0.00")
        print("BM25 MRR@10 (100 dev samples): 0.0000")
        return {
            "total_triples": 0.0,
            "pct_query_has_num": 0.0,
            "pct_pos_has_num": 0.0,
            "mean_query_spans": 0.0,
            "bm25_mrr10_dev_100": 0.0,
        }

    query_has_num = sum(1 for item in dev_triples if "[num]" in item["query_text"])
    pos_has_num = sum(1 for item in dev_triples if "[num]" in item["pos_doc_text"])
    mean_query_spans = sum(len(item["query_spans"]) for item in dev_triples) / float(total)
    bm25_mrr10 = _bm25_mrr_at_10(dev_triples, sample_size=100, seed=seed)

    print(f"Total triples: {total}")
    print(f"% [num] in query_text: {100.0 * query_has_num / total:.2f}")
    print(f"% [num] in pos_doc_text: {100.0 * pos_has_num / total:.2f}")
    print(f"Mean query spans: {mean_query_spans:.2f}")
    print(f"BM25 MRR@10 (100 dev samples): {bm25_mrr10:.4f}")
    verify_hard_negatives(dev_triples, sample_n=100, seed=seed)

    return {
        "total_triples": float(total),
        "pct_query_has_num": float(100.0 * query_has_num / total),
        "pct_pos_has_num": float(100.0 * pos_has_num / total),
        "mean_query_spans": float(mean_query_spans),
        "bm25_mrr10_dev_100": float(bm25_mrr10),
    }


def _save_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    """Write one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_and_save_splits(
    output_dir: str = "data/finquant",
    seed: int = 42,
    n_negatives: int = 5,
    max_examples: int | None = None,
    skip_cqe: bool = False,
    streaming: bool = False,
    cache_dir: str | None = None,
) -> dict[str, int]:
    """Build retrieval triples and save train/dev/test JSONL files."""
    random.seed(seed)
    if streaming and max_examples is None:
        raise ValueError("Streaming mode requires --max_examples so we do not iterate the full remote corpus.")

    load_kwargs: dict[str, Any] = {}
    if cache_dir is not None:
        load_kwargs["cache_dir"] = cache_dir
    if streaming:
        load_kwargs["streaming"] = True

    dataset = load_dataset("chancefocus/flare-finqa", **load_kwargs)
    cqe: CQEWrapper | None = None
    output_path = Path(output_dir)
    cache_path = output_path / "cqe_cache.pkl"
    cache_stats = {"hits": 0, "misses": 0}

    span_cache: dict[str, list[dict[str, Any]]] = {}
    if not skip_cqe:
        cqe = _make_cqe_wrapper()
        span_cache = _load_cache(cache_path)
        if span_cache:
            print(f"Loaded CQE cache with {len(span_cache)} entries from {cache_path}")

    train_split = dataset["train"]
    dev_split = dataset["valid"] if "valid" in dataset else dataset.get("validation", dataset["test"])
    test_split = dataset["test"] if "test" in dataset else dev_split

    if streaming:
        assert max_examples is not None
        print(
            f"Streaming FLARE-FinQA from Hugging Face with max_examples={max_examples}. "
            "This avoids downloading the full dataset cache."
        )
        train_split = _materialize_streaming_split(train_split, max_examples=max_examples, split_name="train")
        dev_cap = min(max_examples, 1000)
        test_cap = min(max_examples, 1000)
        dev_split = _materialize_streaming_split(dev_split, max_examples=dev_cap, split_name="dev")
        test_split = _materialize_streaming_split(test_split, max_examples=test_cap, split_name="test")
    else:
        train_split = _limit_split(train_split, max_examples)
        dev_split = _limit_split(dev_split, max_examples)
        test_split = _limit_split(test_split, max_examples)

    train_triples = convert_split_to_triples(
        train_split,
        cqe,
        n_negatives=n_negatives,
        skip_cqe=skip_cqe,
        span_cache=span_cache,
        cache_stats=cache_stats,
        split_name="train",
    )
    dev_triples = convert_split_to_triples(
        dev_split,
        cqe,
        n_negatives=n_negatives,
        skip_cqe=skip_cqe,
        span_cache=span_cache,
        cache_stats=cache_stats,
        split_name="dev",
    )
    test_triples = convert_split_to_triples(
        test_split,
        cqe,
        n_negatives=n_negatives,
        skip_cqe=skip_cqe,
        span_cache=span_cache,
        cache_stats=cache_stats,
        split_name="test",
    )

    train_path = output_path / "train.jsonl"
    dev_path = output_path / "dev.jsonl"
    test_path = output_path / "test.jsonl"

    _save_jsonl(train_triples, train_path)
    _save_jsonl(dev_triples, dev_path)
    _save_jsonl(test_triples, test_path)

    if not skip_cqe:
        _save_cache(span_cache, cache_path)
        print(f"Saved CQE cache with {len(span_cache)} entries to {cache_path}")
        print(f"CQE cache stats: hits={cache_stats['hits']}, misses={cache_stats['misses']}")
    else:
        print("CQE extraction skipped (--skip_cqe). Used regex [num] replacement only.")

    print(f"Saved {len(train_triples)} triples to {train_path}")
    print(f"Saved {len(dev_triples)} triples to {dev_path}")
    print(f"Saved {len(test_triples)} triples to {test_path}")

    verify(dev_triples, seed=seed)

    return {
        "train": len(train_triples),
        "dev": len(dev_triples),
        "test": len(test_triples),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset conversion."""
    parser = argparse.ArgumentParser(description="Build retrieval triples from FLARE-FinQA")
    parser.add_argument("--output_dir", type=str, default="data/finquant")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_negatives", type=int, default=5)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--skip_cqe", action="store_true")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    counts = build_and_save_splits(
        output_dir=args.output_dir,
        seed=args.seed,
        n_negatives=args.n_negatives,
        max_examples=args.max_examples,
        skip_cqe=args.skip_cqe,
        streaming=args.streaming,
        cache_dir=args.cache_dir,
    )
    print(
        "Saved splits:",
        f"train={counts['train']}, dev={counts['dev']}, test={counts['test']}",
    )
