"""FinQuant corpus extension with decomposed arithmetic-aware variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, Tuple

QUESTION_PATTERN = re.compile(r"Question:\s*(.*?)\s*Answer:", flags=re.IGNORECASE | re.DOTALL)
CONTEXT_PATTERN = re.compile(r"Context:\s*(.*?)\s*Question:", flags=re.IGNORECASE | re.DOTALL)
NUMBER_PATTERN = re.compile(
    r"(?P<prefix>\$)?(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>%|bn|billion|million|mm|m|x)?",
    flags=re.IGNORECASE,
)
PERCENT_PATTERN = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*%", flags=re.IGNORECASE)
PE_PATTERN = re.compile(r"(?:p/e|pe ratio|price-to-earnings)", flags=re.IGNORECASE)
GROWTH_QUERY_PATTERN = re.compile(r"(?:grew|growth|increase(?:d)?|change)", flags=re.IGNORECASE)
REVENUE_PATTERN = re.compile(r"revenue|sales", flags=re.IGNORECASE)
GROWTH_VALUE_PATTERN = re.compile(r"revenue|sales|earnings|income", flags=re.IGNORECASE)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows from disk."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    """Write JSONL rows to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_qrels(queries: Iterable[dict[str, Any]], path: Path) -> None:
    """Write TREC-style qrels from query records containing relevant_doc_ids."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for query in queries:
            query_id = str(query["query_id"])
            for doc_id in query.get("relevant_doc_ids", []):
                handle.write(f"{query_id}\t0\t{doc_id}\t1\n")


def _slugify(text: str, max_len: int = 24) -> str:
    """Create a short alphanumeric slug from free text."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:max_len] or "query"


def _doc_hash(text: str) -> str:
    """Create a stable short hash for a document text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _coerce_doc_id(doc: dict[str, Any], fallback_text: str) -> str:
    """Resolve a stable document id."""
    for key in ["doc_id", "document_id", "id"]:
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"finquant_doc_{_doc_hash(fallback_text)}"


def _normalized_query_record(query_id: str, query_text: str, relevant_doc_ids: list[str], split: str) -> dict[str, Any]:
    """Build a standard query record."""
    return {
        "query_id": query_id,
        "query_text": query_text,
        "relevant_doc_ids": list(dict.fromkeys(relevant_doc_ids)),
        "split": split,
    }


def _safe_text(value: Any) -> str:
    """Normalize unknown values to a clean string."""
    if not isinstance(value, str):
        return ""
    return value.strip()


def _extract_flare_question_and_context(example: dict[str, Any]) -> tuple[str, str]:
    """Extract question/context from a raw FLARE-FinQA row."""
    prompt = _safe_text(example.get("query"))
    question = _safe_text(example.get("text")) or _safe_text(example.get("question"))
    context = _safe_text(example.get("context")) or _safe_text(example.get("passage"))

    if prompt:
        question_match = QUESTION_PATTERN.search(prompt)
        context_match = CONTEXT_PATTERN.search(prompt)
        if question_match:
            candidate = _safe_text(question_match.group(1))
            if candidate:
                question = candidate
        if context_match:
            candidate = _safe_text(context_match.group(1))
            if candidate:
                context = candidate

    if not question:
        question = prompt
    if not context:
        context = prompt
    return question, context


def _parse_qrels(path: Path) -> dict[str, list[str]]:
    """Parse a TREC qrels file into query -> relevant doc ids."""
    qrels: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) >= 4:
                query_id, _, doc_id, relevance = parts[:4]
            elif len(parts) == 3:
                query_id, doc_id, relevance = parts
            else:
                continue
            try:
                rel_value = float(relevance)
            except ValueError:
                continue
            if rel_value <= 0:
                continue
            qrels.setdefault(query_id, []).append(doc_id)
    return qrels


def _load_trec_style(data_path: Path) -> Tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Load documents and queries from a TREC-style directory if present."""
    query_candidates = [data_path / "queries.jsonl", data_path / "queries.json"]
    corpus_candidates = [data_path / "corpus.jsonl", data_path / "documents.jsonl", data_path / "corpus.json"]
    qrels_path = data_path / "qrels.tsv"

    query_path = next((path for path in query_candidates if path.exists()), None)
    corpus_path = next((path for path in corpus_candidates if path.exists()), None)
    if query_path is None or corpus_path is None or not qrels_path.exists():
        return None

    queries = _load_jsonl(query_path)
    documents = _load_jsonl(corpus_path)
    qrels = _parse_qrels(qrels_path)
    for query in queries:
        query_id = str(query.get("query_id") or query.get("id") or "")
        query["query_id"] = query_id
        query["query_text"] = str(query.get("query_text") or query.get("text") or "")
        query["relevant_doc_ids"] = list(dict.fromkeys(qrels.get(query_id, query.get("relevant_doc_ids", []))))
        query.setdefault("split", "test")
    for doc in documents:
        doc["doc_id"] = _coerce_doc_id(doc, str(doc.get("text", "")))
        doc["text"] = str(doc.get("text") or doc.get("doc_text") or "")
    return documents, queries


def _load_retrieval_triples(data_path: Path) -> Tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the repo's retrieval triples and convert them into corpus/query records."""
    documents_by_text: dict[str, dict[str, Any]] = {}
    queries: list[dict[str, Any]] = []
    split_files = [("train", data_path / "train.jsonl"), ("dev", data_path / "dev.jsonl"), ("test", data_path / "test.jsonl")]
    query_counter = 0

    for split_name, split_path in split_files:
        if not split_path.exists():
            continue
        rows = _load_jsonl(split_path)
        for row in rows:
            query_text = str(row.get("query_text", "")).strip()
            pos_text = str(row.get("pos_doc_text", "")).strip()
            if not query_text or not pos_text:
                continue

            doc_id = documents_by_text.get(pos_text, {}).get("doc_id")
            if doc_id is None:
                doc_id = f"finquant_doc_{_doc_hash(pos_text)}"
                documents_by_text[pos_text] = {
                    "doc_id": doc_id,
                    "text": pos_text,
                    "source": split_name,
                    "origin": "retrieval_triple_positive",
                }

            for neg_text in row.get("neg_doc_texts", []):
                neg_text = str(neg_text).strip()
                if not neg_text or neg_text in documents_by_text:
                    continue
                documents_by_text[neg_text] = {
                    "doc_id": f"finquant_doc_{_doc_hash(neg_text)}",
                    "text": neg_text,
                    "source": split_name,
                    "origin": "retrieval_triple_negative",
                }

            query_id = f"finquant_{split_name}_q_{query_counter:06d}"
            query_counter += 1
            queries.append(_normalized_query_record(query_id, query_text, [doc_id], split_name))

    documents = list(documents_by_text.values())
    return documents, queries


def _load_raw_flare_finqa_from_hf() -> Tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load raw FLARE-FinQA directly from Hugging Face so numeric values are preserved."""
    from datasets import load_dataset

    split_mapping = [("train", "train"), ("valid", "dev"), ("test", "test")]
    documents_by_text: dict[str, dict[str, Any]] = {}
    queries: list[dict[str, Any]] = []

    for hf_split, local_split in split_mapping:
        split = load_dataset("chancefocus/flare-finqa", split=hf_split)
        for idx, example in enumerate(split):
            query_text, doc_text = _extract_flare_question_and_context(dict(example))
            if not query_text or not doc_text:
                continue
            doc_id = documents_by_text.get(doc_text, {}).get("doc_id")
            if doc_id is None:
                doc_id = f"flare_finqa_doc_{_doc_hash(doc_text)}"
                documents_by_text[doc_text] = {
                    "doc_id": doc_id,
                    "text": doc_text,
                    "source": local_split,
                    "origin": "flare_finqa_raw",
                }
            queries.append(
                _normalized_query_record(
                    query_id=f"flare_finqa_{local_split}_q_{idx:06d}",
                    query_text=query_text,
                    relevant_doc_ids=[doc_id],
                    split=local_split,
                )
            )

    return list(documents_by_text.values()), queries


def load_finquant(data_path: str) -> Tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load FinQuant corpus and queries from TREC-style files or retrieval triples."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"FinQuant data path does not exist: {path}")

    trec_loaded = _load_trec_style(path)
    if trec_loaded is not None:
        return trec_loaded

    try:
        return _load_raw_flare_finqa_from_hf()
    except Exception:
        return _load_retrieval_triples(path)


def _scaled_number_matches(text: str) -> list[dict[str, Any]]:
    """Extract numeric mentions with scale-aware values."""
    matches: list[dict[str, Any]] = []
    for match in NUMBER_PATTERN.finditer(text):
        value = float(match.group("value"))
        unit = (match.group("unit") or "").lower()
        scale = 1.0
        normalized_unit = unit
        if unit in {"bn", "billion"}:
            scale = 1e9
            normalized_unit = "bn"
        elif unit in {"million", "mm"}:
            scale = 1e6
            normalized_unit = "million"
        elif unit == "m" and match.group("prefix") == "$":
            scale = 1e6
            normalized_unit = "million"
        elif unit == "%":
            scale = 1.0
            normalized_unit = "%"
        elif unit == "x":
            scale = 1.0
            normalized_unit = "x"
        matches.append(
            {
                "raw": match.group(0),
                "value": value,
                "unit": normalized_unit,
                "prefix": match.group("prefix") or "",
                "absolute_value": value * scale,
            }
        )
    return matches


def _extract_primary_money_mention(text: str) -> dict[str, Any] | None:
    """Find the most salient currency-valued mention in a document."""
    money_mentions = [item for item in _scaled_number_matches(text) if item["prefix"] == "$"]
    if not money_mentions:
        return None
    return max(money_mentions, key=lambda item: item["absolute_value"])


def _extract_primary_percent(text: str) -> float | None:
    """Find the first percentage in text."""
    match = PERCENT_PATTERN.search(text)
    if match is None:
        return None
    return float(match.group("value"))


def _extract_primary_ratio(text: str) -> float | None:
    """Find a plausible ratio value for PE-style queries."""
    for item in _scaled_number_matches(text):
        if item["prefix"] == "$":
            continue
        if item["unit"] in {"%", "bn", "million"}:
            continue
        if 0.1 <= item["value"] <= 200.0:
            return float(item["value"])
    return None


def _format_scaled_currency(value_absolute: float, display_unit: str) -> str:
    """Format an absolute money value into a human-readable scaled currency string."""
    if display_unit == "bn":
        return f"${value_absolute / 1e9:.2f}bn"
    if display_unit == "million":
        return f"${value_absolute / 1e6:.2f} million"
    return f"${value_absolute:.2f}"


def _make_variant_doc(doc: dict[str, Any], query: dict[str, Any], index: int, text: str, arith_type: str) -> dict[str, Any]:
    """Clone metadata from the source doc into a new decomposed variant record."""
    base = dict(doc)
    base["source_doc_id"] = str(doc["doc_id"])
    base["doc_id"] = f"{doc['doc_id']}__{query['query_id']}__decomp_{index}"
    base["text"] = text
    base["arith_type"] = arith_type
    base["is_decomposed_variant"] = True
    return base


def _make_quarter_weights(seed_text: str) -> list[float]:
    """Create four quarterly weights with +/-5% noise and exact normalization."""
    rng = random.Random(seed_text)
    raw = [0.25 + rng.uniform(-0.05, 0.05) for _ in range(4)]
    total = sum(raw)
    return [value / total for value in raw]


def _infer_growth_metric_label(query_text: str, doc_text: str) -> str:
    """Infer a metric label for growth-rate decompositions from the query/doc text."""
    label_patterns = [
        ("share price", r"share price|stock price"),
        ("cash flow", r"cash flow"),
        ("balance", r"balance"),
        ("assets", r"assets?"),
        ("debt", r"debt"),
        ("revenue", r"revenue|sales"),
        ("earnings", r"earnings"),
        ("income", r"net income|income"),
    ]
    for text in [query_text.lower(), doc_text.lower()]:
        for label, pattern in label_patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return label
    return "value"


def _revenue_variants(doc: dict[str, Any], query: dict[str, Any]) -> list[dict[str, Any]]:
    """Create two TYPE_A quarterly revenue variants."""
    money = _extract_primary_money_mention(str(doc.get("text", ""))) or _extract_primary_money_mention(
        str(query.get("query_text", ""))
    )
    if money is None:
        return []
    display_unit = money["unit"] if money["unit"] in {"bn", "million"} else "bn"
    total_absolute = money["absolute_value"]
    variants: list[dict[str, Any]] = []
    for idx in range(2):
        weights = _make_quarter_weights(f"{doc['doc_id']}|{query['query_id']}|revenue|{idx}")
        quarter_values = [total_absolute * weight for weight in weights]
        text = (
            f"Q1 revenue: {_format_scaled_currency(quarter_values[0], display_unit)}. "
            f"Q2 revenue: {_format_scaled_currency(quarter_values[1], display_unit)}. "
            f"Q3 revenue: {_format_scaled_currency(quarter_values[2], display_unit)}. "
            f"Q4 revenue: {_format_scaled_currency(quarter_values[3], display_unit)}."
        )
        variants.append(_make_variant_doc(doc, query, idx, text, "TYPE_A"))
    return variants


def _pe_ratio_variants(doc: dict[str, Any], query: dict[str, Any]) -> list[dict[str, Any]]:
    """Create two TYPE_B PE-ratio variants."""
    ratio = _extract_primary_ratio(str(doc.get("text", ""))) or _extract_primary_ratio(str(query.get("query_text", "")))
    if ratio is None or ratio <= 0.0:
        return []
    rng = random.Random(f"{doc['doc_id']}|{query['query_id']}|pe")
    variants: list[dict[str, Any]] = []
    for idx in range(2):
        eps = rng.uniform(2.0, 20.0)
        price = ratio * eps
        text = f"Share price: ${price:.2f}. Earnings per share: ${eps:.2f}."
        variants.append(_make_variant_doc(doc, query, idx, text, "TYPE_B"))
    return variants


def _growth_rate_variants(doc: dict[str, Any], query: dict[str, Any]) -> list[dict[str, Any]]:
    """Create two TYPE_B growth-rate variants."""
    percent = _extract_primary_percent(str(doc.get("text", ""))) or _extract_primary_percent(str(query.get("query_text", "")))
    if percent is None:
        return []
    query_text = str(query.get("query_text", ""))
    doc_text = str(doc.get("text", ""))
    metric_label = _infer_growth_metric_label(query_text, doc_text)
    money = _extract_primary_money_mention(str(doc.get("text", ""))) or _extract_primary_money_mention(
        str(query.get("query_text", ""))
    )
    display_unit = "bn"
    if money is not None and money["unit"] in {"bn", "million"}:
        display_unit = money["unit"]
    current_absolute = money["absolute_value"] if money is not None else 5e9
    if 1.0 + percent / 100.0 <= 1e-8:
        return []
    previous_absolute = current_absolute / (1.0 + percent / 100.0)
    rng = random.Random(f"{doc['doc_id']}|{query['query_id']}|growth")
    variants: list[dict[str, Any]] = []
    for idx in range(2):
        scale = rng.uniform(0.9, 1.1)
        prev = previous_absolute * scale
        curr = prev * (1.0 + percent / 100.0)
        text = (
            f"Previous period {metric_label}: {_format_scaled_currency(prev, display_unit)}. "
            f"Current period {metric_label}: {_format_scaled_currency(curr, display_unit)}."
        )
        variants.append(_make_variant_doc(doc, query, idx, text, "TYPE_B"))
    return variants


def create_decomposed_variants(doc: dict[str, Any], query: dict[str, Any]) -> list[dict[str, Any]]:
    """Create arithmetic decomposition variants tailored to one query-document pair."""
    query_text = str(query.get("query_text", ""))
    doc_text = str(doc.get("text", ""))
    combined = f"{query_text} {doc_text}"

    if PE_PATTERN.search(combined):
        return _pe_ratio_variants(doc, query)
    if (
        GROWTH_QUERY_PATTERN.search(query_text)
        and GROWTH_VALUE_PATTERN.search(combined)
        and _extract_primary_percent(combined) is not None
    ):
        return _growth_rate_variants(doc, query)
    if REVENUE_PATTERN.search(combined):
        return _revenue_variants(doc, query)
    return []


def build_finquant_extension(finquant_dir: str, output_dir: str) -> dict[str, Any]:
    """Extend the FinQuant corpus by adding decomposed variants for relevant test documents."""
    documents, queries = load_finquant(finquant_dir)
    docs_by_id = {str(doc["doc_id"]): dict(doc) for doc in documents}
    extended_documents = [dict(doc) for doc in documents]
    extended_queries = [dict(query) for query in queries]
    added_variants = 0

    for query in extended_queries:
        if str(query.get("split", "test")) != "test":
            continue
        relevant_doc_ids = list(query.get("relevant_doc_ids", []))
        new_doc_ids: list[str] = []
        for doc_id in relevant_doc_ids:
            source_doc = docs_by_id.get(str(doc_id))
            if source_doc is None:
                continue
            variants = create_decomposed_variants(source_doc, query)
            for variant in variants[:2]:
                if variant["doc_id"] in docs_by_id:
                    continue
                docs_by_id[variant["doc_id"]] = variant
                extended_documents.append(variant)
                new_doc_ids.append(str(variant["doc_id"]))
                added_variants += 1
        query["relevant_doc_ids"] = list(dict.fromkeys(relevant_doc_ids + new_doc_ids))

    output_path = Path(output_dir)
    corpus_path = output_path / "corpus.jsonl"
    queries_path = output_path / "queries.jsonl"
    qrels_path = output_path / "qrels.tsv"
    _write_jsonl(extended_documents, corpus_path)
    _write_jsonl(extended_queries, queries_path)
    _write_qrels(extended_queries, qrels_path)

    summary = {
        "input_documents": len(documents),
        "input_queries": len(queries),
        "extended_documents": len(extended_documents),
        "extended_queries": len(extended_queries),
        "added_decomposed_documents": added_variants,
        "corpus_path": str(corpus_path),
        "queries_path": str(queries_path),
        "qrels_path": str(qrels_path),
    }
    print(f"Added {added_variants} decomposed documents to FinQuant corpus")
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build the WideQuant FinQuant extension corpus")
    parser.add_argument("--finquant_dir", type=str, default="data/finquant")
    parser.add_argument("--output_dir", type=str, default="data/finquant_extension")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = build_finquant_extension(finquant_dir=args.finquant_dir, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2))
