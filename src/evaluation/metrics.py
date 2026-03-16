"""Evaluation metrics and full retrieval evaluation for WideQuant."""

from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - FAISS fallback
    faiss = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.encoding.cqe_wrapper import CQEWrapper, QuantitySpan, setup_tokenizer

NUMBER_PATTERN = re.compile(r"(?<!\w)(?P<value>\d+(?:\.\d+)?|\.\d+)(?!\w)")
UNIT_AFTER_NUMBER_PATTERN = re.compile(r"^\s*(?P<unit>%|\$|[A-Za-z]+)")
TOKEN_PATTERN = re.compile(r"[A-Za-z]+")


def _scientific_components(value: float) -> tuple[float, int]:
    """Convert one scalar to mantissa/exponent form."""
    if abs(value) < 1e-12:
        return 0.0, 0
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = float(value / (10.0 ** exponent))
    return mantissa, exponent


def _coerce_spans(raw_spans: Any) -> list[QuantitySpan]:
    """Normalize unknown span payloads into QuantitySpan objects."""
    if raw_spans is None:
        return []

    if isinstance(raw_spans, Sequence) and raw_spans and isinstance(raw_spans[0], list):
        if len(raw_spans) != 1:
            raise ValueError("Nested quantity span payloads must have batch size 1.")
        raw_spans = raw_spans[0]

    spans: list[QuantitySpan] = []
    for raw in raw_spans:
        if isinstance(raw, QuantitySpan):
            spans.append(raw)
        elif isinstance(raw, Mapping):
            spans.append(QuantitySpan(**dict(raw)))
    return spans


def _infer_concept(text: str, start_char: int) -> str:
    """Infer a lightweight concept label from nearby tokens."""
    window_start = max(0, start_char - 40)
    left_context = text[window_start:start_char]
    tokens = TOKEN_PATTERN.findall(left_context.lower())
    if not tokens:
        return ""
    stopwords = {"the", "a", "an", "of", "with", "for", "than", "under", "over", "above", "below", "and", "per"}
    for token in reversed(tokens):
        if token not in stopwords:
            return token
    return tokens[-1]


def _regex_extract_spans(text: str) -> list[QuantitySpan]:
    """Fallback numeric span extractor when CQE is unavailable."""
    spans: list[QuantitySpan] = []
    for match in NUMBER_PATTERN.finditer(text):
        raw_value = match.group("value")
        try:
            value = float(raw_value)
        except ValueError:
            continue
        unit_match = UNIT_AFTER_NUMBER_PATTERN.match(text[match.end() :])
        unit = unit_match.group("unit") if unit_match is not None else ""
        mantissa, exponent = _scientific_components(value)
        spans.append(
            QuantitySpan(
                text=raw_value,
                mantissa=mantissa,
                exponent=exponent,
                unit=unit,
                concept=_infer_concept(text, match.start()),
                start_char=match.start(),
                end_char=match.end(),
            )
        )
    return spans


def _extract_spans(text: str, cqe_wrapper: CQEWrapper | None) -> list[QuantitySpan]:
    """Extract quantity spans with CQE when available, else regex fallback."""
    if cqe_wrapper is not None:
        try:
            spans = cqe_wrapper.extract(text)
            if spans:
                return spans
        except Exception:
            pass
    return _regex_extract_spans(text)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records from disk."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Parse a TREC-style qrels file into graded relevance by query id."""
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            query_id, _, doc_id, relevance = parts[:4]
            try:
                qrels[str(query_id)][str(doc_id)] = int(float(relevance))
            except ValueError:
                continue
    return dict(qrels)


def _resolve_dataset_dict(test_dataset: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Normalize a dataset source into documents, queries, and qrels."""
    if isinstance(test_dataset, (str, Path)):
        dataset_path = Path(test_dataset)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        doc_path = None
        for candidate in [dataset_path / "documents.jsonl", dataset_path / "corpus.jsonl", dataset_path / "corpus.json"]:
            if candidate.exists():
                doc_path = candidate
                break
        query_path = None
        for candidate in [dataset_path / "queries.jsonl", dataset_path / "queries.json"]:
            if candidate.exists():
                query_path = candidate
                break
        qrels_path = dataset_path / "qrels.tsv"
        if doc_path is None or query_path is None or not qrels_path.exists():
            raise FileNotFoundError(
                "Expected dataset directory with documents/corpus, queries, and qrels.tsv files."
            )
        documents = _load_jsonl(doc_path)
        queries = _load_jsonl(query_path)
        qrels = _parse_qrels(qrels_path)
        return documents, queries, qrels

    if isinstance(test_dataset, Mapping):
        documents = list(test_dataset.get("documents", []))
        queries = list(test_dataset.get("queries", []))
        raw_qrels = test_dataset.get("qrels", {})
        qrels: dict[str, dict[str, int]] = {}
        if isinstance(raw_qrels, Mapping):
            for query_id, doc_map in raw_qrels.items():
                if isinstance(doc_map, Mapping):
                    qrels[str(query_id)] = {str(doc_id): int(score) for doc_id, score in doc_map.items()}
        return documents, queries, qrels

    if hasattr(test_dataset, "documents") and hasattr(test_dataset, "queries"):
        documents = list(getattr(test_dataset, "documents"))
        queries = list(getattr(test_dataset, "queries"))
        raw_qrels = getattr(test_dataset, "qrels", {})
        qrels = {}
        if isinstance(raw_qrels, Mapping):
            for query_id, doc_map in raw_qrels.items():
                if isinstance(doc_map, Mapping):
                    qrels[str(query_id)] = {str(doc_id): int(score) for doc_id, score in doc_map.items()}
        return documents, queries, qrels

    raise TypeError("Unsupported test_dataset format. Use a dataset directory, dict, or object with documents/queries.")


def _relevant_lists_from_queries(
    queries: Sequence[dict[str, Any]],
    qrels: Mapping[str, Mapping[str, int]],
) -> tuple[list[list[str]], list[dict[str, int]]]:
    """Build relevant-doc lists and graded relevance maps for each query."""
    relevant_docs: list[list[str]] = []
    relevance_maps: list[dict[str, int]] = []
    for query in queries:
        query_id = str(query.get("query_id") or query.get("id") or "")
        if query_id in qrels:
            rel_map = {str(doc_id): int(score) for doc_id, score in qrels[query_id].items() if int(score) > 0}
        else:
            rel_ids = [str(doc_id) for doc_id in query.get("relevant_doc_ids", [])]
            rel_map = {doc_id: 1 for doc_id in rel_ids}
        relevant_docs.append(list(rel_map.keys()))
        relevance_maps.append(rel_map)
    return relevant_docs, relevance_maps


def _get_model_device(model: Any, device: torch.device | str | None) -> torch.device:
    """Resolve the device to use for evaluation."""
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _get_model_tokenizer(model: Any) -> Any:
    """Return the model tokenizer, creating one if needed."""
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    encoder_name = "bert-base-uncased"
    if hasattr(model, "config"):
        encoder_name = str(getattr(model, "config", {}).get("model", {}).get("encoder", encoder_name))
    tokenizer, _ = setup_tokenizer(encoder_name=encoder_name)
    return tokenizer


def _prepare_payload(
    model: Any,
    tokenizer: Any,
    text: str,
    raw_spans: Any = None,
    max_length: int = 256,
    cqe_wrapper: CQEWrapper | None = None,
) -> dict[str, Any]:
    """Tokenize text and attach quantity spans for model encoding."""
    spans = _coerce_spans(raw_spans)
    if spans:
        processed_text = text if "[num]" in text else CQEWrapper.replace_with_num_tokens(text, spans)
    elif "[num]" in text:
        processed_text = text
    else:
        spans = _extract_spans(text, cqe_wrapper)
        processed_text = CQEWrapper.replace_with_num_tokens(text, spans)

    encoded = tokenizer(
        processed_text,
        truncation=True,
        padding="max_length",
        max_length=int(max_length),
        return_tensors="pt",
    )

    return {
        "input_ids": encoded["input_ids"].squeeze(0),
        "attention_mask": encoded["attention_mask"].squeeze(0),
        "quantity_spans": spans,
        "text": processed_text,
    }


def _to_device(payload: Any, device: torch.device) -> Any:
    """Recursively move tensors to a target device."""
    if isinstance(payload, Tensor):
        return payload.to(device)
    if isinstance(payload, dict):
        return {key: _to_device(value, device) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_to_device(value, device) for value in payload]
    if isinstance(payload, tuple):
        return tuple(_to_device(value, device) for value in payload)
    return payload


def _encode_document_vector(model: Any, payload: dict[str, Any]) -> Tensor:
    """Encode one document payload and return its normalized CLS vector."""
    encoded = model.encode_document(
        payload["input_ids"],
        payload["attention_mask"],
        payload.get("quantity_spans", []),
    )
    return F.normalize(encoded["cls"], dim=0)


def _encode_query_vector(model: Any, payload: dict[str, Any]) -> Tensor:
    """Encode one query payload and return its normalized CLS vector."""
    encoded = model.encode_query(
        payload["input_ids"],
        payload["attention_mask"],
        payload.get("quantity_spans", []),
    )
    return F.normalize(encoded["cls"], dim=0)


def _forward_score(model: Any, query_payload: dict[str, Any], doc_payload: dict[str, Any]) -> float:
    """Compute one query-document score with either DeepQuant or WideQuant."""
    if model.__class__.__name__ == "WideQuant":
        outputs = model(query_payload, doc_payload, None, use_aan=True)
    else:
        outputs = model(query_payload, doc_payload, None)
    return float(outputs["final_score"].detach().cpu())


def _normalize_relevance_inputs(
    rankings: Sequence[Sequence[str]],
    relevance_scores: Sequence[Mapping[str, int]] | Mapping[str, int],
) -> list[dict[str, int]]:
    """Normalize NDCG inputs to one relevance map per query."""
    if isinstance(relevance_scores, Mapping):
        if len(rankings) != 1:
            raise ValueError("Single relevance mapping can only be used with a single ranking.")
        return [{str(doc_id): int(score) for doc_id, score in relevance_scores.items()}]

    if len(rankings) != len(relevance_scores):
        raise ValueError("rankings and relevance_scores must have the same length.")
    return [{str(doc_id): int(score) for doc_id, score in rel_map.items()} for rel_map in relevance_scores]


def mrr_at_k(
    rankings: list[list[str]],
    relevant_docs: list[list[str]],
    k: int = 10,
) -> tuple[float, list[float]]:
    """Compute mean reciprocal rank at k and the per-query reciprocal ranks."""
    if len(rankings) != len(relevant_docs):
        raise ValueError("rankings and relevant_docs must have the same length.")

    reciprocal_ranks: list[float] = []
    for ranked_docs, rel_docs in zip(rankings, relevant_docs, strict=False):
        rel_set = set(str(doc_id) for doc_id in rel_docs)
        rr = 0.0
        for rank, doc_id in enumerate(ranked_docs[: int(k)], start=1):
            if str(doc_id) in rel_set:
                rr = 1.0 / float(rank)
                break
        reciprocal_ranks.append(rr)

    mean_mrr = float(sum(reciprocal_ranks) / max(len(reciprocal_ranks), 1))
    return mean_mrr, reciprocal_ranks


def ndcg_at_k(
    rankings: list[list[str]],
    relevance_scores: Sequence[Mapping[str, int]] | Mapping[str, int],
    k: int = 10,
) -> float:
    """Compute mean NDCG@k over one or more query rankings."""
    relevance_maps = _normalize_relevance_inputs(rankings, relevance_scores)

    ndcg_values: list[float] = []
    for ranked_docs, rel_map in zip(rankings, relevance_maps, strict=False):
        dcg = 0.0
        for rank, doc_id in enumerate(ranked_docs[: int(k)], start=1):
            gain = float(rel_map.get(str(doc_id), 0))
            if gain > 0.0:
                dcg += gain / math.log2(rank + 1)

        ideal_gains = sorted((float(score) for score in rel_map.values() if score > 0), reverse=True)[: int(k)]
        idcg = 0.0
        for rank, gain in enumerate(ideal_gains, start=1):
            idcg += gain / math.log2(rank + 1)
        ndcg_values.append((dcg / idcg) if idcg > 0.0 else 0.0)

    return float(sum(ndcg_values) / max(len(ndcg_values), 1))


def precision_at_k(
    rankings: list[list[str]],
    relevant_docs: list[list[str]],
    k: int = 10,
) -> float:
    """Compute mean precision at k over all queries."""
    if len(rankings) != len(relevant_docs):
        raise ValueError("rankings and relevant_docs must have the same length.")

    precisions: list[float] = []
    for ranked_docs, rel_docs in zip(rankings, relevant_docs, strict=False):
        rel_set = set(str(doc_id) for doc_id in rel_docs)
        hits = sum(1 for doc_id in ranked_docs[: int(k)] if str(doc_id) in rel_set)
        precisions.append(float(hits) / float(max(int(k), 1)))

    return float(sum(precisions) / max(len(precisions), 1))


def recall_at_k(
    rankings: list[list[str]],
    relevant_docs: list[list[str]],
    k: int = 100,
) -> float:
    """Compute mean recall at k over all queries."""
    if len(rankings) != len(relevant_docs):
        raise ValueError("rankings and relevant_docs must have the same length.")

    recalls: list[float] = []
    for ranked_docs, rel_docs in zip(rankings, relevant_docs, strict=False):
        rel_set = set(str(doc_id) for doc_id in rel_docs)
        if not rel_set:
            recalls.append(0.0)
            continue
        hits = sum(1 for doc_id in ranked_docs[: int(k)] if str(doc_id) in rel_set)
        recalls.append(float(hits) / float(len(rel_set)))

    return float(sum(recalls) / max(len(recalls), 1))


def mcnemar_test(per_query_scores_a: list[float], per_query_scores_b: list[float]) -> dict[str, float | bool]:
    """Run McNemar's test from per-query success indicators."""
    if len(per_query_scores_a) != len(per_query_scores_b):
        raise ValueError("McNemar inputs must have the same length.")

    a_wins = 0
    b_wins = 0
    for score_a, score_b in zip(per_query_scores_a, per_query_scores_b, strict=False):
        a_correct = float(score_a) > 0.0
        b_correct = float(score_b) > 0.0
        if a_correct and not b_correct:
            a_wins += 1
        elif b_correct and not a_correct:
            b_wins += 1

    discordant = a_wins + b_wins
    if discordant == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = ((abs(a_wins - b_wins) - 1.0) ** 2) / float(discordant)
        p_value = math.erfc(math.sqrt(max(statistic, 0.0) / 2.0))

    return {
        "p_value": float(p_value),
        "statistic": float(statistic),
        "significant": bool(p_value < 0.05),
        "a_only_correct": float(a_wins),
        "b_only_correct": float(b_wins),
    }


def _infer_query_type(query: Mapping[str, Any]) -> str | None:
    """Infer the evaluation subset label for a query."""
    for key in ["query_type", "arith_type"]:
        value = query.get(key)
        if isinstance(value, str) and value:
            normalized = value.strip()
            if normalized in {"atomic", "typeA", "typeB", "typeC"}:
                return normalized
            upper_map = {"TYPE_A": "typeA", "TYPE_B": "typeB", "TYPE_C": "typeC", "ATOMIC": "atomic"}
            if normalized in upper_map:
                return upper_map[normalized]
    return None


def _compute_metric_bundle(
    rankings: list[list[str]],
    relevant_docs: list[list[str]],
    relevance_maps: list[dict[str, int]],
    mrr_k: int,
    recall_k: int,
) -> dict[str, Any]:
    """Compute the full metric bundle for one query subset."""
    mrr_value, reciprocal_ranks = mrr_at_k(rankings, relevant_docs, k=mrr_k)
    return {
        f"mrr{mrr_k}": mrr_value,
        f"ndcg{mrr_k}": ndcg_at_k(rankings, relevance_maps, k=mrr_k),
        f"p{mrr_k}": precision_at_k(rankings, relevant_docs, k=mrr_k),
        f"r{recall_k}": recall_at_k(rankings, relevant_docs, k=recall_k),
        "per_query_rr": reciprocal_ranks,
    }


def run_full_evaluation(
    model: Any,
    test_dataset: Any,
    device: torch.device | str | None,
    k_values: list[int] | None = None,
) -> dict[str, Any]:
    """Run full retrieval evaluation with FAISS retrieval and per-type breakdowns."""
    if k_values is None:
        k_values = [10, 100]
    mrr_k = 10 if 10 in k_values else int(min(k_values))
    recall_k = 100 if 100 in k_values else int(max(k_values))
    retrieval_k = max(int(value) for value in k_values + [100])

    documents, all_queries, qrels = _resolve_dataset_dict(test_dataset)
    queries = [query for query in all_queries if str(query.get("split", "test")) == "test"]
    if not queries:
        queries = list(all_queries)

    relevant_docs, relevance_maps = _relevant_lists_from_queries(queries, qrels)
    docs_by_id = {
        str(doc.get("doc_id") or doc.get("id") or doc.get("document_id") or f"doc_{idx:06d}"): dict(doc)
        for idx, doc in enumerate(documents)
    }

    eval_device = _get_model_device(model, device)
    model.to(eval_device)
    model.eval()

    tokenizer = _get_model_tokenizer(model)
    cqe_wrapper: CQEWrapper | None
    try:
        cqe_wrapper = CQEWrapper()
    except Exception:
        cqe_wrapper = None

    max_length = int(getattr(model, "config", {}).get("evaluation", {}).get("max_length", 256))
    doc_payloads: dict[str, dict[str, Any]] = {}
    doc_vectors: list[np.ndarray] = []
    doc_ids = list(docs_by_id.keys())

    with torch.no_grad():
        for start in tqdm(range(0, len(doc_ids), 64), desc="Encode docs", leave=False):
            batch_doc_ids = doc_ids[start : start + 64]
            for doc_id in batch_doc_ids:
                doc = docs_by_id[doc_id]
                text = str(doc.get("text") or doc.get("doc_text") or "")
                spans = doc.get("quantity_spans") or doc.get("spans")
                payload = _prepare_payload(
                    model=model,
                    tokenizer=tokenizer,
                    text=text,
                    raw_spans=spans,
                    max_length=max_length,
                    cqe_wrapper=cqe_wrapper,
                )
                doc_payloads[doc_id] = payload
                payload_device = _to_device(payload, eval_device)
                vector = _encode_document_vector(model, payload_device).detach().cpu().numpy().astype(np.float32)
                doc_vectors.append(vector)

    if not doc_vectors:
        raise ValueError("run_full_evaluation() found no documents to index.")

    doc_matrix = np.stack(doc_vectors, axis=0)
    if faiss is not None:
        index = faiss.IndexFlatIP(doc_matrix.shape[1])
        index.add(doc_matrix)
    else:  # pragma: no cover - FAISS fallback
        index = None

    rankings: list[list[str]] = []
    query_types: list[str | None] = []
    query_ids: list[str] = []
    stage1_rankings: list[list[str]] = []

    with torch.no_grad():
        for query in tqdm(queries, desc="Evaluate queries", leave=False):
            query_id = str(query.get("query_id") or query.get("id") or f"query_{len(query_ids):06d}")
            query_text = str(query.get("query_text") or query.get("text") or "")
            query_payload = _prepare_payload(
                model=model,
                tokenizer=tokenizer,
                text=query_text,
                raw_spans=query.get("query_spans") or query.get("quantity_spans"),
                max_length=max_length,
                cqe_wrapper=cqe_wrapper,
            )
            query_payload_device = _to_device(query_payload, eval_device)
            query_vector = _encode_query_vector(model, query_payload_device).detach().cpu().numpy().astype(np.float32)

            top_k = min(retrieval_k, len(doc_ids))
            if faiss is not None:
                _, retrieved = index.search(query_vector[None, :], top_k)
                candidate_indices = retrieved[0].tolist()
            else:  # pragma: no cover - FAISS fallback
                sims = np.matmul(doc_matrix, query_vector)
                candidate_indices = np.argsort(-sims)[:top_k].tolist()

            stage1_doc_ids = [doc_ids[int(idx)] for idx in candidate_indices]
            stage1_rankings.append(stage1_doc_ids)

            reranked: list[tuple[float, str]] = []
            for doc_id in stage1_doc_ids:
                doc_payload_device = _to_device(doc_payloads[doc_id], eval_device)
                score = _forward_score(model, query_payload_device, doc_payload_device)
                reranked.append((score, doc_id))
            reranked.sort(key=lambda item: item[0], reverse=True)

            rankings.append([doc_id for _, doc_id in reranked])
            query_types.append(_infer_query_type(query))
            query_ids.append(query_id)

    overall_metrics = _compute_metric_bundle(rankings, relevant_docs, relevance_maps, mrr_k=mrr_k, recall_k=recall_k)

    subset_indices: dict[str, list[int]] = {
        "overall": list(range(len(queries))),
        "atomic_only": [idx for idx, query_type in enumerate(query_types) if query_type == "atomic"],
        "typeA_only": [idx for idx, query_type in enumerate(query_types) if query_type == "typeA"],
        "typeB_only": [idx for idx, query_type in enumerate(query_types) if query_type == "typeB"],
        "typeC_only": [idx for idx, query_type in enumerate(query_types) if query_type == "typeC"],
    }

    metrics: dict[str, Any] = {"overall": overall_metrics}
    for subset_name, indices in subset_indices.items():
        if subset_name == "overall":
            continue
        if not indices:
            metrics[subset_name] = {
                f"mrr{mrr_k}": 0.0,
                f"ndcg{mrr_k}": 0.0,
                f"p{mrr_k}": 0.0,
                f"r{recall_k}": 0.0,
                "per_query_rr": [],
            }
            continue

        subset_rankings = [rankings[idx] for idx in indices]
        subset_relevant = [relevant_docs[idx] for idx in indices]
        subset_relevance = [relevance_maps[idx] for idx in indices]
        metrics[subset_name] = _compute_metric_bundle(
            subset_rankings,
            subset_relevant,
            subset_relevance,
            mrr_k=mrr_k,
            recall_k=recall_k,
        )

    metrics["query_ids"] = query_ids
    metrics["query_types"] = query_types
    metrics["rankings"] = rankings
    metrics["stage1_rankings"] = stage1_rankings
    metrics["model_name"] = model.__class__.__name__

    print("System       | MRR@10 | NDCG@10 | P@10 | R@100")
    print("------------ | ------ | ------- | ---- | -----")
    for row_name, row_metrics in [
        (metrics["model_name"], metrics["overall"]),
        ("atomic_only", metrics["atomic_only"]),
        ("typeA_only", metrics["typeA_only"]),
        ("typeB_only", metrics["typeB_only"]),
        ("typeC_only", metrics["typeC_only"]),
    ]:
        print(
            f"{row_name:<12} | "
            f"{row_metrics.get('mrr10', 0.0):6.3f} | "
            f"{row_metrics.get('ndcg10', 0.0):7.3f} | "
            f"{row_metrics.get('p10', 0.0):4.2f} | "
            f"{row_metrics.get('r100', 0.0):5.2f}"
        )

    return metrics


if __name__ == "__main__":
    rankings = [["d1", "d2", "d3"], ["d4", "d5", "d6"], ["d7", "d8", "d9"]]
    relevant = [["d2"], ["d5", "d6"], ["dx"]]
    relevance = [{"d2": 1}, {"d5": 1, "d6": 1}, {"dx": 1}]

    mrr_value, per_query_rr = mrr_at_k(rankings, relevant, k=3)
    ndcg_value = ndcg_at_k(rankings, relevance, k=3)
    p_value = precision_at_k(rankings, relevant, k=3)
    r_value = recall_at_k(rankings, relevant, k=3)
    mc = mcnemar_test([1.0, 0.0, 1.0], [0.0, 0.0, 1.0])

    assert abs(mrr_value - ((0.5 + 0.5 + 0.0) / 3.0)) < 1e-6
    assert len(per_query_rr) == 3
    assert 0.0 <= ndcg_value <= 1.0
    assert abs(p_value - (1.0 / 3.0)) < 1e-6
    assert abs(r_value - ((1.0 + 1.0 + 0.0) / 3.0)) < 1e-6
    assert "p_value" in mc and "statistic" in mc and "significant" in mc
    print("Metrics smoke test: PASS")
