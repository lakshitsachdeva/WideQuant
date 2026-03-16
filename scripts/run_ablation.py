"""Run WideQuant ablations across datasets and save aggregate results."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - FAISS fallback
    faiss = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ecommerce_synthetic import build_dataset as build_ecommerce_dataset
from src.data.finquant_extension import build_finquant_extension
from src.data.openfoodfacts import build_dataset as build_openfoodfacts_dataset
from src.evaluation.metrics import (
    _get_model_device,
    _get_model_tokenizer,
    _prepare_payload,
    _relevant_lists_from_queries,
    _resolve_dataset_dict,
    mcnemar_test,
    mrr_at_k,
)
from src.models.deepquant import DeepQuant
from src.models.widequant import WideQuant


@dataclass(slots=True)
class VariantConfig:
    """One ablation variant definition."""

    key: str
    label: str
    model_kind: str
    use_aan: bool = True
    allowed_types: tuple[str, ...] | None = None
    oracle_detector: bool = False


@dataclass(slots=True)
class DatasetConfig:
    """One dataset definition for ablation runs."""

    key: str
    label: str
    path: Path


VARIANTS: list[VariantConfig] = [
    VariantConfig("baseline", "DeepQuant (baseline)", model_kind="deepquant", use_aan=False),
    VariantConfig("type_c_only", "+ Type C only", model_kind="widequant", use_aan=True, allowed_types=("TYPE_C",)),
    VariantConfig("type_a_only", "+ Type A only", model_kind="widequant", use_aan=True, allowed_types=("TYPE_A",)),
    VariantConfig("type_b_only", "+ Type B only", model_kind="widequant", use_aan=True, allowed_types=("TYPE_B",)),
    VariantConfig("reencode", "+ All types, Re-Encode", model_kind="widequant", use_aan=False),
    VariantConfig("aan_full", "+ All types, AAN (FULL)", model_kind="widequant", use_aan=True),
    VariantConfig("oracle", "+ Oracle Detector", model_kind="widequant", use_aan=True, oracle_detector=True),
]

ORACLE_QUERY_TYPE_MAP = {
    "atomic": "ATOMIC",
    "typeA": "TYPE_A",
    "typeB": "TYPE_B",
    "typeC": "TYPE_C",
    "ATOMIC": "ATOMIC",
    "TYPE_A": "TYPE_A",
    "TYPE_B": "TYPE_B",
    "TYPE_C": "TYPE_C",
}
DOC_TYPE_TO_ORACLE = {
    "atomic": "ATOMIC",
    "typeA": "TYPE_A",
    "typeB": "TYPE_B",
    "typeC": "TYPE_C",
    "TYPE_A": "TYPE_A",
    "TYPE_B": "TYPE_B",
    "TYPE_C": "TYPE_C",
}
DECOMP_QUERY_TYPES = {"typeA", "typeB", "typeC", "TYPE_A", "TYPE_B", "TYPE_C"}
ATOMIC_QUERY_TYPES = {"atomic", "ATOMIC"}


class RestrictedDetector:
    """Detector wrapper that suppresses arithmetic types outside an allowed set."""

    def __init__(self, base_detector: Any, allowed_types: Sequence[str]) -> None:
        """Store the wrapped detector and permitted labels."""
        self.base_detector = base_detector
        self.allowed_types = {str(label) for label in allowed_types}

    def detect(self, quantities: list[Any], query_unit: str, query_concept: str) -> str:
        """Return the base detector label when allowed, else ATOMIC."""
        label = str(self.base_detector.detect(quantities, query_unit=query_unit, query_concept=query_concept))
        return label if label in self.allowed_types else "ATOMIC"


class OracleDetector:
    """Detector wrapper that returns a query-provided oracle arithmetic type."""

    def __init__(self) -> None:
        """Initialize with ATOMIC as the default label."""
        self.current_label = "ATOMIC"

    def set_label(self, label: str) -> None:
        """Update the oracle label for the next query evaluation."""
        self.current_label = str(label)

    def detect(self, quantities: list[Any], query_unit: str, query_concept: str) -> str:
        """Ignore inputs and emit the current oracle label."""
        _ = (quantities, query_unit, query_concept)
        return self.current_label


class VariantRuntime:
    """Runtime bundle containing a configured model plus variant controls."""

    def __init__(self, model: torch.nn.Module, variant: VariantConfig, oracle_detector: OracleDetector | None) -> None:
        """Store model and ablation-specific options."""
        self.model = model
        self.variant = variant
        self.oracle_detector = oracle_detector

    def set_oracle_label(self, label: str) -> None:
        """Set the current oracle detector label when this variant uses one."""
        if self.oracle_detector is not None:
            self.oracle_detector.set_label(label)

    def score_pair(self, query_payload: dict[str, Any], doc_payload: dict[str, Any]) -> float:
        """Score one query-document pair under the current ablation settings."""
        if isinstance(self.model, WideQuant):
            outputs = self.model(query_payload, doc_payload, None, use_aan=self.variant.use_aan)
        else:
            outputs = self.model(query_payload, doc_payload, None)
        return float(outputs["final_score"].detach().cpu().item())


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for ablation runs."""
    parser = argparse.ArgumentParser(description="Run WideQuant ablations across datasets")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--deepquant_ckpt", type=str, default="checkpoints/best_deepquant.pt")
    parser.add_argument("--widequant_ckpt", type=str, default="checkpoints/best_widequant.pt")
    parser.add_argument("--openfoodfacts_dir", type=str, default="data/openfoodfacts")
    parser.add_argument("--finquant_dir", type=str, default="data/finquant")
    parser.add_argument("--finquant_extension_dir", type=str, default="data/finquant_extension")
    parser.add_argument("--ecommerce_dir", type=str, default="data/ecommerce")
    parser.add_argument("--datasets", nargs="*", choices=["openfoodfacts", "finquant_extension", "ecommerce"], default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[13, 42, 101])
    parser.add_argument("--output", type=str, default="ablation_results.json")
    parser.add_argument("--build_missing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_queries_per_dataset", type=int, default=None)
    parser.add_argument("--allow_downloads", action="store_true")
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--openfoodfacts_n_products", type=int, default=10000)
    parser.add_argument("--dry_run_openfoodfacts_n_products", type=int, default=100)
    parser.add_argument("--ecommerce_n_phones", type=int, default=5000)
    parser.add_argument("--ecommerce_n_laptops", type=int, default=5000)
    parser.add_argument("--ecommerce_n_prices", type=int, default=10000)
    parser.add_argument("--dry_run_ecommerce_n_phones", type=int, default=8)
    parser.add_argument("--dry_run_ecommerce_n_laptops", type=int, default=8)
    parser.add_argument("--dry_run_ecommerce_n_prices", type=int, default=8)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set deterministic random seeds for Python, NumPy, and Torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    """Load YAML config and inject cache/download controls."""
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config.setdefault("model", {})
    config["model"]["local_files_only"] = not bool(args.allow_downloads)
    if args.hf_cache_dir is not None:
        config["model"]["cache_dir"] = args.hf_cache_dir
    return config


def ensure_dataset(dataset_key: str, args: argparse.Namespace) -> DatasetConfig:
    """Ensure one ablation dataset exists, optionally building it."""
    if dataset_key == "openfoodfacts":
        path = Path(args.openfoodfacts_dir)
        required = [path / "documents.jsonl", path / "queries.jsonl", path / "qrels.tsv"]
        if not all(item.exists() for item in required):
            if not args.build_missing:
                raise FileNotFoundError(
                    f"Missing OpenFoodFacts dataset at {path}. Pass --build_missing to create it."
                )
            n_products = int(args.dry_run_openfoodfacts_n_products if args.dry_run else args.openfoodfacts_n_products)
            build_openfoodfacts_dataset(n_products=n_products, output_dir=str(path))
        return DatasetConfig("openfoodfacts", "OpenFoodFacts", path)

    if dataset_key == "finquant_extension":
        path = Path(args.finquant_extension_dir)
        required = [path / "corpus.jsonl", path / "queries.jsonl", path / "qrels.tsv"]
        if not all(item.exists() for item in required):
            if not args.build_missing:
                raise FileNotFoundError(
                    f"Missing FinQuant Extension dataset at {path}. Pass --build_missing to create it."
                )
            build_finquant_extension(args.finquant_dir, str(path))
        return DatasetConfig("finquant_extension", "FinQuant Extension", path)

    if dataset_key == "ecommerce":
        path = Path(args.ecommerce_dir)
        required = [path / "documents.jsonl", path / "queries.jsonl", path / "qrels.tsv"]
        if not all(item.exists() for item in required):
            if not args.build_missing:
                raise FileNotFoundError(
                    f"Missing E-Commerce dataset at {path}. Pass --build_missing to create it."
                )
            n_phones = int(args.dry_run_ecommerce_n_phones if args.dry_run else args.ecommerce_n_phones)
            n_laptops = int(args.dry_run_ecommerce_n_laptops if args.dry_run else args.ecommerce_n_laptops)
            n_prices = int(args.dry_run_ecommerce_n_prices if args.dry_run else args.ecommerce_n_prices)
            build_ecommerce_dataset(
                output_dir=str(path),
                n_phones=n_phones,
                n_laptops=n_laptops,
                n_prices=n_prices,
                queries_per_product=3,
                seed=42,
            )
        return DatasetConfig("ecommerce", "E-Commerce", path)

    raise ValueError(f"Unsupported dataset key: {dataset_key}")


def load_checkpoint_payload(path: Path) -> dict[str, Any]:
    """Load a checkpoint payload from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return torch.load(path, map_location="cpu")


def instantiate_variant(
    variant: VariantConfig,
    config: dict[str, Any],
    deepquant_ckpt: Path,
    widequant_ckpt: Path,
) -> VariantRuntime:
    """Instantiate a baseline or WideQuant ablation model."""
    oracle_detector: OracleDetector | None = None
    if variant.model_kind == "deepquant":
        model = DeepQuant(config)
        checkpoint = load_checkpoint_payload(deepquant_ckpt)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        return VariantRuntime(model=model, variant=variant, oracle_detector=None)

    model = WideQuant(config)
    ckpt_path = widequant_ckpt if widequant_ckpt.exists() else deepquant_ckpt
    checkpoint = load_checkpoint_payload(ckpt_path)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)

    if variant.oracle_detector:
        oracle_detector = OracleDetector()
        model.detector = oracle_detector
    elif variant.allowed_types is not None:
        model.detector = RestrictedDetector(model.detector, variant.allowed_types)

    return VariantRuntime(model=model, variant=variant, oracle_detector=oracle_detector)


def infer_query_subset(
    query: Mapping[str, Any],
    relevant_doc_ids: Sequence[str],
    docs_by_id: Mapping[str, Mapping[str, Any]],
) -> str:
    """Infer whether a query belongs to the atomic or decomposed subset."""
    query_type = str(query.get("query_type") or query.get("arith_type") or "").strip()
    if query_type in ATOMIC_QUERY_TYPES:
        return "atomic"
    if query_type in DECOMP_QUERY_TYPES:
        return "decomp"
    if query_type == "mixed":
        return "skip"

    for doc_id in relevant_doc_ids:
        doc = docs_by_id.get(str(doc_id))
        if doc is None:
            continue
        doc_type = str(doc.get("arith_type") or doc.get("doc_type") or "").strip()
        if bool(doc.get("is_decomposed_variant")) or doc_type in DECOMP_QUERY_TYPES:
            return "decomp"
    return "atomic"


def infer_oracle_label(
    query: Mapping[str, Any],
    relevant_doc_ids: Sequence[str],
    docs_by_id: Mapping[str, Mapping[str, Any]],
) -> str:
    """Infer the oracle arithmetic type for one query."""
    query_type = str(query.get("query_type") or query.get("arith_type") or "").strip()
    if query_type in ORACLE_QUERY_TYPE_MAP:
        return ORACLE_QUERY_TYPE_MAP[query_type]

    for doc_id in relevant_doc_ids:
        doc = docs_by_id.get(str(doc_id))
        if doc is None:
            continue
        doc_type = str(doc.get("arith_type") or doc.get("doc_type") or "").strip()
        if doc_type in DOC_TYPE_TO_ORACLE:
            return DOC_TYPE_TO_ORACLE[doc_type]
    return "ATOMIC"


def maybe_sample_queries(
    queries: list[dict[str, Any]],
    relevant_docs: list[list[str]],
    docs_by_id: Mapping[str, Mapping[str, Any]],
    max_queries: int | None,
    seed: int,
) -> tuple[list[dict[str, Any]], list[list[str]]]:
    """Optionally sample a smaller query set while keeping subset diversity."""
    if max_queries is None or len(queries) <= max_queries:
        return queries, relevant_docs

    rng = random.Random(seed)
    atomic_indices = [
        idx for idx, (query, rel_docs) in enumerate(zip(queries, relevant_docs, strict=False))
        if infer_query_subset(query, rel_docs, docs_by_id) == "atomic"
    ]
    decomp_indices = [
        idx for idx, (query, rel_docs) in enumerate(zip(queries, relevant_docs, strict=False))
        if infer_query_subset(query, rel_docs, docs_by_id) == "decomp"
    ]
    skip_indices = [
        idx for idx, (query, rel_docs) in enumerate(zip(queries, relevant_docs, strict=False))
        if infer_query_subset(query, rel_docs, docs_by_id) == "skip"
    ]

    selected: list[int] = []
    if atomic_indices and decomp_indices:
        atomic_target = min(len(atomic_indices), max_queries // 2)
        decomp_target = min(len(decomp_indices), max_queries - atomic_target)
        selected.extend(rng.sample(atomic_indices, atomic_target))
        selected.extend(rng.sample(decomp_indices, decomp_target))
    else:
        pool = atomic_indices + decomp_indices + skip_indices
        selected.extend(rng.sample(pool, min(len(pool), max_queries)))

    while len(selected) < min(max_queries, len(queries)):
        remaining = [idx for idx in range(len(queries)) if idx not in selected]
        if not remaining:
            break
        selected.append(rng.choice(remaining))

    selected = sorted(set(selected))[: max_queries]
    return [queries[idx] for idx in selected], [relevant_docs[idx] for idx in selected]


def encode_document_index(
    model: torch.nn.Module,
    documents: list[dict[str, Any]],
    docs_by_id: Mapping[str, Mapping[str, Any]],
    doc_ids: list[str],
    device: torch.device,
    max_length: int,
) -> tuple[dict[str, dict[str, Any]], np.ndarray]:
    """Encode all documents for ANN stage-1 retrieval."""
    tokenizer = _get_model_tokenizer(model)
    doc_payloads: dict[str, dict[str, Any]] = {}
    vectors: list[np.ndarray] = []

    with torch.no_grad():
        for start in tqdm(range(0, len(doc_ids), 64), desc="Encode docs", leave=False):
            for doc_id in doc_ids[start : start + 64]:
                doc = docs_by_id[doc_id]
                payload = _prepare_payload(
                    model=model,
                    tokenizer=tokenizer,
                    text=str(doc.get("text") or doc.get("doc_text") or ""),
                    raw_spans=doc.get("quantity_spans") or doc.get("spans"),
                    max_length=max_length,
                )
                doc_payloads[doc_id] = payload
                payload_device = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in payload.items()
                }
                encoded = model.encode_document(
                    payload_device["input_ids"],
                    payload_device["attention_mask"],
                    payload_device.get("quantity_spans", []),
                )
                vectors.append(F.normalize(encoded["cls"], dim=0).detach().cpu().numpy().astype(np.float32))

    if not vectors:
        raise ValueError("No documents were encoded for stage-1 retrieval.")
    return doc_payloads, np.stack(vectors, axis=0)


def build_stage1_index(doc_matrix: np.ndarray) -> Any:
    """Build a FAISS stage-1 index when available, else return None."""
    if faiss is None:
        return None
    index = faiss.IndexFlatIP(doc_matrix.shape[1])
    index.add(doc_matrix)
    return index


def search_stage1(index: Any, doc_matrix: np.ndarray, query_vector: np.ndarray, top_k: int) -> list[int]:
    """Retrieve top-k document indices using a prebuilt FAISS or NumPy fallback."""
    top_k = min(int(top_k), int(doc_matrix.shape[0]))
    if index is not None:
        _, retrieved = index.search(query_vector[None, :], top_k)
        return retrieved[0].tolist()
    scores = np.matmul(doc_matrix, query_vector)
    return np.argsort(-scores)[:top_k].tolist()


def align_scores(
    ids_a: Sequence[str],
    scores_a: Sequence[float],
    ids_b: Sequence[str],
    scores_b: Sequence[float],
) -> tuple[list[float], list[float]]:
    """Align two per-query score lists by query id."""
    score_map_a = {str(query_id): float(score) for query_id, score in zip(ids_a, scores_a, strict=False)}
    score_map_b = {str(query_id): float(score) for query_id, score in zip(ids_b, scores_b, strict=False)}
    shared_ids = [query_id for query_id in ids_a if query_id in score_map_b]
    return [score_map_a[query_id] for query_id in shared_ids], [score_map_b[query_id] for query_id in shared_ids]


def mean_std(values: Sequence[float]) -> dict[str, float]:
    """Compute mean and sample standard deviation with safe defaults."""
    if not values:
        return {"mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {"mean": float(statistics.mean(values)), "std": float(statistics.pstdev(values))}


def format_stat(values: Mapping[str, float]) -> str:
    """Render mean ± std for tables."""
    return f"{values.get('mean', 0.0):.3f}±{values.get('std', 0.0):.2f}"


def evaluate_variant_on_dataset(
    runtime: VariantRuntime,
    dataset: DatasetConfig,
    max_queries_per_dataset: int | None,
    seed: int,
) -> dict[str, Any]:
    """Evaluate one ablation variant on one dataset split."""
    model = runtime.model
    device = _get_model_device(model, None)
    model.to(device)
    model.eval()

    documents, all_queries, qrels = _resolve_dataset_dict(dataset.path)
    queries = [query for query in all_queries if str(query.get("split", "test")) == "test"]
    if not queries:
        queries = list(all_queries)

    relevant_docs, _ = _relevant_lists_from_queries(queries, qrels)
    docs_by_id = {
        str(doc.get("doc_id") or doc.get("id") or doc.get("document_id") or f"doc_{idx:06d}"): dict(doc)
        for idx, doc in enumerate(documents)
    }
    doc_ids = list(docs_by_id.keys())

    queries, relevant_docs = maybe_sample_queries(
        queries=queries,
        relevant_docs=relevant_docs,
        docs_by_id=docs_by_id,
        max_queries=max_queries_per_dataset,
        seed=seed,
    )

    max_length = int(getattr(model, "config", {}).get("evaluation", {}).get("max_length", 256))
    doc_payloads, doc_matrix = encode_document_index(
        model=model,
        documents=documents,
        docs_by_id=docs_by_id,
        doc_ids=doc_ids,
        device=device,
        max_length=max_length,
    )
    index = build_stage1_index(doc_matrix)

    tokenizer = _get_model_tokenizer(model)
    rankings: list[list[str]] = []
    query_ids: list[str] = []
    subset_labels: list[str] = []
    stage1_rankings: list[list[str]] = []

    with torch.no_grad():
        for query, rel_docs in tqdm(list(zip(queries, relevant_docs, strict=False)), desc=f"Eval {dataset.label}", leave=False):
            query_id = str(query.get("query_id") or query.get("id") or f"query_{len(query_ids):06d}")
            subset_label = infer_query_subset(query, rel_docs, docs_by_id)
            oracle_label = infer_oracle_label(query, rel_docs, docs_by_id)
            runtime.set_oracle_label(oracle_label)

            query_payload = _prepare_payload(
                model=model,
                tokenizer=tokenizer,
                text=str(query.get("query_text") or query.get("text") or ""),
                raw_spans=query.get("query_spans") or query.get("quantity_spans"),
                max_length=max_length,
            )
            query_payload_device = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in query_payload.items()
            }
            query_encoded = model.encode_query(
                query_payload_device["input_ids"],
                query_payload_device["attention_mask"],
                query_payload_device.get("quantity_spans", []),
            )
            query_vector = F.normalize(query_encoded["cls"], dim=0).detach().cpu().numpy().astype(np.float32)

            candidate_indices = search_stage1(index, doc_matrix, query_vector, top_k=100)
            candidate_doc_ids = [doc_ids[int(index)] for index in candidate_indices]
            stage1_rankings.append(candidate_doc_ids)

            reranked: list[tuple[float, str]] = []
            for doc_id in candidate_doc_ids:
                doc_payload = doc_payloads[doc_id]
                doc_payload_device = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in doc_payload.items()
                }
                reranked.append((runtime.score_pair(query_payload_device, doc_payload_device), doc_id))
            reranked.sort(key=lambda item: item[0], reverse=True)

            query_ids.append(query_id)
            subset_labels.append(subset_label)
            rankings.append([doc_id for _, doc_id in reranked])

    _, per_query_rr = mrr_at_k(rankings, relevant_docs, k=10)

    decomp_query_ids = [query_ids[idx] for idx, label in enumerate(subset_labels) if label == "decomp"]
    decomp_rrs = [float(per_query_rr[idx]) for idx, label in enumerate(subset_labels) if label == "decomp"]
    atomic_query_ids = [query_ids[idx] for idx, label in enumerate(subset_labels) if label == "atomic"]
    atomic_rrs = [float(per_query_rr[idx]) for idx, label in enumerate(subset_labels) if label == "atomic"]

    decomp_mrr = float(sum(decomp_rrs) / max(len(decomp_rrs), 1))
    atomic_mrr = float(sum(atomic_rrs) / max(len(atomic_rrs), 1))

    return {
        "dataset": dataset.key,
        "dataset_label": dataset.label,
        "query_ids": query_ids,
        "subset_labels": subset_labels,
        "stage1_rankings": stage1_rankings,
        "rankings": rankings,
        "per_query_rr": [float(value) for value in per_query_rr],
        "decomp_query_ids": decomp_query_ids,
        "decomp_rrs": decomp_rrs,
        "atomic_query_ids": atomic_query_ids,
        "atomic_rrs": atomic_rrs,
        "decomp_mrr10": decomp_mrr,
        "atomic_mrr10": atomic_mrr,
    }


def aggregate_variant_results(
    dataset_name: str,
    variant_key: str,
    seed_results: list[dict[str, Any]],
    baseline_seed_results: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Aggregate one variant's results across seeds for a single dataset."""
    decomp_values = [float(item["decomp_mrr10"]) for item in seed_results]
    atomic_values = [float(item["atomic_mrr10"]) for item in seed_results]

    p_values: list[float] = []
    mcnemar_payloads: list[dict[str, Any]] = []
    if baseline_seed_results is not None:
        for variant_result, baseline_result in zip(seed_results, baseline_seed_results, strict=False):
            aligned_variant, aligned_baseline = align_scores(
                variant_result["decomp_query_ids"],
                variant_result["decomp_rrs"],
                baseline_result["decomp_query_ids"],
                baseline_result["decomp_rrs"],
            )
            if aligned_variant and aligned_baseline:
                test_result = mcnemar_test(aligned_variant, aligned_baseline)
                p_values.append(float(test_result["p_value"]))
                mcnemar_payloads.append(test_result)

    summary = {
        "dataset": dataset_name,
        "variant": variant_key,
        "decomp_mrr10": mean_std(decomp_values),
        "atomic_mrr10": mean_std(atomic_values),
        "p_value": mean_std(p_values) if p_values else {"mean": 1.0, "std": 0.0},
        "seed_runs": seed_results,
        "mcnemar": mcnemar_payloads,
    }
    return summary


def build_overall_summary(dataset_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate variant results across all datasets."""
    variant_keys = list(VARIANTS[i].key for i in range(len(VARIANTS)))
    overall: dict[str, Any] = {}
    for variant_key in variant_keys:
        decomp_values: list[float] = []
        atomic_values: list[float] = []
        p_values: list[float] = []
        for dataset_name, result_map in dataset_results.items():
            variant_result = result_map[variant_key]
            decomp_values.append(float(variant_result["decomp_mrr10"]["mean"]))
            atomic_values.append(float(variant_result["atomic_mrr10"]["mean"]))
            if variant_key != "baseline":
                p_values.append(float(variant_result["p_value"]["mean"]))
        overall[variant_key] = {
            "decomp_mrr10": mean_std(decomp_values),
            "atomic_mrr10": mean_std(atomic_values),
            "p_value": mean_std(p_values) if p_values else {"mean": 1.0, "std": 0.0},
        }
    return overall


def print_table(title: str, result_map: Mapping[str, Any]) -> None:
    """Print one ablation result table."""
    print(title)
    print("Variant                      | MRR@10 Decomp | MRR@10 Atomic | p-value")
    print("---------------------------- | ------------- | ------------- | -------")
    for variant in VARIANTS:
        row = result_map[variant.key]
        p_text = "-"
        if variant.key != "baseline":
            p_text = format_stat(row["p_value"])
        print(
            f"{variant.label:<28} | {format_stat(row['decomp_mrr10']):>13} | "
            f"{format_stat(row['atomic_mrr10']):>13} | {p_text}"
        )


def gate_result(dataset_seed_results: dict[str, dict[str, Any]], overall_summary: dict[str, Any]) -> None:
    """Evaluate the final phase-6 gate using the full variant and baseline."""
    baseline_rrs: list[float] = []
    main_rrs: list[float] = []
    for dataset_name, result_map in dataset_seed_results.items():
        baseline_runs = result_map["baseline"]["seed_runs"]
        main_runs = result_map["aan_full"]["seed_runs"]
        for baseline_run, main_run in zip(baseline_runs, main_runs, strict=False):
            aligned_main, aligned_base = align_scores(
                main_run["decomp_query_ids"],
                main_run["decomp_rrs"],
                baseline_run["decomp_query_ids"],
                baseline_run["decomp_rrs"],
            )
            main_rrs.extend(aligned_main)
            baseline_rrs.extend(aligned_base)

    gate_test = mcnemar_test(main_rrs, baseline_rrs) if main_rrs and baseline_rrs else {"p_value": 1.0}
    baseline_decomp = float(overall_summary["baseline"]["decomp_mrr10"]["mean"])
    main_decomp = float(overall_summary["aan_full"]["decomp_mrr10"]["mean"])
    improvement = main_decomp - baseline_decomp

    if float(gate_test.get("p_value", 1.0)) < 0.05 and improvement > 0.05:
        print("PHASE 6 COMPLETE. RESULTS READY FOR PAPER.")
    else:
        print(
            "Gate failed: "
            f"p-value={float(gate_test.get('p_value', 1.0)):.4f}, "
            f"decomp improvement={improvement:.4f}."
        )
        if float(gate_test.get("p_value", 1.0)) >= 0.05:
            print("Investigate variance across seeds/datasets or weak effect size in decomposed retrieval.")
        if improvement <= 0.05:
            print("Investigate resolver coverage, detector quality, or AAN vs re-encode scoring gains.")


def main() -> None:
    """Run all configured ablations and save aggregated outputs."""
    args = parse_args()
    config = load_config(args)

    dataset_keys = list(args.datasets) if args.datasets is not None else ["openfoodfacts", "finquant_extension", "ecommerce"]
    seeds = list(args.seeds)
    if args.dry_run and seeds:
        seeds = [int(seeds[0])]
    max_queries = args.max_queries_per_dataset
    if args.dry_run and max_queries is None:
        max_queries = 12

    datasets = [ensure_dataset(dataset_key, args) for dataset_key in dataset_keys]
    deepquant_ckpt = Path(args.deepquant_ckpt)
    widequant_ckpt = Path(args.widequant_ckpt)

    dataset_seed_results: dict[str, dict[str, Any]] = {dataset.key: {} for dataset in datasets}

    for dataset in datasets:
        print(f"\nDataset: {dataset.label} ({dataset.path})")
        baseline_seed_results: list[dict[str, Any]] = []
        dataset_variant_results: dict[str, Any] = {}

        for variant in VARIANTS:
            print(f"Running variant: {variant.label}")
            seed_runs: list[dict[str, Any]] = []
            for seed in seeds:
                set_seed(int(seed))
                runtime = instantiate_variant(
                    variant=variant,
                    config=config,
                    deepquant_ckpt=deepquant_ckpt,
                    widequant_ckpt=widequant_ckpt,
                )
                seed_result = evaluate_variant_on_dataset(
                    runtime=runtime,
                    dataset=dataset,
                    max_queries_per_dataset=max_queries,
                    seed=int(seed),
                )
                seed_result["seed"] = int(seed)
                seed_runs.append(seed_result)
                print(
                    f"  seed={seed} | decomp MRR@10={seed_result['decomp_mrr10']:.4f} | "
                    f"atomic MRR@10={seed_result['atomic_mrr10']:.4f}"
                )
                del runtime
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if variant.key == "baseline":
                baseline_seed_results = seed_runs
                dataset_variant_results[variant.key] = aggregate_variant_results(
                    dataset_name=dataset.key,
                    variant_key=variant.key,
                    seed_results=seed_runs,
                    baseline_seed_results=None,
                )
            else:
                dataset_variant_results[variant.key] = aggregate_variant_results(
                    dataset_name=dataset.key,
                    variant_key=variant.key,
                    seed_results=seed_runs,
                    baseline_seed_results=baseline_seed_results,
                )

        dataset_seed_results[dataset.key] = dataset_variant_results
        print_table(f"\n{dataset.label} Results", dataset_variant_results)

    overall_summary = build_overall_summary(dataset_seed_results)
    print_table("\nOverall Results", overall_summary)

    gate_result(dataset_seed_results, overall_summary)

    output_payload = {
        "datasets": dataset_seed_results,
        "overall": overall_summary,
        "seeds": seeds,
        "datasets_run": [dataset.key for dataset in datasets],
        "dry_run": bool(args.dry_run),
        "max_queries_per_dataset": max_queries,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2)
    print(f"Saved ablation results to {output_path}")


if __name__ == "__main__":
    main()
