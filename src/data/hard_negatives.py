"""Hard-negative generation utilities for WideQuant datasets."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Sequence

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover - optional dependency fallback
    BM25Okapi = None

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", flags=re.IGNORECASE)
QUERY_TO_DOC_TYPE = {"atomic": "atomic", "typeA": "typeA", "typeB": "typeB", "typeC": "typeC", "mixed": "atomic"}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows from disk."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 and lexical similarity."""
    return TOKEN_PATTERN.findall(text.lower())


def _lexical_rank(query_tokens: list[str], corpus_tokens: list[list[str]]) -> list[int]:
    """Fallback ranking by token overlap when BM25 is unavailable."""
    query_set = set(query_tokens)
    scored: list[tuple[int, int]] = []
    for idx, tokens in enumerate(corpus_tokens):
        scored.append((len(query_set.intersection(tokens)), idx))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [idx for _, idx in scored]


class HardNegativeGenerator:
    """Generate challenging non-relevant documents for arithmetic retrieval."""

    def __init__(self) -> None:
        """Initialize reusable helper state."""
        self._rng = random.Random(42)

    @staticmethod
    def _group_documents(all_products: Sequence[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
        """Group document records by product id and document type."""
        grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for doc in all_products:
            grouped[str(doc["product_id"])][str(doc["doc_type"])] = dict(doc)
        return dict(grouped)

    @staticmethod
    def _docs_by_id(corpus: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Map doc ids to document rows."""
        return {str(doc["doc_id"]): dict(doc) for doc in corpus}

    @staticmethod
    def _name_similarity(name_a: str, name_b: str) -> float:
        """Compute a rough category similarity based on names."""
        tokens_a = set(_tokenize(name_a))
        tokens_b = set(_tokenize(name_b))
        overlap = 0.0
        if tokens_a or tokens_b:
            overlap = len(tokens_a.intersection(tokens_b)) / max(1, len(tokens_a.union(tokens_b)))
        ratio = SequenceMatcher(None, name_a.lower(), name_b.lower()).ratio()
        return 0.6 * overlap + 0.4 * ratio

    @staticmethod
    def _query_thresholds(query: dict[str, Any]) -> dict[str, float]:
        """Normalize query threshold payload into a simple numeric dictionary."""
        threshold_value = query.get("threshold_value", 0.0)
        if isinstance(threshold_value, dict):
            return {str(key): float(value) for key, value in threshold_value.items()}
        query_type = str(query.get("query_type", ""))
        if query_type in {"typeA", "typeC"}:
            return {"energy_kcal": float(threshold_value)}
        if query_type == "typeB":
            return {"protein_pct": float(threshold_value)}
        if query_type == "atomic":
            return {"fat_g": float(threshold_value)}
        return {"value": float(threshold_value)}

    @staticmethod
    def _product_metrics(product_docs: dict[str, dict[str, Any]]) -> dict[str, float]:
        """Compute the arithmetic quantities used to test query satisfaction."""
        atomic = product_docs.get("atomic") or next(iter(product_docs.values()))
        energy_kcal = float(atomic.get("energy_kcal", 0.0))
        protein_g = float(atomic.get("protein_g", 0.0))
        fat_g = float(atomic.get("fat_g", 0.0))
        protein_kcal = float(atomic.get("protein_kcal", protein_g * 4.0))
        fat_kcal = float(atomic.get("fat_kcal", fat_g * 9.0))
        carbs_kcal = float(atomic.get("carbs_kcal", 0.0))
        energy_kj = float(atomic.get("energy_kj", energy_kcal * 4.184))
        protein_pct = (protein_kcal / energy_kcal * 100.0) if energy_kcal > 1e-8 else 0.0
        return {
            "energy_kcal": energy_kcal,
            "protein_g": protein_g,
            "fat_g": fat_g,
            "energy_kj": energy_kj,
            "protein_pct": protein_pct,
            "typeA_energy": protein_kcal + fat_kcal + carbs_kcal,
        }

    def _reference_doc(self, query: dict[str, Any], grouped: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any] | None:
        """Resolve the first relevant document for the query."""
        relevant_ids = [str(doc_id) for doc_id in query.get("relevant_doc_ids", query.get("relevant_product_ids", []))]
        if not relevant_ids:
            return None
        product_id = relevant_ids[0].split("__", 1)[0]
        preferred_doc_type = QUERY_TO_DOC_TYPE.get(str(query.get("query_type", "mixed")), "atomic")
        return grouped.get(product_id, {}).get(preferred_doc_type) or grouped.get(product_id, {}).get("atomic")

    def _candidate_doc(self, product_docs: dict[str, dict[str, Any]], query_type: str) -> dict[str, Any] | None:
        """Pick the document variant that matches the intended reasoning type."""
        preferred = QUERY_TO_DOC_TYPE.get(query_type, "atomic")
        return product_docs.get(preferred) or product_docs.get("atomic")

    def _fails_query(self, query: dict[str, Any], metrics: dict[str, float]) -> bool:
        """Check whether a product fails the query constraint."""
        query_type = str(query.get("query_type", ""))
        operator = str(query.get("operator", "gt"))
        thresholds = self._query_thresholds(query)

        if query_type in {"typeA", "typeC"}:
            value = metrics["energy_kcal"]
            threshold = thresholds["energy_kcal"]
            return value < threshold if operator == "gt" else value > threshold
        if query_type == "typeB":
            value = metrics["protein_pct"]
            threshold = thresholds["protein_pct"]
            return value < threshold if operator == "gt" else value > threshold
        if query_type == "atomic":
            value = metrics["fat_g"]
            threshold = thresholds["fat_g"]
            return value < threshold if operator == "gt" else value > threshold
        if query_type == "mixed":
            return (
                metrics["energy_kcal"] < thresholds["energy_kcal"]
                or metrics["protein_g"] < thresholds["protein_g"]
            )
        return False

    def _sorted_candidate_products(
        self,
        query: dict[str, Any],
        all_products: Sequence[dict[str, Any]],
    ) -> list[tuple[float, str, dict[str, dict[str, Any]], dict[str, float]]]:
        """Score candidate products by category/name similarity to the query's positive example."""
        grouped = self._group_documents(all_products)
        reference = self._reference_doc(query, grouped)
        if reference is None:
            return []
        reference_product_id = str(reference["product_id"])
        reference_name = str(reference.get("name", reference_product_id))
        scored: list[tuple[float, str, dict[str, dict[str, Any]], dict[str, float]]] = []
        for product_id, docs in grouped.items():
            if product_id == reference_product_id:
                continue
            atomic = docs.get("atomic") or next(iter(docs.values()))
            score = self._name_similarity(reference_name, str(atomic.get("name", product_id)))
            scored.append((score, product_id, docs, self._product_metrics(docs)))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def generate_threshold_violation(
        self,
        query: dict[str, Any],
        all_products: Sequence[dict[str, Any]],
        n: int = 5,
    ) -> list[dict[str, Any]]:
        """Generate same-category negatives that fail the query threshold."""
        query_type = str(query.get("query_type", "mixed"))
        negatives: list[dict[str, Any]] = []
        for _, _, docs, metrics in self._sorted_candidate_products(query, all_products):
            if not self._fails_query(query, metrics):
                continue
            candidate = self._candidate_doc(docs, query_type)
            if candidate is None:
                continue
            negatives.append(candidate)
            if len(negatives) >= n:
                break
        return negatives

    def generate_unit_mismatch(
        self,
        query: dict[str, Any],
        all_products: Sequence[dict[str, Any]],
        n: int = 5,
    ) -> list[dict[str, Any]]:
        """Generate typeC negatives where kJ looks large textually but the kcal value still fails."""
        thresholds = self._query_thresholds(query)
        if "energy_kcal" not in thresholds:
            return []
        threshold = thresholds["energy_kcal"]
        negatives: list[dict[str, Any]] = []
        for _, _, docs, metrics in self._sorted_candidate_products(query, all_products):
            type_c = docs.get("typeC")
            if type_c is None:
                continue
            if metrics["energy_kj"] <= threshold:
                continue
            if not self._fails_query(query, metrics):
                continue
            negatives.append(type_c)
            if len(negatives) >= n:
                break
        return negatives

    def generate_wrong_arithmetic(
        self,
        query: dict[str, Any],
        all_products: Sequence[dict[str, Any]],
        n: int = 5,
    ) -> list[dict[str, Any]]:
        """Generate typeA negatives whose arithmetic sum fails the query threshold."""
        thresholds = self._query_thresholds(query)
        if "energy_kcal" not in thresholds:
            return []
        threshold = thresholds["energy_kcal"]
        negatives: list[dict[str, Any]] = []
        for _, _, docs, metrics in self._sorted_candidate_products(query, all_products):
            type_a = docs.get("typeA")
            if type_a is None:
                continue
            if metrics["typeA_energy"] >= threshold:
                continue
            negatives.append(type_a)
            if len(negatives) >= n:
                break
        return negatives

    def generate_bm25_negatives(
        self,
        query: dict[str, Any],
        corpus: Sequence[dict[str, Any]],
        n: int = 10,
    ) -> list[dict[str, Any]]:
        """Take top lexical non-relevant documents as additional hard negatives."""
        query_text = str(query.get("query_text", ""))
        relevant_ids = set(str(doc_id) for doc_id in query.get("relevant_doc_ids", query.get("relevant_product_ids", [])))
        tokenized_corpus = [_tokenize(str(doc.get("text", ""))) for doc in corpus]
        query_tokens = _tokenize(query_text)

        if BM25Okapi is not None:
            bm25 = BM25Okapi(tokenized_corpus)
            scores = bm25.get_scores(query_tokens)
            ranked = sorted(range(len(scores)), key=lambda idx: float(scores[idx]), reverse=True)
        else:
            ranked = _lexical_rank(query_tokens, tokenized_corpus)

        negatives: list[dict[str, Any]] = []
        for idx in ranked:
            candidate = corpus[idx]
            if str(candidate["doc_id"]) in relevant_ids:
                continue
            negatives.append(dict(candidate))
            if len(negatives) >= n:
                break
        return negatives

    def generate_all(
        self,
        query: dict[str, Any],
        all_products: Sequence[dict[str, Any]],
        corpus: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Combine all negative generators, deduplicate, and cap at 20 documents."""
        combined = (
            self.generate_threshold_violation(query, all_products, n=5)
            + self.generate_unit_mismatch(query, all_products, n=5)
            + self.generate_wrong_arithmetic(query, all_products, n=5)
            + self.generate_bm25_negatives(query, corpus, n=10)
        )
        deduped: list[dict[str, Any]] = []
        seen_doc_ids: set[str] = set()
        seen_products: set[str] = set()
        relevant_ids = set(str(doc_id) for doc_id in query.get("relevant_doc_ids", query.get("relevant_product_ids", [])))

        for doc in combined:
            doc_id = str(doc["doc_id"])
            product_id = str(doc["product_id"])
            if doc_id in relevant_ids or doc_id in seen_doc_ids or product_id in seen_products:
                continue
            deduped.append(dict(doc))
            seen_doc_ids.add(doc_id)
            seen_products.add(product_id)
            if len(deduped) >= 20:
                break
        return deduped

    def verify_hard_negatives(self, dataset_dir: str, sample_n: int = 100) -> float:
        """Measure BM25 MRR@10 when positives compete against generated hard negatives."""
        dataset_path = Path(dataset_dir)
        documents = _load_jsonl(dataset_path / "documents.jsonl")
        queries = _load_jsonl(dataset_path / "queries.jsonl")
        docs_by_id = self._docs_by_id(documents)

        rng = random.Random(42)
        sampled_queries = queries if len(queries) <= sample_n else rng.sample(queries, sample_n)
        reciprocal_ranks: list[float] = []

        for query in sampled_queries:
            negatives = self.generate_all(query, documents, documents)
            relevant_docs = [
                docs_by_id[doc_id]
                for doc_id in query.get("relevant_doc_ids", query.get("relevant_product_ids", []))
                if doc_id in docs_by_id
            ]
            candidate_docs = relevant_docs + negatives
            if not candidate_docs:
                continue

            tokenized_corpus = [_tokenize(str(doc.get("text", ""))) for doc in candidate_docs]
            query_tokens = _tokenize(str(query.get("query_text", "")))
            if BM25Okapi is not None:
                bm25 = BM25Okapi(tokenized_corpus)
                scores = bm25.get_scores(query_tokens)
                ranked = sorted(range(len(scores)), key=lambda idx: float(scores[idx]), reverse=True)
            else:
                ranked = _lexical_rank(query_tokens, tokenized_corpus)

            relevant_set = set(str(doc["doc_id"]) for doc in relevant_docs)
            rr = 0.0
            for rank, idx in enumerate(ranked[:10], start=1):
                if str(candidate_docs[idx]["doc_id"]) in relevant_set:
                    rr = 1.0 / float(rank)
                    break
            reciprocal_ranks.append(rr)

        mrr = float(sum(reciprocal_ranks) / max(1, len(reciprocal_ranks)))
        print(f"Hard negatives BM25 MRR@10: {mrr:.4f}")
        if mrr > 0.30:
            print("WARNING: negatives too easy")
        else:
            print("HARD NEGATIVES: PASS")
        return mrr


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for hard-negative verification."""
    parser = argparse.ArgumentParser(description="Verify WideQuant hard negatives")
    parser.add_argument("--dataset_dir", type=str, default="data/openfoodfacts")
    parser.add_argument("--sample_n", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generator = HardNegativeGenerator()
    generator.verify_hard_negatives(dataset_dir=args.dataset_dir, sample_n=int(args.sample_n))
