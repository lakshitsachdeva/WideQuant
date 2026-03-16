"""Failure analysis utilities for WideQuant retrieval errors."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import (
    _get_model_device,
    _get_model_tokenizer,
    _prepare_payload,
    _relevant_lists_from_queries,
    _resolve_dataset_dict,
    run_full_evaluation,
)

FAILURE_LABELS = [
    "DETECTOR_FAILURE",
    "RESOLVER_FAILURE",
    "SCORING_FAILURE",
    "TYPE_D_OUTOFSCOPE",
    "STAGE1_MISS",
]
QUERY_TYPE_TO_GROUND_TRUTH = {
    "atomic": "ATOMIC",
    "typeA": "TYPE_A",
    "typeB": "TYPE_B",
    "typeC": "TYPE_C",
    "mixed": "TYPE_D_OUTOFSCOPE",
}


class FailureCaseAnalyzer:
    """Categorize low-performing queries into detector/resolver/scoring buckets."""

    def __init__(self) -> None:
        """Initialize reusable analyzer state."""
        self.score_threshold = 0.10
        self.relative_tolerance = 0.10
        self.absolute_tolerance = 1e-4

    def _ground_truth_type(self, query: Mapping[str, Any]) -> str:
        """Infer the intended arithmetic type for one query."""
        query_type = str(query.get("query_type") or query.get("arith_type") or "").strip()
        if query_type in QUERY_TYPE_TO_GROUND_TRUTH:
            return QUERY_TYPE_TO_GROUND_TRUTH[query_type]
        if query_type in {"ATOMIC", "TYPE_A", "TYPE_B", "TYPE_C", "TYPE_D_OUTOFSCOPE"}:
            return query_type
        return "ATOMIC"

    def _is_out_of_scope(self, query: Mapping[str, Any], ground_truth_type: str) -> bool:
        """Return True when the query is contextual arithmetic outside current resolver support."""
        if ground_truth_type == "TYPE_D_OUTOFSCOPE":
            return True
        operator = str(query.get("operator") or "").strip().lower()
        threshold_value = query.get("threshold_value")
        return operator == "and" or isinstance(threshold_value, Mapping)

    def _query_constraint_satisfied(self, query: Mapping[str, Any], resolved_value: float | None) -> bool:
        """Check whether one resolved scalar satisfies the query threshold/operator."""
        if resolved_value is None:
            return False

        threshold = query.get("threshold_value")
        if isinstance(threshold, Mapping):
            return False
        if threshold is None:
            return True

        try:
            target = float(threshold)
        except (TypeError, ValueError):
            return False

        operator = str(query.get("operator") or "").strip().lower()
        if operator == "gt":
            return float(resolved_value) > target
        if operator == "lt":
            return float(resolved_value) < target
        if operator == "eq":
            return math.isclose(float(resolved_value), target, rel_tol=self.relative_tolerance, abs_tol=self.absolute_tolerance)
        return False

    def _value_matches_ground_truth(self, resolved_value: float | None, ground_truth_value: float | None) -> bool:
        """Return True when resolved and reference values match within tolerance."""
        if resolved_value is None or ground_truth_value is None:
            return False
        return math.isclose(
            float(resolved_value),
            float(ground_truth_value),
            rel_tol=self.relative_tolerance,
            abs_tol=self.absolute_tolerance,
        )

    def categorize_failure(
        self,
        query: Mapping[str, Any],
        retrieved_docs: list[str],
        relevant_docs: list[str],
        model_internals: dict[str, Any],
    ) -> str:
        """Return a coarse failure category for one query."""
        relevant_set = {str(doc_id) for doc_id in relevant_docs}
        stage1_candidates = [str(doc_id) for doc_id in model_internals.get("stage1_candidates", [])]
        stage1_set = set(stage1_candidates)
        ground_truth_type = str(model_internals.get("ground_truth_type") or self._ground_truth_type(query))
        detected_type = str(model_internals.get("detected_type") or "ATOMIC")
        resolved_value = model_internals.get("resolved_value")
        ground_truth_value = model_internals.get("ground_truth_value")
        resolved_candidate_score = float(model_internals.get("resolved_candidate_score", 0.0) or 0.0)

        if stage1_candidates and relevant_set and relevant_set.isdisjoint(stage1_set):
            return "STAGE1_MISS"

        if self._is_out_of_scope(query, ground_truth_type):
            return "TYPE_D_OUTOFSCOPE"

        if ground_truth_type != "ATOMIC" and detected_type != ground_truth_type:
            return "DETECTOR_FAILURE"

        if ground_truth_type != "ATOMIC":
            if not self._query_constraint_satisfied(query, resolved_value):
                return "RESOLVER_FAILURE"
            if ground_truth_value is not None and not self._value_matches_ground_truth(resolved_value, ground_truth_value):
                return "RESOLVER_FAILURE"

        if relevant_set and retrieved_docs:
            top_10 = {str(doc_id) for doc_id in retrieved_docs[:10]}
            if relevant_set.isdisjoint(top_10):
                return "SCORING_FAILURE"

        if ground_truth_type != "ATOMIC" and resolved_candidate_score < self.score_threshold:
            return "SCORING_FAILURE"

        return "SCORING_FAILURE"

    def _preferred_relevant_doc_id(
        self,
        query: Mapping[str, Any],
        relevant_doc_ids: list[str],
        docs_by_id: Mapping[str, Mapping[str, Any]],
    ) -> str | None:
        """Choose the relevant document variant that best matches the query type."""
        if not relevant_doc_ids:
            return None

        desired_type = str(query.get("query_type") or "").strip()
        for doc_id in relevant_doc_ids:
            doc = docs_by_id.get(str(doc_id))
            if doc is None:
                continue
            if str(doc.get("doc_type") or "") == desired_type:
                return str(doc_id)
        return str(relevant_doc_ids[0])

    def _reference_value(self, query: Mapping[str, Any]) -> float | None:
        """Return a simple numeric reference value for explanation/debugging."""
        threshold = query.get("threshold_value")
        if isinstance(threshold, Mapping) or threshold is None:
            return None
        try:
            return float(threshold)
        except (TypeError, ValueError):
            return None

    def _build_model_internals(
        self,
        model: Any,
        query: Mapping[str, Any],
        doc: Mapping[str, Any],
        stage1_candidates: list[str],
        device: torch.device,
    ) -> dict[str, Any]:
        """Extract detector/resolver/score internals for one query-document pair."""
        tokenizer = _get_model_tokenizer(model)
        query_payload = _prepare_payload(
            model=model,
            tokenizer=tokenizer,
            text=str(query.get("query_text") or query.get("text") or ""),
            raw_spans=query.get("query_spans") or query.get("quantity_spans"),
        )
        doc_payload = _prepare_payload(
            model=model,
            tokenizer=tokenizer,
            text=str(doc.get("text") or doc.get("doc_text") or ""),
            raw_spans=doc.get("quantity_spans") or doc.get("spans"),
        )

        query_payload_device = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in query_payload.items()
        }
        doc_payload_device = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in doc_payload.items()
        }

        outputs: dict[str, Any]
        with torch.no_grad():
            if model.__class__.__name__ == "WideQuant":
                outputs = model(query_payload_device, doc_payload_device, None, use_aan=True)
            else:
                outputs = model(query_payload_device, doc_payload_device, None)

        query_spans = list(query_payload.get("quantity_spans", []))
        doc_spans = list(doc_payload.get("quantity_spans", []))
        query_unit = str(query_spans[0].unit) if query_spans else str(query.get("query_unit") or "")
        query_concept = str(query_spans[0].concept) if query_spans else str(query.get("query_concept") or query.get("query_type") or "")

        detected_type = "ATOMIC"
        resolved_value: float | None = None
        if hasattr(model, "detector") and callable(getattr(model.detector, "detect", None)):
            try:
                detected_type = str(model.detector.detect(doc_spans, query_unit=query_unit, query_concept=query_concept))
            except Exception:
                detected_type = "ATOMIC"

        if detected_type != "ATOMIC" and hasattr(model, "resolvers"):
            resolver = getattr(model, "resolvers", {}).get(detected_type)
            if resolver is not None and callable(getattr(resolver, "resolve", None)):
                try:
                    if detected_type == "TYPE_A":
                        candidate = resolver.resolve(doc_spans, query_unit=query_unit, query_concept=query_concept)
                    else:
                        candidate = resolver.resolve(doc_spans, query_unit=query_unit)
                    if candidate is not None:
                        resolved_value = float(candidate.value)
                except Exception:
                    resolved_value = None

        resolved_candidate_score = 0.0
        if isinstance(outputs.get("resolved_candidate_scores"), torch.Tensor):
            scores_tensor = outputs["resolved_candidate_scores"]
            if scores_tensor.numel() > 0:
                resolved_candidate_score = float(scores_tensor.max().detach().cpu().item())

        return {
            "detected_type": detected_type,
            "ground_truth_type": self._ground_truth_type(query),
            "resolved_value": resolved_value,
            "ground_truth_value": self._reference_value(query),
            "resolved_candidate_score": resolved_candidate_score,
            "stage1_candidates": list(stage1_candidates),
            "final_score": float(outputs["final_score"].detach().cpu().item()) if isinstance(outputs.get("final_score"), torch.Tensor) else None,
        }

    def _explanation(
        self,
        category: str,
        query: Mapping[str, Any],
        model_internals: Mapping[str, Any],
        reciprocal_rank: float,
    ) -> str:
        """Build a short human-readable explanation for one failure."""
        if category == "STAGE1_MISS":
            return "Relevant document never appeared in the stage-1 top-100 candidate set."
        if category == "TYPE_D_OUTOFSCOPE":
            return "Query needs contextual or mixed arithmetic outside the current Type A/B/C resolver set."
        if category == "DETECTOR_FAILURE":
            return (
                f"Detector predicted {model_internals.get('detected_type', 'ATOMIC')} but expected "
                f"{model_internals.get('ground_truth_type', self._ground_truth_type(query))}."
            )
        if category == "RESOLVER_FAILURE":
            return (
                f"Resolver output {model_internals.get('resolved_value')} did not satisfy the query threshold "
                f"{query.get('operator')} {query.get('threshold_value')}."
            )
        return (
            f"Relevant doc still ranked low (RR={reciprocal_rank:.3f}) despite arithmetic path; "
            f"candidate score={float(model_internals.get('resolved_candidate_score', 0.0) or 0.0):.4f}."
        )

    def analyze_n_failures(self, model: Any, test_dataset: Any, n: int = 100) -> dict[str, Any]:
        """Analyze the worst-performing queries and summarize failure categories."""
        evaluation = run_full_evaluation(model, test_dataset, device=_get_model_device(model, None), k_values=[10, 100])
        documents, all_queries, qrels = _resolve_dataset_dict(test_dataset)
        queries = [query for query in all_queries if str(query.get("split", "test")) == "test"]
        if not queries:
            queries = list(all_queries)

        query_by_id = {
            str(query.get("query_id") or query.get("id") or f"query_{idx:06d}"): query
            for idx, query in enumerate(queries)
        }
        docs_by_id = {
            str(doc.get("doc_id") or doc.get("id") or doc.get("document_id") or f"doc_{idx:06d}"): doc
            for idx, doc in enumerate(documents)
        }
        relevant_docs_list, _ = _relevant_lists_from_queries(queries, qrels)
        relevant_docs_by_query_id = {
            str(query.get("query_id") or query.get("id") or f"query_{idx:06d}"): relevant_docs_list[idx]
            for idx, query in enumerate(queries)
        }

        query_ids = list(evaluation.get("query_ids", []))
        rankings = list(evaluation.get("rankings", []))
        stage1_rankings = list(evaluation.get("stage1_rankings", []))
        reciprocal_ranks = list(evaluation.get("overall", {}).get("per_query_rr", []))

        order = sorted(range(len(query_ids)), key=lambda idx: reciprocal_ranks[idx] if idx < len(reciprocal_ranks) else 0.0)
        worst_indices = order[: min(int(n), len(order))]

        counts = {label: 0 for label in FAILURE_LABELS}
        examples: dict[str, list[dict[str, Any]]] = {label: [] for label in FAILURE_LABELS}
        device = _get_model_device(model, None)

        for idx in worst_indices:
            query_id = str(query_ids[idx])
            query = query_by_id.get(query_id)
            if query is None:
                continue
            relevant_doc_ids = [str(doc_id) for doc_id in relevant_docs_by_query_id.get(query_id, [])]
            focus_doc_id = self._preferred_relevant_doc_id(query, relevant_doc_ids, docs_by_id)
            focus_doc = docs_by_id.get(str(focus_doc_id)) if focus_doc_id is not None else None

            model_internals = {
                "detected_type": "ATOMIC",
                "ground_truth_type": self._ground_truth_type(query),
                "resolved_value": None,
                "ground_truth_value": self._reference_value(query),
                "resolved_candidate_score": 0.0,
                "stage1_candidates": list(stage1_rankings[idx]) if idx < len(stage1_rankings) else [],
            }
            if focus_doc is not None:
                try:
                    model_internals = self._build_model_internals(
                        model=model,
                        query=query,
                        doc=focus_doc,
                        stage1_candidates=list(stage1_rankings[idx]) if idx < len(stage1_rankings) else [],
                        device=device,
                    )
                except Exception:
                    pass

            category = self.categorize_failure(
                query=query,
                retrieved_docs=list(rankings[idx]) if idx < len(rankings) else [],
                relevant_docs=relevant_doc_ids,
                model_internals=model_internals,
            )
            counts[category] += 1

            if len(examples[category]) < 3:
                examples[category].append(
                    {
                        "query_id": query_id,
                        "query_text": str(query.get("query_text") or query.get("text") or ""),
                        "category": category,
                        "reciprocal_rank": float(reciprocal_ranks[idx]) if idx < len(reciprocal_ranks) else 0.0,
                        "retrieved_top10": list(rankings[idx][:10]) if idx < len(rankings) else [],
                        "relevant_docs": relevant_doc_ids,
                        "explanation": self._explanation(
                            category=category,
                            query=query,
                            model_internals=model_internals,
                            reciprocal_rank=float(reciprocal_ranks[idx]) if idx < len(reciprocal_ranks) else 0.0,
                        ),
                    }
                )

        total = max(len(worst_indices), 1)
        print(f"Failure Analysis (n={len(worst_indices)}):")
        print(f"|- Detector Failure: {counts['DETECTOR_FAILURE']:3d} cases ({100.0 * counts['DETECTOR_FAILURE'] / total:5.1f}%)")
        print(f"|- Resolver Failure: {counts['RESOLVER_FAILURE']:3d} cases ({100.0 * counts['RESOLVER_FAILURE'] / total:5.1f}%)")
        print(f"|- Scoring Failure:  {counts['SCORING_FAILURE']:3d} cases ({100.0 * counts['SCORING_FAILURE'] / total:5.1f}%)")
        print(f"|- Type D (OOS):    {counts['TYPE_D_OUTOFSCOPE']:3d} cases ({100.0 * counts['TYPE_D_OUTOFSCOPE'] / total:5.1f}%)")
        print(f"`- Stage 1 Miss:    {counts['STAGE1_MISS']:3d} cases ({100.0 * counts['STAGE1_MISS'] / total:5.1f}%)")

        return {
            "DETECTOR_FAILURE": counts["DETECTOR_FAILURE"],
            "RESOLVER_FAILURE": counts["RESOLVER_FAILURE"],
            "SCORING_FAILURE": counts["SCORING_FAILURE"],
            "TYPE_D_OUTOFSCOPE": counts["TYPE_D_OUTOFSCOPE"],
            "STAGE1_MISS": counts["STAGE1_MISS"],
            "examples": examples,
        }


if __name__ == "__main__":
    analyzer = FailureCaseAnalyzer()

    detector_case = analyzer.categorize_failure(
        query={"query_type": "typeA", "operator": "gt", "threshold_value": 100.0},
        retrieved_docs=["d1", "d2"],
        relevant_docs=["d2"],
        model_internals={
            "detected_type": "ATOMIC",
            "ground_truth_type": "TYPE_A",
            "resolved_value": None,
            "ground_truth_value": 100.0,
            "resolved_candidate_score": 0.0,
            "stage1_candidates": ["d1", "d2"],
        },
    )
    resolver_case = analyzer.categorize_failure(
        query={"query_type": "typeB", "operator": "gt", "threshold_value": 20.0},
        retrieved_docs=["d1", "d2"],
        relevant_docs=["d2"],
        model_internals={
            "detected_type": "TYPE_B",
            "ground_truth_type": "TYPE_B",
            "resolved_value": 10.0,
            "ground_truth_value": 20.0,
            "resolved_candidate_score": 0.8,
            "stage1_candidates": ["d1", "d2"],
        },
    )
    scoring_case = analyzer.categorize_failure(
        query={"query_type": "typeC", "operator": "lt", "threshold_value": 250.0},
        retrieved_docs=["d1", "d3"],
        relevant_docs=["d2"],
        model_internals={
            "detected_type": "TYPE_C",
            "ground_truth_type": "TYPE_C",
            "resolved_value": 200.0,
            "ground_truth_value": None,
            "resolved_candidate_score": 0.7,
            "stage1_candidates": ["d1", "d2", "d3"],
        },
    )
    oos_case = analyzer.categorize_failure(
        query={"query_type": "mixed", "operator": "and", "threshold_value": {"a": 1}},
        retrieved_docs=["d1"],
        relevant_docs=["d1"],
        model_internals={
            "detected_type": "ATOMIC",
            "ground_truth_type": "TYPE_D_OUTOFSCOPE",
            "resolved_value": None,
            "ground_truth_value": None,
            "resolved_candidate_score": 0.0,
            "stage1_candidates": ["d1"],
        },
    )
    stage1_case = analyzer.categorize_failure(
        query={"query_type": "atomic", "operator": "gt", "threshold_value": 5.0},
        retrieved_docs=["d1", "d3"],
        relevant_docs=["d2"],
        model_internals={
            "detected_type": "ATOMIC",
            "ground_truth_type": "ATOMIC",
            "resolved_value": None,
            "ground_truth_value": 5.0,
            "resolved_candidate_score": 0.0,
            "stage1_candidates": ["d1", "d3"],
        },
    )

    assert detector_case == "DETECTOR_FAILURE"
    assert resolver_case == "RESOLVER_FAILURE"
    assert scoring_case == "SCORING_FAILURE"
    assert oos_case == "TYPE_D_OUTOFSCOPE"
    assert stage1_case == "STAGE1_MISS"
    print("FailureCaseAnalyzer smoke test: PASS")
