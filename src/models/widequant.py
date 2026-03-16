"""Full WideQuant model built on top of the DeepQuant baseline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.encoding.cqe_wrapper import QuantitySpan, ResolvedCandidate
from src.encoding.quantity_encoder import gaussian_mantissa_encoding
from src.models.arith_aggregation import AANTrainingObjective, ArithAggregationNetwork
from src.models.deepquant import DeepQuant
from src.resolution.decomposition_detector import RuleBasedDetector, build_arithmetic_candidates
from src.resolution.type_a_resolver import TypeAResolver
from src.resolution.type_b_resolver import TypeBResolver
from src.resolution.type_c_resolver import TypeCResolver

TYPE_TO_INDEX = {"TYPE_A": 0, "TYPE_B": 1, "TYPE_C": 2}


class WideQuant(DeepQuant):
    """Arithmetic-aware extension of DeepQuant with resolved candidate scoring."""

    def __init__(self, config: dict) -> None:
        """Initialize DeepQuant backbone plus arithmetic-specific modules."""
        super().__init__(config)
        self.aan = ArithAggregationNetwork(hidden_dim=self.hidden_dim)
        self.detector = RuleBasedDetector()
        self.resolvers = {
            "TYPE_A": TypeAResolver(),
            "TYPE_B": TypeBResolver(),
            "TYPE_C": TypeCResolver(),
        }
        self.aan_loss = AANTrainingObjective()

    def _quantity_key_to_metadata(self, key: tuple[Any, ...]) -> dict[str, Any]:
        """Recover quantity metadata from DeepQuant's stable dictionary key."""
        _, text, start_char, end_char, unit, concept = key
        return {
            "text": text,
            "start_char": int(start_char),
            "end_char": int(end_char),
            "unit": str(unit),
            "concept": str(concept),
        }

    def _build_span_embedding_lookup(self, spans: list[QuantitySpan], doc_enc: dict) -> dict[tuple[Any, ...], Tensor]:
        """Map span identity fields to document-side quantity embeddings."""
        embeddings = list(doc_enc.get("quantity_outputs", {}).values())
        if len(spans) != len(embeddings):
            return {}

        lookup: dict[tuple[Any, ...], Tensor] = {}
        for span, embedding in zip(spans, embeddings):
            span_key = (
                str(span.text),
                int(span.start_char),
                int(span.end_char),
                str(span.unit),
                str(span.concept),
            )
            lookup[span_key] = embedding
        return lookup

    def _resolved_candidate_embedding(
        self,
        resolved_candidate: ResolvedCandidate,
        doc_enc: dict,
        doc_quantities: list[QuantitySpan],
        use_aan: bool,
    ) -> Tensor | None:
        """Convert a resolved arithmetic candidate into an embedding."""
        if use_aan:
            span_lookup = self._build_span_embedding_lookup(doc_quantities, doc_enc)
            sub_embeddings: list[Tensor] = []
            for span in resolved_candidate.source_spans:
                span_key = (
                    str(span.text),
                    int(span.start_char),
                    int(span.end_char),
                    str(span.unit),
                    str(span.concept),
                )
                embedding = span_lookup.get(span_key)
                if embedding is None:
                    return None
                sub_embeddings.append(embedding)
            if not sub_embeddings:
                return None
            return self.aan(
                torch.stack(sub_embeddings, dim=0),
                arith_type=TYPE_TO_INDEX[resolved_candidate.source_type],
            )

        device = self.alpha_weights.device
        num_token_tensor = torch.tensor(self.num_token_id, dtype=torch.long, device=device)
        base_num_embedding = self.bert.get_input_embeddings()(num_token_tensor).to(dtype=torch.float32)
        exponent_vector = self.exponent_embedding(int(resolved_candidate.exponent)).to(device=device, dtype=torch.float32)
        mantissa_vector = gaussian_mantissa_encoding(
            float(resolved_candidate.mantissa),
            J_m=self.J_m,
            sigma=1.0,
        ).to(device=device, dtype=torch.float32)
        quantity_vector = torch.cat([exponent_vector, mantissa_vector], dim=0)
        return base_num_embedding + quantity_vector

    def _score_candidates(self, y_a: Tensor, candidates: list[Tensor]) -> tuple[Tensor, Tensor]:
        """Score a set of candidate document embeddings against one query quantity embedding."""
        if not candidates:
            zero = torch.tensor(0.0, dtype=torch.float32, device=y_a.device)
            return zero, zero.unsqueeze(0)[:0]

        candidate_tensor = torch.stack(candidates, dim=0)
        unit_compat = self.unit_compatibility_scorer(y_a, candidate_tensor)
        p_op = self.comparator_predictor(y_a.unsqueeze(0)).squeeze(0)

        candidate_scores: list[Tensor] = []
        for idx, y_c in enumerate(candidates):
            n_lt = self.comparator_pair_scorers["lt"](y_a, y_c)
            n_eq = self.comparator_pair_scorers["eq"](y_a, y_c)
            n_gt = self.comparator_pair_scorers["gt"](y_a, y_c)
            op_score = p_op[0] * n_lt + p_op[1] * n_eq + p_op[2] * n_gt
            candidate_scores.append(unit_compat[idx] * op_score)

        scores_tensor = torch.stack(candidate_scores)
        return scores_tensor.max(), scores_tensor

    def compute_quantity_score_arith(
        self,
        query_enc: dict,
        doc_quantities: list[QuantitySpan],
        doc_enc: dict,
        query_unit: str | None = None,
        query_concept: str | None = None,
        use_aan: bool = True,
    ) -> tuple[Tensor, list[Tensor], list[ResolvedCandidate]]:
        """Score a document using both original and resolved arithmetic candidates."""
        query_items = list(query_enc.get("quantity_outputs", {}).items())
        original_candidates = list(doc_enc.get("quantity_outputs", {}).values())
        device = query_enc["cls"].device
        if not query_items or not original_candidates:
            zero = torch.tensor(0.0, dtype=torch.float32, device=device)
            return zero, [], []

        total_score = torch.tensor(0.0, dtype=torch.float32, device=device)
        all_resolved_candidate_scores: list[Tensor] = []
        all_resolved_candidates: list[ResolvedCandidate] = []

        for key, y_a in query_items:
            metadata = self._quantity_key_to_metadata(key)
            unit_for_query = query_unit if query_unit is not None else metadata["unit"]
            concept_for_query = query_concept if query_concept is not None else metadata["concept"]
            resolved_candidates = build_arithmetic_candidates(
                doc_quantities=doc_quantities,
                query_unit=unit_for_query,
                query_concept=concept_for_query,
                detector=self.detector,
            )

            arithmetic_embeddings: list[Tensor] = []
            valid_resolved_candidates: list[ResolvedCandidate] = []
            for resolved_candidate in resolved_candidates:
                candidate_embedding = self._resolved_candidate_embedding(
                    resolved_candidate=resolved_candidate,
                    doc_enc=doc_enc,
                    doc_quantities=doc_quantities,
                    use_aan=use_aan,
                )
                if candidate_embedding is None:
                    continue
                arithmetic_embeddings.append(candidate_embedding)
                valid_resolved_candidates.append(resolved_candidate)

            best_score, all_candidate_scores = self._score_candidates(y_a, original_candidates + arithmetic_embeddings)
            total_score = total_score + best_score
            if arithmetic_embeddings:
                arithmetic_only = all_candidate_scores[-len(arithmetic_embeddings) :]
                all_resolved_candidate_scores.extend(list(arithmetic_only))
                all_resolved_candidates.extend(valid_resolved_candidates)

        return total_score, all_resolved_candidate_scores, all_resolved_candidates

    def _compute_optional_aan_loss(
        self,
        doc_batch: dict,
        doc_enc: dict,
    ) -> Tensor:
        """Compute optional AAN supervision if an atomic paired document is present."""
        atomic_doc_batch = doc_batch.get("atomic_doc_batch")
        if atomic_doc_batch is None:
            return torch.tensor(0.0, dtype=torch.float32, device=self.alpha_weights.device)

        doc_quantities = self._coerce_spans(doc_batch.get("quantity_spans", []))
        if not doc_quantities:
            return torch.tensor(0.0, dtype=torch.float32, device=self.alpha_weights.device)

        query_unit = str(doc_quantities[0].unit)
        query_concept = str(doc_quantities[0].concept)
        resolved_candidates = build_arithmetic_candidates(
            doc_quantities=doc_quantities,
            query_unit=query_unit,
            query_concept=query_concept,
            detector=self.detector,
        )
        if not resolved_candidates:
            return torch.tensor(0.0, dtype=torch.float32, device=self.alpha_weights.device)

        candidate = resolved_candidates[0]
        sub_embedding = self._resolved_candidate_embedding(candidate, doc_enc, doc_quantities, use_aan=True)
        if sub_embedding is None:
            return torch.tensor(0.0, dtype=torch.float32, device=self.alpha_weights.device)

        atomic_doc_enc = self.encode_document(
            input_ids=atomic_doc_batch["input_ids"],
            attention_mask=atomic_doc_batch["attention_mask"],
            quantity_spans=atomic_doc_batch.get("quantity_spans", []),
        )
        atomic_quantity_outputs = list(atomic_doc_enc.get("quantity_outputs", {}).values())
        if not atomic_quantity_outputs:
            return torch.tensor(0.0, dtype=torch.float32, device=self.alpha_weights.device)
        return self.aan_loss.compute_loss(sub_embedding, atomic_quantity_outputs[0])

    def forward(
        self,
        query_batch: dict,
        doc_pos_batch: dict,
        doc_neg_batch: dict | None = None,
        use_aan: bool = True,
    ) -> dict:
        """Run WideQuant forward pass with arithmetic-aware quantity scoring."""
        query_enc = self.encode_query(
            input_ids=query_batch["input_ids"],
            attention_mask=query_batch["attention_mask"],
            quantity_spans=query_batch.get("quantity_spans", []),
        )
        doc_pos_enc = self.encode_document(
            input_ids=doc_pos_batch["input_ids"],
            attention_mask=doc_pos_batch["attention_mask"],
            quantity_spans=doc_pos_batch.get("quantity_spans", []),
        )
        doc_pos_quantities = self._coerce_spans(doc_pos_batch.get("quantity_spans", []))

        pos_quantity_score, pos_resolved_scores, pos_resolved_candidates = self.compute_quantity_score_arith(
            query_enc=query_enc,
            doc_quantities=doc_pos_quantities,
            doc_enc=doc_pos_enc,
            use_aan=use_aan,
        )
        pos_text_score = self.compute_text_score(query_enc, doc_pos_enc)
        alpha = self.compute_alpha(query_enc)
        pos_final_score = (1.0 - alpha) * pos_quantity_score + alpha * pos_text_score

        resolved_candidate_scores = torch.stack(pos_resolved_scores) if pos_resolved_scores else torch.empty(
            0, dtype=torch.float32, device=self.alpha_weights.device
        )
        is_satisfying_mask = torch.ones_like(resolved_candidate_scores, dtype=torch.bool)
        aan_loss_value = self._compute_optional_aan_loss(doc_pos_batch, doc_pos_enc)

        result: dict[str, Any] = {
            "query_enc": query_enc,
            "doc_pos_enc": doc_pos_enc,
            "doc_neg_enc": None,
            "alpha": alpha,
            "quantity_score": pos_quantity_score,
            "text_score": pos_text_score,
            "final_score": pos_final_score,
            "pos_quantity_score": pos_quantity_score,
            "pos_text_score": pos_text_score,
            "pos_final_score": pos_final_score,
            "neg_quantity_score": None,
            "neg_text_score": None,
            "neg_final_score": None,
            "resolved_candidate_scores": resolved_candidate_scores,
            "is_satisfying_mask": is_satisfying_mask,
            "resolved_candidates": pos_resolved_candidates,
            "L_AAN": aan_loss_value,
        }

        if doc_neg_batch is not None:
            doc_neg_enc = self.encode_document(
                input_ids=doc_neg_batch["input_ids"],
                attention_mask=doc_neg_batch["attention_mask"],
                quantity_spans=doc_neg_batch.get("quantity_spans", []),
            )
            doc_neg_quantities = self._coerce_spans(doc_neg_batch.get("quantity_spans", []))
            neg_quantity_score, neg_resolved_scores, neg_resolved_candidates = self.compute_quantity_score_arith(
                query_enc=query_enc,
                doc_quantities=doc_neg_quantities,
                doc_enc=doc_neg_enc,
                use_aan=use_aan,
            )
            neg_text_score = self.compute_text_score(query_enc, doc_neg_enc)
            neg_final_score = (1.0 - alpha) * neg_quantity_score + alpha * neg_text_score
            neg_aan_loss = self._compute_optional_aan_loss(doc_neg_batch, doc_neg_enc)

            if neg_resolved_scores:
                neg_scores_tensor = torch.stack(neg_resolved_scores)
                result["resolved_candidate_scores"] = (
                    torch.cat([resolved_candidate_scores, neg_scores_tensor], dim=0)
                    if resolved_candidate_scores.numel() > 0
                    else neg_scores_tensor
                )
                result["is_satisfying_mask"] = torch.cat(
                    [
                        is_satisfying_mask,
                        torch.ones_like(neg_scores_tensor, dtype=torch.bool),
                    ],
                    dim=0,
                )
                result["resolved_candidates"] = pos_resolved_candidates + neg_resolved_candidates

            result.update(
                {
                    "doc_neg_enc": doc_neg_enc,
                    "neg_quantity_score": neg_quantity_score,
                    "neg_text_score": neg_text_score,
                    "neg_final_score": neg_final_score,
                    "L_AAN": result["L_AAN"] + neg_aan_loss,
                }
            )

        return result

    def two_stage_retrieve(self, query: dict, corpus_index: Any, top_k: int = 100) -> list[dict]:
        """Run DeepQuant retrieval first, then arithmetic-aware re-ranking over the top-k candidates.

        Expected corpus_index interface:
        - retrieve(query, top_k) -> list[dict]
        Each returned candidate dict should contain:
        - `doc_batch`
        - `doc_quantities`
        - optionally `score`
        """
        initial_candidates = corpus_index.retrieve(query, top_k=top_k)
        query_enc = self.encode_query(
            input_ids=query["input_ids"],
            attention_mask=query["attention_mask"],
            quantity_spans=query.get("quantity_spans", []),
        )
        reranked: list[dict] = []

        for candidate in initial_candidates:
            doc_batch = candidate["doc_batch"]
            doc_quantities = self._coerce_spans(candidate.get("doc_quantities", doc_batch.get("quantity_spans", [])))
            doc_enc = self.encode_document(
                input_ids=doc_batch["input_ids"],
                attention_mask=doc_batch["attention_mask"],
                quantity_spans=doc_batch.get("quantity_spans", []),
            )
            quantity_score, _, _ = self.compute_quantity_score_arith(
                query_enc=query_enc,
                doc_quantities=doc_quantities,
                doc_enc=doc_enc,
                use_aan=True,
            )
            text_score = self.compute_text_score(query_enc, doc_enc)
            alpha = self.compute_alpha(query_enc)
            final_score = (1.0 - alpha) * quantity_score + alpha * text_score
            updated_candidate = dict(candidate)
            updated_candidate["rerank_score"] = float(final_score.detach().cpu())
            reranked.append(updated_candidate)

        reranked.sort(key=lambda item: float(item.get("rerank_score", item.get("score", 0.0))), reverse=True)
        return reranked[:top_k]


if __name__ == "__main__":
    cfg = {
        "model": {
            "encoder": "bert-base-uncased",
            "hidden_dim": 768,
            "J_m": 691,
            "J_e": 77,
            "num_exponent_classes": 41,
        }
    }
    model = WideQuant(cfg)
    print(f"WideQuant initialized, num_token_id={model.num_token_id}")
