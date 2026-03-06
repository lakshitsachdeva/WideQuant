"""DeepQuant baseline model with quantity-aware and text-aware scoring."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel, BertConfig

from src.encoding.cqe_wrapper import QuantitySpan, setup_tokenizer
from src.encoding.quantity_encoder import ExponentEmbedding, QuantityInjector
from src.models.scoring_networks import (
    ComparatorPairScorer,
    ComparatorPredictor,
    UnitCompatibilityScorer,
)


class DeepQuant(nn.Module):
    """DeepQuant model with quantity comparison and ColBERT-style text scoring."""

    def __init__(self, config: dict) -> None:
        """Initialize BERT backbone and all DeepQuant scoring components."""
        super().__init__()
        self.config = config

        model_cfg = config.get("model", {})
        encoder_name = str(model_cfg.get("encoder", "bert-base-uncased"))
        local_files_only = bool(model_cfg.get("local_files_only", True))
        self.hidden_dim = int(model_cfg.get("hidden_dim", 768))
        self.J_m = int(model_cfg.get("J_m", 691))
        self.J_e = int(model_cfg.get("J_e", 77))
        self.num_exponent_classes = int(model_cfg.get("num_exponent_classes", 41))

        try:
            self.bert = AutoModel.from_pretrained(
                encoder_name,
                local_files_only=local_files_only,
            )
        except Exception:
            try:
                bert_cfg = AutoConfig.from_pretrained(
                    encoder_name,
                    local_files_only=local_files_only,
                )
            except Exception:
                bert_cfg = BertConfig(hidden_size=self.hidden_dim)
            self.bert = AutoModel.from_config(bert_cfg)

        self.tokenizer, self.num_token_id = setup_tokenizer(
            encoder_name=encoder_name,
            local_files_only=local_files_only,
            model=self.bert,
        )
        unk_id = int(self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 100)
        print(f"[DeepQuant] num_token_id={self.num_token_id}, unk_token_id={unk_id}")
        if self.num_token_id == unk_id:
            raise RuntimeError("[num] token id equals [UNK] id; tokenizer special token setup failed.")

        self.quantity_injector = QuantityInjector(
            base_bert_embeddings=self.bert.get_input_embeddings(),
            num_token_id=self.num_token_id,
            J=self.hidden_dim,
            J_m=self.J_m,
            J_e=self.J_e,
        )
        self.exponent_embedding = ExponentEmbedding(
            num_classes=self.num_exponent_classes,
            J_e=self.J_e,
        )
        self.comparator_predictor = ComparatorPredictor()
        self.unit_compatibility_scorer = UnitCompatibilityScorer()
        self.comparator_pair_scorers = nn.ModuleDict(
            {
                "lt": ComparatorPairScorer("lt"),
                "eq": ComparatorPairScorer("eq"),
                "gt": ComparatorPairScorer("gt"),
            }
        )
        self.alpha_weights = nn.Parameter(torch.zeros(self.hidden_dim, dtype=torch.float32))
        self.text_projection = nn.Linear(self.hidden_dim, 128, bias=False)

    def _ensure_2d(self, tensor: Tensor) -> Tensor:
        """Ensure input tensor is 2D (batch, seq_len)."""
        if tensor.ndim == 1:
            return tensor.unsqueeze(0)
        if tensor.ndim == 2:
            return tensor
        raise ValueError("Expected tensor with shape (seq_len,) or (batch, seq_len).")

    def _coerce_spans(self, quantity_spans: Any) -> List[QuantitySpan]:
        """Normalize raw quantity span payloads into QuantitySpan list."""
        if quantity_spans is None:
            return []

        # Support [[...]] for batch-size-1 payloads.
        if isinstance(quantity_spans, Sequence) and quantity_spans and isinstance(quantity_spans[0], list):
            if len(quantity_spans) != 1:
                raise ValueError("Batch size > 1 quantity span payloads are not supported in this implementation.")
            quantity_spans = quantity_spans[0]

        spans: list[QuantitySpan] = []
        for item in quantity_spans:
            if isinstance(item, QuantitySpan):
                spans.append(item)
            elif isinstance(item, Mapping):
                spans.append(QuantitySpan(**item))
            else:
                raise TypeError(f"Unsupported quantity span type: {type(item)!r}")
        return spans

    @staticmethod
    def _span_key(span: QuantitySpan, idx: int) -> tuple[Any, ...]:
        """Create a stable dictionary key for quantity span outputs."""
        return (
            idx,
            span.text,
            span.start_char,
            span.end_char,
            span.unit,
            span.concept,
        )

    def _encode(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        quantity_spans: Any,
    ) -> dict:
        """Shared encoder path used by query/document encoding."""
        input_ids = self._ensure_2d(input_ids).to(dtype=torch.long, device=self.alpha_weights.device)
        attention_mask = self._ensure_2d(attention_mask).to(dtype=torch.long, device=self.alpha_weights.device)
        if input_ids.shape[0] != 1:
            raise ValueError("encode_query/encode_document currently require batch size 1.")

        spans = self._coerce_spans(quantity_spans)
        num_positions = (input_ids[0] == self.num_token_id).nonzero(as_tuple=False).flatten()
        if not spans and int(num_positions.numel()) > 0:
            spans = [
                QuantitySpan(
                    text="[num]",
                    mantissa=0.0,
                    exponent=0,
                    unit="",
                    concept="",
                    start_char=0,
                    end_char=0,
                )
                for _ in range(int(num_positions.numel()))
            ]
        modified_embeddings = self.quantity_injector(input_ids=input_ids, quantity_spans=spans)

        outputs = self.bert(
            inputs_embeds=modified_embeddings,
            attention_mask=attention_mask,
            return_dict=True,
        )

        token_embeddings = outputs.last_hidden_state[0].to(dtype=torch.float32)  # (seq_len, 768)
        cls_embedding = token_embeddings[0]  # (768,)

        if len(spans) != int(num_positions.numel()):
            raise ValueError(
                f"Found {int(num_positions.numel())} [num] tokens but received {len(spans)} quantity spans."
            )

        quantity_outputs: dict[tuple[Any, ...], Tensor] = {}
        for idx, (span, pos) in enumerate(zip(spans, num_positions)):
            key = self._span_key(span, idx)
            quantity_outputs[key] = token_embeddings[int(pos.item())]

        return {
            "cls": cls_embedding,
            "token_embeddings": token_embeddings,
            "attention_mask": attention_mask[0],
            "quantity_outputs": quantity_outputs,
        }

    def encode_query(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        quantity_spans: Any,
    ) -> dict:
        """Encode query sequence with quantity-aware [num] replacement."""
        return self._encode(input_ids=input_ids, attention_mask=attention_mask, quantity_spans=quantity_spans)

    def encode_document(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        quantity_spans: Any,
    ) -> dict:
        """Encode document sequence with quantity-aware [num] replacement."""
        return self._encode(input_ids=input_ids, attention_mask=attention_mask, quantity_spans=quantity_spans)

    def compute_quantity_score(self, query_enc: dict, doc_enc: dict) -> Tensor:
        """Compute DeepQuant quantity comparison score for one query-document pair."""
        query_quantities = list(query_enc["quantity_outputs"].values())
        doc_quantities = list(doc_enc["quantity_outputs"].values())

        device = query_enc["cls"].device
        if not query_quantities or not doc_quantities:
            return torch.tensor(0.0, device=device, dtype=torch.float32)

        total_score = torch.tensor(0.0, device=device, dtype=torch.float32)
        y_b_set = torch.stack(doc_quantities, dim=0)

        for y_a in query_quantities:
            unit_compat = self.unit_compatibility_scorer(y_a, y_b_set)  # (N_docs,)
            p_op = self.comparator_predictor(y_a.unsqueeze(0)).squeeze(0)  # (3,)

            candidate_scores: list[Tensor] = []
            for b_idx, y_b in enumerate(doc_quantities):
                n_lt = self.comparator_pair_scorers["lt"](y_a, y_b)
                n_eq = self.comparator_pair_scorers["eq"](y_a, y_b)
                n_gt = self.comparator_pair_scorers["gt"](y_a, y_b)
                op_score = p_op[0] * n_lt + p_op[1] * n_eq + p_op[2] * n_gt
                score_b = unit_compat[b_idx] * op_score
                candidate_scores.append(score_b)

            best_score = torch.max(torch.stack(candidate_scores))
            total_score = total_score + best_score

        return total_score

    def compute_text_score(self, query_enc: dict, doc_enc: dict) -> Tensor:
        """Compute ColBERT MaxSim text score over projected token embeddings."""
        q_tokens = query_enc["token_embeddings"]  # (Lq, 768)
        d_tokens = doc_enc["token_embeddings"]  # (Ld, 768)
        q_mask = query_enc["attention_mask"].to(dtype=torch.bool)  # (Lq,)
        d_mask = doc_enc["attention_mask"].to(dtype=torch.bool)  # (Ld,)

        q_proj = self.text_projection(q_tokens)
        d_proj = self.text_projection(d_tokens)
        q_norm = F.normalize(q_proj, p=2, dim=-1)
        d_norm = F.normalize(d_proj, p=2, dim=-1)

        sim = torch.matmul(q_norm, d_norm.T)  # (Lq, Ld)
        sim = sim.masked_fill(~d_mask.unsqueeze(0), float("-inf"))
        max_per_query_token = sim.max(dim=1).values
        max_per_query_token = torch.where(
            torch.isfinite(max_per_query_token),
            max_per_query_token,
            torch.zeros_like(max_per_query_token),
        )

        return (max_per_query_token * q_mask.to(dtype=max_per_query_token.dtype)).sum()

    def compute_alpha(self, query_enc: dict) -> Tensor:
        """Compute alpha = sigmoid(N_alpha · y_CLS)."""
        y_cls = query_enc["cls"]
        alpha_logit = torch.dot(self.alpha_weights, y_cls)
        return torch.sigmoid(alpha_logit)

    def forward(
        self,
        query_batch: dict,
        doc_pos_batch: dict,
        doc_neg_batch: dict | None = None,
    ) -> dict:
        """Run DeepQuant forward and return intermediate scores for loss computation."""
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

        pos_quantity_score = self.compute_quantity_score(query_enc, doc_pos_enc)
        pos_text_score = self.compute_text_score(query_enc, doc_pos_enc)
        alpha = self.compute_alpha(query_enc)
        pos_final_score = (1.0 - alpha) * pos_quantity_score + alpha * pos_text_score

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
        }

        if doc_neg_batch is not None:
            doc_neg_enc = self.encode_document(
                input_ids=doc_neg_batch["input_ids"],
                attention_mask=doc_neg_batch["attention_mask"],
                quantity_spans=doc_neg_batch.get("quantity_spans", []),
            )
            neg_quantity_score = self.compute_quantity_score(query_enc, doc_neg_enc)
            neg_text_score = self.compute_text_score(query_enc, doc_neg_enc)
            neg_final_score = (1.0 - alpha) * neg_quantity_score + alpha * neg_text_score

            result.update(
                {
                    "doc_neg_enc": doc_neg_enc,
                    "neg_quantity_score": neg_quantity_score,
                    "neg_text_score": neg_text_score,
                    "neg_final_score": neg_final_score,
                }
            )

        return result


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
    model = DeepQuant(cfg)
    print(f"DeepQuant initialized, num_token_id={model.num_token_id}")
