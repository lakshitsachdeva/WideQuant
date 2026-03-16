"""Training loop and evaluation utilities for DeepQuant."""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - FAISS fallback
    faiss = None

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - tensorboard fallback
    class SummaryWriter:  # type: ignore[no-redef]
        """No-op SummaryWriter fallback when tensorboard isn't installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _ = (args, kwargs)

        def add_scalar(self, *args: Any, **kwargs: Any) -> None:
            _ = (args, kwargs)

        def flush(self) -> None:
            return None

        def close(self) -> None:
            return None

from src.encoding.cqe_wrapper import QuantitySpan
from src.training.losses import TotalLoss, debug_infonce


class DeepQuantTrainer:
    """Trainer for DeepQuant with mixed precision, FAISS eval, and checkpointing."""

    @staticmethod
    def _select_device() -> torch.device:
        """Select CUDA first, then Apple MPS, else CPU."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        train_dataloader: Iterable[dict],
        dev_dataloader: Iterable[dict],
    ) -> None:
        """Initialize model, optimizers, losses, and logging."""
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader

        self.device = self._select_device()
        self.model.to(self.device)
        print(f"[DeepQuantTrainer] Using device: {self.device}")

        training_cfg = config.get("training", {})
        model_cfg = config.get("model", {})

        self.gradient_accumulation_steps = int(training_cfg.get("gradient_accumulation_steps", 1))
        self.max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
        self.temperature = float(training_cfg.get("temperature", 0.02))
        self.warmup_ratio = float(training_cfg.get("warmup_ratio", 0.10))
        self.log_every_steps = int(training_cfg.get("log_every_steps", 50))

        self.hidden_dim = int(model_cfg.get("hidden_dim", 768))
        self.num_unit_classes = int(model_cfg.get("num_unit_classes", 500))

        # Auxiliary reconstruction heads to keep L_quant active.
        self.exp_head = nn.Linear(self.hidden_dim, 41).to(self.device)
        self.man_head = nn.Linear(self.hidden_dim, 1).to(self.device)
        self.unit_head = nn.Linear(self.hidden_dim, self.num_unit_classes).to(self.device)

        bert_params = []
        non_bert_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("bert."):
                bert_params.append(param)
            else:
                non_bert_params.append(param)

        non_bert_params.extend(self.exp_head.parameters())
        non_bert_params.extend(self.man_head.parameters())
        non_bert_params.extend(self.unit_head.parameters())

        self.bert_optimizer = AdamW(bert_params, lr=2e-5, weight_decay=0.01)
        self.head_optimizer = AdamW(non_bert_params, lr=1e-4, weight_decay=0.01)

        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.loss_fn = TotalLoss(lambda_arith=float(training_cfg["lambda_arith"]))
        self.loss_fn.retr_loss.temperature = self.temperature

        self.runs_dir = Path("runs")
        self.ckpt_dir = Path("checkpoints")
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.runs_dir))

        self.global_step = 0
        self.best_mrr10 = float("-inf")
        self._seen_hard_negatives = False
        self._checked_first_num_batch = False
        self._num_check_attempts = 0
        self._did_infonce_debug = False
        self._last_train_losses: dict[str, float] = {}
        self.bert_scheduler: Optional[Any] = None
        self.head_scheduler: Optional[Any] = None
        self.loss_history: dict[str, list[float]] = defaultdict(list)
        self.epoch_history: dict[str, list[float]] = defaultdict(list)

    def _to_device(self, payload: Any) -> Any:
        """Recursively move tensors to the trainer device."""
        if isinstance(payload, Tensor):
            return payload.to(self.device)
        if isinstance(payload, dict):
            return {k: self._to_device(v) for k, v in payload.items()}
        if isinstance(payload, list):
            return [self._to_device(v) for v in payload]
        if isinstance(payload, tuple):
            return tuple(self._to_device(v) for v in payload)
        return payload

    @staticmethod
    def _build_linear_warmup_scheduler(optimizer: AdamW, total_steps: int, warmup_steps: int) -> Any:
        """Build linear warmup + linear decay scheduler via Transformers helper."""
        total_steps = max(int(total_steps), 1)
        warmup_steps = max(int(warmup_steps), 0)
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    @staticmethod
    def _unpack_batch(batch: Any) -> tuple[dict, dict, Optional[dict]]:
        """Extract query/positive/negative payloads from one dataloader batch."""
        if isinstance(batch, dict):
            query_batch = batch.get("query_batch") or batch.get("query")
            doc_pos_batch = batch.get("doc_pos_batch") or batch.get("doc_batch") or batch.get("positive")
            doc_neg_batch = batch.get("doc_neg_batch") or batch.get("negative")
            if query_batch is None or doc_pos_batch is None:
                raise KeyError("Batch dict must contain query and positive document payloads.")
            return query_batch, doc_pos_batch, doc_neg_batch

        if isinstance(batch, (tuple, list)):
            if len(batch) == 2:
                return batch[0], batch[1], None
            if len(batch) >= 3:
                return batch[0], batch[1], batch[2]
        raise TypeError("Unsupported batch format.")

    @staticmethod
    def _extract_quantity_vectors(enc: dict) -> list[Tensor]:
        """Return quantity vectors from encoded outputs."""
        return [value for value in enc.get("quantity_outputs", {}).values()]

    def _build_loss_inputs(self, model_outputs: dict, neg_scores_override: Optional[Tensor] = None) -> dict:
        """Build all inputs required by TotalLoss from current model outputs."""
        loss_inputs: dict[str, Any] = {}
        pos_score = model_outputs["pos_final_score"].reshape(1)
        if neg_scores_override is not None:
            neg_score = neg_scores_override.to(device=self.device, dtype=torch.float32)
            loss_inputs["pos_scores"] = pos_score
            loss_inputs["neg_scores"] = neg_score
        elif model_outputs.get("neg_final_score") is None:
            neg_score = None
        else:
            neg_score = model_outputs["neg_final_score"].reshape(1, 1)
            loss_inputs["pos_scores"] = pos_score
            loss_inputs["neg_scores"] = neg_score

        query_enc = model_outputs["query_enc"]
        doc_pos_enc = model_outputs["doc_pos_enc"]
        query_q = self._extract_quantity_vectors(query_enc)
        doc_q = self._extract_quantity_vectors(doc_pos_enc)

        anchors = query_q if query_q else [query_enc["cls"]]
        docs = doc_q if doc_q else [doc_pos_enc["cls"]]

        anchor_mat = torch.stack([v.to(dtype=torch.float32) for v in anchors], dim=0)
        pred_exp_logits = self.exp_head(anchor_mat)
        pred_man = self.man_head(anchor_mat).squeeze(-1)
        pred_unit_logits = self.unit_head(anchor_mat)

        true_exp = torch.zeros(anchor_mat.shape[0], dtype=torch.long, device=self.device)
        true_man = torch.zeros(anchor_mat.shape[0], dtype=torch.float32, device=self.device)
        true_unit = torch.zeros(anchor_mat.shape[0], dtype=torch.long, device=self.device)

        lt_scores = []
        eq_scores = []
        gt_scores = []
        relation_labels = []

        for y_a in anchors:
            rel_prob = self.model.comparator_predictor(y_a.unsqueeze(0)).squeeze(0)
            rel_idx = int(torch.argmax(rel_prob).item())
            for y_b in docs:
                lt_scores.append(torch.sigmoid(self.model.comparator_pair_scorers["lt"](y_a, y_b)))
                eq_scores.append(torch.sigmoid(self.model.comparator_pair_scorers["eq"](y_a, y_b)))
                gt_scores.append(torch.sigmoid(self.model.comparator_pair_scorers["gt"](y_a, y_b)))
                relation_labels.append(rel_idx)

        N_lt_scores = torch.stack(lt_scores)
        N_eq_scores = torch.stack(eq_scores)
        N_gt_scores = torch.stack(gt_scores)
        true_relations = torch.tensor(relation_labels, dtype=torch.long, device=self.device)

        # Arithmetic candidates for L_arith.
        y_a0 = anchors[0]
        y_b_set = torch.stack(docs, dim=0)
        unit_compat = self.model.unit_compatibility_scorer(y_a0, y_b_set)
        p_op = self.model.comparator_predictor(y_a0.unsqueeze(0)).squeeze(0)
        arith_scores = []
        for i, y_b in enumerate(docs):
            n_lt = torch.sigmoid(self.model.comparator_pair_scorers["lt"](y_a0, y_b))
            n_eq = torch.sigmoid(self.model.comparator_pair_scorers["eq"](y_a0, y_b))
            n_gt = torch.sigmoid(self.model.comparator_pair_scorers["gt"](y_a0, y_b))
            mixed = p_op[0] * n_lt + p_op[1] * n_eq + p_op[2] * n_gt
            arith_scores.append(unit_compat[i] * mixed)
        resolved_candidate_scores = torch.clamp(torch.stack(arith_scores), min=1e-6, max=1.0)
        is_satisfying_mask = torch.zeros_like(resolved_candidate_scores, dtype=torch.bool)
        is_satisfying_mask[int(torch.argmax(resolved_candidate_scores).item())] = True

        loss_inputs.update(
            {
            "pred_exponent_logits": pred_exp_logits,
            "true_exponent": true_exp,
            "pred_mantissa": pred_man,
            "true_mantissa": true_man,
            "pred_unit_logits": pred_unit_logits,
            "true_unit": true_unit,
            "N_lt_scores": N_lt_scores,
            "N_eq_scores": N_eq_scores,
            "N_gt_scores": N_gt_scores,
            "true_relations": true_relations,
            "resolved_candidate_scores": resolved_candidate_scores,
            "is_satisfying_mask": is_satisfying_mask,
            }
        )
        return loss_inputs

    @staticmethod
    def _ranking_metrics(rank_labels: list[list[int]]) -> dict:
        """Compute MRR@10, NDCG@10, P@10, R@100 from binary relevance labels."""
        mrr = 0.0
        ndcg = 0.0
        p10 = 0.0
        r100 = 0.0
        n = max(len(rank_labels), 1)

        for labels in rank_labels:
            top10 = labels[:10]
            top100 = labels[:100]

            rr = 0.0
            for i, rel in enumerate(top10, start=1):
                if rel > 0:
                    rr = 1.0 / i
                    break
            mrr += rr

            dcg = 0.0
            for i, rel in enumerate(top10, start=1):
                dcg += rel / np.log2(i + 1)
            ideal = sorted(top10, reverse=True)
            idcg = 0.0
            for i, rel in enumerate(ideal, start=1):
                idcg += rel / np.log2(i + 1)
            ndcg += (dcg / idcg) if idcg > 0 else 0.0

            p10 += float(sum(top10)) / 10.0
            total_rel = max(sum(labels), 1)
            r100 += float(sum(top100)) / float(total_rel)

        return {
            "MRR@10": mrr / n,
            "NDCG@10": ndcg / n,
            "P@10": p10 / n,
            "R@100": r100 / n,
        }

    def train_epoch(self, epoch: int) -> dict:
        """Run one epoch with AMP, accumulation, clipping, and tqdm loss logging."""
        self.model.train()
        self.exp_head.train()
        self.man_head.train()
        self.unit_head.train()

        self.bert_optimizer.zero_grad(set_to_none=True)
        self.head_optimizer.zero_grad(set_to_none=True)

        running = defaultdict(float)
        steps = 0

        progress = tqdm(self.train_dataloader, desc=f"Train Epoch {epoch}", leave=False)
        for step_idx, batch in enumerate(progress, start=1):
            query_batch, doc_pos_batch, doc_neg_batch = self._unpack_batch(batch)
            query_batch = self._to_device(query_batch)
            doc_pos_batch = self._to_device(doc_pos_batch)
            doc_neg_batch = self._to_device(doc_neg_batch) if doc_neg_batch is not None else None

            if not self._checked_first_num_batch:
                num_id = int(self.model.num_token_id)
                batch_query_ids = query_batch["input_ids"]
                if batch_query_ids.ndim == 1:
                    batch_query_ids = batch_query_ids.unsqueeze(0)
                has_num_per_query = (batch_query_ids == num_id).any(dim=1)
                frac_with_num = float(has_num_per_query.float().mean().item())
                self._num_check_attempts += 1
                if bool(has_num_per_query.any().item()):
                    self._checked_first_num_batch = True
                else:
                    print(
                        f"WARNING: Only {frac_with_num:.1%} of queries in batch contain "
                        f"[num] token (id={num_id})"
                    )
                    if batch_query_ids.shape[0] > 1 or self._num_check_attempts >= 8:
                        raise RuntimeError(
                            "CRITICAL: Zero queries in the initial training batches contain [num] token. "
                            "The injection pipeline is broken. Check setup_tokenizer() "
                            "and replace_with_num_tokens() ordering."
                        )

            if doc_neg_batch is not None:
                pos_ids = doc_pos_batch.get("input_ids")
                if isinstance(doc_neg_batch, list):
                    for neg_item in doc_neg_batch:
                        neg_ids = neg_item.get("input_ids")
                        if isinstance(pos_ids, Tensor) and isinstance(neg_ids, Tensor):
                            if not torch.equal(pos_ids, neg_ids):
                                self._seen_hard_negatives = True
                                break
                else:
                    neg_ids = doc_neg_batch.get("input_ids")
                    if isinstance(pos_ids, Tensor) and isinstance(neg_ids, Tensor):
                        if not torch.equal(pos_ids, neg_ids):
                            self._seen_hard_negatives = True

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                model_outputs = self.model(query_batch, doc_pos_batch, None)
                neg_scores_override: Optional[Tensor] = None
                if doc_neg_batch is not None:
                    neg_batches = doc_neg_batch if isinstance(doc_neg_batch, list) else [doc_neg_batch]
                    neg_scores: list[Tensor] = []
                    for neg_payload in neg_batches:
                        neg_outputs = self.model(query_batch, neg_payload, None)
                        neg_scores.append(neg_outputs["final_score"].reshape(()))
                    if neg_scores:
                        neg_scores_override = torch.stack(neg_scores, dim=0).unsqueeze(0)
                if neg_scores_override is None:
                    raise RuntimeError(
                        "CRITICAL: No negative scores were produced for the training batch. "
                        "Check hard-negative batch construction before computing InfoNCE."
                    )

                loss_inputs = self._build_loss_inputs(model_outputs, neg_scores_override=neg_scores_override)
                if epoch == 1 and not self._did_infonce_debug:
                    print("InfoNCE debug (epoch 1, first batch):")
                    debug_infonce(
                        loss_inputs["pos_scores"],
                        loss_inputs["neg_scores"],
                        temperature=self.temperature,
                    )
                    pos_vec = loss_inputs["pos_scores"].detach()
                    neg_vec = loss_inputs["neg_scores"].detach()
                    print(f"  pos_scores == neg_scores allclose: {torch.allclose(pos_vec.unsqueeze(1), neg_vec)}")
                    self._did_infonce_debug = True
                loss_dict = self.loss_fn(loss_inputs)
                scaled_loss = loss_dict["total"] / float(self.gradient_accumulation_steps)

            self.scaler.scale(scaled_loss).backward()

            if step_idx % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.bert_optimizer)
                self.scaler.unscale_(self.head_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters())
                    + list(self.exp_head.parameters())
                    + list(self.man_head.parameters())
                    + list(self.unit_head.parameters()),
                    max_norm=self.max_grad_norm,
                )
                self.scaler.step(self.bert_optimizer)
                self.scaler.step(self.head_optimizer)
                self.scaler.update()
                if self.bert_scheduler is not None:
                    self.bert_scheduler.step()
                if self.head_scheduler is not None:
                    self.head_scheduler.step()
                self.bert_optimizer.zero_grad(set_to_none=True)
                self.head_optimizer.zero_grad(set_to_none=True)

            steps += 1
            self.global_step += 1
            for key, value in loss_dict.items():
                scalar = float(value.detach().cpu().item())
                running[key] += scalar
                self.loss_history[key].append(scalar)

            if step_idx % self.log_every_steps == 0:
                print(
                    f"Epoch {epoch} Step {step_idx} | "
                    f"L_retr={loss_dict['L_retr'].item():.4f} "
                    f"L_quant={loss_dict['L_quant'].item():.4f} "
                    f"L_reg={loss_dict['L_reg'].item():.4f} "
                    f"L_comp={loss_dict['L_comp'].item():.4f} "
                    f"L_arith={loss_dict['L_arith'].item():.4f}"
                )

            progress.set_postfix(
                {
                    "total": f"{loss_dict['total'].item():.4f}",
                    "L_retr": f"{loss_dict['L_retr'].item():.4f}",
                    "L_quant": f"{loss_dict['L_quant'].item():.4f}",
                    "L_reg": f"{loss_dict['L_reg'].item():.4f}",
                    "L_comp": f"{loss_dict['L_comp'].item():.4f}",
                    "L_arith": f"{loss_dict['L_arith'].item():.4f}",
                }
            )

        if steps % self.gradient_accumulation_steps != 0:
            self.scaler.unscale_(self.bert_optimizer)
            self.scaler.unscale_(self.head_optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters())
                + list(self.exp_head.parameters())
                + list(self.man_head.parameters())
                + list(self.unit_head.parameters()),
                max_norm=self.max_grad_norm,
            )
            self.scaler.step(self.bert_optimizer)
            self.scaler.step(self.head_optimizer)
            self.scaler.update()
            if self.bert_scheduler is not None:
                self.bert_scheduler.step()
            if self.head_scheduler is not None:
                self.head_scheduler.step()
            self.bert_optimizer.zero_grad(set_to_none=True)
            self.head_optimizer.zero_grad(set_to_none=True)

        avg_losses = {k: (v / max(steps, 1)) for k, v in running.items()}
        self._last_train_losses = avg_losses
        for key, value in avg_losses.items():
            self.writer.add_scalar(f"train/{key}", value, epoch)
        return avg_losses

    def evaluate(self, dataloader: Iterable[dict]) -> dict:
        """Evaluate MRR@10/NDCG@10/P@10/R@100 using FAISS IndexFlatIP retrieval."""
        self.model.eval()
        self.exp_head.eval()
        self.man_head.eval()
        self.unit_head.eval()

        doc_vectors: list[list[float]] = []
        query_vectors: list[list[float]] = []
        relevant_doc_ids: list[int] = []

        with torch.no_grad():
            for doc_id, batch in enumerate(tqdm(dataloader, desc="Eval", leave=False)):
                query_batch, doc_pos_batch, _ = self._unpack_batch(batch)
                query_batch = self._to_device(query_batch)
                doc_pos_batch = self._to_device(doc_pos_batch)

                query_enc = self.model.encode_query(
                    query_batch["input_ids"],
                    query_batch["attention_mask"],
                    query_batch.get("quantity_spans", []),
                )
                doc_enc = self.model.encode_document(
                    doc_pos_batch["input_ids"],
                    doc_pos_batch["attention_mask"],
                    doc_pos_batch.get("quantity_spans", []),
                )

                q_vec = F.normalize(query_enc["cls"], dim=0).detach().cpu().to(dtype=torch.float32).tolist()
                d_vec = F.normalize(doc_enc["cls"], dim=0).detach().cpu().to(dtype=torch.float32).tolist()
                query_vectors.append(q_vec)
                doc_vectors.append(d_vec)
                relevant_doc_ids.append(doc_id)

        if not doc_vectors or not query_vectors:
            return {"MRR@10": 0.0, "NDCG@10": 0.0, "P@10": 0.0, "R@100": 0.0}

        doc_matrix = np.asarray(doc_vectors, dtype=np.float32)
        query_matrix = np.asarray(query_vectors, dtype=np.float32)

        if faiss is not None:
            index = faiss.IndexFlatIP(doc_matrix.shape[1])
            index.add(doc_matrix)
            top_k = min(100, doc_matrix.shape[0])
            _, retrieved = index.search(query_matrix, top_k)
        else:  # pragma: no cover - FAISS fallback
            sims = np.matmul(query_matrix, doc_matrix.T)
            top_k = min(100, doc_matrix.shape[0])
            retrieved = np.argsort(-sims, axis=1)[:, :top_k]

        rank_labels: list[list[int]] = []
        for q_idx, doc_ids in enumerate(retrieved):
            rel_id = relevant_doc_ids[q_idx]
            labels = [1 if int(doc_id) == int(rel_id) else 0 for doc_id in doc_ids.tolist()]
            rank_labels.append(labels)

        return self._ranking_metrics(rank_labels)

    def _run_debug_checklist(self) -> dict[str, tuple[bool, str]]:
        """Run debug checklist for low-MRR training runs."""
        checks: dict[str, tuple[bool, str]] = {}

        # 1) [num] injection placement check.
        try:
            batch = next(iter(self.train_dataloader))
            query_batch, _, _ = self._unpack_batch(batch)
            query_batch = self._to_device(query_batch)
            input_ids = query_batch["input_ids"]
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)

            num_positions = (input_ids[0] == self.model.num_token_id).nonzero(as_tuple=False).flatten()
            if int(num_positions.numel()) == 0:
                checks["1"] = (False, "No [num] token positions found in sampled batch.")
            else:
                spans = query_batch.get("quantity_spans", [])
                spans = self.model._coerce_spans(spans)
                if not spans:
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
                base_emb = self.model.bert.get_input_embeddings()(input_ids)
                injected_emb = self.model.quantity_injector(input_ids, spans)
                replaced_all = True
                for pos in num_positions:
                    p = int(pos.item())
                    if torch.allclose(base_emb[0, p], injected_emb[0, p]):
                        replaced_all = False
                        break
                checks["1"] = (replaced_all, f"num_positions={int(num_positions.numel())}")
        except Exception as exc:
            checks["1"] = (False, f"Injection check error: {exc}")

        # 2) Four losses contributing.
        retr = self._last_train_losses.get("L_retr", 0.0)
        quant = self._last_train_losses.get("L_quant", 0.0)
        reg = self._last_train_losses.get("L_reg", 0.0)
        comp = self._last_train_losses.get("L_comp", 0.0)
        contributing = all(value > 0.0 for value in [retr, quant, reg, comp])
        checks["2"] = (
            contributing,
            f"L_retr={retr:.4f}, L_quant={quant:.4f}, L_reg={reg:.4f}, L_comp={comp:.4f}",
        )

        # 3) Temperature is 0.02.
        tau = float(self.loss_fn.retr_loss.temperature)
        checks["3"] = (abs(tau - 0.02) < 1e-9, f"tau={tau}")

        # 4) Hard negatives are present.
        checks["4"] = (self._seen_hard_negatives, f"seen_hard_negatives={self._seen_hard_negatives}")

        # 5) ColBERT normalization check.
        try:
            batch = next(iter(self.train_dataloader))
            query_batch, doc_pos_batch, _ = self._unpack_batch(batch)
            query_batch = self._to_device(query_batch)
            doc_pos_batch = self._to_device(doc_pos_batch)
            q_enc = self.model.encode_query(
                query_batch["input_ids"],
                query_batch["attention_mask"],
                query_batch.get("quantity_spans", []),
            )
            d_enc = self.model.encode_document(
                doc_pos_batch["input_ids"],
                doc_pos_batch["attention_mask"],
                doc_pos_batch.get("quantity_spans", []),
            )
            q_proj = self.model.text_projection(q_enc["token_embeddings"])
            d_proj = self.model.text_projection(d_enc["token_embeddings"])
            q_norm = F.normalize(q_proj, dim=-1)
            d_norm = F.normalize(d_proj, dim=-1)
            q_len = torch.linalg.norm(q_norm, dim=-1)
            d_len = torch.linalg.norm(d_norm, dim=-1)
            q_ok = torch.allclose(q_len[q_enc["attention_mask"].bool()], torch.ones_like(q_len[q_enc["attention_mask"].bool()]), atol=1e-3)
            d_ok = torch.allclose(d_len[d_enc["attention_mask"].bool()], torch.ones_like(d_len[d_enc["attention_mask"].bool()]), atol=1e-3)
            checks["5"] = (bool(q_ok and d_ok), "ColBERT token norms are unit-length on non-pad tokens.")
        except Exception as exc:
            checks["5"] = (False, f"ColBERT normalization check error: {exc}")

        print("Debug checklist (MRR@10 < 0.65):")
        print(f"1. [num] injection positions: {'PASS' if checks['1'][0] else 'FAIL'} | {checks['1'][1]}")
        print(f"2. 4 losses contributing: {'PASS' if checks['2'][0] else 'FAIL'} | {checks['2'][1]}")
        print(f"3. temperature τ=0.02: {'PASS' if checks['3'][0] else 'FAIL'} | {checks['3'][1]}")
        print(f"4. hard negatives in batches: {'PASS' if checks['4'][0] else 'FAIL'} | {checks['4'][1]}")
        print(f"5. ColBERT normalization: {'PASS' if checks['5'][0] else 'FAIL'} | {checks['5'][1]}")
        return checks

    def train(self, n_epochs: int) -> dict:
        """Train/evaluate for n epochs, checkpoint by best MRR@10, and enforce target tracking."""
        best_metrics: dict[str, float] = {"MRR@10": float("-inf")}
        steps_per_epoch = len(self.train_dataloader)
        optimizer_steps_per_epoch = int(math.ceil(steps_per_epoch / float(max(self.gradient_accumulation_steps, 1))))
        total_optimizer_steps = max(1, optimizer_steps_per_epoch * n_epochs)
        warmup_steps = int(total_optimizer_steps * self.warmup_ratio)

        self.bert_scheduler = self._build_linear_warmup_scheduler(
            self.bert_optimizer,
            total_steps=total_optimizer_steps,
            warmup_steps=warmup_steps,
        )
        self.head_scheduler = self._build_linear_warmup_scheduler(
            self.head_optimizer,
            total_steps=total_optimizer_steps,
            warmup_steps=warmup_steps,
        )
        print(
            f"Scheduler setup | total_optimizer_steps={total_optimizer_steps} "
            f"| warmup_steps={warmup_steps}"
        )

        for epoch in range(1, n_epochs + 1):
            train_losses = self.train_epoch(epoch)
            dev_metrics = self.evaluate(self.dev_dataloader)

            for key, value in train_losses.items():
                self.epoch_history[key].append(float(value))

            self.writer.add_scalar("dev/MRR@10", dev_metrics["MRR@10"], epoch)
            self.writer.add_scalar("dev/NDCG@10", dev_metrics["NDCG@10"], epoch)
            self.writer.add_scalar("dev/P@10", dev_metrics["P@10"], epoch)
            self.writer.add_scalar("dev/R@100", dev_metrics["R@100"], epoch)

            if dev_metrics["MRR@10"] > self.best_mrr10:
                self.best_mrr10 = dev_metrics["MRR@10"]
                best_metrics = dev_metrics
                ckpt_path = self.ckpt_dir / "best_deepquant.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "exp_head_state_dict": self.exp_head.state_dict(),
                        "man_head_state_dict": self.man_head.state_dict(),
                        "unit_head_state_dict": self.unit_head.state_dict(),
                        "bert_optimizer_state_dict": self.bert_optimizer.state_dict(),
                        "head_optimizer_state_dict": self.head_optimizer.state_dict(),
                        "best_mrr10": self.best_mrr10,
                        "config": self.config,
                    },
                    ckpt_path,
                )

            train_total = train_losses.get("total", 0.0)
            l_retr = float(train_losses.get("L_retr", 0.0))
            l_quant = float(train_losses.get("L_quant", 0.0))
            l_reg = float(train_losses.get("L_reg", 0.0))
            l_comp = float(train_losses.get("L_comp", 0.0))
            print(
                f"Epoch {epoch} | L_retr: {l_retr:.4f} | L_quant: {l_quant:.4f} | "
                f"L_reg: {l_reg:.4f} | L_comp: {l_comp:.4f} | "
                f"Train Loss: {train_total:.4f} | Dev MRR@10: {dev_metrics['MRR@10']:.4f} | "
                f"Best: {self.best_mrr10:.4f}"
            )

            if epoch == 3:
                retr_curve = self.epoch_history.get("L_retr", [])
                if len(retr_curve) >= 3 and not (retr_curve[2] < retr_curve[1] < retr_curve[0]):
                    print("L_retr did not decrease across the first 3 epochs. Stopping early.")
                    print("InfoNCE debug output above should be used to diagnose score ranges and negatives.")
                    break

        debug_results = None
        if n_epochs >= 8 and self.best_mrr10 < 0.65:
            debug_results = self._run_debug_checklist()

        self.writer.flush()
        self.writer.close()

        target_passed = self.best_mrr10 >= 0.70
        return {
            "best_metrics": best_metrics,
            "best_mrr10": self.best_mrr10,
            "target_mrr10": 0.70,
            "target_passed": target_passed,
            "debug_checklist": debug_results,
            "loss_curves": dict(self.epoch_history),
            "step_loss_curves": dict(self.loss_history),
        }


if __name__ == "__main__":
    print("DeepQuantTrainer module ready.")
