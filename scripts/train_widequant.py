"""Train WideQuant by warm-starting from a DeepQuant checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    class SummaryWriter:  # type: ignore[no-redef]
        """No-op writer fallback."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _ = (args, kwargs)

        def add_scalar(self, *args: Any, **kwargs: Any) -> None:
            _ = (args, kwargs)

        def flush(self) -> None:
            return None

        def close(self) -> None:
            return None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.openfoodfacts import build_dataset as build_openfoodfacts_dataset
from src.encoding.cqe_wrapper import CQEWrapper, QuantitySpan
from src.models.arith_aggregation import verify_aan_quality
from src.models.deepquant import DeepQuant
from src.models.widequant import WideQuant
from src.training.losses import TotalLoss
from src.training.trainer import DeepQuantTrainer
from scripts.verify_num_injection import run_verification

QUERY_SPAN_PATTERNS = {
    "atomic": [(re.compile(r"greater than (?P<value>\d+(?:\.\d+)?)g", flags=re.IGNORECASE), "g", "fat_g")],
    "typeA": [
        (re.compile(r"greater than (?P<value>\d+(?:\.\d+)?)\s*kcal", flags=re.IGNORECASE), "kcal", "energy_kcal")
    ],
    "typeC": [(re.compile(r"under (?P<value>\d+(?:\.\d+)?)\s*kcal", flags=re.IGNORECASE), "kcal", "energy_kcal")],
    "typeB": [
        (
            re.compile(r"exceeds (?P<value>\d+(?:\.\d+)?)%", flags=re.IGNORECASE),
            "%",
            "protein_share",
        )
    ],
    "mixed": [
        (re.compile(r"energy above (?P<value>\d+(?:\.\d+)?)\s*kcal", flags=re.IGNORECASE), "kcal", "energy_kcal"),
        (re.compile(r"protein above (?P<value>\d+(?:\.\d+)?)g", flags=re.IGNORECASE), "g", "protein_g"),
    ],
}
FIELD_PATTERNS = {
    "energy_kcal": re.compile(r"Energy:\s*(?P<value>\d+(?:\.\d+)?)\s*kcal", flags=re.IGNORECASE),
    "protein_g": re.compile(r"Protein:\s*(?P<value>\d+(?:\.\d+)?)g", flags=re.IGNORECASE),
    "fat_g": re.compile(r"Fat:\s*(?P<value>\d+(?:\.\d+)?)g", flags=re.IGNORECASE),
    "carbs_g": re.compile(r"Carbs:\s*(?P<value>\d+(?:\.\d+)?)g", flags=re.IGNORECASE),
    "energy_from_protein": re.compile(r"energy from protein:\s*(?P<value>\d+(?:\.\d+)?)\s*kcal", flags=re.IGNORECASE),
    "energy_from_fat": re.compile(r"energy from fat:\s*(?P<value>\d+(?:\.\d+)?)\s*kcal", flags=re.IGNORECASE),
    "energy_from_carbs": re.compile(r"energy from carbohydrates:\s*(?P<value>\d+(?:\.\d+)?)\s*kcal", flags=re.IGNORECASE),
    "protein_kcal": re.compile(r"providing\s*(?P<value>\d+(?:\.\d+)?)\s*kcal", flags=re.IGNORECASE),
    "total_kcal": re.compile(r"Total energy:\s*(?P<value>\d+(?:\.\d+)?)\s*kcal", flags=re.IGNORECASE),
    "energy_kj": re.compile(r"energy:\s*(?P<value>\d+(?:\.\d+)?)\s*kJ", flags=re.IGNORECASE),
}
DECOMP_TYPE_TO_ARITH = {"typeA": "TYPE_A", "typeB": "TYPE_B", "typeC": "TYPE_C"}


def parse_args() -> argparse.Namespace:
    """Parse training arguments."""
    parser = argparse.ArgumentParser(description="Train WideQuant from a DeepQuant checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_dir", type=str, default="data/openfoodfacts")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_deepquant.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_eval_examples", type=int, default=None)
    parser.add_argument("--build_if_missing", action="store_true")
    parser.add_argument("--build_n_products", type=int, default=10000)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set deterministic seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows from disk."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_qrels(path: Path) -> dict[str, list[str]]:
    """Parse TREC qrels into a query -> doc ids mapping."""
    qrels: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            query_id, _, doc_id, relevance = parts[:4]
            try:
                rel_value = float(relevance)
            except ValueError:
                continue
            if rel_value > 0:
                qrels[query_id].append(doc_id)
    return dict(qrels)


def _to_scientific(value: float) -> tuple[float, int]:
    """Convert a scalar value into mantissa/exponent form."""
    if abs(value) < 1e-12:
        return 0.0, 0
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = float(value / (10.0 ** exponent))
    return mantissa, exponent


def _span_to_json(span: QuantitySpan) -> dict[str, Any]:
    """Serialize QuantitySpan for model payloads."""
    return asdict(span)


def _replace_with_num_tokens(text: str, spans: list[QuantitySpan]) -> str:
    """Replace numeric mentions with [num] while preserving surrounding unit context."""
    replaced = CQEWrapper.replace_with_num_tokens(text, spans)
    if len(spans) != replaced.count("[num]"):
        raise ValueError(
            f"Mismatch between quantity spans ({len(spans)}) and [num] tokens ({replaced.count('[num]')})."
        )
    return replaced


def _span_from_match(text: str, match: re.Match[str], unit: str, concept: str) -> QuantitySpan:
    """Create one QuantitySpan from a regex match."""
    raw = match.group("value")
    value = float(raw)
    mantissa, exponent = _to_scientific(value)
    return QuantitySpan(
        text=raw,
        mantissa=mantissa,
        exponent=exponent,
        unit=unit,
        concept=concept,
        start_char=match.start("value"),
        end_char=match.end("value"),
    )


def _query_spans_openfoodfacts(query: dict[str, Any]) -> tuple[str, list[QuantitySpan]]:
    """Build quantity spans for one OpenFoodFacts query."""
    text = str(query["query_text"])
    query_type = str(query.get("query_type", ""))
    spans: list[QuantitySpan] = []
    specs = QUERY_SPAN_PATTERNS.get(query_type, [])
    if not specs:
        raise ValueError(f"Unsupported query_type={query_type} for query_id={query['query_id']}")

    for pattern, unit, concept in specs:
        match = pattern.search(text)
        if match is None:
            raise ValueError(f"Could not extract query number for query_id={query['query_id']} with pattern={pattern.pattern}")
        spans.append(_span_from_match(text, match, unit=unit, concept=concept))
    return _replace_with_num_tokens(text, spans), spans


def _doc_spans_openfoodfacts(doc: dict[str, Any]) -> tuple[str, list[QuantitySpan]]:
    """Build quantity spans for one OpenFoodFacts document."""
    text = str(doc["text"])
    doc_type = str(doc["doc_type"])
    if doc_type == "atomic":
        specs = [
            ("energy_kcal", "kcal", "energy_kcal"),
            ("protein_g", "g", "protein_g"),
            ("fat_g", "g", "fat_g"),
            ("carbs_g", "g", "carbs_g"),
        ]
    elif doc_type == "typeA":
        specs = [
            ("energy_from_protein", "kcal", "energy_from_protein"),
            ("energy_from_fat", "kcal", "energy_from_fat"),
            ("energy_from_carbs", "kcal", "energy_from_carbs"),
        ]
    elif doc_type == "typeB":
        specs = [
            ("protein_g", "g", "protein_g"),
            ("protein_kcal", "kcal", "protein_kcal"),
            ("total_kcal", "kcal", "total_kcal"),
        ]
    elif doc_type == "typeC":
        specs = [
            ("energy_kj", "kJ", "energy_kcal"),
            ("protein_g", "g", "protein_g"),
            ("fat_g", "g", "fat_g"),
        ]
    else:
        raise ValueError(f"Unsupported OpenFoodFacts doc_type: {doc_type}")

    spans: list[QuantitySpan] = []
    for pattern_key, unit, concept in specs:
        match = FIELD_PATTERNS[pattern_key].search(text)
        if match is None:
            raise ValueError(f"Missing field pattern {pattern_key} in doc_id={doc['doc_id']}")
        spans.append(_span_from_match(text, match, unit=unit, concept=concept))
    return _replace_with_num_tokens(text, spans), spans


class OpenFoodFactsPayloadBuilder:
    """Prepare cached query/document payloads with [num] replacement and spans."""

    def __init__(self, tokenizer: Any, max_length: int = 256) -> None:
        """Initialize tokenizer and payload caches."""
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.query_cache: dict[str, dict[str, Any]] = {}
        self.doc_cache: dict[str, dict[str, Any]] = {}

    def _tokenize(self, text: str, spans: list[QuantitySpan]) -> dict[str, Any]:
        """Tokenize prepared text and attach quantity spans."""
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "quantity_spans": [_span_to_json(span) for span in spans],
        }

    def query_payload(self, query: dict[str, Any]) -> dict[str, Any]:
        """Return cached tokenized query payload."""
        query_id = str(query["query_id"])
        if query_id not in self.query_cache:
            modified_text, spans = _query_spans_openfoodfacts(query)
            self.query_cache[query_id] = self._tokenize(modified_text, spans)
        cached = self.query_cache[query_id]
        return {
            "input_ids": cached["input_ids"].clone(),
            "attention_mask": cached["attention_mask"].clone(),
            "quantity_spans": list(cached["quantity_spans"]),
        }

    def doc_payload(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Return cached tokenized document payload."""
        doc_id = str(doc["doc_id"])
        if doc_id not in self.doc_cache:
            modified_text, spans = _doc_spans_openfoodfacts(doc)
            self.doc_cache[doc_id] = self._tokenize(modified_text, spans)
        cached = self.doc_cache[doc_id]
        return {
            "input_ids": cached["input_ids"].clone(),
            "attention_mask": cached["attention_mask"].clone(),
            "quantity_spans": list(cached["quantity_spans"]),
        }


def _preferred_decomp_types(query: dict[str, Any]) -> list[str]:
    """Return decomposed document types appropriate for one query."""
    query_type = str(query.get("query_type", ""))
    if query_type in {"typeA", "typeB", "typeC"}:
        return [query_type]
    if query_type == "mixed":
        return ["typeA", "typeB", "typeC"]
    return []


def _attach_relevant_doc_ids(queries: list[dict[str, Any]], qrels: dict[str, list[str]]) -> None:
    """Ensure query rows carry relevant doc ids from qrels."""
    for query in queries:
        query_id = str(query["query_id"])
        if "relevant_doc_ids" not in query or not query["relevant_doc_ids"]:
            query["relevant_doc_ids"] = list(qrels.get(query_id, []))


def _build_examples(
    queries: list[dict[str, Any]],
    docs_by_id: dict[str, dict[str, Any]],
    split: str,
    mode: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Build training or evaluation examples from OpenFoodFacts corpus/query data."""
    docs_by_type: dict[str, list[str]] = defaultdict(list)
    for doc_id, doc in docs_by_id.items():
        docs_by_type[str(doc["doc_type"])].append(doc_id)

    examples: list[dict[str, Any]] = []
    for query in queries:
        if str(query.get("split", "")) != split:
            continue
        relevant_ids = [doc_id for doc_id in query.get("relevant_doc_ids", []) if doc_id in docs_by_id]
        if not relevant_ids:
            continue
        relevant_docs = [docs_by_id[doc_id] for doc_id in relevant_ids]
        atomic_doc = next((doc for doc in relevant_docs if str(doc["doc_type"]) == "atomic"), None)
        if atomic_doc is None:
            continue

        def sample_negative(target_doc_type: str, product_id: str) -> dict[str, Any]:
            pool = [
                doc_id
                for doc_id in docs_by_type.get(target_doc_type, [])
                if doc_id not in relevant_ids and str(docs_by_id[doc_id]["product_id"]) != product_id
            ]
            if not pool:
                pool = [
                    doc_id
                    for doc_id in docs_by_id
                    if doc_id not in relevant_ids and str(docs_by_id[doc_id]["product_id"]) != product_id
                ]
            return docs_by_id[rng.choice(pool)]

        if mode == "train":
            examples.append(
                {
                    "query": query,
                    "pos_doc": atomic_doc,
                    "neg_doc": sample_negative("atomic", str(atomic_doc["product_id"])),
                    "atomic_doc": None,
                    "doc_id": str(atomic_doc["doc_id"]),
                    "arith_type": None,
                }
            )
            for doc_type in _preferred_decomp_types(query):
                decomp_doc = next((doc for doc in relevant_docs if str(doc["doc_type"]) == doc_type), None)
                if decomp_doc is None:
                    continue
                examples.append(
                    {
                        "query": query,
                        "pos_doc": decomp_doc,
                        "neg_doc": sample_negative(doc_type, str(decomp_doc["product_id"])),
                        "atomic_doc": atomic_doc,
                        "doc_id": str(decomp_doc["doc_id"]),
                        "arith_type": DECOMP_TYPE_TO_ARITH[doc_type],
                    }
                )
        elif mode == "atomic_eval":
            if str(query.get("query_type", "")) != "atomic":
                continue
            examples.append(
                {
                    "query": query,
                    "pos_doc": atomic_doc,
                    "atomic_doc": None,
                    "doc_id": str(atomic_doc["doc_id"]),
                    "arith_type": None,
                }
            )
        elif mode == "decomp_eval":
            for doc_type in _preferred_decomp_types(query):
                decomp_doc = next((doc for doc in relevant_docs if str(doc["doc_type"]) == doc_type), None)
                if decomp_doc is None:
                    continue
                examples.append(
                    {
                        "query": query,
                        "pos_doc": decomp_doc,
                        "atomic_doc": atomic_doc,
                        "doc_id": str(decomp_doc["doc_id"]),
                        "arith_type": DECOMP_TYPE_TO_ARITH[doc_type],
                    }
                )
        else:
            raise ValueError(f"Unsupported example mode: {mode}")

    return examples


class WideQuantOpenFoodFactsDataset(Dataset):
    """Training dataset for WideQuant over OpenFoodFacts examples."""

    def __init__(self, examples: list[dict[str, Any]], payload_builder: OpenFoodFactsPayloadBuilder) -> None:
        """Store abstract examples and payload builder."""
        self.examples = examples
        self.payload_builder = payload_builder

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one WideQuant training example."""
        example = self.examples[idx]
        query_batch = self.payload_builder.query_payload(example["query"])
        doc_pos_batch = self.payload_builder.doc_payload(example["pos_doc"])
        if example.get("atomic_doc") is not None:
            doc_pos_batch["atomic_doc_batch"] = self.payload_builder.doc_payload(example["atomic_doc"])

        doc_neg_batch = None
        if example.get("neg_doc") is not None:
            doc_neg_batch = self.payload_builder.doc_payload(example["neg_doc"])
            neg_doc_type = str(example["neg_doc"]["doc_type"])
            if neg_doc_type in {"typeA", "typeB", "typeC"} and example.get("atomic_doc") is not None:
                doc_neg_batch["atomic_doc_batch"] = self.payload_builder.doc_payload(example["atomic_doc"])

        return {
            "query_batch": query_batch,
            "doc_pos_batch": doc_pos_batch,
            "doc_neg_batch": doc_neg_batch,
            "meta": {
                "doc_id": str(example["doc_id"]),
                "arith_type": example.get("arith_type"),
            },
        }


def _collate_identity(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep sample-level dictionaries unchanged for batch size 1."""
    return batch[0]


def _trim(rows: list[Any], max_examples: int | None) -> list[Any]:
    """Trim a list if max_examples is provided."""
    if max_examples is None:
        return rows
    return rows[: max(0, int(max_examples))]


class WideQuantTrainer(DeepQuantTrainer):
    """WideQuant-specific trainer with AAN optimization and subset evaluation."""

    def __init__(
        self,
        model: WideQuant,
        config: dict,
        train_dataloader: Iterable[dict],
        dev_dataloader: Iterable[dict],
        atomic_eval_examples: list[dict[str, Any]],
        decomp_eval_examples: list[dict[str, Any]],
        aan_eval_examples: list[dict[str, Any]],
    ) -> None:
        """Initialize WideQuant trainer state."""
        super().__init__(model=model, config=config, train_dataloader=train_dataloader, dev_dataloader=dev_dataloader)
        self.atomic_eval_examples = atomic_eval_examples
        self.decomp_eval_examples = decomp_eval_examples
        self.aan_eval_examples = aan_eval_examples

        bert_params = []
        head_params = []
        aan_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("bert."):
                bert_params.append(param)
            elif name.startswith("aan."):
                aan_params.append(param)
            else:
                head_params.append(param)

        head_params.extend(self.exp_head.parameters())
        head_params.extend(self.man_head.parameters())
        head_params.extend(self.unit_head.parameters())

        training_cfg = config.get("training", {})
        self.bert_optimizer = AdamW(bert_params, lr=2e-5, weight_decay=0.01)
        self.head_optimizer = AdamW(head_params, lr=1e-4, weight_decay=0.01)
        self.aan_optimizer = AdamW(aan_params, lr=float(training_cfg.get("lr_aan", 1e-4)), weight_decay=0.01)
        self.loss_fn = TotalLoss(lambda_arith=float(training_cfg.get("lambda_arith", 0.5)))
        self.loss_fn.retr_loss.temperature = float(training_cfg.get("temperature", 0.02))
        self.runs_dir = Path("runs") / "widequant"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.runs_dir))

    def _call_model(self, model: nn.Module, query_batch: dict, doc_batch: dict, use_aan: bool = True) -> dict:
        """Call either DeepQuant or WideQuant with the appropriate signature."""
        if isinstance(model, WideQuant):
            return model(query_batch, doc_batch, None, use_aan=use_aan)
        return model(query_batch, doc_batch, None)

    def train_epoch(self, epoch: int) -> dict:
        """Run one WideQuant epoch with an additional AAN loss term."""
        self.model.train()
        self.exp_head.train()
        self.man_head.train()
        self.unit_head.train()

        self.bert_optimizer.zero_grad(set_to_none=True)
        self.head_optimizer.zero_grad(set_to_none=True)
        self.aan_optimizer.zero_grad(set_to_none=True)

        running = defaultdict(float)
        steps = 0
        progress = tqdm(self.train_dataloader, desc=f"WideQuant Train Epoch {epoch}", leave=False)

        for step_idx, batch in enumerate(progress, start=1):
            query_batch, doc_pos_batch, doc_neg_batch = self._unpack_batch(batch)
            query_batch = self._to_device(query_batch)
            doc_pos_batch = self._to_device(doc_pos_batch)
            doc_neg_batch = self._to_device(doc_neg_batch) if doc_neg_batch is not None else None

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                model_outputs = self.model(query_batch, doc_pos_batch, None, use_aan=True)
                neg_scores_override: Tensor | None = None
                if doc_neg_batch is not None:
                    neg_outputs = self.model(query_batch, doc_neg_batch, None, use_aan=True)
                    neg_scores_override = neg_outputs["final_score"].reshape(1, 1)

                loss_inputs = self._build_loss_inputs(model_outputs, neg_scores_override=neg_scores_override)
                loss_dict = self.loss_fn(loss_inputs)
                l_aan = model_outputs.get("L_AAN", torch.tensor(0.0, dtype=torch.float32, device=self.device))
                total = loss_dict["total"] + 0.5 * l_aan

            scaled_total = total / float(self.gradient_accumulation_steps)
            self.scaler.scale(scaled_total).backward()

            if step_idx % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.bert_optimizer)
                self.scaler.unscale_(self.head_optimizer)
                self.scaler.unscale_(self.aan_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters())
                    + list(self.exp_head.parameters())
                    + list(self.man_head.parameters())
                    + list(self.unit_head.parameters()),
                    max_norm=self.max_grad_norm,
                )
                self.scaler.step(self.bert_optimizer)
                self.scaler.step(self.head_optimizer)
                self.scaler.step(self.aan_optimizer)
                self.scaler.update()
                self.bert_optimizer.zero_grad(set_to_none=True)
                self.head_optimizer.zero_grad(set_to_none=True)
                self.aan_optimizer.zero_grad(set_to_none=True)

            steps += 1
            for key, value in loss_dict.items():
                running[key] += float(value.detach().cpu())
            running["L_AAN"] += float(l_aan.detach().cpu())
            running["total"] += float(total.detach().cpu())

            progress.set_postfix(
                {
                    "total": f"{float(total.detach().cpu()):.4f}",
                    "L_retr": f"{float(loss_dict['L_retr'].detach().cpu()):.4f}",
                    "L_quant": f"{float(loss_dict['L_quant'].detach().cpu()):.4f}",
                    "L_reg": f"{float(loss_dict['L_reg'].detach().cpu()):.4f}",
                    "L_comp": f"{float(loss_dict['L_comp'].detach().cpu()):.4f}",
                    "L_arith": f"{float(loss_dict['L_arith'].detach().cpu()):.4f}",
                    "L_AAN": f"{float(l_aan.detach().cpu()):.4f}",
                }
            )

        averages = {key: value / max(steps, 1) for key, value in running.items()}
        for key, value in averages.items():
            self.writer.add_scalar(f"train/{key}", value, epoch)
        return averages

    def _evaluate_examples(self, model: nn.Module, examples: list[dict[str, Any]], use_aan: bool = True) -> float:
        """Evaluate MRR@10 on a query/doc subset using ANN stage-1 and pairwise re-ranking."""
        if not examples:
            return 0.0
        model.eval()

        unique_docs: dict[str, dict[str, Any]] = {}
        for example in examples:
            unique_docs[str(example["doc_id"])] = example["doc_pos_batch"]

        doc_ids = list(unique_docs.keys())
        doc_vectors: list[np.ndarray] = []
        query_vectors: list[np.ndarray] = []

        with torch.no_grad():
            for doc_id in doc_ids:
                doc_batch = self._to_device(unique_docs[doc_id])
                doc_enc = model.encode_document(
                    doc_batch["input_ids"],
                    doc_batch["attention_mask"],
                    doc_batch.get("quantity_spans", []),
                )
                doc_vec = F.normalize(doc_enc["cls"], dim=0).detach().cpu().numpy().astype(np.float32)
                doc_vectors.append(doc_vec)

            for example in examples:
                query_batch = self._to_device(example["query_batch"])
                query_enc = model.encode_query(
                    query_batch["input_ids"],
                    query_batch["attention_mask"],
                    query_batch.get("quantity_spans", []),
                )
                query_vec = F.normalize(query_enc["cls"], dim=0).detach().cpu().numpy().astype(np.float32)
                query_vectors.append(query_vec)

        doc_matrix = np.stack(doc_vectors, axis=0)
        query_matrix = np.stack(query_vectors, axis=0)
        top_k = min(100, doc_matrix.shape[0])

        if faiss is not None:
            index = faiss.IndexFlatIP(doc_matrix.shape[1])
            index.add(doc_matrix)
            _, retrieved = index.search(query_matrix, top_k)
        else:  # pragma: no cover
            sims = np.matmul(query_matrix, doc_matrix.T)
            retrieved = np.argsort(-sims, axis=1)[:, :top_k]

        reciprocal_ranks: list[float] = []
        with torch.no_grad():
            for query_idx, example in enumerate(examples):
                query_batch = self._to_device(example["query_batch"])
                retrieved_doc_ids = [doc_ids[int(idx)] for idx in retrieved[query_idx]]
                reranked: list[tuple[float, str]] = []
                for doc_id in retrieved_doc_ids:
                    doc_batch = self._to_device(unique_docs[doc_id])
                    outputs = self._call_model(model, query_batch, doc_batch, use_aan=use_aan)
                    reranked.append((float(outputs["final_score"].detach().cpu()), doc_id))
                reranked.sort(key=lambda item: item[0], reverse=True)

                rr = 0.0
                for rank, (_, doc_id) in enumerate(reranked[:10], start=1):
                    if doc_id == str(example["doc_id"]):
                        rr = 1.0 / float(rank)
                        break
                reciprocal_ranks.append(rr)

        return float(sum(reciprocal_ranks) / max(1, len(reciprocal_ranks)))

    def train(
        self,
        n_epochs: int,
        baseline_decomp_mrr: float,
        baseline_atomic_mrr: float,
    ) -> dict[str, float]:
        """Train WideQuant and report decomposed/atomic performance after each epoch."""
        best_decomp_mrr = float("-inf")
        best_atomic_mrr = 0.0
        best_aan_mean = 0.0

        for epoch in range(1, int(n_epochs) + 1):
            train_losses = self.train_epoch(epoch)
            decomp_mrr = self._evaluate_examples(self.model, self.decomp_eval_examples, use_aan=True)
            atomic_mrr = self._evaluate_examples(self.model, self.atomic_eval_examples, use_aan=True)
            aan_metrics = verify_aan_quality(self.model, self.model.aan, self.aan_eval_examples, threshold=0.80)
            aan_mean = float(aan_metrics["mean_cosine"])

            self.writer.add_scalar("eval/decomp_mrr10", decomp_mrr, epoch)
            self.writer.add_scalar("eval/atomic_mrr10", atomic_mrr, epoch)
            self.writer.add_scalar("eval/aan_mean_cosine", aan_mean, epoch)

            if decomp_mrr > best_decomp_mrr:
                best_decomp_mrr = decomp_mrr
                best_atomic_mrr = atomic_mrr
                best_aan_mean = aan_mean
                ckpt_path = Path("checkpoints") / "best_widequant.pt"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "exp_head_state_dict": self.exp_head.state_dict(),
                        "man_head_state_dict": self.man_head.state_dict(),
                        "unit_head_state_dict": self.unit_head.state_dict(),
                        "bert_optimizer_state_dict": self.bert_optimizer.state_dict(),
                        "head_optimizer_state_dict": self.head_optimizer.state_dict(),
                        "aan_optimizer_state_dict": self.aan_optimizer.state_dict(),
                        "config": self.config,
                        "best_decomp_mrr": best_decomp_mrr,
                    },
                    ckpt_path,
                )

            print(
                f"Epoch {epoch} | total={train_losses['total']:.4f} | "
                f"L_retr={train_losses['L_retr']:.4f} | L_quant={train_losses['L_quant']:.4f} | "
                f"L_reg={train_losses['L_reg']:.4f} | L_comp={train_losses['L_comp']:.4f} | "
                f"L_arith={train_losses['L_arith']:.4f} | L_AAN={train_losses['L_AAN']:.4f} | "
                f"Decomp MRR@10={decomp_mrr:.4f} | Atomic MRR@10={atomic_mrr:.4f} | "
                f"AAN mean cos={aan_mean:.4f}"
            )

        self.writer.flush()
        self.writer.close()

        atomic_ok = best_atomic_mrr + 1e-6 >= baseline_atomic_mrr - 1e-4

        if best_aan_mean > 0.80 and best_decomp_mrr > baseline_decomp_mrr and atomic_ok:
            print("PHASE 5 COMPLETE. AAN QUALITY VERIFIED. PROCEED TO PHASE 6")
        else:
            if best_aan_mean <= 0.80:
                print(f"FAILURE: AAN cosine similarity {best_aan_mean:.4f} did not exceed 0.80.")
            if best_decomp_mrr <= baseline_decomp_mrr:
                print(
                    f"FAILURE: Decomposed MRR@10 {best_decomp_mrr:.4f} did not beat DeepQuant baseline "
                    f"{baseline_decomp_mrr:.4f}."
                )
            if not atomic_ok:
                print(
                    f"FAILURE: Atomic MRR@10 regressed from DeepQuant baseline {baseline_atomic_mrr:.4f} "
                    f"to {best_atomic_mrr:.4f}."
                )
            print("Suggested fix: inspect training examples, AAN supervision pairs, and decomposed retrieval scoring.")

        return {
            "best_decomp_mrr10": best_decomp_mrr,
            "best_atomic_mrr10": best_atomic_mrr,
            "best_aan_mean_cosine": best_aan_mean,
            "baseline_decomp_mrr10": baseline_decomp_mrr,
            "baseline_atomic_mrr10": baseline_atomic_mrr,
        }


def _ensure_dataset_ready(data_dir: Path, build_if_missing: bool, build_n_products: int) -> None:
    """Ensure OpenFoodFacts dataset files exist, optionally building them."""
    required = [data_dir / "documents.jsonl", data_dir / "queries.jsonl", data_dir / "qrels.tsv"]
    if all(path.exists() for path in required):
        return
    if not build_if_missing:
        raise FileNotFoundError(
            f"Missing dataset files in {data_dir}. Expected documents.jsonl, queries.jsonl, qrels.tsv. "
            "Run src/data/openfoodfacts.py or pass --build_if_missing."
        )
    build_openfoodfacts_dataset(n_products=int(build_n_products), output_dir=str(data_dir))


def _load_warm_start(model: WideQuant, checkpoint_path: Path) -> dict[str, Any]:
    """Load DeepQuant checkpoint weights into WideQuant."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(
        f"Loaded DeepQuant weights from {checkpoint_path} | "
        f"missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}"
    )
    return checkpoint


def _prepare_eval_examples(
    abstract_examples: list[dict[str, Any]],
    payload_builder: OpenFoodFactsPayloadBuilder,
) -> list[dict[str, Any]]:
    """Convert abstract examples into encoded evaluation examples."""
    prepared: list[dict[str, Any]] = []
    for example in abstract_examples:
        query_batch = payload_builder.query_payload(example["query"])
        doc_pos_batch = payload_builder.doc_payload(example["pos_doc"])
        if example.get("atomic_doc") is not None:
            doc_pos_batch["atomic_doc_batch"] = payload_builder.doc_payload(example["atomic_doc"])
        prepared.append(
            {
                "query_batch": query_batch,
                "doc_pos_batch": doc_pos_batch,
                "doc_id": example["doc_id"],
                "arith_type": example.get("arith_type"),
            }
        )
    return prepared


def _prepare_aan_examples(
    abstract_examples: list[dict[str, Any]],
    payload_builder: OpenFoodFactsPayloadBuilder,
) -> list[dict[str, Any]]:
    """Prepare decomposed/atomic pairs for verify_aan_quality()."""
    pairs: list[dict[str, Any]] = []
    for example in abstract_examples:
        if example.get("atomic_doc") is None or example.get("arith_type") is None:
            continue
        pairs.append(
            {
                "decomposed_doc_batch": payload_builder.doc_payload(example["pos_doc"]),
                "atomic_doc_batch": payload_builder.doc_payload(example["atomic_doc"]),
                "arith_type": example["arith_type"],
            }
        )
    return pairs


def main() -> None:
    """Run WideQuant training from a DeepQuant warm start."""
    args = parse_args()
    set_seed(args.seed)

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config.setdefault("model", {})
    config["model"].setdefault("local_files_only", False)

    run_verification(config)

    data_dir = Path(args.data_dir)
    _ensure_dataset_ready(data_dir, build_if_missing=args.build_if_missing, build_n_products=args.build_n_products)

    documents = _load_jsonl(data_dir / "documents.jsonl")
    queries = _load_jsonl(data_dir / "queries.jsonl")
    qrels = _parse_qrels(data_dir / "qrels.tsv")
    _attach_relevant_doc_ids(queries, qrels)
    docs_by_id = {str(doc["doc_id"]): doc for doc in documents}

    rng = random.Random(args.seed)
    train_examples = _build_examples(queries, docs_by_id, split="train", mode="train", rng=rng)
    atomic_eval_abstract = _build_examples(queries, docs_by_id, split="test", mode="atomic_eval", rng=rng)
    decomp_eval_abstract = _build_examples(queries, docs_by_id, split="test", mode="decomp_eval", rng=rng)

    if args.dry_run:
        train_examples = _trim(train_examples, 8)
        atomic_eval_abstract = _trim(atomic_eval_abstract, 8)
        decomp_eval_abstract = _trim(decomp_eval_abstract, 8)
    else:
        train_examples = _trim(train_examples, args.max_train_examples)
        atomic_eval_abstract = _trim(atomic_eval_abstract, args.max_eval_examples)
        decomp_eval_abstract = _trim(decomp_eval_abstract, args.max_eval_examples)

    model = WideQuant(config)
    checkpoint = _load_warm_start(model, Path(args.checkpoint))
    payload_builder = OpenFoodFactsPayloadBuilder(model.tokenizer, max_length=args.max_length)

    train_dataset = WideQuantOpenFoodFactsDataset(train_examples, payload_builder)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=_collate_identity)

    atomic_eval_examples = _prepare_eval_examples(atomic_eval_abstract, payload_builder)
    decomp_eval_examples = _prepare_eval_examples(decomp_eval_abstract, payload_builder)
    aan_eval_examples = _prepare_aan_examples(decomp_eval_abstract[:100], payload_builder)

    trainer = WideQuantTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        dev_dataloader=train_dataloader,
        atomic_eval_examples=atomic_eval_examples,
        decomp_eval_examples=decomp_eval_examples,
        aan_eval_examples=aan_eval_examples,
    )

    if "exp_head_state_dict" in checkpoint:
        trainer.exp_head.load_state_dict(checkpoint["exp_head_state_dict"], strict=False)
    if "man_head_state_dict" in checkpoint:
        trainer.man_head.load_state_dict(checkpoint["man_head_state_dict"], strict=False)
    if "unit_head_state_dict" in checkpoint:
        trainer.unit_head.load_state_dict(checkpoint["unit_head_state_dict"], strict=False)

    baseline_model = DeepQuant(config)
    baseline_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    baseline_model.to(trainer.device)
    baseline_decomp_mrr = trainer._evaluate_examples(baseline_model, decomp_eval_examples, use_aan=False)
    baseline_atomic_mrr = trainer._evaluate_examples(baseline_model, atomic_eval_examples, use_aan=False)
    print(f"DeepQuant decomposed baseline MRR@10: {baseline_decomp_mrr:.4f}")
    print(f"DeepQuant atomic baseline MRR@10: {baseline_atomic_mrr:.4f}")

    n_epochs = int(args.epochs) if args.epochs is not None else int(config.get("training", {}).get("epochs", 8))

    if args.dry_run:
        print("DRY RUN - running one short WideQuant epoch.")
        results = trainer.train(
            n_epochs=1,
            baseline_decomp_mrr=baseline_decomp_mrr,
            baseline_atomic_mrr=baseline_atomic_mrr,
        )
    else:
        results = trainer.train(
            n_epochs=n_epochs,
            baseline_decomp_mrr=baseline_decomp_mrr,
            baseline_atomic_mrr=baseline_atomic_mrr,
        )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
