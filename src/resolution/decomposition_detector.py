"""Decomposition detection utilities for WideQuant arithmetic retrieval."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.encoding.cqe_wrapper import QuantitySpan, ResolvedCandidate
from src.resolution.conversion_table import convert, normalize_unit, units_are_compatible
from src.resolution.type_a_resolver import ADDITIVE_SUBCONCEPT_VOCABULARY, TypeAResolver
from src.resolution.type_b_resolver import RATIO_VOCABULARY, TypeBResolver
from src.resolution.type_c_resolver import TypeCResolver

DETECTION_LABELS = ["ATOMIC", "TYPE_A", "TYPE_B", "TYPE_C"]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(DETECTION_LABELS)}


class RuleBasedDetector:
    """Detect decomposition type using resolver-aware symbolic rules."""

    def detect(self, quantities: list[QuantitySpan], query_unit: str, query_concept: str) -> str:
        """Return one of ATOMIC, TYPE_A, TYPE_B, or TYPE_C with fixed precedence."""
        if self._is_type_a(quantities, query_unit, query_concept):
            return "TYPE_A"
        if self._is_type_b(quantities):
            return "TYPE_B"
        if self._is_type_c(quantities, query_unit):
            return "TYPE_C"
        return "ATOMIC"

    def _is_type_a(self, quantities: list[QuantitySpan], query_unit: str, query_concept: str) -> bool:
        """Return True when at least two additive subconcepts match the query concept."""
        normalized_query_concept = str(query_concept).strip().lower()
        allowed = {
            concept.lower()
            for concept in ADDITIVE_SUBCONCEPT_VOCABULARY.get(normalized_query_concept, [])
        }
        if not allowed:
            return False

        matches = [
            span
            for span in quantities
            if str(span.concept).strip().lower() in allowed
            and units_are_compatible(span.unit, query_unit)
        ]
        return len(matches) >= 2

    def _is_type_b(self, quantities: list[QuantitySpan]) -> bool:
        """Return True when a known numerator/denominator concept pair appears."""
        concepts = {str(span.concept).strip().lower() for span in quantities}
        for concept_num, concept_den, _ in RATIO_VOCABULARY:
            if concept_num.lower() in concepts and concept_den.lower() in concepts:
                return True
        return False

    def _is_type_c(self, quantities: list[QuantitySpan], query_unit: str) -> bool:
        """Return True when exactly one non-query-unit quantity is convertible to the query unit."""
        normalized_query_unit = normalize_unit(query_unit)
        non_query_unit_spans = [
            span for span in quantities if normalize_unit(span.unit) != normalized_query_unit
        ]
        if len(non_query_unit_spans) != 1:
            return False
        return convert(1.0, non_query_unit_spans[0].unit, normalized_query_unit) is not None


class LearnedDetector(nn.Module):
    """BERT-based learned decomposition classifier over {ATOMIC, TYPE_A, TYPE_B, TYPE_C}."""

    def __init__(self, model_path: str | None = None) -> None:
        """Initialize tokenizer, encoder, and classification head."""
        super().__init__()
        self.encoder_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
        self.bert = AutoModel.from_pretrained(self.encoder_name)
        hidden_size = int(self.bert.config.hidden_size)
        self.classifier = nn.Linear(hidden_size, len(DETECTION_LABELS))
        if model_path is not None and Path(model_path).exists():
            payload = torch.load(model_path, map_location="cpu")
            self.load_state_dict(payload["model_state_dict"])

    def _encode_inputs(
        self,
        doc_text: str,
        query_concept: str,
        query_unit: str,
    ) -> dict[str, Tensor]:
        """Tokenize the detector input sequence."""
        combined = f"{doc_text} [SEP] {query_concept} [SEP] {query_unit}"
        return self.tokenizer(
            combined,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        )

    def logits(self, doc_text: str, query_concept: str, query_unit: str) -> Tensor:
        """Return raw class logits for one detector input."""
        encoded = self._encode_inputs(doc_text, query_concept, query_unit)
        device = next(self.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = self.bert(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            return_dict=True,
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_embedding).squeeze(0)

    def forward(self, doc_text: str, query_concept: str, query_unit: str) -> str:
        """Return the predicted class string for one detector input."""
        with torch.no_grad():
            logits = self.logits(doc_text, query_concept, query_unit)
            prediction = int(torch.argmax(logits).item())
        return DETECTION_LABELS[prediction]

    def train_classifier(
        self,
        train_data: list[dict[str, str]],
        dev_data: list[dict[str, str]],
        epochs: int = 3,
        save_path: str = "checkpoints/learned_detector.pt",
    ) -> dict[str, float]:
        """Fine-tune the detector and save the best checkpoint by dev accuracy."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        optimizer = AdamW(self.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss()
        best_dev_accuracy = 0.0

        for epoch in range(1, int(epochs) + 1):
            self.train()
            for sample in train_data:
                optimizer.zero_grad()
                logits = self.logits(
                    sample["doc_text"],
                    sample["query_concept"],
                    sample["query_unit"],
                ).unsqueeze(0)
                label = torch.tensor([LABEL_TO_INDEX[sample["label"]]], dtype=torch.long, device=device)
                loss = criterion(logits, label)
                loss.backward()
                optimizer.step()

            self.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for sample in dev_data:
                    logits = self.logits(
                        sample["doc_text"],
                        sample["query_concept"],
                        sample["query_unit"],
                    )
                    pred = int(torch.argmax(logits).item())
                    gold = LABEL_TO_INDEX[sample["label"]]
                    correct += int(pred == gold)
                    total += 1
            dev_accuracy = correct / max(total, 1)
            if dev_accuracy > best_dev_accuracy:
                best_dev_accuracy = dev_accuracy
                save_target = Path(save_path)
                save_target.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state_dict": self.state_dict()}, save_target)
            print(f"Epoch {epoch} | Dev Accuracy: {dev_accuracy:.4f}")

        return {"best_dev_accuracy": best_dev_accuracy}


def build_arithmetic_candidates(
    doc_quantities: list[QuantitySpan],
    query_unit: str,
    query_concept: str,
    detector: RuleBasedDetector,
) -> list[ResolvedCandidate]:
    """Detect arithmetic type and return zero or one resolved candidates."""
    arith_type = detector.detect(doc_quantities, query_unit, query_concept)
    if arith_type == "ATOMIC":
        return []
    if arith_type == "TYPE_A":
        candidate = TypeAResolver().resolve(doc_quantities, query_unit=query_unit, query_concept=query_concept)
        return [candidate] if candidate is not None else []
    if arith_type == "TYPE_B":
        candidate = TypeBResolver().resolve(doc_quantities, query_unit=query_unit)
        return [candidate] if candidate is not None else []
    if arith_type == "TYPE_C":
        candidate = TypeCResolver().resolve(doc_quantities, query_unit=query_unit)
        return [candidate] if candidate is not None else []
    return []


if __name__ == "__main__":
    detector = RuleBasedDetector()

    atomic_quantities = [
        QuantitySpan(
            text="256",
            mantissa=2.56,
            exponent=2,
            unit="GB",
            concept="storage",
            start_char=0,
            end_char=3,
        )
    ]
    atomic_result = detector.detect(atomic_quantities, query_unit="GB", query_concept="storage")
    atomic_pass = atomic_result == "ATOMIC"
    print(f"Test ATOMIC: {'PASS' if atomic_pass else 'FAIL'} ({atomic_result})")

    type_a_quantities = [
        QuantitySpan(
            text="256",
            mantissa=2.56,
            exponent=2,
            unit="GB",
            concept="ssd_storage",
            start_char=0,
            end_char=3,
        ),
        QuantitySpan(
            text="1000",
            mantissa=1.0,
            exponent=3,
            unit="GB",
            concept="hdd_storage",
            start_char=4,
            end_char=8,
        ),
    ]
    type_a_result = detector.detect(type_a_quantities, query_unit="GB", query_concept="storage")
    type_a_pass = type_a_result == "TYPE_A"
    print(f"Test TYPE_A: {'PASS' if type_a_pass else 'FAIL'} ({type_a_result})")

    type_b_quantities = [
        QuantitySpan(
            text="$150",
            mantissa=1.5,
            exponent=2,
            unit="$",
            concept="share_price",
            start_char=0,
            end_char=4,
        ),
        QuantitySpan(
            text="$10",
            mantissa=1.0,
            exponent=1,
            unit="$",
            concept="earnings_per_share",
            start_char=5,
            end_char=8,
        ),
    ]
    type_b_result = detector.detect(type_b_quantities, query_unit="x", query_concept="valuation")
    type_b_pass = type_b_result == "TYPE_B"
    print(f"Test TYPE_B: {'PASS' if type_b_pass else 'FAIL'} ({type_b_result})")

    type_c_quantities = [
        QuantitySpan(
            text="1046",
            mantissa=1.046,
            exponent=3,
            unit="kJ",
            concept="energy",
            start_char=0,
            end_char=4,
        )
    ]
    type_c_result = detector.detect(type_c_quantities, query_unit="kcal", query_concept="energy_kcal")
    type_c_pass = type_c_result == "TYPE_C"
    print(f"Test TYPE_C: {'PASS' if type_c_pass else 'FAIL'} ({type_c_result})")

    precedence_quantities = [
        QuantitySpan(
            text="256",
            mantissa=2.56,
            exponent=2,
            unit="GB",
            concept="ssd_storage",
            start_char=0,
            end_char=3,
        ),
        QuantitySpan(
            text="1",
            mantissa=1.0,
            exponent=0,
            unit="TB",
            concept="hdd_storage",
            start_char=4,
            end_char=5,
        ),
    ]
    precedence_result = detector.detect(precedence_quantities, query_unit="GB", query_concept="storage")
    precedence_pass = precedence_result == "TYPE_A"
    print(f"Test precedence: {'PASS' if precedence_pass else 'FAIL'} ({precedence_result})")

    overall = atomic_pass and type_a_pass and type_b_pass and type_c_pass and precedence_pass
    print(f"OVERALL {'PASS' if overall else 'FAIL'}")
