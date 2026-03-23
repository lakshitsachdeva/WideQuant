"""CQE integration utilities for extracting quantity spans."""

from __future__ import annotations

import importlib
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence

from transformers import BertTokenizer

NUMBER_PATTERN = re.compile(r"(?<!\w)(?:\d+\.?\d*|\.\d+)(?!\w)")
REGEX_NUM_REPLACEMENT_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?(?:%|billion|million|thousand)?\b",
    flags=re.IGNORECASE,
)
RECONSTRUCT_NUMBER_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\b")


def setup_tokenizer(
    model_name: str = "bert-base-uncased",
    local_files_only: bool = False,
    cache_dir: str | None = None,
) -> tuple[BertTokenizer, int]:
    """Create the canonical BERT tokenizer with [num] registered as a special token."""
    tokenizer = BertTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
    )
    special_tokens = {"additional_special_tokens": ["[num]"]}
    num_added = tokenizer.add_special_tokens(special_tokens)
    num_token_id = int(tokenizer.convert_tokens_to_ids("[num]"))
    assert num_token_id != 100, (
        f"CRITICAL: [num] mapped to [UNK] id=100. "
        f"Special token was not added correctly. num_added={num_added}"
    )
    assert num_token_id == len(tokenizer) - num_added or num_token_id > 30000, (
        f"CRITICAL: [num] token id={num_token_id} looks wrong. "
        f"Expected id > 30000 for a newly added special token."
    )
    return tokenizer, num_token_id


def replace_with_num_tokens_regex(text: str) -> str:
    """Replace numeric mentions with [num] using a lightweight regex fallback."""
    replaced = REGEX_NUM_REPLACEMENT_PATTERN.sub("[num]", text)
    replaced = replaced.replace("[num]", " [num] ")
    return re.sub(r"\s+", " ", replaced).strip()


def no_numbers_in_text(text: str) -> bool:
    """Return True when the text contains no numeric literals."""
    return NUMBER_PATTERN.search(text) is None


@dataclass(slots=True)
class QuantitySpan:
    """Container for one extracted quantity mention."""

    text: str
    mantissa: float
    exponent: int
    unit: str
    concept: str
    start_char: int
    end_char: int


def reconstruct_spans_from_num_tokens(
    text: str,
    tokenizer: Any | None = None,
    num_token_id: int | None = None,
) -> list[QuantitySpan]:
    """Reconstruct lightweight QuantitySpans directly from numeric literals in raw text.

    The tokenizer and num_token_id arguments are accepted for compatibility with
    training/data-pipeline callers, but the reconstruction itself works from the
    original text before `[num]` replacement.
    """
    _ = (tokenizer, num_token_id)
    spans: list[QuantitySpan] = []
    for match in RECONSTRUCT_NUMBER_PATTERN.finditer(text):
        raw_value = match.group(1)
        try:
            value = float(raw_value)
        except ValueError:
            continue

        if value > 0.0:
            exponent = int(math.floor(math.log10(value)))
            mantissa = float(value / (10**exponent))
        else:
            exponent = 0
            mantissa = 0.0
        mantissa = max(-10.0, min(10.0, mantissa))

        spans.append(
            QuantitySpan(
                text=raw_value,
                mantissa=mantissa,
                exponent=exponent,
                unit="UNK",
                concept="UNK",
                start_char=int(match.start(1)),
                end_char=int(match.end(1)),
            )
        )
    return spans


@dataclass(slots=True)
class ResolvedCandidate:
    """Container for one arithmetic-resolved quantity candidate."""

    value: float
    unit: str
    mantissa: float
    exponent: int
    source_type: str
    source_spans: list[QuantitySpan]


class CQEWrapper:
    """Thin wrapper around CQE output for WideQuant-ready quantity spans."""

    def __init__(self, overload: bool = True, spacy_model: str = "en_core_web_sm") -> None:
        """Initialize the CQE parser instance."""
        parser_cls = self._load_cqe_parser_class()
        try:
            self._parser = parser_cls(overload=overload, spacy_model=spacy_model)
        except TypeError:
            self._parser = parser_cls(overload=overload)

    @staticmethod
    def install_cqe(clone_dir: str | Path | None = None) -> None:
        """Install CQE with pip, or clone from GitHub if pip install fails.

        Preferred install:
            pip install cqe

        GitHub fallback:
            git clone https://github.com/vivkm/CQE
            pip install ./CQE
        """
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "cqe"],
                check=True,
                capture_output=True,
                text=True,
            )
            return
        except subprocess.CalledProcessError:
            if clone_dir is None:
                raise RuntimeError(
                    "Failed to install CQE with pip. "
                    "Run `git clone https://github.com/vivkm/CQE` and install manually."
                )

        clone_path = Path(clone_dir).expanduser().resolve()
        repo_dir = clone_path / "CQE"
        clone_path.mkdir(parents=True, exist_ok=True)

        if not repo_dir.exists():
            subprocess.run(
                ["git", "clone", "https://github.com/vivkm/CQE", str(repo_dir)],
                check=True,
            )

        subprocess.run(
            [sys.executable, "-m", "pip", "install", str(repo_dir)],
            check=True,
        )

    @staticmethod
    def _load_cqe_parser_class() -> Any:
        """Resolve the CQE parser class from supported install layouts."""
        errors: list[str] = []

        try:
            module = importlib.import_module("CQE.CQE")
            parser_cls = getattr(module, "CQE", None)
            if parser_cls is not None:
                return parser_cls
        except Exception as exc:  # pragma: no cover - import fallback path
            errors.append(f"CQE.CQE import failed: {exc}")

        try:
            package = importlib.import_module("CQE")
            cqe_attr = getattr(package, "CQE", None)
            if hasattr(cqe_attr, "CQE"):
                return cqe_attr.CQE
            if isinstance(cqe_attr, type):
                return cqe_attr
        except Exception as exc:  # pragma: no cover - import fallback path
            errors.append(f"CQE import failed: {exc}")

        try:
            module = importlib.import_module("cqe")
            parser_cls = getattr(module, "CQE", None)
            if parser_cls is not None:
                return parser_cls
        except Exception as exc:  # pragma: no cover - import fallback path
            errors.append(f"cqe import failed: {exc}")

        raise ImportError(
            "CQE is not installed or importable. "
            "Install via `pip install cqe` or clone `https://github.com/vivkm/CQE`.\n"
            + "\n".join(errors)
        )

    @staticmethod
    def _to_mantissa_exponent(value: float) -> tuple[float, int]:
        """Convert scalar value to scientific mantissa/exponent pair."""
        if value == 0.0:
            return 0.0, 0
        exponent = int(math.floor(math.log10(abs(value))))
        mantissa = float(value / (10**exponent))
        return mantissa, exponent

    @staticmethod
    def _normalize_indices(value_indices: Any) -> list[tuple[int, int]]:
        """Normalize CQE index output into a list of integer span tuples."""
        spans: list[tuple[int, int]] = []
        if value_indices is None:
            return spans

        if isinstance(value_indices, tuple) and len(value_indices) == 2:
            a, b = value_indices
            if isinstance(a, int) and isinstance(b, int):
                return [(a, b)]

        if isinstance(value_indices, list):
            for item in value_indices:
                if isinstance(item, tuple) and len(item) == 2:
                    a, b = item
                    if isinstance(a, int) and isinstance(b, int):
                        spans.append((a, b))
        return spans

    @staticmethod
    def _extract_scalar(quantity_obj: Any) -> float:
        """Extract a scalar numeric value from CQE quantity output."""
        value_obj = getattr(quantity_obj, "value", None)
        if value_obj is None:
            return 0.0

        if hasattr(value_obj, "value"):
            return float(value_obj.value)
        if hasattr(value_obj, "lower"):
            return float(value_obj.lower)
        return float(str(value_obj))

    @staticmethod
    def _extract_unit(quantity_obj: Any) -> str:
        """Extract a normalized unit string from CQE quantity output."""
        unit_obj = getattr(quantity_obj, "unit", None)
        if unit_obj is None:
            return ""

        norm_unit = getattr(unit_obj, "norm_unit", None)
        if isinstance(norm_unit, str) and norm_unit != "-":
            return norm_unit

        unit_tokens = getattr(unit_obj, "unit", None)
        if isinstance(unit_tokens, list):
            tokens = [getattr(token, "text", str(token)) for token in unit_tokens]
            return " ".join(token for token in tokens if token).strip()

        return ""

    @staticmethod
    def _extract_concept(quantity_obj: Any) -> str:
        """Extract referred concept text from CQE quantity output."""
        concepts = getattr(quantity_obj, "referred_concepts", None)
        if concepts is None:
            return ""

        def to_text(tokens_or_obj: Any) -> str:
            if isinstance(tokens_or_obj, list):
                tokens = [getattr(token, "text", str(token)) for token in tokens_or_obj]
                return " ".join(token for token in tokens if token).strip()
            return str(tokens_or_obj)

        if hasattr(concepts, "get_nouns"):
            nouns = concepts.get_nouns()
            if nouns:
                return to_text(nouns[0])

        noun_attr = getattr(concepts, "noun", None)
        if isinstance(noun_attr, dict) and noun_attr:
            first_key = sorted(noun_attr.keys())[0]
            return to_text(noun_attr[first_key])
        if isinstance(noun_attr, str):
            return "" if noun_attr == "-" else noun_attr

        if isinstance(concepts, dict):
            values = list(concepts.values())
        else:
            values = concepts

        if not values:
            return ""

        first = values[0]
        if isinstance(first, list):
            tokens = [getattr(token, "text", str(token)) for token in first]
            return " ".join(token for token in tokens if token).strip()

        return str(first)

    def extract(self, text: str) -> List[QuantitySpan]:
        """Run CQE on input text and return QuantitySpan outputs."""
        raw_quantities = self._parser.parse(text)
        output: list[QuantitySpan] = []

        for item in raw_quantities:
            indices_dict = item.get_char_indices() if hasattr(item, "get_char_indices") else {}
            value_indices = self._normalize_indices(indices_dict.get("value"))
            if not value_indices:
                continue

            start_char = min(start for start, _ in value_indices)
            end_char = max(end for _, end in value_indices)
            if not (0 <= start_char < end_char <= len(text)):
                continue

            numeric_text = text[start_char:end_char]
            scalar_value = self._extract_scalar(item)
            mantissa, exponent = self._to_mantissa_exponent(scalar_value)

            output.append(
                QuantitySpan(
                    text=numeric_text,
                    mantissa=mantissa,
                    exponent=exponent,
                    unit=self._extract_unit(item),
                    concept=self._extract_concept(item),
                    start_char=start_char,
                    end_char=end_char,
                )
            )

        return output

    @staticmethod
    def replace_with_num_tokens(text: str, spans: List[QuantitySpan]) -> str:
        """Replace quantity numeric spans with [num], keeping unit words untouched."""
        if not spans:
            # CQE can miss certain numeric mentions; keep [num] replacement robust.
            return replace_with_num_tokens_regex(text)

        ordered = sorted(spans, key=lambda span: span.start_char)
        parts: list[str] = []
        cursor = 0

        for span in ordered:
            start = max(0, min(len(text), span.start_char))
            end = max(start, min(len(text), span.end_char))
            if start < cursor:
                continue
            parts.append(text[cursor:start])
            span_text = text[start:end]
            span_replaced = NUMBER_PATTERN.sub("[num]", span_text)
            if "[num]" not in span_replaced:
                span_replaced = "[num]"
            parts.append(span_replaced)
            cursor = end

        parts.append(text[cursor:])
        replaced = "".join(parts)
        replaced = NUMBER_PATTERN.sub("[num]", replaced)
        # Keep [num] as a standalone token so tokenizer doesn't split/drop it.
        replaced = replaced.replace("[num]", " [num] ")
        return re.sub(r"\s+", " ", replaced).strip()


if __name__ == "__main__":
    sample_text = "laptop storage over 256 GB"
    pipeline_text = "company revenue over 5 billion dollars"

    try:
        wrapper = CQEWrapper()
    except ImportError as exc:
        print("CQE not found. Attempting `pip install cqe`...")
        try:
            CQEWrapper.install_cqe()
            wrapper = CQEWrapper()
        except Exception as install_exc:  # pragma: no cover - install path
            print(f"Automatic install failed: {install_exc}")
            print("Fallback: git clone https://github.com/vivkm/CQE")
            raise SystemExit(1) from exc

    extracted_spans = wrapper.extract(sample_text)
    modified_text = wrapper.replace_with_num_tokens(sample_text, extracted_spans)

    print("Extracted spans:")
    for span in extracted_spans:
        print(span)
    print("Modified text:")
    print(modified_text)

    # Pipeline-order verification.
    raw_text = pipeline_text
    spans = wrapper.extract(raw_text)
    modified = wrapper.replace_with_num_tokens(raw_text, spans)
    print(f"Step 1 raw_text: {raw_text}")
    print(f"Step 2 spans count: {len(spans)}")
    print(f"Step 3 modified_text: {modified}")
    if "[num]" not in modified:
        print("BUG: replace_with_num_tokens() did not inject [num].")

    tokenizer, num_id = setup_tokenizer()
    ids = tokenizer(modified, add_special_tokens=True)["input_ids"]
    appears = num_id in ids
    print(f"Step 4 num_token_id: {num_id}")
    print(f"Step 4 appears in input_ids: {appears}")
    if not appears:
        print(
            "CRITICAL: [num] token not appearing in input_ids — "
            "replacement is happening after tokenization, not before"
        )

    verify_text = "laptop storage over [num] GB"
    verify_ids = tokenizer(verify_text, add_special_tokens=True)["input_ids"]
    verify_num_id = tokenizer.convert_tokens_to_ids("[num]")
    assert verify_num_id in verify_ids, "FAIL"
    assert verify_num_id != 100, "FAIL — [num] mapped to [UNK]"
    print(f"PASS — [num] token id: {verify_num_id}, position: {verify_ids.index(verify_num_id)}")
