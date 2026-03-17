"""Synthetic quantity-aware retrieval triples for DeepQuant baseline training."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.msmarco_loader import verify_hard_negatives
from src.encoding.cqe_wrapper import replace_with_num_tokens_regex

QUERY_TEMPLATES = [
    "product with {concept} greater than {threshold} {unit}",
    "find items where {concept} exceeds {threshold} {unit}",
    "show {concept} above {threshold} {unit}",
    "{concept} over {threshold} {unit}",
]

POSITIVE_TEMPLATES = [
    "This product has {concept}: {value} {unit}.",
    "Specification sheet lists {concept} at {value} {unit}.",
    "Rated {concept}: {value} {unit}.",
]

LEXICAL_NEGATIVE_TEMPLATES = [
    "This product lists {concept}: {value} {unit}, close to the advertised target.",
    "Spec summary shows {concept} at {value} {unit} with the same feature set.",
    "The catalog highlights {concept} of {value} {unit} for this item.",
]

TARGET_NEGATIVES = 7

CONCEPT_SPECS: dict[str, dict[str, Any]] = {
    "battery life": {
        "units": {"hours": 1.0, "hr": 1.0, "minutes": 1.0 / 60.0},
        "threshold_range": (4.0, 18.0),
    },
    "storage capacity": {
        "units": {"GB": 1.0, "gigabytes": 1.0, "MB": 1.0 / 1024.0, "TB": 1024.0},
        "threshold_range": (64.0, 512.0),
    },
    "weight": {
        "units": {"kg": 1.0, "lbs": 0.453592, "grams": 0.001},
        "threshold_range": (0.5, 25.0),
    },
    "revenue": {
        "units": {"million": 1.0, "billion": 1000.0},
        "threshold_range": (25.0, 5000.0),
    },
    "price": {
        "units": {"dollars": 1.0, "USD": 1.0, "cents": 0.01},
        "threshold_range": (50.0, 2500.0),
    },
    "energy": {
        "units": {"kcal": 1.0, "calories": 1.0, "kJ": 1.0 / 4.184},
        "threshold_range": (80.0, 1200.0),
    },
    "speed": {
        "units": {"mph": 1.0, "km/h": 1.0 / 1.60934, "GHz": 1.0},
        "threshold_range": (20.0, 180.0),
        "convertible_units": ["mph", "km/h"],
    },
    "memory": {
        "units": {"GB": 1.0, "MB": 1.0 / 1024.0, "TB": 1024.0},
        "threshold_range": (4.0, 256.0),
    },
}


def _format_number(value: float) -> str:
    """Format floats compactly for text generation."""
    formatted = f"{value:.1f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def _save_jsonl(rows: Sequence[dict[str, Any]], path: Path) -> None:
    """Write rows to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows from disk."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _convert_units(value: float, from_unit: str, to_unit: str, spec: dict[str, Any]) -> float:
    """Convert values between units within one concept family."""
    factors = spec["units"]
    canonical = float(value) * float(factors[from_unit])
    return canonical / float(factors[to_unit])


def _choose_query_unit(spec: dict[str, Any], rng: random.Random) -> str:
    """Choose the query unit for one concept."""
    candidates = list(spec.get("convertible_units", spec["units"].keys()))
    return str(rng.choice(candidates))


def _choose_wrong_unit(query_unit: str, spec: dict[str, Any], rng: random.Random) -> str | None:
    """Choose a different compatible unit for a wrong-unit negative."""
    candidates = [
        str(unit)
        for unit in spec.get("convertible_units", spec["units"].keys())
        if str(unit) != query_unit and float(spec["units"][unit]) != float(spec["units"][query_unit])
    ]
    if not candidates:
        return None
    return str(rng.choice(candidates))


def _build_negative_text(concept: str, value: float, unit: str, rng: random.Random) -> str:
    """Build a lexical negative sentence."""
    template = rng.choice(LEXICAL_NEGATIVE_TEMPLATES)
    return template.format(concept=concept, value=_format_number(value), unit=unit)


def generate_numeric_triples(n: int = 10000, seed: int = 42) -> list[dict[str, Any]]:
    """Generate synthetic quantity-aware retrieval triples."""
    rng = random.Random(seed)
    concepts = list(CONCEPT_SPECS.keys())
    triples: list[dict[str, Any]] = []

    for idx in range(int(n)):
        concept = str(rng.choice(concepts))
        spec = CONCEPT_SPECS[concept]
        query_unit = _choose_query_unit(spec, rng)
        threshold = rng.uniform(*spec["threshold_range"])
        positive_value = threshold * rng.uniform(1.1, 3.0)

        query_raw = rng.choice(QUERY_TEMPLATES).format(
            concept=concept,
            threshold=_format_number(threshold),
            unit=query_unit,
        )
        positive_raw = rng.choice(POSITIVE_TEMPLATES).format(
            concept=concept,
            value=_format_number(positive_value),
            unit=query_unit,
        )

        threshold_violation_negs: list[str] = []
        for _ in range(2):
            negative_value = threshold * rng.uniform(0.3, 0.9)
            threshold_violation_negs.append(
                rng.choice(POSITIVE_TEMPLATES).format(
                    concept=concept,
                    value=_format_number(negative_value),
                    unit=query_unit,
                )
            )

        wrong_unit_negs: list[str] = []
        wrong_unit = _choose_wrong_unit(query_unit, spec, rng)
        if wrong_unit is not None:
            for _ in range(2):
                wrong_canonical = threshold * float(spec["units"][query_unit]) * rng.uniform(0.4, 0.95)
                wrong_value = wrong_canonical / float(spec["units"][wrong_unit])
                wrong_unit_negs.append(
                    f"Technical sheet records {concept}: {_format_number(wrong_value)} {wrong_unit}."
                )
        else:
            for _ in range(2):
                fallback_value = threshold * rng.uniform(0.35, 0.85)
                wrong_unit_negs.append(
                    f"Technical sheet records {concept}: {_format_number(fallback_value)} {query_unit}."
                )

        lexical_negs: list[str] = []
        for _ in range(3):
            lexical_value = threshold * rng.uniform(0.5, 0.98)
            lexical_negs.append(_build_negative_text(concept, lexical_value, query_unit, rng))

        query_text = replace_with_num_tokens_regex(query_raw)
        pos_doc_text = replace_with_num_tokens_regex(positive_raw)
        neg_doc_texts = [
            replace_with_num_tokens_regex(text)
            for text in [*threshold_violation_negs, *wrong_unit_negs, *lexical_negs]
        ]
        neg_doc_texts = neg_doc_texts[:TARGET_NEGATIVES]

        triples.append(
            {
                "id": f"synthetic:{idx}",
                "source": "synthetic",
                "query_text": query_text,
                "query_spans": [],
                "pos_doc_text": pos_doc_text,
                "pos_doc_spans": [],
                "neg_doc_texts": neg_doc_texts,
                "neg_doc_spans": [[] for _ in neg_doc_texts],
                "concept": concept,
                "query_unit": query_unit,
            }
        )

    return triples


def _split_rows(rows: Sequence[dict[str, Any]], seed: int = 42) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows into train/dev/test using 80/10/10."""
    shuffled = list(rows)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_n = max(1, int(0.8 * total))
    remaining = max(2, total - train_n)
    dev_n = max(1, remaining // 2)
    test_n = max(1, total - train_n - dev_n)
    return (
        shuffled[:train_n],
        shuffled[train_n : train_n + dev_n],
        shuffled[train_n + dev_n : train_n + dev_n + test_n],
    )


def _mix_with_ratio(
    msmarco_rows: Sequence[dict[str, Any]],
    synthetic_rows: Sequence[dict[str, Any]],
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Mix two sources into 80% MS MARCO and 20% synthetic."""
    rng = random.Random(seed)
    msmarco_pool = list(msmarco_rows)
    synthetic_pool = list(synthetic_rows)
    rng.shuffle(msmarco_pool)
    rng.shuffle(synthetic_pool)

    synthetic_target = min(len(synthetic_pool), max(1, len(msmarco_pool) // 4))
    msmarco_target = min(len(msmarco_pool), synthetic_target * 4)
    synthetic_target = min(len(synthetic_pool), max(1, msmarco_target // 4))

    msmarco_selected = [
        {**row, "source": "msmarco"}
        for row in msmarco_pool[:msmarco_target]
    ]
    synthetic_selected = [
        {**row, "source": "synthetic"}
        for row in synthetic_pool[:synthetic_target]
    ]
    combined = [*msmarco_selected, *synthetic_selected]
    rng.shuffle(combined)
    return combined, msmarco_selected, synthetic_selected


def _summarize_source_mix(rows: Sequence[dict[str, Any]]) -> tuple[float, float]:
    """Return source percentages for combined rows."""
    total = max(len(rows), 1)
    msmarco_pct = 100.0 * sum(str(row.get("source", "")) == "msmarco" for row in rows) / float(total)
    synthetic_pct = 100.0 * sum(str(row.get("source", "")) == "synthetic" for row in rows) / float(total)
    return msmarco_pct, synthetic_pct


def build_combined_dataset(
    msmarco_dir: str = "data/msmarco",
    synthetic_n: int = 10000,
    output_dir: str = "data/combined",
    seed: int = 42,
) -> dict[str, int]:
    """Build a combined MS MARCO + synthetic quantity dataset."""
    msmarco_path = Path(msmarco_dir)
    train_path = msmarco_path / "train.jsonl"
    dev_path = msmarco_path / "dev.jsonl"
    test_path = msmarco_path / "test.jsonl"
    for path in (train_path, dev_path, test_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing MS MARCO split at {path}. "
                "Run src/data/msmarco_loader.py first."
            )

    msmarco_train = _load_jsonl(train_path)
    msmarco_dev = _load_jsonl(dev_path)
    msmarco_test = _load_jsonl(test_path)

    synthetic_rows = generate_numeric_triples(n=synthetic_n, seed=seed)
    synthetic_train, synthetic_dev, synthetic_test = _split_rows(synthetic_rows, seed=seed)

    combined_train, train_msmarco_only, train_synthetic_only = _mix_with_ratio(msmarco_train, synthetic_train, seed=seed)
    combined_dev, dev_msmarco_only, dev_synthetic_only = _mix_with_ratio(msmarco_dev, synthetic_dev, seed=seed + 1)
    combined_test, test_msmarco_only, test_synthetic_only = _mix_with_ratio(msmarco_test, synthetic_test, seed=seed + 2)

    output_path = Path(output_dir)
    _save_jsonl(combined_train, output_path / "train.jsonl")
    _save_jsonl(combined_dev, output_path / "dev.jsonl")
    _save_jsonl(combined_test, output_path / "test.jsonl")
    _save_jsonl(dev_msmarco_only, output_path / "dev_msmarco.jsonl")
    _save_jsonl(dev_synthetic_only, output_path / "dev_synthetic.jsonl")
    _save_jsonl(test_msmarco_only, output_path / "test_msmarco.jsonl")
    _save_jsonl(test_synthetic_only, output_path / "test_synthetic.jsonl")
    _save_jsonl(train_synthetic_only, output_path / "train_synthetic.jsonl")

    msmarco_pct, synthetic_pct = _summarize_source_mix(combined_train)
    total_train = len(combined_train)
    total_dev = len(combined_dev)
    total_test = len(combined_test)
    pct_query_has_num = 100.0 * sum("[num]" in row["query_text"] for row in combined_train) / float(max(total_train, 1))
    pct_pos_has_num = 100.0 * sum("[num]" in row["pos_doc_text"] for row in combined_train) / float(max(total_train, 1))

    print(f"Total triples: train={total_train} dev={total_dev} test={total_test}")
    print(f"% from MS MARCO vs synthetic: {msmarco_pct:.2f}% / {synthetic_pct:.2f}%")
    print(f"% with [num] in query: {pct_query_has_num:.2f}")
    print(f"% with [num] in positive doc: {pct_pos_has_num:.2f}")
    verify_hard_negatives(combined_dev, sample_n=min(100, len(combined_dev)), seed=seed)

    return {
        "train": total_train,
        "dev": total_dev,
        "test": total_test,
        "dev_msmarco": len(dev_msmarco_only),
        "dev_synthetic": len(dev_synthetic_only),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build synthetic quantity triples and combined datasets")
    parser.add_argument("--synthetic_n", type=int, default=10000)
    parser.add_argument("--msmarco_dir", type=str, default="data/msmarco")
    parser.add_argument("--output_dir", type=str, default="data/combined")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    counts = build_combined_dataset(
        msmarco_dir=args.msmarco_dir,
        synthetic_n=args.synthetic_n,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(
        "Saved combined dataset:",
        f"train={counts['train']}, dev={counts['dev']}, test={counts['test']}, "
        f"dev_msmarco={counts['dev_msmarco']}, dev_synthetic={counts['dev_synthetic']}",
    )
