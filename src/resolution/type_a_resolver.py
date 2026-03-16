"""Type A additive resolver for WideQuant decomposed quantities."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.encoding.cqe_wrapper import QuantitySpan, ResolvedCandidate
from src.resolution.conversion_table import convert, normalize_unit

ADDITIVE_SUBCONCEPT_VOCABULARY: dict[str, list[str]] = {
    "battery": [
        "screen_on_time",
        "active_battery",
        "usage_time",
        "talk_time",
        "standby_time",
    ],
    "storage": [
        "ssd_storage",
        "hdd_storage",
        "internal_storage",
        "expandable_storage",
    ],
    "revenue": [
        "q1_revenue",
        "q2_revenue",
        "q3_revenue",
        "q4_revenue",
    ],
    "energy_kcal": [
        "energy_from_protein",
        "energy_from_fat",
        "energy_from_carbs",
    ],
    "calories": [
        "protein_calories",
        "fat_calories",
        "carb_calories",
    ],
}


def _span_value(span: QuantitySpan) -> float:
    """Recover a scalar quantity value from mantissa/exponent representation."""
    return float(span.mantissa) * (10.0 ** int(span.exponent))


def _mantissa_exponent(value: float) -> tuple[float, int]:
    """Convert a scalar value into WideQuant mantissa/exponent form."""
    if abs(value) < 1e-12:
        return 0.0, 0
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = float(value / (10.0 ** exponent))
    mantissa = max(-10.0, min(10.0, mantissa))
    return mantissa, exponent


class TypeAResolver:
    """Resolve additive multi-span quantities into a single candidate."""

    def identify_subconcepts(self, quantities: list[QuantitySpan], query_concept: str) -> list[QuantitySpan]:
        """Return spans whose concepts match additive subconcepts for the query concept."""
        normalized_query_concept = str(query_concept).strip().lower()
        allowed = {
            concept.lower()
            for concept in ADDITIVE_SUBCONCEPT_VOCABULARY.get(normalized_query_concept, [])
        }
        if not allowed:
            return []

        matches: list[QuantitySpan] = []
        for span in quantities:
            if str(span.concept).strip().lower() in allowed:
                matches.append(span)
        return matches

    def resolve(
        self,
        quantities: list[QuantitySpan],
        query_unit: str,
        query_concept: str,
    ) -> Optional[ResolvedCandidate]:
        """Resolve additive quantity spans into one candidate in the query unit."""
        subconcept_spans = self.identify_subconcepts(quantities, query_concept)
        if len(subconcept_spans) < 2:
            return None

        normalized_values: list[float] = []
        for span in subconcept_spans:
            converted = convert(_span_value(span), span.unit, query_unit)
            if converted is None:
                return None
            normalized_values.append(float(converted))

        resolved_value = float(sum(normalized_values))
        mantissa, exponent = _mantissa_exponent(resolved_value)
        return ResolvedCandidate(
            value=resolved_value,
            unit=normalize_unit(query_unit),
            mantissa=mantissa,
            exponent=exponent,
            source_type="TYPE_A",
            source_spans=subconcept_spans,
        )


if __name__ == "__main__":
    resolver = TypeAResolver()

    battery_spans = [
        QuantitySpan(
            text="8",
            mantissa=8.0,
            exponent=0,
            unit="hr",
            concept="screen_on_time",
            start_char=0,
            end_char=1,
        ),
        QuantitySpan(
            text="120",
            mantissa=1.2,
            exponent=2,
            unit="hr",
            concept="standby_time",
            start_char=2,
            end_char=5,
        ),
    ]
    battery_candidate = resolver.resolve(battery_spans, query_unit="hr", query_concept="battery")
    battery_pass = battery_candidate is not None and abs(battery_candidate.value - 128.0) <= 1e-6
    print(
        f"Test 1 - Battery: {'PASS' if battery_pass else 'FAIL'} "
        f"(value={None if battery_candidate is None else battery_candidate.value})"
    )

    storage_spans = [
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
    storage_candidate = resolver.resolve(storage_spans, query_unit="GB", query_concept="storage")
    storage_pass = storage_candidate is not None and abs(storage_candidate.value - 1256.0) <= 1e-6
    print(
        f"Test 2 - Storage: {'PASS' if storage_pass else 'FAIL'} "
        f"(value={None if storage_candidate is None else storage_candidate.value})"
    )

    energy_spans = [
        QuantitySpan(
            text="80",
            mantissa=8.0,
            exponent=1,
            unit="kcal",
            concept="energy_from_protein",
            start_char=0,
            end_char=2,
        ),
        QuantitySpan(
            text="90",
            mantissa=9.0,
            exponent=1,
            unit="kcal",
            concept="energy_from_fat",
            start_char=3,
            end_char=5,
        ),
        QuantitySpan(
            text="120",
            mantissa=1.2,
            exponent=2,
            unit="kcal",
            concept="energy_from_carbs",
            start_char=6,
            end_char=9,
        ),
    ]
    energy_candidate = resolver.resolve(energy_spans, query_unit="kcal", query_concept="energy_kcal")
    energy_pass = energy_candidate is not None and abs(energy_candidate.value - 290.0) <= 1e-6
    print(
        f"Test 3 - Energy: {'PASS' if energy_pass else 'FAIL'} "
        f"(value={None if energy_candidate is None else energy_candidate.value})"
    )

    overall = battery_pass and storage_pass and energy_pass
    print(f"OVERALL {'PASS' if overall else 'FAIL'}")
