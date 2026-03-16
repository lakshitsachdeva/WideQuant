"""Type B ratio/division resolver for WideQuant decomposed quantities."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.encoding.cqe_wrapper import QuantitySpan, ResolvedCandidate
from src.resolution.conversion_table import normalize_unit

RATIO_VOCABULARY: list[tuple[str, str, str]] = [
    ("current_revenue", "previous_revenue", "growth_rate"),
    ("share_price", "earnings_per_share", "pe_ratio"),
    ("protein_kcal", "total_kcal", "percentage"),
    ("fat_kcal", "total_kcal", "percentage"),
    ("carb_kcal", "total_kcal", "percentage"),
    ("discount_amount", "original_price", "discount_percentage"),
]


def _span_value(span: QuantitySpan) -> float:
    """Recover a scalar value from WideQuant mantissa/exponent encoding."""
    return float(span.mantissa) * (10.0 ** int(span.exponent))


def _mantissa_exponent(value: float) -> tuple[float, int]:
    """Convert a scalar value into WideQuant mantissa/exponent form."""
    if abs(value) < 1e-12:
        return 0.0, 0
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = float(value / (10.0 ** exponent))
    mantissa = max(-10.0, min(10.0, mantissa))
    return mantissa, exponent


class TypeBResolver:
    """Resolve ratio-based decomposed quantities into a single candidate."""

    def identify_ratio_pair(
        self,
        quantities: list[QuantitySpan],
    ) -> Optional[tuple[QuantitySpan, QuantitySpan, str]]:
        """Return the first matching numerator/denominator concept pair."""
        concepts = {str(span.concept).strip().lower(): span for span in quantities}
        for concept_num, concept_den, formula_type in RATIO_VOCABULARY:
            numerator = concepts.get(concept_num.lower())
            denominator = concepts.get(concept_den.lower())
            if numerator is not None and denominator is not None:
                return numerator, denominator, formula_type
        return None

    def resolve(
        self,
        quantities: list[QuantitySpan],
        query_unit: str,
    ) -> Optional[ResolvedCandidate]:
        """Resolve ratio-compatible quantities into a single candidate."""
        ratio_pair = self.identify_ratio_pair(quantities)
        if ratio_pair is None:
            return None

        numerator, denominator, formula_type = ratio_pair
        num_value = _span_value(numerator)
        den_value = _span_value(denominator)
        if abs(den_value) < 1e-8:
            return None

        if formula_type == "growth_rate":
            resolved_value = (num_value - den_value) / den_value * 100.0
            resolved_unit = "%"
        elif formula_type in {"percentage", "discount_percentage"}:
            resolved_value = num_value / den_value * 100.0
            resolved_unit = "%"
        elif formula_type == "pe_ratio":
            resolved_value = num_value / den_value
            resolved_unit = normalize_unit(query_unit) if str(query_unit).strip() else "x"
        else:
            return None

        if resolved_value > 1000.0:
            resolved_value = 999.0

        mantissa, exponent = _mantissa_exponent(resolved_value)
        return ResolvedCandidate(
            value=float(resolved_value),
            unit=resolved_unit,
            mantissa=mantissa,
            exponent=exponent,
            source_type="TYPE_B",
            source_spans=[numerator, denominator],
        )


if __name__ == "__main__":
    resolver = TypeBResolver()

    growth_spans = [
        QuantitySpan(
            text="$5bn",
            mantissa=5.0,
            exponent=9,
            unit="$",
            concept="current_revenue",
            start_char=0,
            end_char=4,
        ),
        QuantitySpan(
            text="$4bn",
            mantissa=4.0,
            exponent=9,
            unit="$",
            concept="previous_revenue",
            start_char=5,
            end_char=9,
        ),
    ]
    growth_candidate = resolver.resolve(growth_spans, query_unit="%")
    growth_pass = growth_candidate is not None and abs(growth_candidate.value - 25.0) <= 1e-6
    print(
        f"Test 1 - Revenue growth: {'PASS' if growth_pass else 'FAIL'} "
        f"(value={None if growth_candidate is None else growth_candidate.value})"
    )

    pe_spans = [
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
    pe_candidate = resolver.resolve(pe_spans, query_unit="x")
    pe_pass = pe_candidate is not None and abs(pe_candidate.value - 15.0) <= 1e-6
    print(
        f"Test 2 - P/E ratio: {'PASS' if pe_pass else 'FAIL'} "
        f"(value={None if pe_candidate is None else pe_candidate.value})"
    )

    zero_den_spans = [
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
            text="0",
            mantissa=0.0,
            exponent=0,
            unit="$",
            concept="earnings_per_share",
            start_char=5,
            end_char=6,
        ),
    ]
    zero_candidate = resolver.resolve(zero_den_spans, query_unit="x")
    zero_pass = zero_candidate is None
    print(f"Test 3 - Division by zero guard: {'PASS' if zero_pass else 'FAIL'}")

    overall = growth_pass and pe_pass and zero_pass
    print(f"OVERALL {'PASS' if overall else 'FAIL'}")
