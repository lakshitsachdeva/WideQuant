"""Type C unit-conversion resolver for WideQuant decomposed quantities."""

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


class TypeCResolver:
    """Resolve one unit-mismatched quantity by converting it into the query unit.

    After conversion, magnitude changes dramatically. For example, 1046 kJ becomes
    roughly 250 kcal. The mantissa/exponent encoding handles this because it stores
    overall magnitude in the exponent and scaled local value in the mantissa.
    """

    def resolve(
        self,
        quantities: list[QuantitySpan],
        query_unit: str,
    ) -> Optional[ResolvedCandidate]:
        """Convert exactly one compatible non-query-unit span into the query unit."""
        normalized_query_unit = normalize_unit(query_unit)
        non_query_unit_spans = [
            span for span in quantities if normalize_unit(span.unit) != normalized_query_unit
        ]
        if len(non_query_unit_spans) != 1:
            return None

        source_span = non_query_unit_spans[0]
        converted = convert(_span_value(source_span), source_span.unit, normalized_query_unit)
        if converted is None:
            return None

        resolved_value = float(converted)
        mantissa, exponent = _mantissa_exponent(resolved_value)
        return ResolvedCandidate(
            value=resolved_value,
            unit=normalized_query_unit,
            mantissa=mantissa,
            exponent=exponent,
            source_type="TYPE_C",
            source_spans=[source_span],
        )


if __name__ == "__main__":
    resolver = TypeCResolver()

    test_1 = [
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
    candidate_1 = resolver.resolve(test_1, query_unit="kcal")
    pass_1 = candidate_1 is not None and abs(candidate_1.value - 250.15) <= 0.5
    print(
        f"Test 1 - 1046 kJ -> kcal: {'PASS' if pass_1 else 'FAIL'} "
        f"(value={None if candidate_1 is None else candidate_1.value})"
    )

    test_2 = [
        QuantitySpan(
            text="500",
            mantissa=5.0,
            exponent=2,
            unit="g",
            concept="mass",
            start_char=0,
            end_char=3,
        )
    ]
    candidate_2 = resolver.resolve(test_2, query_unit="lb")
    pass_2 = candidate_2 is not None and abs(candidate_2.value - 1.102) <= 0.01
    print(
        f"Test 2 - 500 g -> lb: {'PASS' if pass_2 else 'FAIL'} "
        f"(value={None if candidate_2 is None else candidate_2.value})"
    )

    test_3 = [
        QuantitySpan(
            text="2048",
            mantissa=2.048,
            exponent=3,
            unit="MB",
            concept="storage",
            start_char=0,
            end_char=4,
        )
    ]
    candidate_3 = resolver.resolve(test_3, query_unit="GB")
    pass_3 = candidate_3 is not None and abs(candidate_3.value - 2.0) <= 1e-6
    print(
        f"Test 3 - 2048 MB -> GB: {'PASS' if pass_3 else 'FAIL'} "
        f"(value={None if candidate_3 is None else candidate_3.value})"
    )

    test_4 = [
        QuantitySpan(
            text="1046",
            mantissa=1.046,
            exponent=3,
            unit="kJ",
            concept="energy",
            start_char=0,
            end_char=4,
        ),
        QuantitySpan(
            text="500",
            mantissa=5.0,
            exponent=2,
            unit="g",
            concept="mass",
            start_char=5,
            end_char=8,
        ),
    ]
    candidate_4 = resolver.resolve(test_4, query_unit="kcal")
    pass_4 = candidate_4 is None
    print(f"Test 4 - Multiple non-query units guard: {'PASS' if pass_4 else 'FAIL'}")

    overall = pass_1 and pass_2 and pass_3 and pass_4
    print(f"OVERALL {'PASS' if overall else 'FAIL'}")
