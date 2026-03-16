"""Unit conversion utilities for WideQuant arithmetic resolution."""

from __future__ import annotations

from typing import Optional

CONVERSION_TABLE: dict[str, dict[str, float]] = {
    "kj": {
        "kcal": 1.0 / 4.184,
    },
    "kcal": {
        "kj": 4.184,
        "cal": 1000.0,
    },
    "cal": {
        "kcal": 1.0 / 1000.0,
    },
    "g": {
        "kg": 1.0 / 1000.0,
        "lb": 1.0 / 453.592,
        "oz": 1.0 / 28.3495,
    },
    "kg": {
        "g": 1000.0,
    },
    "lb": {
        "g": 453.592,
    },
    "oz": {
        "g": 28.3495,
    },
    "kb": {
        "mb": 1.0 / 1024.0,
    },
    "mb": {
        "kb": 1024.0,
        "gb": 1.0 / 1024.0,
    },
    "gb": {
        "mb": 1024.0,
        "tb": 1.0 / 1024.0,
    },
    "tb": {
        "gb": 1024.0,
    },
    "sec": {
        "min": 1.0 / 60.0,
        "hr": 1.0 / 3600.0,
    },
    "min": {
        "sec": 60.0,
        "hr": 1.0 / 60.0,
    },
    "hr": {
        "min": 60.0,
        "sec": 3600.0,
    },
}

UNIT_ALIASES: dict[str, str] = {
    "kilocalorie": "kcal",
    "kilocalories": "kcal",
    "kilojoule": "kj",
    "kilojoules": "kj",
    "gigabyte": "gb",
    "gigabytes": "gb",
    "megabyte": "mb",
    "megabytes": "mb",
    "terabyte": "tb",
    "terabytes": "tb",
    "hour": "hr",
    "hours": "hr",
    "hrs": "hr",
    "minute": "min",
    "minutes": "min",
    "second": "sec",
    "seconds": "sec",
    "gram": "g",
    "grams": "g",
    "kilogram": "kg",
    "kilograms": "kg",
    "pound": "lb",
    "pounds": "lb",
    "ounce": "oz",
    "ounces": "oz",
}


def normalize_unit(unit: str) -> str:
    """Normalize a unit string for table lookup."""
    normalized = str(unit).strip().lower()
    return UNIT_ALIASES.get(normalized, normalized)


def convert(value: float, from_unit: str, to_unit: str) -> Optional[float]:
    """Convert a scalar value between supported units if a mapping exists."""
    from_norm = normalize_unit(from_unit)
    to_norm = normalize_unit(to_unit)

    if not from_norm or not to_norm:
        return None
    if from_norm == to_norm:
        return float(value)

    direct = CONVERSION_TABLE.get(from_norm, {}).get(to_norm)
    if direct is not None:
        return float(value) * float(direct)

    inverse = CONVERSION_TABLE.get(to_norm, {}).get(from_norm)
    if inverse is not None and inverse != 0.0:
        return float(value) / float(inverse)

    return None


def units_are_compatible(unit_a: str, unit_b: str) -> bool:
    """Return True when a conversion path exists between the two units."""
    return convert(1.0, unit_a, unit_b) is not None


def _check(name: str, actual: Optional[float], expected: float, tolerance: float) -> bool:
    """Print a PASS/FAIL line for one numeric conversion check."""
    if actual is None:
        print(f"{name}: FAIL (conversion returned None)")
        return False
    passed = abs(actual - expected) <= tolerance
    print(
        f"{name}: {'PASS' if passed else 'FAIL'} "
        f"(actual={actual:.4f}, expected={expected:.4f}, tol={tolerance:.4f})"
    )
    return passed


if __name__ == "__main__":
    checks = [
        _check("1046 kJ -> kcal", convert(1046.0, "kJ", "kcal"), 250.0, 1.0),
        _check("500 g -> lb", convert(500.0, "g", "lb"), 1.102, 0.01),
        _check("2048 MB -> GB", convert(2048.0, "MB", "GB"), 2.0, 1e-6),
        _check("120 min -> hr", convert(120.0, "min", "hr"), 2.0, 1e-6),
        _check("kilocalorie alias", convert(1.0, "kilocalorie", "kcal"), 1.0, 1e-6),
    ]
    overall = all(checks)
    print(f"OVERALL {'PASS' if overall else 'FAIL'}")
