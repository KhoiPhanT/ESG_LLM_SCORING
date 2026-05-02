"""
Shared scoring utility functions.
"""
from __future__ import annotations

import re


def parse_score_value(raw) -> float:
    """Parse a score value from VNSI workbook logic text.

    Handles:
    - Normal numbers: "1", "-0.5", "+0.125"
    - Vietnamese comma-as-decimal: "0,125" → 0.125
    - Missing decimal: "0125" → 0.125
    """
    text = str(raw or "").strip().replace(",", ".")
    if not text:
        return 0.0
    sign = -1.0 if text.startswith("-") else 1.0
    unsigned = text.lstrip("+-")
    if "." not in unsigned and unsigned.startswith("0") and len(unsigned) > 1:
        unsigned = "0." + unsigned.lstrip("0")
    try:
        return sign * float(unsigned)
    except ValueError:
        return 0.0
