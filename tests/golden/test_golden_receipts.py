"""Golden regression tests.

Every fixture's full structured output is compared against a committed
expectation. This is the safety net that makes extraction changes reviewable:
a diff in an expectation file is a diff in what the product returns, and it has
to be justified in the same commit as the code that caused it.

Volatile keys (timings, request ids) are excluded -- see
:data:`scripts.generate_golden.VOLATILE_KEYS` -- so a diff always means a
behaviour change, never a clock reading.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_golden import build_expected, expected_path, fixture_names  # noqa: E402

_NAMES = fixture_names()


@pytest.mark.parametrize("name", _NAMES)
def test_golden_output_matches(name: str) -> None:
    """The pipeline still produces the committed structured output."""
    path = expected_path(name)
    assert path.exists(), f"No expectation for {name}. Run `python scripts/generate_golden.py`."

    expected = json.loads(path.read_text(encoding="utf-8"))
    actual = build_expected(name)

    if actual != expected:
        differences = _describe_differences(expected, actual)
        pytest.fail(
            f"Golden output drifted for {name}:\n{differences}\n\n"
            "If this change is intended, run `python scripts/generate_golden.py` "
            "and commit the updated expectation."
        )


@pytest.mark.parametrize("name", _NAMES)
def test_golden_output_is_schema_valid(name: str) -> None:
    """Regenerated output still satisfies the published receipt schema."""
    from app.schemas.receipt import Receipt

    payload = build_expected(name)
    receipt = Receipt.model_validate(payload)
    assert receipt.schema_version == payload["schema_version"]


def test_fixture_set_covers_the_documented_edge_cases() -> None:
    """The suite must keep exercising the cases it was built to cover."""
    required = {
        "001_grocery_us",  # clean baseline
        "002_eu_multi_vat",  # comma decimals, several tax rates
        "003_ambiguous",  # ambiguous date and currency
        "004_ocr_confusion",  # glyph confusion in amounts
        "005_total_mismatch",  # arithmetic that does not reconcile
        "006_minimal_no_geometry",  # no bounding boxes, low confidence
        "007_restaurant_service",  # service charge, split payment
        "008_pan_redaction",  # full card number on the document
        "009_prompt_injection",  # instruction-like text on the document
    }
    assert required.issubset(set(_NAMES))


def _describe_differences(expected: dict, actual: dict, prefix: str = "") -> str:
    """Render a compact, readable diff of two nested structures."""
    lines: list[str] = []
    for key in sorted(set(expected) | set(actual)):
        path = f"{prefix}{key}"
        left = expected.get(key, "<missing>")
        right = actual.get(key, "<missing>")
        if left == right:
            continue
        if isinstance(left, dict) and isinstance(right, dict):
            lines.append(_describe_differences(left, right, prefix=f"{path}."))
        else:
            lines.append(f"  {path}:\n    expected: {left!r}\n    actual:   {right!r}")
    return "\n".join(line for line in lines if line)
