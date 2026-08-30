#!/usr/bin/env python
"""Regenerate golden expectation files from recorded OCR fixtures.

Usage::

    python scripts/generate_golden.py                 # regenerate all
    python scripts/generate_golden.py --fixture 001_grocery_us
    python scripts/generate_golden.py --check         # fail if any drifted

``--check`` is what CI runs: it regenerates in memory and reports any fixture
whose output no longer matches its committed expectation, without writing
anything. Regenerating and committing is a deliberate act that says "this
behaviour change is intended", and the diff belongs in the same commit as the
code that caused it.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "receipts"

#: Fixed "today" so that future-date validation stays reproducible as real
#: time moves on. Without it, every golden file would eventually drift.
REFERENCE_DATE = date(2026, 9, 1)

#: Keys excluded from golden comparison because they vary per run rather than
#: per behaviour. Timing and request ids would make every diff meaningless.
VOLATILE_KEYS = frozenset({"processing"})


def build_expected(name: str) -> dict[str, Any]:
    """Run the pipeline over one fixture and return its comparable output."""
    import numpy as np

    from app.confidence.scorer import ConfidenceScorer
    from app.core.config import Settings
    from app.extraction.receipt import RuleBasedReceiptExtractor
    from app.ocr.base import OCRRequest
    from app.ocr.factory import create_ocr_provider
    from app.pipeline.assembly import build_receipt
    from app.validation.engine import ValidationEngine

    settings = Settings(
        app_env="test",
        ocr_provider="fixture",
        fixture_ocr_dir=str(FIXTURE_DIR),
        log_level="CRITICAL",
    )

    provider = create_ocr_provider("fixture", settings)
    ocr = provider.extract(
        OCRRequest(image=np.zeros((10, 10), dtype=np.uint8), hints={"fixture": name})
    )

    extraction = RuleBasedReceiptExtractor(settings).extract(ocr)
    validation = ValidationEngine(settings).validate(extraction, ocr=ocr, today=REFERENCE_DATE)
    scored = ConfidenceScorer(settings).score(
        extraction, validation, ocr_confidence=ocr.mean_confidence
    )
    receipt = build_receipt(
        extraction,
        settings=settings,
        ocr=ocr,
        validation=validation,
        confidence=scored.report,
        processing=None,
        review_required=scored.review_required,
        review_reasons=scored.review_reasons,
    )

    payload = receipt.model_dump(mode="json")
    return {key: value for key, value in payload.items() if key not in VOLATILE_KEYS}


def fixture_names() -> list[str]:
    return sorted(p.name.removesuffix(".ocr.json") for p in FIXTURE_DIR.glob("*.ocr.json"))


def expected_path(name: str) -> Path:
    return FIXTURE_DIR / f"{name}.expected.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", help="Regenerate only this fixture.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report drift without writing. Exit code 1 when anything drifted.",
    )
    args = parser.parse_args()

    names = [args.fixture] if args.fixture else fixture_names()
    if not names:
        print("No fixtures found.", file=sys.stderr)
        return 1

    drifted: list[str] = []
    for name in names:
        actual = build_expected(name)
        path = expected_path(name)

        if args.check:
            if not path.exists():
                print(f"MISSING  {name}")
                drifted.append(name)
                continue
            committed = json.loads(path.read_text(encoding="utf-8"))
            if committed != actual:
                print(f"DRIFTED  {name}")
                drifted.append(name)
            else:
                print(f"ok       {name}")
            continue

        path.write_text(
            json.dumps(actual, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote    {path.relative_to(REPO_ROOT)}")

    if args.check and drifted:
        print(
            f"\n{len(drifted)} fixture(s) drifted. If the change is intended, run "
            "`python scripts/generate_golden.py` and commit the diff.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
