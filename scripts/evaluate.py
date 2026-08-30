#!/usr/bin/env python
"""Run the evaluation harness over the fixture dataset.

Usage::

    python scripts/evaluate.py
    python scripts/evaluate.py --json
    python scripts/evaluate.py --min-exact-match 0.8   # gate CI on accuracy

Reports per-field accuracy, item F1 and the exact-match rate, and lists every
document that failed with the specific fields that differed. Use
``--min-exact-match`` in CI to make an accuracy regression fail the build
rather than merely be noted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from app.evaluation.metrics import EvaluationSummary, score_document  # noqa: E402
from generate_golden import build_expected, expected_path, fixture_names  # noqa: E402


def evaluate() -> EvaluationSummary:
    """Score every fixture whose expectation is committed."""
    summary = EvaluationSummary()

    for name in fixture_names():
        path = expected_path(name)
        if not path.exists():
            print(f"skipping {name}: no expectation committed", file=sys.stderr)
            continue
        expected = json.loads(path.read_text(encoding="utf-8"))
        actual = build_expected(name)
        summary.documents.append(score_document(name, expected, actual))

    return summary


def render(summary: EvaluationSummary) -> str:
    """Render a human-readable report."""
    lines = [
        "Receipt extraction evaluation",
        "=" * 60,
        f"documents        : {len(summary.documents)}",
        f"exact match rate : {summary.exact_match_rate:.1%}",
        f"item F1          : {summary.item_f1:.3f}",
        "",
        "Field accuracy",
        "-" * 60,
    ]
    for name, accuracy in summary.all_field_accuracies().items():
        marker = "ok  " if accuracy >= 0.999 else "WARN"
        lines.append(f"  {marker} {name:24} {accuracy:6.1%}")

    failures = [d for d in summary.documents if not d.exact_match]
    if failures:
        lines += ["", "Documents with differences", "-" * 60]
        for document in failures:
            lines.append(f"  {document.name}")
            for score in document.fields:
                if not score.correct:
                    lines.append(
                        f"      {score.field}: expected {score.expected!r}, got {score.actual!r}"
                    )
            if document.item_f1 < 1.0:
                lines.append(
                    f"      items: F1={document.item_f1:.2f} "
                    f"(expected {document.expected_items}, got {document.actual_items})"
                )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument(
        "--min-exact-match",
        type=float,
        default=None,
        help="Exit non-zero when the exact-match rate falls below this (0-1).",
    )
    args = parser.parse_args()

    summary = evaluate()

    if args.json:
        print(json.dumps(summary.to_dict(), indent=2))
    else:
        print(render(summary))

    if args.min_exact_match is not None and summary.exact_match_rate < args.min_exact_match:
        print(
            f"\nFAIL: exact-match rate {summary.exact_match_rate:.1%} is below the "
            f"required {args.min_exact_match:.1%}.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
