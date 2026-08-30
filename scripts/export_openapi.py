#!/usr/bin/env python
"""Export the OpenAPI schema and a sample response to files.

Written for handover. A backend developer integrating this service should not
have to run it just to see the contract: they can import ``openapi.json`` into
Postman or Insomnia, generate a typed client from it, and code against
``sample-response.json`` before the service is even deployed.

Usage::

    python scripts/export_openapi.py                 # writes to contract/
    python scripts/export_openapi.py --out-dir docs/contract
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

#: The fixture whose output is exported as the worked example. A clean receipt
#: with every major field populated makes the most useful reference.
SAMPLE_FIXTURE = "001_grocery_us"


def export_openapi(out_dir: Path) -> Path:
    """Write the OpenAPI schema without starting a server."""
    from app.core.config import Settings
    from app.main import create_app

    settings = Settings(app_env="development", ocr_provider="fixture", log_level="CRITICAL")
    schema = create_app(settings).openapi()

    path = out_dir / "openapi.json"
    path.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def export_sample_response(out_dir: Path) -> Path:
    """Write a complete, realistic success response.

    Built from a recorded fixture so it is genuine pipeline output rather than
    a hand-written example that could drift from what the service returns.
    """
    from generate_golden import build_expected

    receipt = build_expected(SAMPLE_FIXTURE)
    envelope = {
        "success": True,
        "data": receipt,
        "warnings": [],
        "errors": [],
        "processing": {
            "request_id": "3f9a2c18-0b7e-4d5a-9c31-2e6f8b4a1d07",
            "api_version": "v1",
            "pipeline_version": receipt["schema_version"] and "1.0.0",
            "schema_version": receipt["schema_version"],
            "processing_time_ms": 812.4,
        },
    }

    path = out_dir / "sample-response.json"
    path.write_text(json.dumps(envelope, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def export_error_response(out_dir: Path) -> Path:
    """Write a representative failure envelope."""
    payload = {
        "success": False,
        "data": None,
        "warnings": [],
        "errors": [
            {
                "code": "UNSUPPORTED_FILE_TYPE",
                "message": "File content does not match any supported image format.",
                "field": None,
                "details": None,
            }
        ],
        "processing": {
            "request_id": "3f9a2c18-0b7e-4d5a-9c31-2e6f8b4a1d07",
            "api_version": "v1",
            "pipeline_version": "1.0.0",
            "schema_version": "1.0",
            "processing_time_ms": 3.1,
        },
    }
    path = out_dir / "sample-error.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="contract",
        help="Directory to write into (default: contract/).",
    )
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in (
        export_openapi(out_dir),
        export_sample_response(out_dir),
        export_error_response(out_dir),
    ):
        print(f"wrote {path.relative_to(REPO_ROOT)}  ({path.stat().st_size:,} bytes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
