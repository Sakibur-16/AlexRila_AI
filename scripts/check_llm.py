#!/usr/bin/env python
"""Verify the configured model names against the provider.

Model names change faster than code does. A name that was current when the
default was written can be retired months later, and the failure surfaces at
the first production request rather than at deploy time. This asks the
provider what it actually offers and checks the configured names against it.

Usage::

    python scripts/check_llm.py              # check LLM_MODEL and VISION_MODEL
    python scripts/check_llm.py --list       # also list every available model
    python scripts/check_llm.py --test       # make one real call to each

Requires ``LLM_API_KEY`` (and ``VISION_API_KEY`` for the vision check) in the
environment or ``.env``. Exit code 1 when a configured model is unavailable,
so this can gate a deploy.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import suppress
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BASE_URL = "https://api.openai.com/v1"


def _list_models(base_url: str, api_key: str) -> list[str]:
    """Ask the provider which models the key can use."""
    import httpx

    with httpx.Client(timeout=30.0) as client:
        response = client.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
    if response.status_code == 401:
        raise SystemExit("  FAIL  the API key was rejected (401).")
    if response.status_code >= 400:
        raise SystemExit(f"  FAIL  provider returned HTTP {response.status_code}.")

    payload = response.json()
    return sorted(entry["id"] for entry in payload.get("data", []) if "id" in entry)


def _closest(name: str, available: list[str], limit: int = 5) -> list[str]:
    """Suggest plausible alternatives for a name that was not found."""
    import difflib

    close = difflib.get_close_matches(name, available, n=limit, cutoff=0.4)
    if close:
        return close
    # Fall back to a prefix match: "gpt-5.1-mini" -> anything starting "gpt-5".
    stem = name.split("-")[0]
    return [candidate for candidate in available if candidate.startswith(stem)][:limit]


def _check(label: str, model: str, available: list[str]) -> bool:
    if not model:
        print(f"  skip  {label}: not configured")
        return True
    if model in available:
        print(f"  ok    {label}: {model}")
        return True
    print(f"  FAIL  {label}: {model!r} is not available to this key")
    suggestions = _closest(model, available)
    if suggestions:
        print(f"        did you mean: {', '.join(suggestions)}")
    return False


def _smoke_test(base_url: str, api_key: str, model: str) -> bool:
    """Make one minimal call, because availability is not the same as usable."""
    import httpx

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Reply with the word: ok"}],
                    "max_tokens": 16,
                },
            )
    except httpx.HTTPError as exc:
        print(f"  FAIL  {model}: {type(exc).__name__}")
        return False

    if response.status_code >= 400:
        detail = ""
        with suppress(ValueError, AttributeError):
            detail = response.json().get("error", {}).get("message", "")[:160]
        print(f"  FAIL  {model}: HTTP {response.status_code} {detail}")
        return False

    try:
        reply = response.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, ValueError):
        print(f"  FAIL  {model}: unexpected response shape")
        return False

    print(f"  ok    {model}: replied {reply[:40]!r}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List every available model.")
    parser.add_argument("--test", action="store_true", help="Make one real call per model.")
    args = parser.parse_args()

    from app.core.config import get_settings

    settings = get_settings()
    llm_key = settings.llm_api_key.get_secret_value()
    vision_key = settings.vision_api_key.get_secret_value()

    if not llm_key and not vision_key:
        print("No API key configured. Set LLM_API_KEY (and VISION_API_KEY) in .env.")
        return 1

    base_url = settings.llm_base_url or DEFAULT_BASE_URL
    key = llm_key or vision_key

    print(f"provider: {base_url}")
    available = _list_models(base_url, key)
    print(f"models available to this key: {len(available)}\n")

    if args.list:
        for name in available:
            print(f"    {name}")
        print()

    ok = True
    ok &= _check("LLM_MODEL   ", settings.llm_model, available)
    ok &= _check("VISION_MODEL", settings.vision_model, available)

    if args.test:
        print("\nlive calls:")
        for model in {settings.llm_model, settings.vision_model} - {""}:
            ok &= _smoke_test(base_url, key, model)

    if not ok:
        print(
            "\nOne or more configured models are unavailable. Update LLM_MODEL / "
            "VISION_MODEL in .env, or run with --list to see the full catalogue."
        )
        return 1

    print("\nAll configured models are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
