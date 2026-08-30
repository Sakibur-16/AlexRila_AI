"""API-key authentication tests.

The property that makes this safe to ship: enforcement is *opt-in*. With no
key configured the service behaves exactly as before, so adding this to a
running deployment changes nothing until an operator sets ``API_KEY``.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.security import API_KEY_HEADER
from app.core.versions import API_VERSION
from app.main import create_app

EXTRACT_URL = f"/api/{API_VERSION}/receipts/extract"
SECRET = "test-secret-key-abc123"


@pytest.fixture
def secured_client(make_settings):
    """A client for an app that requires an API key."""
    with TestClient(create_app(make_settings(api_key=SECRET))) as client:
        yield client


@pytest.fixture
def open_client(settings):
    """A client for an app with no key configured (the default)."""
    with TestClient(create_app(settings)) as client:
        yield client


def _upload(image: bytes):
    return {
        "files": {"image": ("receipt.png", image, "image/png")},
        "data": {"fixture": "001_grocery_us"},
    }


# ------------------------------------------------------------- opt-in default
def test_no_key_configured_means_no_authentication(open_client, receipt_image) -> None:
    """Existing deployments must not break when this code lands."""
    response = open_client.post(EXTRACT_URL, **_upload(receipt_image))
    assert response.status_code == 200


# ------------------------------------------------------------- enforcement
def test_request_without_a_key_is_rejected(secured_client, receipt_image) -> None:
    response = secured_client.post(EXTRACT_URL, **_upload(receipt_image))
    body = response.json()

    assert response.status_code == 401
    assert body["success"] is False
    assert body["data"] is None
    assert body["errors"][0]["code"] == "UNAUTHORIZED"


def test_request_with_the_wrong_key_is_rejected(secured_client, receipt_image) -> None:
    response = secured_client.post(
        EXTRACT_URL, headers={API_KEY_HEADER: "wrong-key"}, **_upload(receipt_image)
    )
    assert response.status_code == 401


def test_request_with_the_correct_key_succeeds(secured_client, receipt_image) -> None:
    response = secured_client.post(
        EXTRACT_URL, headers={API_KEY_HEADER: SECRET}, **_upload(receipt_image)
    )
    assert response.status_code == 200
    assert response.json()["data"]["total"] == "17.28"


def test_a_key_that_is_a_prefix_of_the_real_one_is_rejected(secured_client, receipt_image) -> None:
    """Guards against a comparison that stops at the first difference."""
    response = secured_client.post(
        EXTRACT_URL, headers={API_KEY_HEADER: SECRET[:-1]}, **_upload(receipt_image)
    )
    assert response.status_code == 401


# ------------------------------------------------------------- probes public
@pytest.mark.parametrize("path", ["/health", "/ready", "/version"])
def test_probes_stay_reachable_without_a_key(secured_client, path: str) -> None:
    """An orchestrator has no credential to present, and these expose nothing."""
    assert secured_client.get(path).status_code in (200, 503)


# ---------------------------------------------------------------- no leakage
def test_the_key_never_appears_in_a_response(secured_client, receipt_image) -> None:
    response = secured_client.post(
        EXTRACT_URL, headers={API_KEY_HEADER: "wrong-key"}, **_upload(receipt_image)
    )
    assert SECRET not in response.text
    assert "wrong-key" not in response.text


def test_the_key_is_not_exposed_by_settings_repr(make_settings) -> None:
    settings = make_settings(api_key=SECRET)
    assert SECRET not in repr(settings)
    assert settings.api_key.get_secret_value() == SECRET


def test_error_envelope_shape_is_unchanged(secured_client, receipt_image) -> None:
    """A 401 uses the same envelope as every other failure."""
    body = secured_client.post(EXTRACT_URL, **_upload(receipt_image)).json()
    assert set(body) == {"success", "data", "warnings", "errors", "processing"}
    assert body["processing"]["request_id"]
