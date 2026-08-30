"""HTTP API tests.

Covers the response contract a backend developer depends on: envelope shape,
status codes, correlation ids, and the guarantee that no internal detail
escapes through an error.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.versions import API_VERSION, PIPELINE_VERSION, SCHEMA_VERSION
from app.main import create_app

EXTRACT_URL = f"/api/{API_VERSION}/receipts/extract"


@pytest.fixture
def client(settings):
    """A test client whose lifespan has run (so the pipeline is built)."""
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def _upload(image: bytes, fixture: str = "001_grocery_us", filename: str = "receipt.png"):
    return {
        "files": {"image": (filename, image, "image/png")},
        "data": {"fixture": fixture},
    }


# ------------------------------------------------------------- operational
def test_health_is_a_liveness_probe(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_reports_component_status(client) -> None:
    response = client.get("/ready")
    body = response.json()
    assert response.status_code == 200
    assert body["ready"] is True
    assert any(c["name"].startswith("ocr:") for c in body["components"])


def test_version_reports_all_three_versions(client) -> None:
    body = client.get("/version").json()
    assert body["api_version"] == API_VERSION
    assert body["pipeline_version"] == PIPELINE_VERSION
    assert body["schema_version"] == SCHEMA_VERSION


def test_metrics_snapshot_is_available(client, receipt_image) -> None:
    client.post(EXTRACT_URL, **_upload(receipt_image))
    body = client.get("/metrics").json()
    assert "counters" in body
    assert any("pipeline" in key for key in body["counters"])


# ------------------------------------------------------------- happy path
def test_extract_returns_the_success_envelope(client, receipt_image) -> None:
    response = client.post(EXTRACT_URL, **_upload(receipt_image))
    assert response.status_code == 200

    body = response.json()
    assert set(body) == {"success", "data", "warnings", "errors", "processing"}
    assert body["success"] is True
    assert body["errors"] == []
    assert body["data"]["total"] == "17.28"
    assert body["data"]["merchant"]["name"] == "GREEN VALLEY MARKET"


def test_response_carries_versions_and_request_id(client, receipt_image) -> None:
    response = client.post(EXTRACT_URL, **_upload(receipt_image))
    processing = response.json()["processing"]

    assert processing["schema_version"] == SCHEMA_VERSION
    assert processing["pipeline_version"] == PIPELINE_VERSION
    assert processing["request_id"] == response.headers["X-Request-ID"]


def test_client_request_id_is_honoured(client, receipt_image) -> None:
    upload = _upload(receipt_image)
    response = client.post(EXTRACT_URL, headers={"X-Request-ID": "trace-abc-123"}, **upload)
    assert response.json()["processing"]["request_id"] == "trace-abc-123"


def test_client_request_id_is_sanitised(client, receipt_image) -> None:
    """An unbounded client string in a log field is an injection vector."""
    upload = _upload(receipt_image)
    response = client.post(EXTRACT_URL, headers={"X-Request-ID": "x" * 500}, **upload)
    assert len(response.json()["processing"]["request_id"]) <= 128


def test_absent_fields_are_null_in_json(client, receipt_image) -> None:
    response = client.post(EXTRACT_URL, **_upload(receipt_image, "006_minimal_no_geometry"))
    data = response.json()["data"]
    assert data["merchant"]["phone"] is None
    assert data["transaction"]["date"] is None


def test_uncertain_result_is_still_a_success(client, receipt_image) -> None:
    """success answers 'did it process', not 'is it certain'."""
    response = client.post(EXTRACT_URL, **_upload(receipt_image, "005_total_mismatch"))
    body = response.json()

    assert response.status_code == 200
    assert body["success"] is True
    assert any(w["code"] == "TOTAL_MISMATCH" for w in body["warnings"])
    assert body["data"]["validation"]["is_valid"] is False
    assert body["data"]["review"]["review_required"] is True


def test_warnings_name_the_field_they_concern(client, receipt_image) -> None:
    response = client.post(EXTRACT_URL, **_upload(receipt_image, "003_ambiguous"))
    warnings = response.json()["warnings"]
    ambiguous = [w for w in warnings if w["code"] == "AMBIGUOUS_DATE_FORMAT"]
    assert ambiguous
    assert ambiguous[0]["field"] == "transaction.date"


# ------------------------------------------------------------------ errors
@pytest.mark.parametrize(
    ("content", "status", "code"),
    [
        (b"not an image", 415, "UNSUPPORTED_FILE_TYPE"),
        (b"", 400, "EMPTY_FILE"),
    ],
)
def test_error_envelope(client, content: bytes, status: int, code: str) -> None:
    response = client.post(EXTRACT_URL, files={"image": ("x.png", content, "image/png")})
    body = response.json()

    assert response.status_code == status
    assert body["success"] is False
    assert body["data"] is None
    assert body["errors"][0]["code"] == code
    assert body["processing"]["request_id"]


def test_oversized_upload_is_rejected(client, make_settings) -> None:
    settings = make_settings(max_file_size_mb=0.01)
    with TestClient(create_app(settings)) as small_client:
        payload = b"\x89PNG\r\n\x1a\n" + b"0" * (200 * 1024)
        response = small_client.post(
            EXTRACT_URL, files={"image": ("big.png", payload, "image/png")}
        )
    assert response.status_code == 413
    assert response.json()["errors"][0]["code"] == "IMAGE_TOO_LARGE"


def test_undersized_image_is_rejected(client) -> None:
    tiny = cv2.imencode(".png", np.full((30, 30, 3), 255, np.uint8))[1].tobytes()
    response = client.post(EXTRACT_URL, files={"image": ("t.png", tiny, "image/png")})
    assert response.status_code == 422
    assert response.json()["errors"][0]["code"] == "IMAGE_TOO_SMALL"


def test_missing_file_field_is_rejected(client) -> None:
    response = client.post(EXTRACT_URL)
    assert response.status_code == 422
    assert response.json()["success"] is False


def test_errors_never_leak_internals(client) -> None:
    """No stack traces, no file paths, no module names."""
    response = client.post(EXTRACT_URL, files={"image": ("x.png", b"not an image", "image/png")})
    rendered = response.text

    for leak in ("Traceback", "site-packages", "app/", "app\\\\", ".py", "__"):
        assert leak not in rendered, f"response leaked {leak!r}"


def test_error_details_hidden_unless_debug_enabled(client, make_settings) -> None:
    response = client.post(EXTRACT_URL, files={"image": ("x.png", b"not an image", "image/png")})
    assert response.json()["errors"][0]["details"] is None

    debug_settings = make_settings(debug_errors=True)
    with TestClient(create_app(debug_settings)) as debug_client:
        debug_response = debug_client.post(
            EXTRACT_URL, files={"image": ("x.png", b"not an image", "image/png")}
        )
    assert debug_response.json()["errors"][0]["details"] is not None


def test_path_traversal_filename_is_harmless(client, receipt_image) -> None:
    response = client.post(
        EXTRACT_URL,
        files={"image": ("../../etc/passwd.png", receipt_image, "image/png")},
        data={"fixture": "001_grocery_us"},
    )
    assert response.status_code == 200


# ------------------------------------------------------------------ schema
def test_openapi_document_is_generated(client) -> None:
    schema = client.get("/openapi.json").json()
    assert f"/api/{API_VERSION}/receipts/extract" in schema["paths"]
    assert "Receipt" in schema["components"]["schemas"]


def test_endpoints_are_versioned(client, receipt_image) -> None:
    assert client.post("/receipts/extract", **_upload(receipt_image)).status_code == 404
    assert client.post(EXTRACT_URL, **_upload(receipt_image)).status_code == 200
