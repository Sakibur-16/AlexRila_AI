# syntax=docker/dockerfile:1.7
# =============================================================================
# Multi-stage build.
#
# The builder installs dependencies into a virtualenv; the runtime copies only
# that virtualenv plus the application. Build toolchains never reach the final
# image, which keeps it smaller and removes compilers from the attack surface.
# =============================================================================

# --------------------------------------------------------------------- builder
FROM python:3.12-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# build-essential is needed by some wheels' fallback source builds; it stays in
# this stage only.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependency metadata is copied alone first, so the (slow) install layer is
# cached and only re-runs when dependencies actually change.
COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install --upgrade pip setuptools wheel \
    && pip install .

# --------------------------------------------------------------------- runtime
FROM python:3.12-slim-bookworm AS runtime

# tesseract-ocr is the OCR engine; the -eng data pack is the default language.
# Add more packs (tesseract-ocr-deu, -fra, -spa, -ben, -hin, -ara) to widen
# OCR_LANGUAGES. libgl1/libglib2.0-0 are OpenCV's runtime dependencies.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Run as an unprivileged user. Created with no login shell and no home content
# so a container escape yields as little as possible.
RUN groupadd --system --gid 1001 appuser \
    && useradd --system --uid 1001 --gid appuser --no-create-home --shell /usr/sbin/nologin appuser

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    APP_ENV=production \
    LOG_FORMAT=json

# PORT: Render, Cloud Run, Railway and Heroku all inject the port to bind on.
#   A hardcoded port makes the service unreachable on every one of them.
# WEB_CONCURRENCY: each worker holds its own OpenCV/Tesseract working set
#   (~140 MB idle), so two workers on a 512 MB instance leave almost no
#   headroom for image processing. Tunable per plan without a rebuild.
ENV PORT=8000 \
    WEB_CONCURRENCY=2

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=appuser:appuser app ./app

# Writable location for optional temporary output. Nothing is written here
# unless ENABLE_RAW_OCR_STORAGE is turned on.
RUN mkdir -p /app/data/output && chown -R appuser:appuser /app/data

USER appuser

EXPOSE 8000

# Liveness only -- /ready is checked by the orchestrator, because a failing
# readiness probe should remove the instance from rotation, not restart it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --fail --silent "http://localhost:${PORT}/health" || exit 1

# No secrets are baked in: every credential arrives via the environment.
#
# Shell form so $PORT and $WEB_CONCURRENCY expand at start-up; `exec` replaces
# the shell with uvicorn so it becomes PID 1 and receives SIGTERM directly,
# which is what makes a container shut down promptly instead of being killed.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port \"${PORT}\" --workers \"${WEB_CONCURRENCY}\""]
