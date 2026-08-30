# Deployment

## Does the target host need Tesseract?

**With Docker: no action needed.** The image installs `tesseract-ocr` and
`tesseract-ocr-eng` itself, so the container is self-contained. Deploy it
anywhere that runs containers.

**Without Docker: yes, you must install it.** `pip install` pulls in
`pytesseract`, which is only a wrapper around a `tesseract` executable — not
the engine. On a bare VM:

```bash
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng   # Debian/Ubuntu
sudo dnf install -y tesseract                             # RHEL/Fedora
```

Either way the failure mode is safe: a host without the engine still starts the
service, logs `ocr_provider_not_ready` at ERROR, and returns **503** from
`/ready`, so no traffic is routed to it. Verify after any deploy:

```bash
curl -fsS http://<host>:8000/ready
```

## Container

```bash
docker build -t receipt-ocr:1.0.0 .
docker run --rm -p 8000:8000 --env-file .env receipt-ocr:1.0.0
```

The image is multi-stage: the builder installs dependencies into a virtualenv,
and the runtime copies only that virtualenv plus the application. Compilers
never reach the final image, which keeps it smaller and removes them from the
attack surface.

Also in the runtime image: Tesseract with the English pack, OpenCV's runtime
libraries, an unprivileged `appuser` (uid 1001, no login shell), and a
`HEALTHCHECK` on `/health`. No secrets are baked in — every credential arrives
through the environment.

### Adding languages

```dockerfile
RUN apt-get install -y --no-install-recommends \
        tesseract-ocr-deu tesseract-ocr-fra tesseract-ocr-ben
```

Then widen `OCR_LANGUAGES=eng,deu,fra,ben`. Each language adds recognition
time, so list only what you actually process.

## Probes

| Probe | Endpoint | Why this one |
|---|---|---|
| liveness | `/health` | Touches no dependency, so a slow engine cannot trigger a restart loop |
| readiness | `/ready` | Returns 503 when OCR is unusable, removing the instance from rotation |
| startup | `/health` | Give it ~15s |

```yaml
livenessProbe:
  httpGet: { path: /health, port: 8000 }
  initialDelaySeconds: 15
  periodSeconds: 30
readinessProbe:
  httpGet: { path: /ready, port: 8000 }
  initialDelaySeconds: 10
  periodSeconds: 10
```

A deployment missing its engine binary starts, logs `ocr_provider_not_ready`,
and fails readiness. That is deliberate: a container that starts and reports
*why* it cannot serve is far more diagnosable than one that refuses to boot.

## Sizing

OCR is CPU-bound and single-threaded per request. Tesseract on a preprocessed
receipt is typically a few hundred milliseconds to a couple of seconds
depending on resolution and core speed — **measure on your own hardware**
rather than trusting a number in a document.

- Start with `--workers 2` per container and 1–2 vCPU per worker.
- Memory: ~300–500 MB per worker. Uploads are held in memory, so
  `MAX_FILE_SIZE_MB` x concurrency is the floor.
- Scale horizontally. The service is stateless; nothing is shared between
  instances.

`MAX_IMAGE_PIXELS` (40 MP default) is a decompression-bomb guard: a small
compressed file can decode to an enormous array, which the file-size check
alone cannot catch.

## Observability

### Logs

Structured JSON to stderr, one object per event, with `request_id` bound to
every line for a request.

```json
{"event": "pipeline_complete", "request_id": "trace-abc-123",
 "ocr_provider": "tesseract", "ocr_confidence": 0.94,
 "overall_confidence": 0.941, "items": 4, "is_valid": true,
 "review_required": false, "processing_time_ms": 812.4,
 "level": "info", "timestamp": "2026-08-25T14:35:02Z"}
```

Two guarantees: keys matching credential patterns are replaced with
`[REDACTED]` before rendering, and document content never appears unless
`LOG_DOCUMENT_CONTENT=true`. **Keep that false in production** — receipts are
PII.

### Metrics

Bind a real sink at startup:

```python
from app.core.metrics import set_metrics_sink
set_metrics_sink(PrometheusMetricsSink())   # implement MetricsSink
```

Worth alerting on:

| Signal | Why |
|---|---|
| `ocr.failure.total / ocr.success.total` | Engine or provider health |
| `pipeline.latency.ms` p95 / p99 | Latency regression |
| `pipeline.low_confidence.total` rate | Input quality degrading, or a bad deploy |
| `validation.failure.total{code=TOTAL_MISMATCH}` | Extraction accuracy regression |
| `pipeline.review_required.total` rate | Review queue load |
| `llm.invoked.total` rate | Cost, and how often rules are falling short |
| `ocr.retry.total` | Transient provider instability |

A rising low-confidence rate with unchanged code usually means input quality
changed — a new client app compressing more aggressively, for instance. That is
worth catching early.

## Security

### Already implemented

- Type detection from magic bytes; extension and client `Content-Type` are
  never trusted.
- Size enforced **while streaming**, so a hostile 1 GB upload is abandoned
  shortly after it exceeds the limit rather than being buffered first.
- Uploads stay in memory and never touch disk — path traversal and temp-file
  cleanup are not concerns.
- Filenames sanitised (Unicode normalised, directory components stripped) and
  used only for logging.
- Card numbers and labelled CVV/PIN values masked out of OCR text before it is
  returned, stored, or sent to any external provider. Only `card_last4` is ever
  captured.
- Errors expose a stable code and a safe message; stack traces and provider
  error text never leave the process.
- Client-supplied `X-Request-ID` is length-capped and stripped of
  non-printable characters before it reaches a log field or a response header.

### Your responsibility

- **Authentication.** The service has none. Put it behind a gateway.
- **Rate limiting.** Configured but not enforced in-process. Enforce at the
  gateway.
- **TLS.** Terminate upstream.
- **Secrets.** Use your platform's secret manager. Never bake them into the
  image or commit a `.env`.
- **Network egress.** With `LLM_ENABLED=false` (the default) the container
  makes no outbound calls; consider restricting egress to match.

### Privacy checklist

```bash
ENABLE_RAW_OCR_STORAGE=false    # persist nothing (default)
LOG_DOCUMENT_CONTENT=false      # no receipt text in logs (default)
REDACT_SENSITIVE_TEXT=true      # mask card data (default)
DEBUG_ERRORS=false              # no error details to clients (default)
INCLUDE_RAW_OCR=false           # consider: omit OCR text from responses
```

`INCLUDE_RAW_OCR` defaults to `true` because it is genuinely useful for
debugging an integration. If your consumer does not need it, turning it off
shrinks the amount of receipt text crossing the network.

## Production configuration

```bash
APP_ENV=production
LOG_LEVEL=INFO
LOG_FORMAT=json
DEBUG_ERRORS=false
LOG_DOCUMENT_CONTENT=false
ENABLE_RAW_OCR_STORAGE=false
OCR_PROVIDER=tesseract
OCR_TIMEOUT_SECONDS=30
MAX_FILE_SIZE_MB=10
# Set only if you know where receipts come from -- it resolves ambiguous
# dates and currency symbols instead of leaving them null.
DEFAULT_COUNTRY=US
DATE_ORDER=MDY
```

That last pair is the highest-leverage configuration in the system. With
`DATE_ORDER=none`, `03/04/2026` returns `null`; with a declared locale it
resolves. Set it only when you genuinely know the source.

## Upgrading

1. Read the `PIPELINE_VERSION` change. A bump means output *values* may differ
   even when the shape has not.
2. Run the golden suite against your own fixtures.
3. Deploy to a canary and compare `pipeline.low_confidence.total` and
   `validation.failure.total` against the previous version.
4. A `SCHEMA_VERSION` major bump requires consumer changes; a minor bump adds
   optional fields only.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `/ready` 503, `PROVIDER_UNAVAILABLE` | Tesseract not on `PATH`. Install it or set `TESSERACT_CMD`. |
| `OCR_EMPTY_RESULT` | Blank, inverted or unreadable image. Check `data.quality` in a successful call for the measured metrics. |
| Everything flagged for review | `CONFIDENCE_THRESHOLD` too high for your input quality, or genuinely poor images. Check `review.reasons`. |
| `AMBIGUOUS_DATE_FORMAT` on every receipt | Working as designed. Set `DATE_ORDER`. |
| `currency: null` with a `$` symbol | Working as designed. Set `DEFAULT_COUNTRY`. |
| `TOTAL_MISMATCH` on correct receipts | Tolerance too tight, or an amount misread. Compare `raw_ocr.text` against the printed receipt. |
| Items missing or merged | Unusual layout. Enable `INCLUDE_EVIDENCE=true` to see which lines were used, and add a fixture. |
| Slow requests | Check `processing.stage_timings_ms`. If `preprocessing` dominates, lower `PREPROCESS_MAX_LONG_EDGE`. |
