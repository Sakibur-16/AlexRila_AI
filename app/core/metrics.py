"""Observability primitives.

A deliberately tiny metrics facade. The pipeline records counters, gauges and
timings against this interface; a deployment can bind a real backend
(Prometheus, OTLP, StatsD) by implementing :class:`MetricsSink` and calling
:func:`set_metrics_sink` at startup -- no pipeline code changes.

The default :class:`InMemoryMetricsSink` keeps bounded reservoirs so that
``/metrics`` is useful in development and tests without pulling in a
dependency, and never grows without limit in a long-running process.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

#: Bound on retained samples per timing series.
_MAX_SAMPLES = 1024


def _key(name: str, labels: Mapping[str, str] | None) -> str:
    """Render a stable series key from a metric name and its labels."""
    if not labels:
        return name
    rendered = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
    return f"{name}{{{rendered}}}"


class MetricsSink(ABC):
    """Destination for pipeline telemetry."""

    @abstractmethod
    def increment(
        self, name: str, value: float = 1.0, labels: Mapping[str, str] | None = None
    ) -> None:
        """Add to a monotonically increasing counter."""

    @abstractmethod
    def observe(self, name: str, value: float, labels: Mapping[str, str] | None = None) -> None:
        """Record a single observation in a distribution (latency, score...)."""

    @abstractmethod
    def snapshot(self) -> dict[str, Any]:
        """Return a serialisable view of current metrics."""


class InMemoryMetricsSink(MetricsSink):
    """Thread-safe, bounded, dependency-free default sink."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=_MAX_SAMPLES))

    def increment(
        self, name: str, value: float = 1.0, labels: Mapping[str, str] | None = None
    ) -> None:
        with self._lock:
            self._counters[_key(name, labels)] += value

    def observe(self, name: str, value: float, labels: Mapping[str, str] | None = None) -> None:
        with self._lock:
            self._samples[_key(name, labels)].append(value)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            samples = {k: list(v) for k, v in self._samples.items()}

        distributions: dict[str, dict[str, float]] = {}
        for key, values in samples.items():
            if not values:
                continue
            ordered = sorted(values)
            distributions[key] = {
                "count": float(len(ordered)),
                "min": ordered[0],
                "max": ordered[-1],
                "mean": sum(ordered) / len(ordered),
                "p50": _percentile(ordered, 0.50),
                "p95": _percentile(ordered, 0.95),
                "p99": _percentile(ordered, 0.99),
            }
        return {"counters": counters, "distributions": distributions}

    def reset(self) -> None:
        """Drop all recorded metrics. Intended for tests."""
        with self._lock:
            self._counters.clear()
            self._samples.clear()


class NullMetricsSink(MetricsSink):
    """Sink that discards everything. Useful for benchmarks.

    The parameters are unused by design -- they exist to satisfy the interface.
    """

    def increment(
        self,
        name: str,  # noqa: ARG002 - null object: the interface requires it
        value: float = 1.0,  # noqa: ARG002
        labels: Mapping[str, str] | None = None,  # noqa: ARG002
    ) -> None:
        return None

    def observe(
        self,
        name: str,  # noqa: ARG002 - null object: the interface requires it
        value: float,  # noqa: ARG002
        labels: Mapping[str, str] | None = None,  # noqa: ARG002
    ) -> None:
        return None

    def snapshot(self) -> dict[str, Any]:
        return {"counters": {}, "distributions": {}}


def _percentile(ordered: list[float], q: float) -> float:
    """Nearest-rank percentile over a pre-sorted list."""
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


_sink: MetricsSink = InMemoryMetricsSink()
_sink_lock = threading.Lock()


def set_metrics_sink(sink: MetricsSink) -> None:
    """Install the process-wide metrics sink."""
    global _sink
    with _sink_lock:
        _sink = sink


def get_metrics_sink() -> MetricsSink:
    """Return the process-wide metrics sink."""
    return _sink


def increment(name: str, value: float = 1.0, labels: Mapping[str, str] | None = None) -> None:
    """Increment a counter on the active sink."""
    _sink.increment(name, value, labels)


def observe(name: str, value: float, labels: Mapping[str, str] | None = None) -> None:
    """Record an observation on the active sink."""
    _sink.observe(name, value, labels)


@contextmanager
def timed(name: str, labels: Mapping[str, str] | None = None) -> Iterator[None]:
    """Time the wrapped block and record it in milliseconds.

    The observation is recorded even when the block raises, so failure latency
    is visible rather than silently excluded.
    """
    started = time.perf_counter()
    try:
        yield
    finally:
        observe(name, (time.perf_counter() - started) * 1000.0, labels)


class MetricNames:
    """Canonical metric names, so call sites cannot drift apart."""

    PIPELINE_REQUESTS = "pipeline.requests.total"
    PIPELINE_SUCCESS = "pipeline.success.total"
    PIPELINE_FAILURE = "pipeline.failure.total"
    PIPELINE_LATENCY_MS = "pipeline.latency.ms"

    PREPROCESS_LATENCY_MS = "preprocess.latency.ms"
    OCR_LATENCY_MS = "ocr.latency.ms"
    EXTRACTION_LATENCY_MS = "extraction.latency.ms"
    VALIDATION_LATENCY_MS = "validation.latency.ms"

    OCR_SUCCESS = "ocr.success.total"
    OCR_FAILURE = "ocr.failure.total"
    OCR_RETRY = "ocr.retry.total"
    OCR_CONFIDENCE = "ocr.confidence"

    LOW_CONFIDENCE = "pipeline.low_confidence.total"
    VALIDATION_FAILURE = "validation.failure.total"
    VALIDATION_WARNING = "validation.warning.total"
    REVIEW_REQUIRED = "pipeline.review_required.total"

    LLM_INVOKED = "llm.invoked.total"
    LLM_SUCCESS = "llm.success.total"
    LLM_FAILURE = "llm.failure.total"
    LLM_LATENCY_MS = "llm.latency.ms"
