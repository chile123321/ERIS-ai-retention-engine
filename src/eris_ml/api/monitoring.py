"""Per-process, low-cardinality Prometheus operational metrics; no employee data."""

from __future__ import annotations

from collections import Counter, defaultdict
from threading import Lock

from fastapi import APIRouter, Depends, Request
from fastapi.responses import PlainTextResponse

from eris_ml.api.auth import require_service_token

router = APIRouter(tags=["operations"], dependencies=[Depends(require_service_token)])

ENDPOINTS = frozenset({
    "/health", "/ready", "/metrics", "/api/v1/model-info",
    "/api/v1/predict", "/api/v1/predict/batch", "/docs", "/openapi.json",
})


def safe_endpoint(path: str) -> str:
    """Use only static allowlisted endpoint labels, never user-controlled paths."""
    return path if path in ENDPOINTS else "unmatched"


class OperationalMetrics:
    """Process-local counters and latency sums; not drift or outcome monitoring."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._requests: Counter[tuple[str, str, str]] = Counter()
        self._latency_count: Counter[tuple[str, str]] = Counter()
        self._latency_sum: defaultdict[tuple[str, str], float] = defaultdict(float)
        self._totals: Counter[str] = Counter()
        self._ready = 0

    def set_ready(self, ready: bool) -> None:
        with self._lock:
            self._ready = int(ready)

    def increment(self, name: str, amount: int = 1) -> None:
        if name not in {
            "prediction_requests", "batch_records", "validation_failures",
            "authentication_failures", "prediction_alerts", "unknown_category_warnings",
            "bundle_load_failures",
        } or amount < 0:
            raise ValueError("Unknown or invalid operational counter.")
        with self._lock:
            self._totals[name] += amount

    def record_request(self, path: str, method: str, status: int, seconds: float) -> None:
        endpoint = safe_endpoint(path)
        safe_method = method if method in {"GET", "POST"} else "OTHER"
        with self._lock:
            self._requests[(endpoint, safe_method, str(status))] += 1
            self._latency_count[(endpoint, safe_method)] += 1
            self._latency_sum[(endpoint, safe_method)] += max(0.0, seconds)
            if status == 401:
                self._totals["authentication_failures"] += 1
            if status == 422:
                self._totals["validation_failures"] += 1
            if safe_method == "POST" and endpoint in {
                "/api/v1/predict", "/api/v1/predict/batch"
            }:
                self._totals["prediction_requests"] += 1

    def render(self) -> str:
        """Expose fixed-series text without request, feature, token or ID labels."""
        with self._lock:
            lines = [
                "# HELP eris_api_ready One when bundle and service auth are ready.",
                "# TYPE eris_api_ready gauge",
                f"eris_api_ready {self._ready}",
                "# HELP eris_http_requests_total HTTP requests by static route, method and status.",
                "# TYPE eris_http_requests_total counter",
            ]
            for (endpoint, method, status), count in sorted(self._requests.items()):
                lines.append(
                    f'eris_http_requests_total{{endpoint="{endpoint}",method="{method}",'
                    f'status="{status}"}} {count}'
                )
            lines.extend([
                "# HELP eris_http_request_duration_seconds Request duration per static route.",
                "# TYPE eris_http_request_duration_seconds summary",
            ])
            for (endpoint, method), count in sorted(self._latency_count.items()):
                label = f'endpoint="{endpoint}",method="{method}"'
                lines.append(
                    f'eris_http_request_duration_seconds_sum{{{label}}} '
                    f'{self._latency_sum[(endpoint, method)]:.9f}'
                )
                lines.append(f'eris_http_request_duration_seconds_count{{{label}}} {count}')
            for name in (
                "prediction_requests", "batch_records", "validation_failures",
                "authentication_failures", "prediction_alerts", "unknown_category_warnings",
                "bundle_load_failures",
            ):
                lines.append(f"# TYPE eris_{name}_total counter")
                lines.append(f"eris_{name}_total {self._totals[name]}")
            return "\n".join(lines) + "\n"


@router.get("/metrics", response_class=PlainTextResponse, include_in_schema=True)
def metrics(request: Request) -> PlainTextResponse:
    """Authenticated, process-local operational metrics without PII labels."""
    return PlainTextResponse(
        request.app.state.metrics.render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
