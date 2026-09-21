"""Structured JSON request logging (P08 deliverable #5) and Prometheus request/latency
instrumentation (P10 deliverable #1), in one middleware: one JSON line per request with a
request id, endpoint, latency, model version, and a prediction *summary* — never the raw input
window — plus a ``pdm_requests_total``/``pdm_request_latency_seconds`` observation for every
response, success or error. Route handlers that want the summary/model-version fields in the log
line stash them on ``request.state`` before returning; the middleware never touches the request
body itself, so there is no raw payload for it to accidentally log even by mistake.

Recording request-count/latency here rather than in each handler (P08's original approach) is
deliberate: a handler that raises (a 422 from schema validation, the 503
``/detect/anomaly`` returns with no anomaly bundle loaded) never reaches its own
``REQUEST_COUNT.inc()`` call, which silently excluded every error response from
``pdm_requests_total`` — undermining the one thing an "error rate" dashboard panel needs. This
middleware wraps every request via ``call_next``, including ones a handler turned into an error
response, so every status code this service can return is counted.
"""

import json
import logging
import sys
import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from pdm.serving.metrics import record_request

logger = logging.getLogger("pdm.serving")


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        latency_seconds = time.perf_counter() - start
        latency_ms = latency_seconds * 1000

        record_request(request.url.path, str(response.status_code), latency_seconds)

        entry = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": round(latency_ms, 3),
        }
        model_version = getattr(request.state, "model_version", None)
        if model_version is not None:
            entry["model_version"] = model_version
        prediction_summary = getattr(request.state, "prediction_summary", None)
        if prediction_summary is not None:
            entry["prediction_summary"] = prediction_summary

        logger.info(json.dumps(entry))
        response.headers["X-Request-ID"] = request_id
        return response
