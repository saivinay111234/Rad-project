"""
Observability: structured logging + Prometheus metrics for the Radiology Assistant.

Sets up:
1. JSON-structured logging via python-json-logger
2. Prometheus metrics via prometheus-fastapi-instrumentator
3. Custom application-level metrics (LLM call counters, agent timings, errors)
"""

import logging
import time
from functools import wraps
from typing import Optional, Callable

# Try to import optional observability libraries gracefully
try:
    from pythonjsonlogger import jsonlogger
    _HAS_JSON_LOGGER = True
except ImportError:
    _HAS_JSON_LOGGER = False

try:
    from prometheus_fastapi_instrumentator import Instrumentator
    from prometheus_client import Counter, Histogram, Gauge
    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False

from .config import Config

# ---------------------------------------------------------------------------
# Structured JSON Logging
# ---------------------------------------------------------------------------

def setup_logging(level: Optional[str] = None) -> None:
    """
    Configure structured JSON logging for the entire application.

    In development (LOG_LEVEL=DEBUG), falls back to human-readable format
    if python-json-logger is not installed.

    Args:
        level: Logging level string (e.g. "INFO", "DEBUG"). Defaults to Config.LOG_LEVEL.
    """
    log_level = getattr(logging, (level or Config.LOG_LEVEL).upper(), logging.INFO)

    if _HAS_JSON_LOGGER:
        handler = logging.StreamHandler()
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.addHandler(handler)
        root_logger.setLevel(log_level)

        logging.getLogger(__name__).info(
            "Structured JSON logging configured",
            extra={"log_level": log_level, "formatter": "json"}
        )
    else:
        # Fallback: plain text logging
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )
        logging.getLogger(__name__).warning(
            "python-json-logger not installed. Using plain text logging. "
            "Install with: pip install python-json-logger"
        )


# ---------------------------------------------------------------------------
# Prometheus Metrics
# ---------------------------------------------------------------------------

# Defined at module level so they persist across requests
if _HAS_PROMETHEUS:
    LLM_REQUEST_TOTAL = Counter(
        "rad_llm_request_total",
        "Total LLM API calls by agent and status",
        labelnames=["agent", "provider", "status"],
    )

    LLM_REQUEST_DURATION = Histogram(
        "rad_llm_request_duration_seconds",
        "LLM response latency in seconds",
        labelnames=["agent", "provider"],
        buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    )

    AGENT_ERRORS_TOTAL = Counter(
        "rad_agent_errors_total",
        "Total agent-level errors by agent and error type",
        labelnames=["agent", "error_type"],
    )

    ACTIVE_REQUESTS = Gauge(
        "rad_active_requests",
        "Number of currently in-flight requests",
    )


def setup_metrics(app) -> None:
    """
    Attach Prometheus instrumentation to the FastAPI app.

    Exposes a `/metrics` endpoint for Prometheus scraping.

    Args:
        app: The FastAPI application instance.
    """
    if not _HAS_PROMETHEUS:
        logging.getLogger(__name__).warning(
            "prometheus-fastapi-instrumentator not installed. Metrics disabled. "
            "Install with: pip install prometheus-fastapi-instrumentator prometheus-client"
        )
        return

    Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_respect_env_var=False,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics", "/health"],
        env_var_name="ENABLE_METRICS",
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

    logging.getLogger(__name__).info("Prometheus metrics exposed at /metrics")


# ---------------------------------------------------------------------------
# Agent Timing Decorator
# ---------------------------------------------------------------------------

def observe_agent(agent_name: str, provider: str = "gemini") -> Callable:
    """
    Decorator that records LLM call duration and success/failure counts.

    Usage:
        @observe_agent("report_drafter")
        def call_llm(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not _HAS_PROMETHEUS:
                return func(*args, **kwargs)

            ACTIVE_REQUESTS.inc()
            start = time.monotonic()
            status = "success"
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                AGENT_ERRORS_TOTAL.labels(
                    agent=agent_name,
                    error_type=type(e).__name__,
                ).inc()
                raise
            finally:
                duration = time.monotonic() - start
                LLM_REQUEST_TOTAL.labels(agent=agent_name, provider=provider, status=status).inc()
                LLM_REQUEST_DURATION.labels(agent=agent_name, provider=provider).observe(duration)
                ACTIVE_REQUESTS.dec()
        return wrapper
    return decorator
