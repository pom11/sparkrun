"""Least-in-flight balancing for the LiteLLM gateway.

LiteLLM's own ``least-busy`` strategy does not balance on the proxy's async
request path in the pinned LiteLLM release: its in-flight counter is only
incremented from a sync hook that ``Router.acompletion`` never calls, so every
deployment reads 0 and the tie always goes to the first ``model_list`` entry.

This module replaces it with a LiteLLM ``CustomLogger`` that LiteLLM loads
from ``litellm_settings.callbacks``.  sparkrun copies this file next to the
generated ``litellm_config.yaml`` (LiteLLM imports callback modules relative
to the config file, inside its own environment), so it must only depend on
the standard library and LiteLLM itself.

How a request is placed:

* ``async_filter_deployments`` runs after LiteLLM has narrowed the model
  group to its healthy deployments.  It picks the backend with the fewest
  requests in flight (random among ties), reserves one slot on it, and hands
  LiteLLM a single deployment, so pick and reservation happen together and
  concurrent requests cannot all see the same idle backend.
* Load is counted per backend ``api_base``, not per deployment id, so a
  model and its aliases share one view of each backend.  Only the
  deployments LiteLLM offers for the requested model group are candidates,
  so a request never spills onto a backend serving a different model.
* The slot is released on the success event for the same request.  A
  retry or fallback re-enters the filter with the same trace id and moves
  its reservation instead of taking a second one.  LiteLLM logs a failure
  event only for the first failed attempt, so a request whose retries all
  fail is released by the proxy's once-per-request failure hook, matched
  on the proxy call id the filter saw.  Reservations that never see either
  expire after ``stale_after`` seconds so a backend cannot stay "busy"
  forever.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Any, Callable

try:  # Present inside the LiteLLM gateway process.
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:  # pragma: no cover - sparkrun imports this file for tests and to copy it
    CustomLogger = object  # type: ignore[assignment,misc]

STRATEGY = "least-in-flight"
STALE_AFTER_S = 900.0


def backend_key(deployment: dict[str, Any]) -> str:
    """Identity of the backend serving *deployment* (its ``api_base``)."""
    params = deployment.get("litellm_params") or {}
    base = str(params.get("api_base") or "").rstrip("/")
    if base:
        return base
    return str((deployment.get("model_info") or {}).get("id") or params.get("model") or "")


def _trace_id(kwargs: dict[str, Any] | None) -> str | None:
    """The request's trace id, as seen by the routing filter and by the logging events.

    The filter receives the router's request kwargs, where it is top-level.
    Logging events carry it in ``litellm_params`` (and the standard logging
    payload); their own top-level ``litellm_trace_id`` is a different,
    per-logging-object id and must not be preferred.
    """
    if not kwargs:
        return None
    params = kwargs.get("litellm_params") or {}
    standard = kwargs.get("standard_logging_object") or {}
    for trace in (params.get("litellm_trace_id"), standard.get("trace_id"), kwargs.get("litellm_trace_id")):
        if trace:
            return str(trace)
    return None


class LeastInFlight(CustomLogger):
    """Route each request to the backend with the fewest requests in flight."""

    def __init__(
        self,
        stale_after: float = STALE_AFTER_S,
        clock: Callable[[], float] = time.monotonic,
        choice: Callable[[list], Any] = random.choice,
    ) -> None:
        if CustomLogger is not object:
            super().__init__()
        self._lock = threading.Lock()
        self._stale_after = stale_after
        self._clock = clock
        self._choice = choice
        self._in_flight: dict[str, int] = {}
        self._held: dict[str, tuple[str, float]] = {}
        self._trace_by_call: dict[str, str] = {}

    # -- bookkeeping ----------------------------------------------------

    def in_flight(self) -> dict[str, int]:
        """Snapshot of backend -> requests in flight (backends at 0 omitted)."""
        with self._lock:
            self._expire_locked()
            return {k: v for k, v in self._in_flight.items() if v > 0}

    def _release_locked(self, trace: str) -> None:
        held = self._held.pop(trace, None)
        if held is None:
            return
        for call in [c for c, t in self._trace_by_call.items() if t == trace]:
            del self._trace_by_call[call]
        backend = held[0]
        remaining = self._in_flight.get(backend, 0) - 1
        if remaining > 0:
            self._in_flight[backend] = remaining
        else:
            self._in_flight.pop(backend, None)

    def _expire_locked(self) -> None:
        cutoff = self._clock() - self._stale_after
        for trace in [t for t, (_, at) in self._held.items() if at < cutoff]:
            self._release_locked(trace)

    def pick(self, deployments: list[dict[str, Any]], trace: str | None, call: str | None = None) -> dict[str, Any]:
        """Choose one deployment and reserve a slot on its backend for *trace*.

        *call* is the proxy's per-request call id, recorded so the slot can
        also be released by :meth:`release_call`.
        """
        with self._lock:
            self._expire_locked()
            if trace is not None:
                self._release_locked(trace)  # retry / fallback moves the slot
            loads = [self._in_flight.get(backend_key(d), 0) for d in deployments]
            lowest = min(loads)
            chosen = self._choice([d for d, load in zip(deployments, loads, strict=True) if load == lowest])
            if trace is not None:
                backend = backend_key(chosen)
                self._in_flight[backend] = self._in_flight.get(backend, 0) + 1
                self._held[trace] = (backend, self._clock())
                if call:
                    self._trace_by_call[call] = trace
            return chosen

    def release(self, trace: str | None) -> None:
        if trace is None:
            return
        with self._lock:
            self._release_locked(trace)

    def release_call(self, call: str | None) -> None:
        """Release the slot reserved by the proxy request with call id *call*."""
        if not call:
            return
        with self._lock:
            trace = self._trace_by_call.get(call)
            if trace is not None:
                self._release_locked(trace)

    # -- LiteLLM hooks --------------------------------------------------

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: list,
        messages: Any = None,
        request_kwargs: dict | None = None,
        parent_otel_span: Any = None,
    ) -> list:
        if not healthy_deployments:
            return healthy_deployments
        kwargs = request_kwargs or {}
        return [self.pick(list(healthy_deployments), _trace_id(kwargs), kwargs.get("litellm_call_id"))]

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        self.release(_trace_id(kwargs))

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time) -> None:
        self.release(_trace_id(kwargs))

    async def async_post_call_failure_hook(self, request_data, original_exception, user_api_key_dict, traceback_str=None):
        # Proxy-only, once per request after every retry and fallback has failed.
        self.release_call((request_data or {}).get("litellm_call_id"))
        return None

    def log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        self.release(_trace_id(kwargs))

    def log_failure_event(self, kwargs, response_obj, start_time, end_time) -> None:
        self.release(_trace_id(kwargs))


handler = LeastInFlight()
