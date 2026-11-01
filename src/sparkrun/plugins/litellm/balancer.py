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
* A request is identified by the proxy's ``litellm_call_id``, which is fresh
  per HTTP request and kept across the router's retries and fallbacks.  The
  trace id is only a fallback: clients set it from ``x-litellm-trace-id`` /
  ``x-litellm-session-id`` to group a whole session, so concurrent requests
  routinely share one, and keying on it would collapse them into one slot.
* Requests of one conversation prefer one backend, so its vLLM prefix
  cache keeps the conversation's prompt (an agent re-sends 100K+ tokens
  of history on every call).  The conversation is the client's
  ``x-litellm-session-id`` when it sends one, else a hash of the start of
  the system prompt plus the first user message.  Its backend is the
  rendezvous-hash winner among the least-faulted candidates, taken only
  while that backend has at most ``affinity_slack`` more requests in
  flight than the least loaded one; otherwise the request falls back to
  least-in-flight.  A backend joining or leaving moves only the
  conversations it wins or held.
* A retry or fallback re-enters the filter with the same id, moves its
  reservation instead of taking a second one, and prefers a backend that
  request has not tried yet.
* A backend whose request failed on the backend's side (anything but a
  malformed or oversized request; see :func:`is_backend_fault`) is ranked
  after the others for ``fault_avoid``
  seconds.  Least-in-flight alone would do the opposite: a backend that
  errors instantly always reads as the least loaded and would draw every
  new request until LiteLLM's cooldown caught up.  Client errors (a 400 for
  an oversized prompt) say nothing about the backend and are not counted.
* The slot is released on the success event for the same request.  LiteLLM
  logs a failure event only for the first failed attempt, so a request whose
  retries all fail is released by the proxy's once-per-request failure hook.
  A failure event naming a backend other than the one currently held is a
  late event for an earlier attempt and leaves the retry's slot alone.
  A client that hangs up mid-stream produces no logging event at all, so
  the proxy's stream wrapper releases those (without blaming the backend).
  Reservations that see none of these expire after ``stale_after`` seconds
  so a backend cannot stay "busy" forever.

Known undercounts, both bounded to the requests involved and never a leak:
a comma-separated ``model`` (LiteLLM's batch completion) fans one call id
out to concurrent requests, which then share one slot; and a late failure
event cannot be told from the current attempt's when the retry went back to
the same backend (a one-backend model, or every backend already tried).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING or __name__ == "sparkrun.plugins.litellm.balancer":
    # Imported by sparkrun itself (tests, and to copy the file), which has no
    # LiteLLM installed.  Inside the gateway LiteLLM loads this file under its
    # own module name and the import below must succeed: LiteLLM only calls
    # callbacks that are ``CustomLogger`` instances, so a fallback there would
    # turn balancing off without a word.
    from builtins import object as CustomLogger
else:  # pragma: no cover - runs inside the LiteLLM gateway process
    from litellm.integrations.custom_logger import CustomLogger  # pyright: ignore[reportMissingImports] - gateway-only dependency

STRATEGY = "least-in-flight"
STALE_AFTER_S = 900.0
FAULT_AVOID_S = 10.0
AFFINITY_SLACK = 2
#: How much of the system prompt identifies a conversation.  Its head is the
#: agent's fixed persona; a tail may carry per-call context (time, memory).
_SYSTEM_HEAD_CHARS = 2000

#: HTTP statuses that describe the request rather than the backend: a
#: malformed or oversized request fails the same way wherever it is sent.
_REQUEST_FAULT_STATUSES = frozenset({400, 413, 422, 499})


def backend_key(deployment: dict[str, Any]) -> str:
    """Identity of the backend serving *deployment* (its ``api_base``)."""
    params = deployment.get("litellm_params") or {}
    base = str(params.get("api_base") or "").rstrip("/")
    if base:
        return base
    return str((deployment.get("model_info") or {}).get("id") or params.get("model") or "")


def _request_keys(kwargs: dict[str, Any] | None) -> list[str]:
    """Identities of one request, most specific first (call id, then trace id).

    The filter receives the router's request kwargs, where both ids are
    top-level.  Logging events carry them in ``litellm_params`` (and the
    standard logging payload); an event's own top-level ``litellm_trace_id``
    is a different, per-logging-object id and must not be preferred.
    """
    if not kwargs:
        return []
    params = kwargs.get("litellm_params") or {}
    standard = kwargs.get("standard_logging_object") or {}
    keys = []
    for call in (params.get("litellm_call_id"), kwargs.get("litellm_call_id")):
        if call:
            keys.append("call:%s" % call)
            break
    for trace in (params.get("litellm_trace_id"), standard.get("trace_id"), kwargs.get("litellm_trace_id")):
        if trace:
            keys.append("trace:%s" % trace)
            break
    return keys


def _text(content: Any) -> str:
    """Plain text of a chat message's ``content`` (a string or a list of parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(p.get("text") or "") if isinstance(p, dict) else str(p) for p in content)
    return ""


def session_key(request_kwargs: dict[str, Any] | None, messages: Any = None) -> str | None:
    """Identity of the conversation a request belongs to, or None.

    An explicit ``x-litellm-session-id`` (``litellm_session_id``) wins; the
    trace id is not used because LiteLLM fills it per request when the
    client sent none.  Otherwise the conversation is the head of its system
    prompt plus its first user message, which an agent loop re-sends
    unchanged on every call until it compacts its history.
    """
    kwargs = request_kwargs or {}
    explicit = kwargs.get("litellm_session_id") or ((kwargs.get("metadata") or {}).get("session_id"))
    if explicit:
        return "session:%s" % explicit
    if not isinstance(messages, list):
        return None
    system = next((m for m in messages if isinstance(m, dict) and m.get("role") == "system"), None)
    first_user = next((m for m in messages if isinstance(m, dict) and m.get("role") == "user"), None)
    if first_user is None:
        return None
    head = _text(system.get("content"))[:_SYSTEM_HEAD_CHARS] if system else ""
    digest = hashlib.blake2b(("%s\0%s" % (head, _text(first_user.get("content")))).encode(), digest_size=16)
    return "prompt:%s" % digest.hexdigest()


def _affinity(session: str, backend: str) -> int:
    """Rendezvous-hash weight of *backend* for *session*; the highest wins."""
    return int.from_bytes(hashlib.blake2b(("%s|%s" % (session, backend)).encode(), digest_size=8).digest(), "big")


def _event_backend(kwargs: dict[str, Any] | None) -> str | None:
    """Backend a logging event's attempt went to, when the event says."""
    base = str(((kwargs or {}).get("litellm_params") or {}).get("api_base") or "").rstrip("/")
    return base or None


def is_backend_fault(exception: Any) -> bool:
    """Whether *exception* says the backend failed, as opposed to the request.

    Only a malformed or oversized request (400, 413, 422) or a client that
    went away (499) is the caller's.
    Everything else is the backend's: no status at all (connection refused,
    reset, DNS), 5xx, timeouts and 429 (overloaded), and also 401/403/404,
    which from an upstream behind the proxy mean a wrong key or a backend
    that does not serve the model it was registered for.  An unknown failure
    counts against the backend too: a wrong guess only ranks it last for
    ``fault_avoid`` seconds.
    """
    status = getattr(exception, "status_code", None)
    return not (isinstance(status, int) and status in _REQUEST_FAULT_STATUSES)


class LeastInFlight(CustomLogger):
    """Route each request to the backend with the fewest requests in flight."""

    def __init__(
        self,
        stale_after: float = STALE_AFTER_S,
        fault_avoid: float = FAULT_AVOID_S,
        clock: Callable[[], float] = time.monotonic,
        choice: Callable[[list], Any] = random.choice,
        affinity_slack: int = AFFINITY_SLACK,
        stats_path: str | None = None,
    ) -> None:
        if CustomLogger is not object:
            super().__init__()
        self._lock = threading.Lock()
        self._stale_after = stale_after
        self._fault_avoid = fault_avoid
        self._affinity_slack = affinity_slack
        #: (reason, backend) -> picks; reason is "choiceN" (the conversation's
        #: N-th preference) or "no-session".  Written to *stats_path* at most
        #: once a minute so an operator can see how traffic is placed.
        self._picks: dict[tuple[str, str], int] = {}
        self._stats_path = stats_path
        self._stats_written = 0.0
        self._clock = clock
        self._choice = choice
        self._in_flight: dict[str, int] = {}
        #: request key -> (backend holding its slot, reserved at)
        self._held: dict[str, tuple[str, float]] = {}
        #: request key -> (backends it has been sent to, last attempt at)
        self._tried: dict[str, tuple[set[str], float]] = {}
        #: backend -> when a request last failed on its side
        self._faulted: dict[str, float] = {}

    # -- bookkeeping ----------------------------------------------------

    def in_flight(self) -> dict[str, int]:
        """Snapshot of backend -> requests in flight (backends at 0 omitted)."""
        with self._lock:
            self._expire_locked()
            return {k: v for k, v in self._in_flight.items() if v > 0}

    def faulted(self) -> set[str]:
        """Backends currently ranked last because of a recent backend-side failure."""
        with self._lock:
            self._expire_locked()
            return set(self._faulted)

    def _held_key_locked(self, keys: list[str]) -> str | None:
        return next((k for k in keys if k in self._held), None)

    def _release_locked(self, key: str) -> str | None:
        held = self._held.pop(key, None)
        if held is None:
            return None
        backend = held[0]
        remaining = self._in_flight.get(backend, 0) - 1
        if remaining > 0:
            self._in_flight[backend] = remaining
        else:
            self._in_flight.pop(backend, None)
        return backend

    def _finish_locked(self, keys: list[str]) -> str | None:
        """Release the request for good: no retry will follow.

        Only the first of *keys* this request is known by is finished.  Its
        trace id may be shared by a whole session, so the fallback key must
        not end some other request that happens to be keyed on it.
        """
        key = next((k for k in keys if k in self._held or k in self._tried), None)
        if key is None:
            return None
        self._tried.pop(key, None)
        return self._release_locked(key)

    def _expire_locked(self) -> None:
        now = self._clock()
        cutoff = now - self._stale_after
        for key in [k for k, (_, at) in self._held.items() if at < cutoff]:
            self._release_locked(key)
        for key in [k for k, (_, at) in self._tried.items() if at < cutoff]:
            del self._tried[key]
        fault_cutoff = now - self._fault_avoid
        for backend in [b for b, at in self._faulted.items() if at < fault_cutoff]:
            del self._faulted[backend]

    def pick_stats(self) -> dict[str, dict[str, int]]:
        """Snapshot of reason -> backend -> picks since start."""
        with self._lock:
            out: dict[str, dict[str, int]] = {}
            for (reason, backend), n in self._picks.items():
                out.setdefault(reason, {})[backend] = n
            return out

    def _record_pick_locked(self, reason: str, deployment: dict[str, Any]) -> None:
        key = (reason, backend_key(deployment))
        self._picks[key] = self._picks.get(key, 0) + 1
        now = self._clock()
        if self._stats_path and now - self._stats_written >= 60.0:
            self._stats_written = now
            stats: dict[str, dict[str, int]] = {}
            for (r, b), n in self._picks.items():
                stats.setdefault(r, {})[b] = n
            try:
                with open(self._stats_path + ".tmp", "w") as f:
                    json.dump({"written_at": time.time(), "picks": stats}, f, indent=1, sort_keys=True)
                os.replace(self._stats_path + ".tmp", self._stats_path)
            except OSError:
                pass  # observability only; never fail a pick over it

    def _mark_fault_locked(self, backend: str | None) -> None:
        if backend:
            self._faulted[backend] = self._clock()

    def pick(self, deployments: list[dict[str, Any]], key: str | None, session: str | None = None) -> dict[str, Any]:
        """Choose one deployment and reserve a slot on its backend for request *key*.

        With a *session*, its rendezvous-hash backend wins while it is within
        ``affinity_slack`` of the least loaded candidate (see the module doc).
        """
        with self._lock:
            self._expire_locked()
            tried: set[str] = set()
            if key is not None:
                self._release_locked(key)  # retry / fallback moves the slot
                tried = self._tried.get(key, (set(), 0.0))[0]
            fresh = [d for d in deployments if backend_key(d) not in tried]
            candidates = fresh or deployments
            ranks = [(backend_key(d) in self._faulted, self._in_flight.get(backend_key(d), 0)) for d in candidates]
            best = min(ranks)
            chosen = None
            if session is not None:
                # The conversation's backends in its own fixed preference
                # order: the first within slack wins, so a busy home spills
                # to the same second choice every time (whose cache then
                # holds the conversation too), never to whichever backend
                # happens to be idle.  The least loaded peer always
                # qualifies, so this always picks.
                peers = [(d, rank) for d, rank in zip(candidates, ranks, strict=True) if rank[0] == best[0]]
                peers.sort(key=lambda p: _affinity(session, backend_key(p[0])), reverse=True)
                for order, (d, rank) in enumerate(peers):
                    if rank[1] <= best[1] + self._affinity_slack:
                        chosen = d
                        self._record_pick_locked("choice%d" % (order + 1), d)
                        break
            if chosen is None:
                chosen = self._choice([d for d, rank in zip(candidates, ranks, strict=True) if rank == best])
                self._record_pick_locked("no-session", chosen)
            if key is not None:
                backend = backend_key(chosen)
                now = self._clock()
                self._in_flight[backend] = self._in_flight.get(backend, 0) + 1
                self._held[key] = (backend, now)
                self._tried[key] = (tried | {backend}, now)
            return chosen

    def release(self, kwargs: dict[str, Any] | None) -> None:
        """The request in logging-event *kwargs* succeeded."""
        keys = _request_keys(kwargs)
        if not keys:
            return
        with self._lock:
            self._finish_locked(keys)

    def release_failed_attempt(self, kwargs: dict[str, Any] | None) -> None:
        """An attempt of the request in logging-event *kwargs* failed; a retry may follow."""
        kwargs = kwargs or {}
        fault = is_backend_fault(kwargs.get("exception"))
        attempt_backend = _event_backend(kwargs)
        with self._lock:
            key = self._held_key_locked(_request_keys(kwargs))
            held = self._held[key][0] if key is not None else None
            if key is not None and (attempt_backend is None or attempt_backend == held):
                self._release_locked(key)
                attempt_backend = held
            if fault:
                self._mark_fault_locked(attempt_backend)

    def release_request(self, request_data: dict[str, Any] | None, exception: Any = None, *, fault: bool | None = None) -> None:
        """The proxy request *request_data* ended without success, after every retry and fallback.

        *fault* says whether to blame the backend it held; by default it is
        decided from *exception*.
        """
        keys = _request_keys(request_data)
        if not keys:
            return
        if fault is None:
            fault = is_backend_fault(exception)
        with self._lock:
            backend = self._finish_locked(keys)
            if backend is not None and fault:
                self._mark_fault_locked(backend)

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
        keys = _request_keys(request_kwargs)
        return [self.pick(list(healthy_deployments), keys[0] if keys else None, session_key(request_kwargs, messages))]

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        self.release(kwargs)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time) -> None:
        self.release_failed_attempt(kwargs)

    async def async_post_call_failure_hook(self, request_data, original_exception, user_api_key_dict, traceback_str=None):
        # Proxy-only, once per request after every retry and fallback has failed.
        self.release_request(request_data, original_exception)
        return None

    async def async_post_call_streaming_iterator_hook(self, user_api_key_dict, response, request_data):
        # Proxy-only.  A client that disconnects mid-stream closes this
        # generator (or cancels the task iterating it) and LiteLLM logs no
        # event for that request at all.  Errors raised by the stream itself
        # still reach the failure hooks, which can blame the backend.
        try:
            async for chunk in response:
                yield chunk
        except (GeneratorExit, asyncio.CancelledError):
            self.release_request(request_data, fault=False)
            raise

    def log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        self.release(kwargs)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time) -> None:
        self.release_failed_attempt(kwargs)


#: Inside the gateway the stats land next to the generated config (the proxy
#: state directory); imported by sparkrun itself, nothing is written.
handler = LeastInFlight(
    stats_path=None
    if __name__ == "sparkrun.plugins.litellm.balancer"
    else os.path.join(os.path.dirname(os.path.abspath(__file__)), "balancer_stats.json")
)
