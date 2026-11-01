"""Least-in-flight balancing callback for the LiteLLM gateway (``plugins/litellm/balancer.py``).

The module is loaded by LiteLLM from beside the generated config, so these
tests drive it directly: no LiteLLM process, no network.
"""

from __future__ import annotations

import asyncio
import collections
import importlib.util
import sys
import types

from sparkrun.plugins.litellm import balancer
from sparkrun.plugins.litellm.balancer import LeastInFlight, backend_key, is_backend_fault


def _dep(i: int, model: str = "org/model-a", alias: str | None = None) -> dict:
    return {
        "model_name": alias or model,
        "litellm_params": {"model": "openai/%s" % model, "api_base": "http://10.0.0.%d:8000/v1" % i},
        "model_info": {"id": "%s-%d" % (alias or model, i)},
    }


def _first(candidates: list) -> dict:
    return candidates[0]


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__("status %d" % status_code)
        self.status_code = status_code


def _filter(lb: LeastInFlight, deployments: list, call: str | None, trace: str | None = None) -> dict:
    kwargs = {}
    if call:
        kwargs["litellm_call_id"] = call
        kwargs["litellm_trace_id"] = trace or "trace-%s" % call
    picked = asyncio.run(lb.async_filter_deployments("g", deployments, None, kwargs))
    assert len(picked) == 1
    return picked[0]


def _event(call: str, api_base: str | None = None, exception: Exception | None = None) -> dict:
    # Logging events carry the request's ids in litellm_params; their own
    # top-level litellm_trace_id is a different id and must be ignored.
    params = {"litellm_call_id": call, "litellm_trace_id": "trace-%s" % call}
    if api_base:
        params["api_base"] = api_base
    return {"litellm_trace_id": "logging-object-id", "litellm_params": params, "exception": exception}


def _success(lb: LeastInFlight, call: str) -> None:
    asyncio.run(lb.async_log_success_event(_event(call), None, 0, 0))


def _attempt_failed(lb: LeastInFlight, call: str, api_base: str | None = None, exception: Exception | None = None) -> None:
    asyncio.run(lb.async_log_failure_event(_event(call, api_base, exception), None, 0, 0))


def test_backend_key_is_api_base_without_trailing_slash():
    assert backend_key(_dep(1)) == "http://10.0.0.1:8000/v1"
    assert backend_key({"litellm_params": {"api_base": "http://10.0.0.1:8000/v1/"}}) == "http://10.0.0.1:8000/v1"
    assert backend_key({"model_info": {"id": "x"}, "litellm_params": {}}) == "x"


def test_concurrent_requests_spread_evenly():
    """8 in flight over 4 backends is 2/2/2/2, wave after wave (LiteLLM's least-busy gives 8/0/0/0)."""
    lb = LeastInFlight(choice=_first)
    deps = [_dep(i) for i in range(1, 5)]
    for wave in range(3):
        traces = ["w%d-%d" % (wave, n) for n in range(8)]
        counts = collections.Counter(backend_key(_filter(lb, deps, t)) for t in traces)
        assert sorted(counts.values()) == [2, 2, 2, 2]
        for t in traces:
            _success(lb, t)
        assert lb.in_flight() == {}


def test_busy_backend_is_avoided_until_released():
    lb = LeastInFlight(choice=_first)
    deps = [_dep(1), _dep(2)]
    assert backend_key(_filter(lb, deps, "a")) == backend_key(deps[0])
    assert backend_key(_filter(lb, deps, "b")) == backend_key(deps[1])
    assert backend_key(_filter(lb, deps, "c")) == backend_key(deps[0])
    _success(lb, "b")
    assert backend_key(_filter(lb, deps, "d")) == backend_key(deps[1])


def test_ties_are_broken_by_choice_not_list_order():
    seen = []

    def choice(candidates):
        seen.append(len(candidates))
        return candidates[-1]

    lb = LeastInFlight(choice=choice)
    deps = [_dep(i) for i in range(1, 4)]
    assert backend_key(_filter(lb, deps, "t")) == backend_key(deps[-1])
    assert seen == [3]


def test_aliases_share_one_view_per_backend():
    """A request through one alias counts against the backend for every alias of it."""
    lb = LeastInFlight(choice=_first)
    worker = [_dep(i, alias="worker-model") for i in (1, 2)]
    orch = [_dep(i, alias="orchestrator-model") for i in (1, 2)]
    assert backend_key(_filter(lb, worker, "w")) == "http://10.0.0.1:8000/v1"
    assert backend_key(_filter(lb, orch, "o")) == "http://10.0.0.2:8000/v1"


def test_only_offered_deployments_are_candidates():
    """The filter never picks outside the model group LiteLLM offers, so models cannot spill onto each other."""
    lb = LeastInFlight(choice=_first)
    family_a = [_dep(1, model="org/model-a"), _dep(2, model="org/model-a")]
    family_b = [_dep(3, model="org/model-b"), _dep(4, model="org/model-b")]
    for n in range(6):
        _filter(lb, family_a, "a%d" % n)
    picks = collections.Counter(backend_key(_filter(lb, family_b, "b%d" % n)) for n in range(4))
    assert set(picks) == {backend_key(d) for d in family_b}
    assert sorted(picks.values()) == [2, 2]
    load = lb.in_flight()
    assert load[backend_key(family_a[0])] == 3 and load[backend_key(family_a[1])] == 3


def test_retry_moves_the_reservation_instead_of_adding_one():
    lb = LeastInFlight(choice=_first)
    deps = [_dep(1), _dep(2)]
    _filter(lb, deps, "t")
    _filter(lb, deps, "t")
    assert sum(lb.in_flight().values()) == 1
    _success(lb, "t")
    assert lb.in_flight() == {}


def test_failure_event_releases():
    lb = LeastInFlight(choice=_first)
    _filter(lb, [_dep(1)], "t")
    _attempt_failed(lb, "t")
    assert lb.in_flight() == {}


def test_proxy_failure_hook_releases_by_call_id():
    """LiteLLM logs a failure only for the first attempt; the proxy hook covers the last retry."""
    lb = LeastInFlight(choice=_first)
    deps = [_dep(1), _dep(2)]
    _filter(lb, deps, "c")
    _attempt_failed(lb, "c")
    _filter(lb, deps, "c")  # the retry reserves again
    assert sum(lb.in_flight().values()) == 1
    asyncio.run(lb.async_post_call_failure_hook({"litellm_call_id": "c"}, RuntimeError("down"), None))
    assert lb.in_flight() == {}
    asyncio.run(lb.async_post_call_failure_hook({"litellm_call_id": "c"}, RuntimeError("down"), None))
    asyncio.run(lb.async_post_call_failure_hook({}, RuntimeError("down"), None))
    assert lb.in_flight() == {}


def test_requests_sharing_a_session_trace_id_are_counted_separately():
    """x-litellm-session-id gives a whole session one trace id; each request still holds its own slot."""
    lb = LeastInFlight(choice=_first)
    deps = [_dep(1), _dep(2)]
    for n in range(4):
        _filter(lb, deps, "c%d" % n, trace="session-1")
    assert lb.in_flight() == {backend_key(deps[0]): 2, backend_key(deps[1]): 2}
    _success(lb, "c0")
    assert sum(lb.in_flight().values()) == 3


def test_trace_id_keys_the_request_when_there_is_no_call_id():
    lb = LeastInFlight(choice=_first)
    picked = asyncio.run(lb.async_filter_deployments("g", [_dep(1)], None, {"litellm_trace_id": "t"}))
    assert backend_key(picked[0]) in lb.in_flight()
    asyncio.run(lb.async_log_success_event({"litellm_params": {"litellm_trace_id": "t"}}, None, 0, 0))
    assert lb.in_flight() == {}


def test_retry_prefers_a_backend_the_request_has_not_tried():
    lb = LeastInFlight(choice=_first)
    deps = [_dep(1), _dep(2)]
    _filter(lb, [deps[1]], "busy")  # backend 2 is busier, so the first attempt goes to 1
    assert backend_key(_filter(lb, deps, "c")) == backend_key(deps[0])
    assert backend_key(_filter(lb, deps, "c")) == backend_key(deps[1])
    # With every backend tried, the retry falls back to the whole group.
    assert backend_key(_filter(lb, deps, "c")) == backend_key(deps[0])
    assert sum(lb.in_flight().values()) == 2


def test_fast_failing_backend_does_not_draw_new_requests():
    """A backend that errors instantly reads as idle; a recent backend-side failure ranks it last."""
    clock = _Clock()
    lb = LeastInFlight(clock=clock, choice=_first, fault_avoid=10)
    broken, healthy = _dep(1), _dep(2)
    for n in range(3):
        _filter(lb, [healthy], "long%d" % n)
    assert backend_key(_filter(lb, [broken, healthy], "r0")) == backend_key(broken)
    _attempt_failed(lb, "r0", backend_key(broken), _StatusError(503))
    asyncio.run(lb.async_post_call_failure_hook({"litellm_call_id": "r0"}, _StatusError(503), None))
    assert lb.faulted() == {backend_key(broken)}
    assert [backend_key(_filter(lb, [broken, healthy], "r%d" % n)) for n in range(1, 4)] == [backend_key(healthy)] * 3
    clock.now += 11
    assert lb.faulted() == set()
    assert backend_key(_filter(lb, [broken, healthy], "r9")) == backend_key(broken)


def test_client_errors_do_not_count_against_the_backend():
    lb = LeastInFlight(choice=_first)
    _filter(lb, [_dep(1), _dep(2)], "c")
    _attempt_failed(lb, "c", backend_key(_dep(1)), _StatusError(400))
    asyncio.run(lb.async_post_call_failure_hook({"litellm_call_id": "c"}, _StatusError(400), None))
    assert lb.faulted() == set()
    assert lb.in_flight() == {}


def test_backend_fault_classification():
    assert is_backend_fault(ConnectionError("refused"))
    assert is_backend_fault(None)
    for status in (500, 503, 408, 429, 401, 404):
        assert is_backend_fault(_StatusError(status))
    for status in (400, 413, 422, 499):
        assert not is_backend_fault(_StatusError(status))


def test_event_api_base_is_compared_without_its_trailing_slash():
    """LiteLLM reports the attempt's api_base with a trailing slash (seen live on 1.82.6)."""
    lb = LeastInFlight(choice=_first)
    _filter(lb, [_dep(1), _dep(2)], "c")
    _attempt_failed(lb, "c", backend_key(_dep(1)) + "/", _StatusError(500))
    assert lb.in_flight() == {}
    assert lb.faulted() == {backend_key(_dep(1))}


def test_finishing_a_request_leaves_others_on_its_shared_trace_id_alone():
    """A request reaching the filter without a call id is keyed on its (session-wide) trace id."""
    lb = LeastInFlight(choice=_first)
    deps = [_dep(1), _dep(2)]
    asyncio.run(lb.async_filter_deployments("g", deps, None, {"litellm_trace_id": "session"}))
    _filter(lb, deps, "c", trace="session")
    assert sum(lb.in_flight().values()) == 2
    asyncio.run(lb.async_log_success_event({"litellm_params": {"litellm_call_id": "c", "litellm_trace_id": "session"}}, None, 0, 0))
    assert sum(lb.in_flight().values()) == 1


async def _chunks(n: int):
    for i in range(n):
        yield i


def test_stream_abandoned_by_the_client_releases_without_blaming_the_backend():
    """A mid-stream disconnect logs no LiteLLM event; the proxy's stream wrapper is the only signal."""
    lb = LeastInFlight(choice=_first)

    async def consume_two_and_hang_up():
        stream = lb.async_post_call_streaming_iterator_hook(None, _chunks(10), {"litellm_call_id": "c"})
        assert [await stream.__anext__(), await stream.__anext__()] == [0, 1]
        await stream.aclose()

    _filter(lb, [_dep(1), _dep(2)], "c")
    asyncio.run(consume_two_and_hang_up())
    assert lb.in_flight() == {}
    assert lb.faulted() == set()


def test_stream_cancelled_with_its_task_releases():
    lb = LeastInFlight(choice=_first)

    async def endless():
        while True:
            yield 0
            await asyncio.sleep(0.01)

    async def cancel_mid_stream():
        async def consume():
            async for _ in lb.async_post_call_streaming_iterator_hook(None, endless(), {"litellm_call_id": "c"}):
                pass

        task = asyncio.ensure_future(consume())
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    _filter(lb, [_dep(1)], "c")
    asyncio.run(cancel_mid_stream())
    assert lb.in_flight() == {}


def test_completed_stream_is_left_to_the_success_event():
    lb = LeastInFlight(choice=_first)

    async def consume_all():
        return [c async for c in lb.async_post_call_streaming_iterator_hook(None, _chunks(3), {"litellm_call_id": "c"})]

    _filter(lb, [_dep(1)], "c")
    assert asyncio.run(consume_all()) == [0, 1, 2]
    assert sum(lb.in_flight().values()) == 1
    _success(lb, "c")
    assert lb.in_flight() == {}


def test_late_failure_event_for_an_earlier_attempt_keeps_the_retry_slot():
    """Failure logging is fire-and-forget, so attempt 1's event can arrive after attempt 2 reserved."""
    lb = LeastInFlight(choice=_first)
    deps = [_dep(1), _dep(2)]
    first = backend_key(_filter(lb, deps, "c"))
    second = backend_key(_filter(lb, deps, "c"))
    assert first != second
    _attempt_failed(lb, "c", first, _StatusError(502))
    assert lb.in_flight() == {second: 1}
    assert lb.faulted() == {first}
    _success(lb, "c")
    assert lb.in_flight() == {}


def test_double_release_never_goes_negative():
    lb = LeastInFlight(choice=_first)
    deps = [_dep(1), _dep(2)]
    _filter(lb, deps, "t")
    _success(lb, "t")
    _success(lb, "t")
    lb.release({"litellm_call_id": "never-reserved"})
    assert lb.in_flight() == {}
    assert backend_key(_filter(lb, deps, "u")) == backend_key(deps[0])


def test_unreleased_reservations_expire():
    clock = _Clock()
    lb = LeastInFlight(stale_after=60, clock=clock, choice=_first)
    _filter(lb, [_dep(1)], "lost")
    assert lb.in_flight() == {"http://10.0.0.1:8000/v1": 1}
    clock.now += 61
    assert lb.in_flight() == {}


def test_without_a_request_id_nothing_is_reserved():
    lb = LeastInFlight(choice=_first)
    _filter(lb, [_dep(1), _dep(2)], None)
    assert lb.in_flight() == {}


def test_empty_candidate_list_is_passed_through():
    lb = LeastInFlight()
    assert asyncio.run(lb.async_filter_deployments("g", [], None, {"litellm_trace_id": "t"})) == []


def _conv(task: str, system: str = "You are a fleet agent.") -> list:
    return [{"role": "system", "content": system}, {"role": "user", "content": task}]


def _filter_conv(lb: LeastInFlight, deployments: list, call: str, messages: list | None = None, session: str | None = None) -> dict:
    kwargs = {"litellm_call_id": call, "litellm_trace_id": "trace-%s" % call}
    if session:
        kwargs["litellm_session_id"] = session
    picked = asyncio.run(lb.async_filter_deployments("g", deployments, messages, kwargs))
    assert len(picked) == 1
    return picked[0]


def test_session_key_prefers_the_explicit_session_id():
    assert balancer.session_key({"litellm_session_id": "s1"}, _conv("task")) == "session:s1"
    assert balancer.session_key({"metadata": {"session_id": "s2"}}) == "session:s2"


def test_session_key_from_the_prompt_ignores_later_turns_and_the_system_tail():
    first = _conv("fix bug 6")
    later = first + [{"role": "assistant", "content": "ok"}, {"role": "tool", "content": "x" * 50}]
    retimed = [{"role": "system", "content": "You are a fleet agent." + " " * 3000 + "now=23:01"}, first[1]]
    retimed_later = [{"role": "system", "content": retimed[0]["content"].replace("23:01", "23:09")}, first[1]]
    key = balancer.session_key({}, first)
    assert key is not None and key.startswith("prompt:")
    assert balancer.session_key({}, later) == key
    assert balancer.session_key({}, retimed) == balancer.session_key({}, retimed_later)
    assert balancer.session_key({}, _conv("fix bug 7")) != key


def test_session_key_needs_a_user_message():
    assert balancer.session_key({}, None) is None
    assert balancer.session_key({}, [{"role": "system", "content": "only"}]) is None
    parts = [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]
    assert balancer.session_key({}, parts) == balancer.session_key({}, [{"role": "user", "content": "ab"}])


def test_a_conversation_stays_on_one_backend_across_calls():
    lb = LeastInFlight(choice=_first)
    deps = [_dep(i) for i in range(1, 5)]
    conv = _conv("card t_1: commit the fix")
    homes = set()
    for n in range(6):
        homes.add(backend_key(_filter_conv(lb, deps, "c%d" % n, conv)))
        _success(lb, "c%d" % n)
    assert len(homes) == 1


def test_conversations_spread_over_the_backends():
    lb = LeastInFlight(choice=_first)
    deps = [_dep(i) for i in range(1, 5)]
    homes = collections.Counter()
    for n in range(40):
        homes[backend_key(_filter_conv(lb, deps, "c%d" % n, _conv("task %d" % n)))] += 1
        _success(lb, "c%d" % n)
    assert len(homes) == 4 and min(homes.values()) >= 4


def test_a_busy_home_spills_to_least_in_flight_beyond_the_slack():
    lb = LeastInFlight(choice=_first, affinity_slack=2)
    deps = [_dep(1), _dep(2)]
    conv = _conv("sticky")
    home = backend_key(_filter_conv(lb, deps, "h", conv))
    _success(lb, "h")
    other = next(backend_key(d) for d in deps if backend_key(d) != home)
    for n in range(2):  # home at 2 in flight, other at 0: still within slack
        _filter_conv(lb, [d for d in deps if backend_key(d) == home], "load%d" % n)
    assert backend_key(_filter_conv(lb, deps, "s1", conv)) == home
    _filter_conv(lb, [d for d in deps if backend_key(d) == home], "load2")  # home now 4 vs 0
    assert backend_key(_filter_conv(lb, deps, "s2", conv)) == other


def test_a_faulted_home_is_not_preferred():
    lb = LeastInFlight(choice=_first)
    deps = [_dep(1), _dep(2)]
    conv = _conv("sticky")
    home = backend_key(_filter_conv(lb, deps, "h", conv))
    _attempt_failed(lb, "h", home, _StatusError(503))
    assert backend_key(_filter_conv(lb, deps, "after", conv)) != home


def test_a_retry_leaves_the_home_it_already_tried():
    lb = LeastInFlight(choice=_first)
    deps = [_dep(1), _dep(2), _dep(3)]
    conv = _conv("sticky")
    home = backend_key(_filter_conv(lb, deps, "r", conv))
    _attempt_failed(lb, "r", home, _StatusError(400))  # client error: no fault, but already tried
    assert backend_key(_filter_conv(lb, deps, "r", conv)) != home
    assert sum(lb.in_flight().values()) == 1


def test_removing_a_backend_moves_only_its_conversations():
    deps = [_dep(i) for i in range(1, 5)]
    convs = [_conv("task %d" % n) for n in range(40)]

    def homes(candidates):
        lb = LeastInFlight(choice=_first, affinity_slack=10**6)
        out = []
        for n, conv in enumerate(convs):
            out.append(backend_key(_filter_conv(lb, candidates, "c%d" % n, conv)))
            _success(lb, "c%d" % n)
        return out

    before, after = homes(deps), homes(deps[:3])
    gone = backend_key(deps[3])
    assert all(a == b for a, b in zip(before, after, strict=True) if a != gone)
    assert any(a == gone for a in before)


def test_module_exposes_the_instance_litellm_loads():
    assert isinstance(balancer.handler, LeastInFlight)


def test_loaded_by_litellm_it_subclasses_the_real_custom_logger(monkeypatch):
    """Inside the gateway the handler must be a CustomLogger, or LiteLLM never calls it."""

    class CustomLogger:
        def __init__(self, *args, **kwargs) -> None:
            self.initialized = True

    for name in ("litellm", "litellm.integrations"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    stub = types.ModuleType("litellm.integrations.custom_logger")
    stub.CustomLogger = CustomLogger
    monkeypatch.setitem(sys.modules, "litellm.integrations.custom_logger", stub)

    spec = importlib.util.spec_from_file_location("sparkrun_balancer", balancer.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert isinstance(module.handler, CustomLogger)
    assert module.handler.initialized
