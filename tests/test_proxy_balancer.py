"""Least-in-flight balancing callback for the LiteLLM gateway (``proxy/balancer.py``).

The module is loaded by LiteLLM from beside the generated config, so these
tests drive it directly: no LiteLLM process, no network.
"""

from __future__ import annotations

import asyncio
import collections

from sparkrun.proxy import balancer
from sparkrun.proxy.balancer import LeastInFlight, backend_key


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


def _filter(lb: LeastInFlight, deployments: list, trace: str | None, call: str | None = None) -> dict:
    kwargs = {"litellm_trace_id": trace}
    if call:
        kwargs["litellm_call_id"] = call
    picked = asyncio.run(lb.async_filter_deployments("g", deployments, None, kwargs))
    assert len(picked) == 1
    return picked[0]


def _success(lb: LeastInFlight, trace: str) -> None:
    # Logging events carry the request's trace id in litellm_params; their own
    # top-level litellm_trace_id is a different id and must be ignored.
    kwargs = {"litellm_trace_id": "logging-object-id", "litellm_params": {"litellm_trace_id": trace}}
    asyncio.run(lb.async_log_success_event(kwargs, None, 0, 0))


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
    asyncio.run(lb.async_log_failure_event({"litellm_params": {"litellm_trace_id": "t"}}, None, 0, 0))
    assert lb.in_flight() == {}


def test_proxy_failure_hook_releases_by_call_id():
    """LiteLLM logs a failure only for the first attempt; the proxy hook covers the last retry."""
    lb = LeastInFlight(choice=_first)
    deps = [_dep(1), _dep(2)]
    _filter(lb, deps, "t", call="c")
    asyncio.run(lb.async_log_failure_event({"litellm_params": {"litellm_trace_id": "t"}}, None, 0, 0))
    _filter(lb, deps, "t", call="c")  # the retry reserves again
    assert sum(lb.in_flight().values()) == 1
    asyncio.run(lb.async_post_call_failure_hook({"litellm_call_id": "c"}, RuntimeError("down"), None))
    assert lb.in_flight() == {}
    asyncio.run(lb.async_post_call_failure_hook({"litellm_call_id": "c"}, RuntimeError("down"), None))
    asyncio.run(lb.async_post_call_failure_hook({}, RuntimeError("down"), None))
    assert lb.in_flight() == {}


def test_double_release_never_goes_negative():
    lb = LeastInFlight(choice=_first)
    deps = [_dep(1), _dep(2)]
    _filter(lb, deps, "t")
    _success(lb, "t")
    _success(lb, "t")
    lb.release("never-reserved")
    assert lb.in_flight() == {}
    assert backend_key(_filter(lb, deps, "u")) == backend_key(deps[0])


def test_unreleased_reservations_expire():
    clock = _Clock()
    lb = LeastInFlight(stale_after=60, clock=clock, choice=_first)
    _filter(lb, [_dep(1)], "lost")
    assert lb.in_flight() == {"http://10.0.0.1:8000/v1": 1}
    clock.now += 61
    assert lb.in_flight() == {}


def test_without_a_trace_id_nothing_is_reserved():
    lb = LeastInFlight(choice=_first)
    _filter(lb, [_dep(1), _dep(2)], None)
    assert lb.in_flight() == {}


def test_empty_candidate_list_is_passed_through():
    lb = LeastInFlight()
    assert asyncio.run(lb.async_filter_deployments("g", [], None, {"litellm_trace_id": "t"})) == []


def test_module_exposes_the_instance_litellm_loads():
    assert isinstance(balancer.handler, LeastInFlight)
