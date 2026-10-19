"""Regression tests for issue #267 — benchmark identity vs. the measured node set.

Two runs of one recipe against *different* nodes shared a benchmark id and
therefore a state directory, so a concurrent pair silently served one node's
cached results as the other's.  The reported cost was real: a genuinely faulty
node (36% slower) was reported as identical to its siblings.

Four separate guards, tested here:

1. hosts participate in ``derive_benchmark_id``
2. state records its host set and refuses reuse against a different one
3. the exported result records the node set (pseudonymously)
4. one live run owns its state directory exclusively
"""

from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path

import pytest
import yaml

from sparkrun.benchmarking.base import host_meta, redact_hosts
from sparkrun.orchestration.job_metadata import INTENT_ID_LEN, PLACEMENT_TOKEN_LEN
from sparkrun.benchmarking.run_state import (
    BenchmarkRunState,
    StateDirLocked,
    canonical_host_key,
    derive_benchmark_id,
    hold_state_dir,
)


COMMON = ("llama-benchy", "default", {"pp": [2048]}, None)


# ---------------------------------------------------------------------------
# 1. hosts are part of the identity
# ---------------------------------------------------------------------------


def test_different_hosts_yield_different_benchmark_ids():
    """The reported collision: same recipe, different node, one state dir."""
    a = derive_benchmark_id("cluster-abc", *COMMON, hosts=["192.168.50.218"])
    b = derive_benchmark_id("cluster-abc", *COMMON, hosts=["192.168.50.221"])
    assert a != b


def test_same_hosts_yield_the_same_benchmark_id():
    """Resume across relaunches must still work when placement is unchanged."""
    a = derive_benchmark_id("cluster-abc", *COMMON, hosts=["node-a", "node-b"])
    b = derive_benchmark_id("cluster-abc", *COMMON, hosts=["node-a", "node-b"])
    assert a == b


def test_host_order_does_not_affect_the_id():
    """Placement may reorder a host list without changing what was measured."""
    a = derive_benchmark_id("cluster-abc", *COMMON, hosts=["node-a", "node-b"])
    b = derive_benchmark_id("cluster-abc", *COMMON, hosts=["node-b", "node-a"])
    assert a == b


def test_a_host_subset_is_a_different_measurement():
    """Two of four nodes is not the same run as all four."""
    a = derive_benchmark_id("cluster-abc", *COMMON, hosts=["a", "b", "c", "d"])
    b = derive_benchmark_id("cluster-abc", *COMMON, hosts=["a", "b"])
    assert a != b


def test_omitting_hosts_preserves_the_legacy_id():
    """An absent/empty host set hashes exactly as before the field existed."""
    legacy = derive_benchmark_id("cluster-abc", *COMMON)
    assert derive_benchmark_id("cluster-abc", *COMMON, hosts=None) == legacy
    assert derive_benchmark_id("cluster-abc", *COMMON, hosts=[]) == legacy


def test_hosts_do_not_reintroduce_placement_token_sensitivity():
    """The random per-launch placement token must still be excluded.

    Hosts and the placement token are easy to conflate; hashing the token
    would break resume across relaunches, which is why only the host set is
    folded in.
    """
    from sparkrun.orchestration.job_metadata import generate_cluster_id

    intent = "a" * INTENT_ID_LEN
    cid_a = generate_cluster_id(intent, "1" * PLACEMENT_TOKEN_LEN)
    cid_b = generate_cluster_id(intent, "2" * PLACEMENT_TOKEN_LEN)
    hosts = ["node-a"]
    assert derive_benchmark_id(cid_a, *COMMON, hosts=hosts) == derive_benchmark_id(cid_b, *COMMON, hosts=hosts)


def test_canonical_host_key_is_sorted_and_empty_for_none():
    assert canonical_host_key(["b", "a"]) == canonical_host_key(["a", "b"])
    assert canonical_host_key(None) == ""
    assert canonical_host_key([]) == ""


# ---------------------------------------------------------------------------
# 2. state records its host set and refuses a mismatched reuse
# ---------------------------------------------------------------------------


def _state(**kw) -> BenchmarkRunState:
    base = dict(
        benchmark_id="bench_000000000000",
        cluster_id="sparkrun_%s_%s" % ("a" * INTENT_ID_LEN, "b" * PLACEMENT_TOKEN_LEN),
        recipe_qualified_name="@r/recipe",
        framework="llama-benchy",
        profile="default",
        base_args={},
        schedule=[],
    )
    base.update(kw)
    return BenchmarkRunState(**base)


def test_matches_hosts_accepts_the_recorded_set_in_any_order():
    st = _state(host_list=["node-a", "node-b"])
    assert st.matches_hosts(["node-b", "node-a"])


def test_matches_hosts_rejects_a_different_set():
    st = _state(host_list=["192.168.50.218"])
    assert not st.matches_hosts(["192.168.50.221"])


def test_matches_hosts_accepts_legacy_state_with_no_recorded_hosts():
    """Pre-#267 state records nothing; refusing every such resume is worse."""
    st = _state(host_list=[])
    assert st.matches_hosts(["anything"])


def test_host_list_round_trips_through_save_and_load(tmp_path: Path):
    st = _state(benchmark_id="bench_abc123abc123", host_list=["node-a", "node-b"])
    st.save(str(tmp_path))

    loaded = BenchmarkRunState.load("bench_abc123abc123", str(tmp_path))
    assert loaded is not None
    assert loaded.host_list == ["node-a", "node-b"]


# ---------------------------------------------------------------------------
# 3. the exported result records the node set, pseudonymously
# ---------------------------------------------------------------------------


def test_host_meta_records_identity_and_count():
    meta = host_meta(["node-a", "node-b"])
    assert meta["hosts"] == ["node-a", "node-b"]
    assert meta["node_count"] == 2
    assert "hosts_redacted" not in meta


def test_redacted_host_meta_leaks_no_address_but_still_distinguishes():
    meta = host_meta(["192.168.50.218", "192.168.50.221"], redact=True)
    assert meta["hosts_redacted"] is True
    assert meta["node_count"] == 2
    blob = json.dumps(meta)
    assert "192.168.50.218" not in blob and "192.168.50.221" not in blob
    # The whole point: two different nodes remain two different entries.
    assert meta["hosts"][0] != meta["hosts"][1]


def test_redaction_is_stable_across_calls():
    """A digest must match itself, or cross-file comparison is impossible."""
    assert redact_hosts(["node-a"]) == redact_hosts(["node-a"])
    assert redact_hosts(["node-a"]) != redact_hosts(["node-b"])


def test_export_records_hosts_and_resume_provenance(tmp_path: Path):
    from sparkrun.benchmarking.base import export_results
    from sparkrun.core.recipe import Recipe

    recipe = Recipe.from_dict({"name": "r", "model": "org/m", "container": "img:1", "runtime": "vllm"})
    out = tmp_path / "result.yaml"
    export_results(
        recipe=recipe,
        hosts=["192.168.50.218"],
        tp=1,
        cluster_id="sparkrun_%s_%s" % ("a" * INTENT_ID_LEN, "b" * PLACEMENT_TOKEN_LEN),
        framework_name="llama-benchy",
        profile_name="default",
        args={},
        results={"rows": []},
        output_path=out,
        resumed=True,
        measured_at="2026-08-20T10:00:00+00:00",
    )

    bench = yaml.safe_load(out.read_text())["sparkrun_benchmark"]
    assert bench["cluster"]["hosts"] == redact_hosts(["192.168.50.218"])
    assert bench["cluster"]["node_count"] == 1
    # A re-emitted result must not read as a fresh measurement: ``timestamp``
    # is when this file was written, ``measured_at`` when the numbers were taken.
    assert bench["benchmark"]["resumed"] is True
    assert bench["benchmark"]["measured_at"] == "2026-08-20T10:00:00+00:00"
    assert "192.168.50.218" not in out.read_text()


def test_export_marks_a_fresh_run_as_not_resumed(tmp_path: Path):
    from sparkrun.benchmarking.base import export_results
    from sparkrun.core.recipe import Recipe

    recipe = Recipe.from_dict({"name": "r", "model": "org/m", "container": "img:1", "runtime": "vllm"})
    out = tmp_path / "result.yaml"
    export_results(
        recipe=recipe,
        hosts=["h1"],
        tp=1,
        cluster_id="sparkrun_%s_%s" % ("a" * INTENT_ID_LEN, "b" * PLACEMENT_TOKEN_LEN),
        framework_name="llama-benchy",
        profile_name=None,
        args={},
        results={"rows": []},
        output_path=out,
    )
    bench = yaml.safe_load(out.read_text())["sparkrun_benchmark"]
    assert bench["benchmark"]["resumed"] is False
    assert "measured_at" not in bench["benchmark"]


# ---------------------------------------------------------------------------
# 4. exclusive ownership of the state directory
# ---------------------------------------------------------------------------


def test_lock_is_created_and_released(tmp_path: Path):
    lock = tmp_path / "benchmarks" / "bench_lock01" / "run.lock"
    with hold_state_dir("bench_lock01", str(tmp_path)):
        assert lock.exists()
    assert not lock.exists()


def test_second_acquisition_is_refused_while_held(tmp_path: Path):
    with hold_state_dir("bench_lock02", str(tmp_path)):
        with pytest.raises(StateDirLocked) as exc:
            with hold_state_dir("bench_lock02", str(tmp_path)):
                pytest.fail("acquired a directory another run holds")
    # The error must name the holder — "something has it" is not actionable.
    assert exc.value.info["pid"] == os.getpid()


def test_distinct_benchmark_ids_do_not_contend(tmp_path: Path):
    with hold_state_dir("bench_lock03", str(tmp_path)):
        with hold_state_dir("bench_lock04", str(tmp_path)):
            pass


def test_lock_is_released_when_the_body_raises(tmp_path: Path):
    lock = tmp_path / "benchmarks" / "bench_lock05" / "run.lock"
    with pytest.raises(ValueError):
        with hold_state_dir("bench_lock05", str(tmp_path)):
            raise ValueError("boom")
    assert not lock.exists()
    # ...and the directory is immediately reusable.
    with hold_state_dir("bench_lock05", str(tmp_path)):
        assert lock.exists()


def test_stale_lock_from_a_dead_local_pid_is_reclaimed(tmp_path: Path):
    from sparkrun.core.pending_ops import lock_hostname

    sdir = tmp_path / "benchmarks" / "bench_lock06"
    sdir.mkdir(parents=True)
    dead_pid = _a_dead_pid()
    (sdir / "run.lock").write_text(
        json.dumps({"benchmark_id": "bench_lock06", "pid": dead_pid, "host": lock_hostname(), "started_at": 0.0})
    )

    with hold_state_dir("bench_lock06", str(tmp_path)):
        held = json.loads((sdir / "run.lock").read_text())
        assert held["pid"] == os.getpid()


def test_lock_from_another_host_is_honored_until_it_ages_out(tmp_path: Path):
    """A remote PID says nothing locally, so only the age ceiling releases it."""
    import time

    from sparkrun.benchmarking.run_state import LOCK_MAX_AGE_SECONDS

    sdir = tmp_path / "benchmarks" / "bench_lock07"
    sdir.mkdir(parents=True)
    lock = sdir / "run.lock"

    lock.write_text(json.dumps({"pid": 1, "host": "some-other-box", "started_at": time.time()}))
    with pytest.raises(StateDirLocked):
        with hold_state_dir("bench_lock07", str(tmp_path)):
            pytest.fail("stole a live lock held on another host")

    lock.write_text(json.dumps({"pid": 1, "host": "some-other-box", "started_at": time.time() - LOCK_MAX_AGE_SECONDS - 1}))
    with hold_state_dir("bench_lock07", str(tmp_path)):
        pass


def test_corrupt_lock_is_reclaimed(tmp_path: Path):
    """A half-written lock must not wedge the directory forever."""
    sdir = tmp_path / "benchmarks" / "bench_lock08"
    sdir.mkdir(parents=True)
    (sdir / "run.lock").write_text("{not json")

    with hold_state_dir("bench_lock08", str(tmp_path)):
        pass


def test_a_run_does_not_delete_another_runs_lock_on_exit(tmp_path: Path):
    """Releasing checks ownership: rmtree+recreate must not cross runs."""
    sdir = tmp_path / "benchmarks" / "bench_lock09"
    lock = sdir / "run.lock"

    with hold_state_dir("bench_lock09", str(tmp_path)):
        # Simulate the directory being reclaimed by a different live run.
        lock.write_text(json.dumps({"pid": 999999, "host": "other-box", "started_at": 1.0}))

    assert lock.exists(), "released a lock owned by another run"


def test_clearing_state_preserves_the_lock(tmp_path: Path):
    """``--fresh`` and the host-mismatch discard both run *inside* the lock.

    Deleting the directory outright would take our own lock with it, leaving
    the benchmark unlocked exactly while it writes its fresh results.
    """
    from sparkrun.benchmarking.run_state import clear_state_dir

    sdir = tmp_path / "benchmarks" / "bench_clear01"
    with hold_state_dir("bench_clear01", str(tmp_path)):
        (sdir / "runs").mkdir(parents=True)
        (sdir / "runs" / "000.json").write_text("{}")
        (sdir / "state.yaml").write_text("{}")

        clear_state_dir("bench_clear01", str(tmp_path))

        assert not (sdir / "state.yaml").exists()
        assert not (sdir / "runs").exists()
        assert (sdir / "run.lock").exists(), "state reset dropped the lock"

        # Still exclusive.
        with pytest.raises(StateDirLocked):
            with hold_state_dir("bench_clear01", str(tmp_path)):
                pytest.fail("acquired a directory whose lock should have survived the reset")


def test_clearing_an_absent_state_dir_is_a_no_op(tmp_path: Path):
    from sparkrun.benchmarking.run_state import clear_state_dir

    clear_state_dir("bench_never_existed", str(tmp_path))


def _acquire_in_child(cache_dir: str, benchmark_id: str, q) -> None:
    try:
        with hold_state_dir(benchmark_id, cache_dir):
            q.put("acquired")
    except StateDirLocked:
        q.put("refused")


def test_a_genuinely_concurrent_process_is_refused(tmp_path: Path):
    """The reported failure mode, across real processes rather than mocks."""
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    with hold_state_dir("bench_lock10", str(tmp_path)):
        proc = ctx.Process(target=_acquire_in_child, args=(str(tmp_path), "bench_lock10", q))
        proc.start()
        proc.join(timeout=60)
        assert proc.exitcode == 0

    assert q.get(timeout=10) == "refused"


def _noop() -> None:
    """Module-level no-op target (lambdas are unpicklable under macOS spawn)."""


def _a_dead_pid() -> int:
    """Return a PID that is not running."""
    proc = multiprocessing.Process(target=_noop)
    proc.start()
    proc.join()
    return proc.pid
