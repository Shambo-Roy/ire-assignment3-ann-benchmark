import json
import os

import faiss
import numpy as np
import psutil
import pytest

from src import threads
from src.timing import BenchmarkLockError, benchmark_lock, time_search


@pytest.fixture(autouse=True)
def restore_threads():
    proc = psutil.Process()
    aff = proc.cpu_affinity()
    omp = faiss.omp_get_max_threads()
    blas = threads.faiss_blas().get()
    yield
    proc.cpu_affinity(aff)
    faiss.omp_set_num_threads(omp)
    if blas is not None:
        threads.faiss_blas().set(blas)


def test_topology_covers_all_logical_cpus():
    topo, source = threads.cpu_topology()
    assert sorted(c.logical for c in topo) == list(range(psutil.cpu_count(logical=True)))
    fast = threads.fast_cpus(topo)
    assert fast and all(c.efficiency_class == max(t.efficiency_class for t in topo) for c in fast)
    assert threads.search_cpu(topo) in fast


def test_search_phase_is_single_threaded_and_pinned():
    st = threads.search_phase()
    assert st.omp_threads == 1 and faiss.omp_get_max_threads() == 1
    assert len(st.affinity) == 1 and psutil.Process().cpu_affinity() == st.affinity
    if st.blas_available:
        assert st.blas_threads == 1 == threads.faiss_blas().get()


def test_build_phase_uses_all_fast_cpus():
    st = threads.build_phase()
    fast = sorted(c.logical for c in threads.fast_cpus(threads.cpu_topology()[0]))
    assert st.affinity == fast
    assert st.omp_threads == len(fast)
    if st.blas_available:
        assert st.blas_threads == len(fast)


def test_faiss_openblas_found_on_this_platform():
    # design revision B found it on the dev machine; other platforms may legitimately lack it
    if os.name == "nt":
        assert threads.faiss_blas().available


def test_time_search_single_and_batch_return_same_ids():
    rng = np.random.default_rng(0)
    x = rng.random((2000, 16), dtype=np.float32)
    idx = faiss.IndexFlatL2(16)
    idx.add(x)
    threads.search_phase()
    s = time_search(idx, x[:30], k=5, repeats=2, mode="single")
    b = time_search(idx, x[:30], k=5, repeats=2, mode="batch")
    assert len(s.qps) == 2 and s.qps_median > 0
    np.testing.assert_array_equal(s.ids, b.ids)
    np.testing.assert_array_equal(s.ids[:, 0], np.arange(30))   # each point is its own NN
    row = s.as_row()
    assert row["search_mode"] == "single" and row["n_timed_queries"] == 30 and row["n_repeats"] == 2


def test_lock_blocks_second_holder_and_cleans_up(tmp_path):
    with benchmark_lock(tmp_path, "a") as p:
        assert json.loads(p.read_text())["pid"] == os.getpid()
        with pytest.raises(BenchmarkLockError):
            with benchmark_lock(tmp_path, "b"):
                pass
    assert not (tmp_path / ".benchmark.lock").exists()


def test_stale_lock_is_replaced(tmp_path):
    (tmp_path / ".benchmark.lock").write_text(json.dumps({"pid": 2**22 + 12345}))
    with benchmark_lock(tmp_path, "c"):
        pass


def test_thread_check_is_inconclusive_for_passes_shorter_than_timer_resolution():
    from src.timing import MIN_CHECK_WALL_S, TimingResult
    ids = np.zeros((1, 1), dtype=np.int64)
    # 1,000 queries at 100k QPS = 10 ms per pass: below 4 process_time ticks, so no verdict
    short = TimingResult("single", 1000, 3, [1e5] * 3, [1.7] * 3, ids)
    assert short.thread_check(True) == "INCONCLUSIVE"
    # same ratio on a long pass is a real failure; a ratio of about 1 passes
    long_fail = TimingResult("single", 1000, 3, [100.0] * 3, [1.7] * 3, ids)
    long_ok = TimingResult("single", 1000, 3, [100.0] * 3, [1.0] * 3, ids)
    assert long_fail.thread_check(True) == "FAIL" and long_ok.thread_check(True) == "PASS"
    assert short.thread_check(False) == "n/a"
    assert 1000 / 100.0 > MIN_CHECK_WALL_S
