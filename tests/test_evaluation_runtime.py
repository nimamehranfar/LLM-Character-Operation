from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time

from src.evaluation import runtime


def test_concurrent_atomic_writers_own_distinct_temporary_files(monkeypatch, tmp_path):
    """Force both writes to finish before either can rename its temporary file."""
    barrier = threading.Barrier(2)
    rename_lock = threading.Lock()
    replace = runtime.os.replace
    sources = []

    def overlapping_replace(source, target):
        sources.append(source)
        barrier.wait(timeout=5)
        # Windows does not guarantee concurrent replacement of one destination.
        # Both files are already written, so this still reproduces the old
        # shared-temp-file bug while matching the ledger's serialized renames.
        with rename_lock:
            replace(source, target)

    monkeypatch.setattr(runtime.os, 'replace', overlapping_replace)
    path = tmp_path / 'state.json'
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(runtime._atomic_write, path, {'writer': i}) for i in range(2)]
        for future in futures:
            future.result(timeout=10)
    assert len(set(sources)) == 2
    assert json.loads(path.read_text())['writer'] in (0, 1)
    assert not list(tmp_path.glob('*.tmp'))


def test_heartbeat_and_foreground_saves_are_serialized(monkeypatch, tmp_path):
    ledger = runtime.ResumeLedger(tmp_path / 'state.json', tmp_path / 'progress.jsonl',
                                  heartbeat_seconds=0.001)
    atomic_write = runtime._atomic_write
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def slow_write(path, payload):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            time.sleep(0.002)
            atomic_write(path, payload)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(runtime, '_atomic_write', slow_write)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(ledger._save) for _ in range(20)]
            for future in futures:
                future.result(timeout=10)
        for i in range(10):
            ledger.append({'example_id': str(i), 'latency_seconds': 0.1})
        ledger.mark_complete()
        saved = json.loads(ledger.state_path.read_text())
        assert saved['status'] == 'complete'
        assert saved['completed_examples'] == 10
        assert maximum_active == 1
        assert len(ledger.progress_path.read_text().splitlines()) == 10
        assert not list(tmp_path.glob('*.tmp'))
    finally:
        ledger.close()
