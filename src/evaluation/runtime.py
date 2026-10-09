from __future__ import annotations

import json
import math
import os
import platform
import time
import threading
import atexit
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from src.model.device import model_dtype, select_device


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Each writer owns its temporary file; heartbeat and foreground saves must
    # never rename or overwrite another writer's in-progress file.
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=path.name + '.', suffix='.tmp', delete=False) as handle:
        tmp = Path(handle.name)
        try:
            handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
        except BaseException:
            handle.close()
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(float(x) for x in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


@dataclass
class ResumeLedger:
    """Persist evaluation progress and interruption-neutral benchmark timing.

    `benchmark_total_seconds` is intentionally stable across interruptions: it is
    first successful setup time + sum of completed-example timings. Repeated model
    setup after a resume is recorded as resume overhead, not benchmark latency.
    `active_operational_seconds` records actual active process time across sessions
    and also excludes downtime between processes.
    """

    state_path: Path
    progress_path: Path
    enabled: bool = True
    heartbeat_seconds: float = 5.0

    def __post_init__(self) -> None:
        self.state_path = Path(self.state_path)
        self.progress_path = Path(self.progress_path)
        self._session_started = time.perf_counter()
        self.state: dict[str, Any] = {
            'version': 1,
            'status': 'running',
            'active_operational_seconds': 0.0,
            'canonical_setup_seconds': None,
            'resume_setup_seconds': 0.0,
            'resume_count': 0,
            'completed_examples': 0,
        }
        existed = self.enabled and self.state_path.exists()
        if existed:
            loaded = json.loads(self.state_path.read_text(encoding='utf-8'))
            if isinstance(loaded, dict):
                self.state.update(loaded)
            self.state['resume_count'] = int(self.state.get('resume_count', 0)) + 1
        self.state['status'] = 'running'
        self._active_base = float(self.state.get('active_operational_seconds', 0.0) or 0.0)
        self._heartbeat_stop = threading.Event()
        self._save_lock = threading.RLock()
        self._heartbeat_thread = None
        if self.enabled and self.heartbeat_seconds > 0:
            self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._heartbeat_thread.start()
        atexit.register(self.close)
        self._save()

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(self.heartbeat_seconds):
            try:
                self._save()
            except Exception:
                pass

    def _active_now(self) -> float:
        return self._active_base + max(0.0, time.perf_counter() - self._session_started)

    def _save(self) -> None:
        if not self.enabled:
            return
        with self._save_lock:
            self.state['active_operational_seconds'] = self._active_now()
            self.state['updated_at_unix'] = time.time()
            _atomic_write(self.state_path, self.state)

    def record_setup(self, seconds: float) -> None:
        seconds = float(seconds)
        if self.state.get('canonical_setup_seconds') is None:
            self.state['canonical_setup_seconds'] = seconds
        else:
            self.state['resume_setup_seconds'] = float(self.state.get('resume_setup_seconds', 0.0) or 0.0) + seconds
        self._save()

    def load_completed(self) -> dict[str, dict[str, Any]]:
        completed: dict[str, dict[str, Any]] = {}
        if not self.enabled or not self.progress_path.exists():
            return completed
        with self.progress_path.open('r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if 'example_id' in row:
                    completed[str(row['example_id'])] = row
        self.state['completed_examples'] = len(completed)
        self._save()
        return completed

    def append(self, row: dict[str, Any]) -> None:
        if self.enabled:
            self.progress_path.parent.mkdir(parents=True, exist_ok=True)
            with self.progress_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        self.state['completed_examples'] = int(self.state.get('completed_examples', 0)) + 1
        self._save()

    def mark_interrupted(self) -> None:
        self.state['status'] = 'interrupted'
        self._save()
        self.close()

    def mark_complete(self) -> None:
        self.state['status'] = 'complete'
        self._save()
        self.close()

    def close(self) -> None:
        stop = getattr(self, '_heartbeat_stop', None)
        if stop is not None:
            stop.set()
        thread = getattr(self, '_heartbeat_thread', None)
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._heartbeat_thread = None

    def snapshot(self) -> dict[str, Any]:
        with self._save_lock:
            self._save()
            return dict(self.state)


def inference_setup_metadata(*, model_repo_id: str, model_path: str | Path | None, quantization: str,
                             max_new_tokens: int, batch_size: int = 1, do_sample: bool = False,
                             use_cache: bool = True, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    meta: dict[str, Any] = {
        'model_repo_id': model_repo_id,
        'model_path': None if model_path is None else str(model_path),
        'quantization': quantization,
        'dtype': str(model_dtype(select_device())).removeprefix('torch.'),
        'device': f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu',
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'batch_size': int(batch_size),
        'max_new_tokens': int(max_new_tokens),
        'do_sample': bool(do_sample),
        'use_cache': bool(use_cache),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        meta.update({
            'gpu_name': props.name,
            'gpu_total_vram_mib': props.total_memory / (1024 ** 2),
        })
    if extra:
        meta.update(extra)
    return meta


def telemetry_summary(rows: list[dict[str, Any]], ledger_state: dict[str, Any], *,
                      latency_field: str = 'latency_seconds', input_field: str = 'input_tokens',
                      output_field: str = 'output_tokens') -> dict[str, Any]:
    latencies = [float(r.get(latency_field, 0.0) or 0.0) for r in rows]
    input_tokens = [int(r.get(input_field, 0) or 0) for r in rows]
    output_tokens = [int(r.get(output_field, 0) or 0) for r in rows]
    inference_seconds = sum(latencies)
    total_in = sum(input_tokens)
    total_out = sum(output_tokens)
    setup = float(ledger_state.get('canonical_setup_seconds') or 0.0)
    active = float(ledger_state.get('active_operational_seconds') or 0.0)
    resume_setup = float(ledger_state.get('resume_setup_seconds') or 0.0)
    return {
        'benchmark_total_seconds': setup + inference_seconds,
        'canonical_model_setup_seconds': setup,
        'inference_seconds': inference_seconds,
        'average_seconds_per_prompt': inference_seconds / len(rows) if rows else None,
        'p50_seconds_per_prompt': _percentile(latencies, 0.50),
        'p95_seconds_per_prompt': _percentile(latencies, 0.95),
        'p99_seconds_per_prompt': _percentile(latencies, 0.99),
        'total_input_tokens': total_in,
        'total_output_tokens': total_out,
        'total_tokens': total_in + total_out,
        'average_input_tokens_per_prompt': total_in / len(rows) if rows else None,
        'average_output_tokens_per_prompt': total_out / len(rows) if rows else None,
        'average_total_tokens_per_prompt': (total_in + total_out) / len(rows) if rows else None,
        'output_tokens_per_second': total_out / inference_seconds if inference_seconds > 0 else None,
        'all_tokens_per_second': (total_in + total_out) / inference_seconds if inference_seconds > 0 else None,
        'active_operational_seconds': active,
        'resume_setup_overhead_seconds': resume_setup,
        'resume_count': int(ledger_state.get('resume_count', 0) or 0),
        'interruption_neutral_timing': True,
    }
