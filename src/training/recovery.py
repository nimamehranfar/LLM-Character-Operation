from __future__ import annotations

import atexit
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import tempfile
import threading
import time
import traceback
from typing import Any, Callable

import torch


RECOVERY_VERSION = 3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def local_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="microseconds")


def safe_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _fsync_path(path: Path) -> None:
    try:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    except OSError:
        pass


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        torch.save(value, tmp)
        _fsync_path(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def append_jsonl_durable(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def process_alive(pid: int, hostname: str | None = None) -> bool:
    if pid <= 0:
        return False
    if hostname and hostname != socket.gethostname():
        return False
    if os.name == "nt":
        try:
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def capture_rng_state(extra_rng_state: Any = None) -> dict[str, Any]:
    import random
    state: dict[str, Any] = {
        "python_random": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "extra_random": extra_rng_state,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any], *, extra_rng: Any = None) -> None:
    import random
    if not state:
        return
    if state.get("python_random") is not None:
        random.setstate(state["python_random"])
    if state.get("torch_cpu") is not None:
        torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if extra_rng is not None and state.get("extra_random") is not None:
        extra_rng.setstate(state["extra_random"])


class GracefulStop(RuntimeError):
    pass


@dataclass
class RunPaths:
    run_dir: Path

    @property
    def state(self) -> Path:
        return self.run_dir / "run_state.json"

    @property
    def events(self) -> Path:
        return self.run_dir / "events.jsonl"

    @property
    def heartbeat(self) -> Path:
        return self.run_dir / "heartbeat.json"

    @property
    def lock(self) -> Path:
        return self.run_dir / "run.lock"

    @property
    def crashes(self) -> Path:
        return self.run_dir / "crashes"

    @property
    def checkpoints(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def cache(self) -> Path:
        return self.run_dir / "cache"

    @property
    def eval(self) -> Path:
        return self.run_dir / "evaluation"

    @property
    def artifacts(self) -> Path:
        return self.run_dir / "artifacts"


class RecoveryManager:
    """Durable run-state manager.

    Catchable signals request a stop and are serviced at the next training safe point.
    Hard process destruction cannot execute Python cleanup; the next launch detects the
    stale lock/heartbeat and records an inferred crash before resuming from the last
    atomic checkpoint.
    """

    def __init__(
        self,
        *,
        run_dir: Path,
        config_fingerprint: str,
        config_path: str,
        heartbeat_seconds: float = 15.0,
        create: bool,
    ) -> None:
        self.paths = RunPaths(run_dir)
        self.config_fingerprint = config_fingerprint
        self.config_path = config_path
        self.heartbeat_seconds = float(heartbeat_seconds)
        self.pid = os.getpid()
        self.hostname = socket.gethostname()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self.stop_requested = False
        self.stop_signal: str | None = None
        self.emergency_saver: Callable[[], None] | None = None
        self._closed = False
        # Active-runtime accounting excludes time spent between stopped/resumed processes.
        self._session_perf_start = time.perf_counter()
        self._active_seconds_base = 0.0

        self.paths.run_dir.mkdir(parents=True, exist_ok=True)
        for directory in (self.paths.crashes, self.paths.checkpoints, self.paths.cache, self.paths.eval, self.paths.artifacts):
            directory.mkdir(parents=True, exist_ok=True)

        if create:
            now = utc_now()
            self.state = {
                "recovery_version": RECOVERY_VERSION,
                "config_fingerprint": config_fingerprint,
                "config_path": config_path,
                "run_id": self.paths.run_dir.name,
                "status": "running",
                "phase": "initializing",
                "stage": "initializing",
                "epoch": None,
                "cursor": 0,
                "total": None,
                "started_at_utc": now,
                "started_at_local": local_now(),
                "updated_at_utc": now,
                "updated_at_local": local_now(),
                "pid": self.pid,
                "hostname": self.hostname,
                "completed_stages": [],
                "active_seconds": 0.0,
            }
            atomic_write_json(self.paths.state, self.state)
        else:
            self.state = load_json(self.paths.state)
            if not isinstance(self.state, dict):
                raise RuntimeError(f"Missing/invalid run state: {self.paths.state}")
            if self.state.get("config_fingerprint") != config_fingerprint:
                raise RuntimeError("Recovery run config fingerprint does not match requested config")
            self._active_seconds_base = float(self.state.get("active_seconds", 0.0) or 0.0)
            self._record_unclean_if_needed()
            self.state.update({"status": "running", "pid": self.pid, "hostname": self.hostname})
            self._save_state()

        self._active_seconds_base = float(self.state.get("active_seconds", self._active_seconds_base) or 0.0)
        self._session_perf_start = time.perf_counter()
        self._acquire_lock()
        self._install_signal_handlers()
        self._start_heartbeat()
        atexit.register(self.close)
        self.log("run_open", resumed=not create, run_dir=str(self.paths.run_dir))

    @staticmethod
    def fingerprint(config: dict[str, Any], *, script_version: str) -> str:
        return canonical_hash({"recovery_version": RECOVERY_VERSION, "script_version": script_version, "config": config})

    @classmethod
    def create_or_resume(
        cls,
        *,
        checkpoint_root: Path,
        config: dict[str, Any],
        config_path: str,
        script_version: str,
        heartbeat_seconds: float,
        fresh: bool,
        explicit_run_dir: Path | None,
    ) -> "RecoveryManager":
        fp = cls.fingerprint(config, script_version=script_version)
        runs_root = checkpoint_root / "runs" / fp
        runs_root.mkdir(parents=True, exist_ok=True)

        if explicit_run_dir is not None:
            run_dir = explicit_run_dir.resolve()
            return cls(
                run_dir=run_dir,
                config_fingerprint=fp,
                config_path=config_path,
                heartbeat_seconds=heartbeat_seconds,
                create=not (run_dir / "run_state.json").exists(),
            )

        if not fresh:
            candidates: list[tuple[float, Path]] = []
            for state_path in runs_root.glob("run_*/run_state.json"):
                state = load_json(state_path, {})
                if state.get("config_fingerprint") != fp:
                    continue
                if state.get("status") == "complete":
                    continue
                candidates.append((state_path.stat().st_mtime, state_path.parent))
            if candidates:
                _, run_dir = max(candidates, key=lambda item: item[0])
                return cls(
                    run_dir=run_dir,
                    config_fingerprint=fp,
                    config_path=config_path,
                    heartbeat_seconds=heartbeat_seconds,
                    create=False,
                )

        run_dir = runs_root / f"run_{safe_stamp()}"
        return cls(
            run_dir=run_dir,
            config_fingerprint=fp,
            config_path=config_path,
            heartbeat_seconds=heartbeat_seconds,
            create=True,
        )

    @classmethod
    def find_latest_run(
        cls,
        *,
        checkpoint_root: Path,
        config: dict[str, Any],
        script_version: str,
    ) -> Path | None:
        fp = cls.fingerprint(config, script_version=script_version)
        runs_root = checkpoint_root / "runs" / fp
        if not runs_root.exists():
            return None
        states = list(runs_root.glob("run_*/run_state.json"))
        if not states:
            return None
        return max(states, key=lambda p: p.stat().st_mtime).parent

    @classmethod
    def print_status(
        cls,
        *,
        checkpoint_root: Path,
        config: dict[str, Any],
        script_version: str,
    ) -> None:
        run_dir = cls.find_latest_run(checkpoint_root=checkpoint_root, config=config, script_version=script_version)
        if run_dir is None:
            print("No recovery run exists for this config.")
            return
        paths = RunPaths(run_dir)
        state = load_json(paths.state, {})
        heartbeat = load_json(paths.heartbeat, {})
        print(f"Run: {run_dir}")
        print(f"Status: {state.get('status')}")
        print(f"Phase/stage: {state.get('phase')} / {state.get('stage')}")
        print(f"Epoch: {state.get('epoch')}")
        print(f"Cursor: {state.get('cursor')} / {state.get('total')}")
        print(f"Updated UTC: {state.get('updated_at_utc')}")
        print(f"Heartbeat UTC: {heartbeat.get('utc')}")
        print(f"PID: {state.get('pid')} on {state.get('hostname')}")
        print(f"Completed stages: {state.get('completed_stages', [])}")
        latest_ckpt = state.get("latest_checkpoint")
        if latest_ckpt:
            print(f"Latest checkpoint: {latest_ckpt}")
        crash_files = sorted(paths.crashes.glob("*.json"))
        print(f"Crash/stop records: {len(crash_files)}")
        for path in crash_files[-5:]:
            row = load_json(path, {})
            print(f"  {path.name}: {row.get('kind')} - {row.get('message')}")
        if paths.events.exists():
            lines = paths.events.read_text(encoding="utf-8").splitlines()[-12:]
            print("Recent events:")
            for line in lines:
                try:
                    row = json.loads(line)
                    print(f"  {row.get('utc')} {row.get('event')} {row.get('message', '')}")
                except Exception:
                    print(f"  {line}")

    def _acquire_lock(self) -> None:
        if self.paths.lock.exists():
            old = load_json(self.paths.lock, {})
            old_pid = int(old.get("pid", -1))
            old_host = old.get("hostname")
            if process_alive(old_pid, old_host):
                raise RuntimeError(f"Run directory is already owned by PID {old_pid} on {old_host}: {self.paths.run_dir}")
            try:
                self.paths.lock.unlink()
            except OSError:
                pass
        atomic_write_json(self.paths.lock, {
            "pid": self.pid,
            "hostname": self.hostname,
            "created_at_utc": utc_now(),
            "created_at_local": local_now(),
        })

    def _record_unclean_if_needed(self) -> None:
        old_status = self.state.get("status")
        old_pid = int(self.state.get("pid") or -1)
        old_host = self.state.get("hostname")
        if old_status == "running" and not process_alive(old_pid, old_host):
            heartbeat = load_json(self.paths.heartbeat, {})
            self.record_crash(
                kind="inferred_unclean_shutdown",
                message="Previous process ended without executing a shutdown handler (possible power loss, OS/RAM kill, driver/process termination, or hard crash).",
                traceback_text=None,
                extra={"previous_pid": old_pid, "previous_hostname": old_host, "last_heartbeat": heartbeat},
                update_status=False,
            )

    def _install_signal_handlers(self) -> None:
        def handler(signum, frame):  # noqa: ARG001
            try:
                name = signal.Signals(signum).name
            except Exception:
                name = str(signum)
            self.stop_requested = True
            self.stop_signal = name
            # Do not raise or perform file I/O inside the signal handler. Training/evaluation
            # loops observe the flag at the next safe point, checkpoint, log, and stop.

        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is not None:
                try:
                    signal.signal(sig, handler)
                except Exception:
                    pass

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(self.heartbeat_seconds):
            try:
                atomic_write_json(self.paths.heartbeat, {
                    "utc": utc_now(),
                    "local": local_now(),
                    "pid": self.pid,
                    "hostname": self.hostname,
                    "phase": self.state.get("phase"),
                    "stage": self.state.get("stage"),
                    "epoch": self.state.get("epoch"),
                    "cursor": self.state.get("cursor"),
                    "total": self.state.get("total"),
                })
            except Exception:
                pass

    def _start_heartbeat(self) -> None:
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, name="dcu-heartbeat", daemon=True)
        self._heartbeat_thread.start()
        atomic_write_json(self.paths.heartbeat, {
            "utc": utc_now(), "local": local_now(), "pid": self.pid, "hostname": self.hostname,
            "phase": self.state.get("phase"), "stage": self.state.get("stage"),
            "epoch": self.state.get("epoch"), "cursor": self.state.get("cursor"), "total": self.state.get("total"),
        })

    def set_emergency_saver(self, saver: Callable[[], None] | None) -> None:
        self.emergency_saver = saver

    def active_elapsed_seconds(self) -> float:
        return max(0.0, self._active_seconds_base + (time.perf_counter() - self._session_perf_start))

    def wall_elapsed_seconds(self) -> float:
        raw = self.state.get("started_at_utc")
        if not raw:
            return 0.0
        try:
            started = datetime.fromisoformat(str(raw))
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            return max(0.0, (datetime.now(timezone.utc) - started.astimezone(timezone.utc)).total_seconds())
        except Exception:
            return 0.0

    def _save_state(self) -> None:
        self.state["active_seconds"] = self.active_elapsed_seconds()
        self.state["updated_at_utc"] = utc_now()
        self.state["updated_at_local"] = local_now()
        atomic_write_json(self.paths.state, self.state)

    def progress(
        self,
        *,
        phase: str,
        stage: str,
        cursor: int,
        total: int | None,
        epoch: int | None = None,
        latest_checkpoint: str | None = None,
        log_event: bool = False,
        **extra: Any,
    ) -> None:
        self.state.update({
            "status": "running",
            "phase": phase,
            "stage": stage,
            "epoch": epoch,
            "cursor": int(cursor),
            "total": None if total is None else int(total),
            "pid": self.pid,
            "hostname": self.hostname,
        })
        if latest_checkpoint is not None:
            self.state["latest_checkpoint"] = latest_checkpoint
        self.state.update(extra)
        self._save_state()
        if log_event:
            self.log("progress", phase=phase, stage=stage, epoch=epoch, cursor=cursor, total=total, **extra)

    def mark_stage_complete(self, stage: str, *, phase: str | None = None, **extra: Any) -> None:
        completed = list(self.state.get("completed_stages", []))
        if stage not in completed:
            completed.append(stage)
        self.state["completed_stages"] = completed
        self.state["stage"] = stage
        if phase is not None:
            self.state["phase"] = phase
        self.state.update(extra)
        self._save_state()
        self.log("stage_complete", stage=stage, phase=self.state.get("phase"), **extra)

    def stage_complete(self, stage: str) -> bool:
        return stage in set(self.state.get("completed_stages", []))

    def log(self, event: str, message: str | None = None, *, echo: bool = False, **fields: Any) -> None:
        """Append a durable timestamped event. Terminal echo is opt-in.

        Normal training progress belongs to tqdm in the training script; keeping the
        recovery journal silent prevents checkpoint/progress events from destroying
        interactive progress bars.
        """
        row = {
            "utc": utc_now(),
            "local": local_now(),
            "event": event,
            "pid": self.pid,
            "phase": self.state.get("phase"),
            "stage": self.state.get("stage"),
            "epoch": self.state.get("epoch"),
            "cursor": self.state.get("cursor"),
            "total": self.state.get("total"),
        }
        if message is not None:
            row["message"] = message
        row.update(fields)
        append_jsonl_durable(self.paths.events, row)
        if echo:
            prefix = row["local"]
            detail = message if message else " ".join(f"{k}={v}" for k, v in fields.items())
            print(f"[{prefix}] {event}: {detail}".rstrip(), flush=True)

    def checkpoint(self, name: str, payload: Any, *, update_state: bool = True) -> Path:
        path = self.paths.checkpoints / name
        atomic_torch_save(path, payload)
        if update_state:
            self.state["latest_checkpoint"] = str(path)
            self._save_state()
        return path

    def load_checkpoint(self, name: str, *, map_location: str | torch.device = "cpu") -> Any | None:
        path = self.paths.checkpoints / name
        if not path.exists():
            return None
        return torch.load(path, map_location=map_location, weights_only=False)

    def record_crash(
        self,
        *,
        kind: str,
        message: str,
        traceback_text: str | None,
        extra: dict[str, Any] | None = None,
        update_status: bool = True,
    ) -> Path:
        row = {
            "utc": utc_now(),
            "local": local_now(),
            "kind": kind,
            "message": message,
            "traceback": traceback_text,
            "pid": self.pid,
            "hostname": self.hostname,
            "run_state": dict(self.state),
            "heartbeat": load_json(self.paths.heartbeat, {}),
            "platform": platform.platform(),
            "python": platform.python_version(),
        }
        if extra:
            row.update(extra)
        path = self.paths.crashes / f"{kind}_{safe_stamp()}.json"
        atomic_write_json(path, row)
        if update_status:
            self.state["status"] = "stopped" if kind == "manual_stop" else "failed"
            self.state["last_crash_record"] = str(path)
            self._save_state()
        return path

    def handle_exception(self, exc: BaseException, *, kind: str | None = None) -> Path:
        if self.emergency_saver is not None:
            try:
                self.emergency_saver()
            except Exception as saver_exc:
                self.log("emergency_checkpoint_failed", message=repr(saver_exc))
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        if kind is None:
            kind = "manual_stop" if isinstance(exc, (KeyboardInterrupt, GracefulStop)) else "exception"
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        path = self.record_crash(kind=kind, message=str(exc) or type(exc).__name__, traceback_text=tb)
        self.log("run_interrupted", message=f"{kind}: {exc}", crash_record=str(path))
        return path

    def check_stop(self, *, saver: Callable[[], None] | None = None) -> None:
        if not self.stop_requested:
            return
        self.log("stop_requested", message=f"Received {self.stop_signal or 'signal'}; stopping at a safe point.", echo=True)
        if saver is not None:
            saver()
        elif self.emergency_saver is not None:
            self.emergency_saver()
        self.emergency_saver = None
        raise GracefulStop(f"Manual stop requested by {self.stop_signal or 'signal'} at a safe point")

    def mark_complete(self, **extra: Any) -> None:
        self.state.update({"status": "complete", "phase": "complete", "stage": "complete", "cursor": 1, "total": 1})
        self.state.update(extra)
        self._save_state()
        self.log("run_complete", **extra)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=2.0)
        try:
            self._save_state()
        except Exception:
            pass
        try:
            if self.paths.lock.exists():
                lock = load_json(self.paths.lock, {})
                if int(lock.get("pid", -1)) == self.pid:
                    self.paths.lock.unlink()
        except Exception:
            pass


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)
