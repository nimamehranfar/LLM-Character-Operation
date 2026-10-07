from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.model.device import model_dtype, select_device

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_REGISTRY_PATH = REPO_ROOT / ".local" / "model_cache.json"


@contextmanager
def isolated_download_cache(directory: Path | None):
    """Prevent Xet's separate shared chunk cache during owned-cache downloads.

    Hub reads its feature flags at import time, so changing the environment
    alone after imports would not disable Xet. Authentication remains intact.
    """
    if directory is None:
        yield
        return
    from huggingface_hub import constants
    original = constants.HF_HUB_DISABLE_XET
    previous = os.environ.get("HF_HUB_DISABLE_XET")
    constants.HF_HUB_DISABLE_XET = True
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    try:
        yield
    finally:
        constants.HF_HUB_DISABLE_XET = original
        if previous is None:
            os.environ.pop("HF_HUB_DISABLE_XET", None)
        else:
            os.environ["HF_HUB_DISABLE_XET"] = previous


def _valid_snapshot(path: Path) -> bool:
    if not (path / "config.json").is_file():
        return False
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = path / name
        if index.is_file():
            try:
                shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
                return bool(shards) and all((path / shard).is_file() and (path / shard).stat().st_size > 0 for shard in shards)
            except (OSError, ValueError, KeyError, TypeError):
                return False
    return any((path / name).is_file() and (path / name).stat().st_size > 0
               for name in ("model.safetensors", "pytorch_model.bin"))


def _load_registry() -> dict[str, object]:
    if MODEL_REGISTRY_PATH.exists():
        return json.loads(MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
    return {}


def _register(repo_id: str, path: Path, revision: str = "main") -> None:
    if os.environ.get("LLM_CHARACTER_SKIP_MODEL_REGISTRY_WRITE") == "1":
        return
    registry = _load_registry()
    registry[repo_id] = {
        "snapshot_path": str(path.resolve()),
        "requested_revision": revision,
        "resolved_revision": path.name,
        "cached_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    MODEL_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement prevents concurrent evaluator workers reading partial JSON.
    fd, name = tempfile.mkstemp(dir=MODEL_REGISTRY_PATH.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(registry, indent=2, ensure_ascii=False) + "\n")
        os.replace(name, MODEL_REGISTRY_PATH)
    finally:
        Path(name).unlink(missing_ok=True)


def ensure_local_model_path(
    repo_id: str,
    explicit_path: str | Path | None = None,
    *,
    auto_download: bool = False,
    cache_dir: str | Path | None = None,
    revision: str = "main",
    use_registry: bool = True,
) -> Path:
    """Resolve a local model, optionally downloading it only when absent.

    Resolution order: explicit path -> project registry -> Hugging Face cache lookup
    (local_files_only) -> optional download. Existing local snapshots are always reused.
    """
    if explicit_path is not None and str(explicit_path).strip():
        path = Path(explicit_path).expanduser().resolve()
        if not _valid_snapshot(path):
            raise FileNotFoundError(f"Explicit model path is incomplete: {path}")
        if use_registry:
            _register(repo_id, path, revision)
        return path

    registry = _load_registry() if use_registry else {}
    entry = registry.get(repo_id)
    if isinstance(entry, dict):
        registered = Path(str(entry.get("snapshot_path", ""))).expanduser().resolve()
        revision_matches = revision in {entry.get("requested_revision"), entry.get("resolved_revision")}
        if revision_matches and _valid_snapshot(registered):
            return registered

    cache_root = None if cache_dir is None or not str(cache_dir).strip() else str(Path(cache_dir).expanduser().resolve())
    try:
        local = Path(snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_root,
            local_files_only=True,
        )).resolve()
        if _valid_snapshot(local):
            if use_registry:
                _register(repo_id, local, revision)
            return local
    except Exception:
        pass

    if not auto_download:
        location = f" in cache_dir={cache_root}" if cache_root else ""
        raise FileNotFoundError(
            f"Model {repo_id!r} is not available locally{location}. "
            "Use --auto-download or python scripts/cache_model.py --repo-id ..."
        )

    downloaded = Path(snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_root,
        local_files_only=False,
    )).resolve()
    if not _valid_snapshot(downloaded):
        raise RuntimeError(f"Downloaded snapshot is incomplete: {downloaded}")
    if use_registry:
        _register(repo_id, downloaded, revision)
    return downloaded


def resolve_local_model_path(
    repo_id: str,
    explicit_path: str | Path | None = None,
) -> Path:
    """Backward-compatible strict local-only resolver."""
    return ensure_local_model_path(repo_id, explicit_path, auto_download=False)


def load_local_causal_lm(
    repo_id: str,
    explicit_path: str | Path | None = None,
    quantization: Literal["4bit", "none"] = "4bit",
    *,
    auto_download: bool = False,
    cache_dir: str | Path | None = None,
    revision: str = "main",
):
    device = select_device(announce=True)
    dtype = model_dtype(device)
    if quantization == "4bit" and device.type != "cuda":
        raise RuntimeError("4-bit evaluation requires CUDA; use quantization='none' for CPU fallback")
    model_path = ensure_local_model_path(
        repo_id, explicit_path, auto_download=auto_download, cache_dir=cache_dir, revision=revision
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    model_kwargs: dict[str, object] = {
        "device_map": {"": device},
        "dtype": dtype,
        "local_files_only": True,
    }
    if quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model_path, tokenizer, model
