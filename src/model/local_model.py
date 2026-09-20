from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_REGISTRY_PATH = REPO_ROOT / ".local" / "model_cache.json"


def _valid_snapshot(path: Path) -> bool:
    return path.exists() and (path / "config.json").is_file()


def _load_registry() -> dict[str, object]:
    if MODEL_REGISTRY_PATH.exists():
        return json.loads(MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
    return {}


def _register(repo_id: str, path: Path, revision: str = "main") -> None:
    registry = _load_registry()
    registry[repo_id] = {
        "snapshot_path": str(path.resolve()),
        "requested_revision": revision,
        "resolved_revision": path.name,
        "cached_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    MODEL_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def ensure_local_model_path(
    repo_id: str,
    explicit_path: str | Path | None = None,
    *,
    auto_download: bool = False,
    cache_dir: str | Path | None = None,
    revision: str = "main",
) -> Path:
    """Resolve a local model, optionally downloading it only when absent.

    Resolution order: explicit path -> project registry -> Hugging Face cache lookup
    (local_files_only) -> optional download. Existing local snapshots are always reused.
    """
    if explicit_path is not None and str(explicit_path).strip():
        path = Path(explicit_path).expanduser().resolve()
        if not _valid_snapshot(path):
            raise FileNotFoundError(f"Explicit model path is incomplete: {path}")
        _register(repo_id, path, revision)
        return path

    registry = _load_registry()
    entry = registry.get(repo_id)
    if isinstance(entry, dict):
        registered = Path(str(entry.get("snapshot_path", ""))).expanduser().resolve()
        if _valid_snapshot(registered):
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
    model_path = ensure_local_model_path(
        repo_id, explicit_path, auto_download=auto_download, cache_dir=cache_dir, revision=revision
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    model_kwargs: dict[str, object] = {
        "device_map": {"": "cuda:0"},
        "dtype": torch.bfloat16,
        "local_files_only": True,
    }
    if quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model_path, tokenizer, model
