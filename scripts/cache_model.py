from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.local_model import MODEL_REGISTRY_PATH  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Explicit Hugging Face download/cache step. Evaluation scripts remain local-only "
            "unless their auto-download option is enabled."
        )
    )
    parser.add_argument("--repo-id", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--revision",
        default="b968826d9c46dd6066d109eabc6255188de91218",
        help="Hub branch/tag/commit to resolve during this explicit cache step.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory. Existing snapshots there are reused.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download instead of reusing the existing HF cache.",
    )
    args = parser.parse_args()

    snapshot = Path(
        snapshot_download(
            repo_id=args.repo_id,
            revision=args.revision,
            force_download=args.force_download,
            cache_dir=args.cache_dir,
        )
    ).resolve()

    if not (snapshot / "config.json").is_file():
        raise RuntimeError(f"Downloaded snapshot is incomplete: {snapshot}")

    registry: dict[str, object] = {}
    if MODEL_REGISTRY_PATH.exists():
        registry = json.loads(MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))

    registry[args.repo_id] = {
        "snapshot_path": str(snapshot),
        "requested_revision": args.revision,
        "resolved_revision": snapshot.name,
        "cached_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    MODEL_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_REGISTRY_PATH.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Registered local model: {args.repo_id}")
    print(f"Resolved revision: {snapshot.name}")
    print(f"Snapshot path: {snapshot}")
    print(f"Registry: {MODEL_REGISTRY_PATH}")


if __name__ == "__main__":
    main()
