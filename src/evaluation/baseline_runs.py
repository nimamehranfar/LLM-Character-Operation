"""Result validation, export and strictly scoped benchmark-cache cleanup."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

MARKER = ".baseline-cache-owner.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def owned_cache(root: Path, run_id: str, slug: str) -> Path:
    if any(Path(part).name != part or part in {"", ".", ".."} for part in (run_id, slug)):
        raise ValueError("Cache identifiers must be single path components")
    directory = root.resolve() / run_id / slug
    if not directory.resolve().is_relative_to(root.resolve() / run_id):
        raise ValueError("Cache resolves outside its run directory")
    owner = {"run_id": run_id, "model": slug}
    marker = directory / MARKER
    if directory.exists():
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != owner:
            raise ValueError(f"Refusing to use an unowned cache: {directory}")
    else:
        directory.mkdir(parents=True)
        save_json(marker, owner)
    return directory


def cleanup_owned_cache(directory: Path, root: Path, run_id: str, slug: str) -> None:
    # Never accept a shared cache root, an arbitrary snapshot, or a symlink as
    # a deletion target. Only the directory carrying this run's marker is owned.
    expected = root.resolve() / run_id / slug
    if directory.is_symlink() or directory.resolve() != expected:
        raise ValueError("Refusing cleanup outside the owned model cache")
    if not expected.is_relative_to(root.resolve() / run_id):
        raise ValueError("Refusing cleanup outside the run cache")
    marker = directory / MARKER
    owner = {"run_id": run_id, "model": slug}
    if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != owner:
        raise ValueError("Refusing cleanup without a matching ownership marker")
    shutil.rmtree(directory)


def validate_report(path: Path, repo_id: str, split: str, expected_ids: list[str]) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = report.get("per_example", [])
    actual = [str(row["example_id"]) for row in rows]
    if (report.get("mode") != "zero_shot" or report.get("model_repo_id") != repo_id
            or report.get("split") != split or report.get("example_count") != len(expected_ids)
            or not expected_ids or len(actual) != len(set(actual)) or set(actual) != set(expected_ids)):
        raise ValueError(f"Incomplete or mismatched baseline report: {path}")
    accuracy = sum(bool(row["exact"]) for row in rows) / len(rows)
    if abs(float(report["exact_match_accuracy"]) - accuracy) > 1e-10:
        raise ValueError(f"Incorrect aggregate accuracy: {path}")
    return report


def export_results(run_dir: Path, manifest: dict) -> Path:
    save_json(run_dir / "manifest.json", manifest)
    columns = ["model", "repo_id", "split", "status", "example_count", "exact_match_accuracy",
               "average_seconds_per_prompt", "report", "error"]
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for slug, model in manifest.get("models", {}).items():
            for split, entry in model.get("splits", {}).items():
                writer.writerow({"model": slug, "repo_id": model["repo_id"], "split": split,
                                 **{name: entry.get(name, "") for name in columns[3:]}})
    archive_path = run_dir.parent / f"{run_dir.name}.zip"
    temporary = archive_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        # Cache files are never exported. Progress, logs, configs and final
        # reports are preserved so interrupted/failed runs are inspectable too.
        files = [run_dir / "manifest.json", run_dir / "summary.csv"]
        for name in ("models", "configs"):
            files.extend(path for path in (run_dir / name).rglob("*") if path.is_file())
        for path in sorted(files):
            if not path.resolve().is_relative_to(run_dir.resolve()):
                raise ValueError(f"Export file resolves outside run: {path}")
            archive.write(path, arcname=f"{run_dir.name}/{path.relative_to(run_dir).as_posix()}")
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Result archive failed integrity validation")
    temporary.replace(archive_path)
    return archive_path
