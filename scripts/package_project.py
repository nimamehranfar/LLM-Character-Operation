"""Package current portable source/data without caches, environments or credentials."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = ("README.md", "requirements.txt", "requirements-dev.txt", "pytest.ini", ".gitignore", ".gitattributes")
DIRECTORIES = ("src", "scripts", "configs", "docs", "tests", "data")


def project_files(root=ROOT):
    files = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for directory in DIRECTORIES:
        files.extend(path for path in (root / directory).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts
                     and path.suffix in {".py", ".toml", ".md", ".txt", ".json", ".jsonl", ".sbatch"})
    for path in files:
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"File resolves outside the project: {path}")
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/LLM-Character-Operation-portable.zip")
    args = parser.parse_args()
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in project_files():
            name = path.relative_to(ROOT).as_posix()
            content = path.read_bytes()
            manifest[name] = hashlib.sha256(content).hexdigest()
            archive.writestr(name, content)
        archive.writestr("PORTABLE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    print(f"Packaged {len(manifest)} files: {output}")


if __name__ == "__main__":
    main()
