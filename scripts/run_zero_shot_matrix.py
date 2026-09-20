from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.local_model import ensure_local_model_path

CONFIG_DIR = REPO_ROOT / "configs" / "baselines" / "local"
DEFAULT_MATRIX_CONFIG = REPO_ROOT / "configs" / "baselines" / "matrix.toml"


def load(path: Path) -> dict:
    with path.open('rb') as handle:
        return tomllib.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete zero-shot local-model baseline matrix. With no arguments: all models, both splits, full data."
    )
    parser.add_argument('--matrix-config', default=str(DEFAULT_MATRIX_CONFIG))
    parser.add_argument('--split', choices=['test', 'heldout', 'both'], default=None)
    parser.add_argument('--examples-per-operation', type=int, default=None)
    parser.add_argument('--auto-download', action=argparse.BooleanOptionalAction, default=None,
                        help='Download missing models; local models are always reused first.')
    parser.add_argument('--model-cache-dir', default=None,
                        help='Directory for Hugging Face model files. Overrides config cache_dir.')
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    matrix = load(Path(args.matrix_config).resolve())
    matrix_eval = matrix.get('evaluation', {})
    matrix_cache = matrix.get('cache', {})
    split = args.split or str(matrix_eval.get('split', 'both'))
    per_op = int(matrix_eval.get('examples_per_operation', -1)) if args.examples_per_operation is None else args.examples_per_operation
    auto_download = bool(matrix_cache.get('auto_download', False)) if args.auto_download is None else bool(args.auto_download)
    cache_dir = args.model_cache_dir if args.model_cache_dir is not None else matrix_cache.get('cache_dir')
    resume = bool(matrix_eval.get('resume', True)) if args.resume is None else bool(args.resume)

    configs = sorted(CONFIG_DIR.glob('*.toml'))
    if not configs:
        raise FileNotFoundError(f'No baseline configs found in {CONFIG_DIR}')
    splits = ['test', 'heldout'] if split == 'both' else [split]
    ran = 0
    skipped: list[tuple[str, str, str]] = []
    for config in configs:
        cfg = load(config)
        model_cfg = cfg['model']
        cache_cfg = cfg.get('cache', {})
        model_auto = auto_download if args.auto_download is not None or 'auto_download' in matrix_cache else bool(cache_cfg.get('auto_download', False))
        model_cache = cache_dir if args.model_cache_dir is not None or str(matrix_cache.get('cache_dir', '')).strip() else cache_cfg.get('cache_dir')
        try:
            path = ensure_local_model_path(
                model_cfg['repo_id'], model_cfg.get('local_path'), auto_download=model_auto,
                cache_dir=model_cache, revision=str(cache_cfg.get('revision', matrix_cache.get('revision', 'main'))),
            )
        except Exception as exc:
            skipped.append((config.stem, model_cfg['repo_id'], str(exc)))
            print(f"SKIP {config.stem}: {exc}")
            continue
        for one_split in splits:
            command = [
                sys.executable, str(REPO_ROOT / 'scripts' / 'evaluate_direct.py'),
                '--config', str(config), '--mode', 'zero_shot', '--split', one_split,
                '--resume' if resume else '--no-resume',
            ]
            if per_op > 0:
                command += ['--examples-per-operation', str(per_op)]
            if model_auto:
                command += ['--auto-download']
            else:
                command += ['--no-auto-download']
            if model_cache and str(model_cache).strip():
                command += ['--model-cache-dir', str(model_cache)]
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            ran += 1

    print(f'Completed evaluations: {ran}')
    if skipped:
        print('Skipped models:')
        for slug, repo_id, reason in skipped:
            print(f'  {slug}: {repo_id} ({reason})')
        if not auto_download:
            print('Rerun with --auto-download to fetch missing ungated models.')


if __name__ == '__main__':
    main()
