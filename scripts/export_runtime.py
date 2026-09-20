from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description='Export chart-ready latency/token/VRAM comparison rows from baseline and trained-pipeline results.')
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--output', default='results/runtime_comparison.csv')
    args = ap.parse_args()
    root = Path(args.results_root).resolve()
    rows: list[dict[str, Any]] = []
    for path in root.rglob('*.json'):
        payload = load(path)
        if not payload or not isinstance(payload.get('runtime'), dict):
            continue
        runtime = payload['runtime']
        setup = payload.get('inference_setup') or {}
        def add_row(split_name, source, source_runtime, accuracy, count):
            rows.append({
                'file': str(path.relative_to(root)),
                'system': payload.get('mode', payload.get('experiment', path.stem)),
                'model': payload.get('model_repo_id', payload.get('model', setup.get('model_repo_id'))),
                'split': split_name,
                'example_count': count,
                'accuracy': accuracy,
                'benchmark_total_seconds': source_runtime.get('benchmark_total_seconds'),
                'canonical_model_setup_seconds': source_runtime.get('canonical_model_setup_seconds'),
                'inference_seconds': source_runtime.get('inference_seconds'),
                'average_seconds_per_prompt': source_runtime.get('average_seconds_per_prompt'),
                'p50_seconds_per_prompt': source_runtime.get('p50_seconds_per_prompt'),
                'p95_seconds_per_prompt': source_runtime.get('p95_seconds_per_prompt'),
                'total_input_tokens': source_runtime.get('total_input_tokens'),
                'total_output_tokens': source_runtime.get('total_output_tokens'),
                'average_total_tokens_per_prompt': source_runtime.get('average_total_tokens_per_prompt'),
                'output_tokens_per_second': source_runtime.get('output_tokens_per_second'),
                'peak_allocated_vram_mib': source_runtime.get('peak_allocated_vram_mib', runtime.get('peak_allocated_vram_mib')),
                'resume_count': source_runtime.get('resume_count', runtime.get('resume_count')),
                'quantization': setup.get('quantization'),
                'gpu_name': setup.get('gpu_name'),
            })
        add_row(payload.get('split', 'both'), payload, runtime,
                payload.get('exact_match_accuracy', payload.get('overall_system_success')), payload.get('example_count'))
        for split_key, split_label in (('test', 'test'), ('heldout_template_challenge', 'heldout')):
            block = payload.get(split_key)
            if isinstance(block, dict) and isinstance(block.get('runtime'), dict):
                add_row(split_label, block, block['runtime'], block.get('overall_system_success'), block.get('example_count'))
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0]) if rows else []
    with out.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        if keys:
            writer.writeheader(); writer.writerows(rows)
    print(f'Saved {len(rows)} rows: {out}')


if __name__ == '__main__':
    main()
