from __future__ import annotations

import json
from pathlib import Path
import re
import tomllib

from src.data.character_dataset import OPERATION_ARGUMENTS, SUPPORTED_OPERATIONS, load_jsonl
from src.data.tool_policy_dataset import call_is_exact, parse_policy_output, render_tool_policy_example
from src.executor.operations import CharacterExecutor, CharacterOperation, ExecutionRequest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "character_operations_50k"


def test_manifest_protocol():
    m = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    assert m["total"] == 50_000
    assert m["split_counts"] == {"train": 35_000, "dev": 5_000, "test": 5_000, "heldout": 5_000}
    assert m["negative_count"] == 18_500
    assert m["negative_fraction"] == 0.37
    assert m["positive_source_counts"] == {
        "real_word": 9_450,
        "augmented_word": 15_750,
        "random_generated": 6_300,
    }
    assert all(v == 2_625 for v in m["operation_counts"].values())
    assert m["exact_prompt_duplicates"] == 0
    assert m["heldout_template_families_disjoint_from_train"] is True


def test_all_rows_match_deterministic_executor():
    executor = CharacterExecutor()
    total = 0
    for name in ("train", "dev", "test", "heldout"):
        rows = load_jsonl(DATA / f"{name}.jsonl")
        total += len(rows)
        for row in rows:
            if not row.is_control:
                assert str(executor.execute(row.request())) == row.expected
    assert total == 50_000


def test_all_operations_roundtrip_through_structured_policy():
    rows = load_jsonl(DATA / "dev.jsonl")
    seen = set()
    for ex in rows:
        if ex.operation == "NONE" or ex.operation in seen:
            continue
        rendered = render_tool_policy_example(ex, seed=20260919, epoch=0, mask_probability=1.0)
        parsed = parse_policy_output(rendered.target, rendered.tool_id_to_operation)
        assert call_is_exact(parsed, ex)
        assert set(parsed["arguments"]) == set(OPERATION_ARGUMENTS[ex.operation])
        seen.add(ex.operation)
    assert seen == set(SUPPORTED_OPERATIONS)
    assert "CHAR_AT" in seen and "CHAR_AT_IN_WORD" in seen


def test_extended_executor_semantics():
    ex = CharacterExecutor()
    assert ex.execute(ExecutionRequest(CharacterOperation.WORD_COUNT, "alpha beta gamma")) == 3
    assert ex.execute(ExecutionRequest(CharacterOperation.WORD_LENGTH_AT, "alpha beta", word_index=2)) == 4
    assert ex.execute(ExecutionRequest(CharacterOperation.INSERT_TEXT, "abcd", insertion="X", index=3)) == "abXcd"
    assert ex.execute(ExecutionRequest(CharacterOperation.CHAR_AT, "abcd", index=3)) == "c"
    assert ex.execute(ExecutionRequest(CharacterOperation.REVERSE, "abcd")) == "dcba"


def test_heldout_templates_are_disjoint():
    train = load_jsonl(DATA / "train.jsonl")
    held = load_jsonl(DATA / "heldout.jsonl")
    assert {x.template_family for x in train}.isdisjoint({x.template_family for x in held})
    assert not ({x.prompt for x in train} & {x.prompt for x in held})


def test_experiment_configs_and_candidate_layers():
    expected_layers = {
        "qwen3_8b": [1, 3, 5, 7, 9, 11, 13, 15],
        "qwen3_4b": [4, 8, 12, 17],
    }
    for model, layers in expected_layers.items():
        root = ROOT / "configs" / "experiments" / model
        for filename in ("tool_policy.toml", "direct_sft.toml", "result_injection.toml"):
            with (root / filename).open("rb") as handle:
                cfg = tomllib.load(handle)
            assert cfg["data"]["train_file"].endswith("train.jsonl")
            assert cfg["cache"]["auto_download"] is True
        with (root / "result_injection.toml").open("rb") as handle:
            cfg = tomllib.load(handle)
        assert cfg["architecture"]["candidate_layers"] == layers
        assert cfg["evaluation"]["run_layerwise_analysis"] is True
        assert cfg["evaluation"]["run_oracle_phase1_analysis"] is True


def test_zero_shot_matrix_has_expected_small_models():
    baseline_dir = ROOT / "configs" / "baselines" / "local"
    expected = {
        "llama2_7b", "llama3_8b", "qwen_7b", "mistral_7b", "yi_6b",
        "chatglm3_6b", "baichuan2_7b", "gemma_7b", "aya23_8b",
        "qwen2_5_7b", "llama3_1_8b", "qwen3_4b", "qwen3_8b",
    }
    assert {p.stem for p in baseline_dir.glob("*.toml")} == expected
    with (ROOT / "configs" / "baselines" / "matrix.toml").open("rb") as handle:
        matrix = tomllib.load(handle)
    assert matrix["evaluation"]["split"] == "both"
    assert matrix["evaluation"]["examples_per_operation"] == -1
    assert matrix["evaluation"]["resume"] is True
    assert "cache_dir" in matrix["cache"]


def test_pipeline_defaults_and_runtime_telemetry_present():
    text = (ROOT / "scripts" / "evaluate_pipeline.py").read_text(encoding="utf-8")
    assert "default=str(DEFAULT_P1)" in text
    assert "default=str(DEFAULT_P2)" in text
    for field in (
        "phase1_seconds", "executor_seconds", "phase2_seconds", "latency_seconds",
        "phase1_input_tokens", "phase1_output_tokens", "phase2_input_tokens", "phase2_output_tokens",
        "runtime_supported_tasks", "runtime_controls",
    ):
        assert field in text


def test_public_layout_has_no_historical_naming_in_paths():
    bad = []
    version_pattern = re.compile(r"(^|[_-])v(?:4|5|6|7)(?:[_-]|$)", re.IGNORECASE)
    for path in ROOT.rglob("*"):
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        low = rel.lower()
        if "__pycache__" in low:
            continue
        if ("i"+"gc") in low or ("pa"+"per") in low or version_pattern.search(low):
            bad.append(rel)
    assert bad == []
