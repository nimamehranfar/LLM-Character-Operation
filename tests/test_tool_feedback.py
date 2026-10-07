from __future__ import annotations

from dataclasses import replace
from contextlib import contextmanager
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.compare_tool_feedback import paired_comparison
from scripts.evaluate_tool_feedback import evaluate_example
from src.data.character_dataset import OPERATION_ARGUMENTS, SUPPORTED_OPERATIONS, load_jsonl
from src.data.tool_policy_dataset import call_is_exact
from src.evaluation.pipeline import execute_parsed_call
from src.evaluation.tool_feedback import (
    build_tools, feedback_messages, initial_messages, parse_native_call,
    render_native, select_examples, summarize,
)
from src.executor.operations import CharacterExecutor

ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    chat_template = "test native template"

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps({"messages": messages, "tools": kwargs.get("tools")}, ensure_ascii=False)


def native_output(name, arguments, interface="qwen"):
    call = {"name": name, "arguments": arguments}
    if interface == "phi":
        return json.dumps([call]) + "<|end|>"
    return "<tool_call>" + json.dumps(call) + "</tool_call><|im_end|>"


class ToolFeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_jsonl(ROOT / "data/character_operations_50k/dev.jsonl")

    def test_all_twelve_operations_execute_through_native_formats(self):
        examples = {row.operation: row for row in self.rows if not row.is_control}
        for interface in ("qwen", "hermes", "phi"):
            for operation in SUPPORTED_OPERATIONS:
                with self.subTest(interface=interface, operation=operation):
                    ex = examples[operation]
                    tools, mapping, masked = build_tools(ex.example_id, seed=20260919, mask_probability=1)
                    name = next(name for name, op in mapping.items() if op == operation)
                    parsed = parse_native_call(native_output(name, ex.arguments, interface), mapping, interface)
                    self.assertTrue(call_is_exact(parsed, ex))
                    self.assertEqual(str(execute_parsed_call(parsed, CharacterExecutor()).result), ex.expected)
                    self.assertTrue(masked)
                    schema = next(tool["function"]["parameters"] for tool in tools if tool["function"]["name"] == name)
                    self.assertEqual(set(schema["required"]), set(OPERATION_ARGUMENTS[operation]))

    def test_rejects_unknown_tools_multiple_calls_and_nonliteral_arguments(self):
        mapping = {"count": "COUNT_CHAR", "at": "CHAR_AT"}
        invalid = [
            native_output("unknown", {"text": "abc"}),
            native_output("count", {"text": ["abc"], "character": "a"}),
            native_output("count", {"text": "abc", "character": "a", "extra": 1}),
            native_output("at", {"text": "abc", "index": True}),
            native_output("at", {"text": "abc", "index": 1.5}),
            '<tool_call>{"name":"count"}',
            native_output("count", {"text": "abc", "character": "a"}) * 2,
        ]
        for output in invalid:
            self.assertFalse(parse_native_call(output, mapping, "qwen")["valid"], output)
        calls = [{"name": "count", "arguments": {"text": "a", "character": "a"}}] * 2
        self.assertFalse(parse_native_call(json.dumps(calls), mapping, "phi")["valid"])

    def test_phi_marker_and_integer_string_accounting(self):
        body = json.dumps([{"name": "at", "arguments": {"text": "abc", "index": "2"}}])
        parsed = parse_native_call("<|tool_call|>" + body + "<|/tool_call|>", {"at": "CHAR_AT"}, "phi")
        self.assertTrue(parsed["valid"])
        self.assertFalse(parsed["schema_valid"])
        self.assertEqual(execute_parsed_call(parsed, CharacterExecutor()).result, "b")

    def test_feedback_preserves_literal_result_and_model_specific_envelopes(self):
        tools, mapping, _ = build_tools("test", seed=1)
        name = next(name for name, op in mapping.items() if op == "COUNT_CHAR")
        for interface in ("qwen", "hermes", "phi"):
            initial = initial_messages("A user prompt", tools, interface)
            parsed = parse_native_call(native_output(name, {"text": "rr", "character": "r"}, interface), mapping, interface)
            messages = feedback_messages(initial, parsed, 'a"b\\c\n', interface)
            self.assertEqual(messages[-1]["content"], 'a"b\\c\n')
            self.assertEqual(len(initial), 2)
            if interface == "phi":
                self.assertIn("tools", initial[0])
            else:
                self.assertIn("tool_calls", messages[-2])
            self.assertIn(name, render_native(FakeTokenizer(), messages, tools, interface))

    def test_wrong_predicted_arguments_are_executed_without_gold_leakage(self):
        ex = next(row for row in self.rows if row.operation == "COUNT_CHAR")
        ex = replace(ex, prompt='Count r in "rr".', arguments={"text": "rr", "character": "r"}, expected="2")
        prompts = []
        outputs = iter([native_output("COUNT_CHAR", {"text": "r", "character": "r"}), "1<|im_end|>"])

        def fake_generate(model, tokenizer, prompt, budget):
            prompts.append(json.loads(prompt))
            return next(outputs), {"seconds": 0.01, "input_tokens": 10, "output_tokens": 5}

        with patch("scripts.evaluate_tool_feedback.generate", side_effect=fake_generate):
            row = evaluate_example(ex, object(), FakeTokenizer(), "qwen", 1, 0, 192, 128)
        self.assertEqual(row["executor_result"], 1)
        self.assertEqual(prompts[1]["messages"][-1]["content"], "1")
        self.assertFalse(row["phase1_exact"])
        self.assertFalse(row["final_exact"])
        self.assertEqual(row["input_tokens"], 20)

    def test_no_call_control_never_runs_executor_or_answer_stage(self):
        ex = next(row for row in self.rows if row.is_control)
        with patch("scripts.evaluate_tool_feedback.generate", return_value=("No matching tool.<|im_end|>",
                {"seconds": 0.01, "input_tokens": 10, "output_tokens": 5})) as generate:
            with patch("scripts.evaluate_tool_feedback.execute_parsed_call") as execute:
                row = evaluate_example(ex, object(), FakeTokenizer(), "qwen", 1, 0, 192, 128)
        generate.assert_called_once()
        execute.assert_not_called()
        self.assertTrue(row["no_call_correct"])
        self.assertTrue(row["system_success"])

    def test_lora_selector_is_disabled_for_text_answer_generation(self):
        ex = next(row for row in self.rows if row.operation == "COUNT_CHAR")
        ex = replace(ex, prompt='Count r in "rr".', arguments={"text": "rr", "character": "r"}, expected="2")

        class FakePolicy:
            adapter_enabled = True

            @contextmanager
            def disable_adapter(self):
                self.adapter_enabled = False
                try:
                    yield
                finally:
                    self.adapter_enabled = True

        model = FakePolicy()
        adapter_states = []
        outputs = iter(['CALL {"tool":"COUNT_CHAR","arguments":{"text":"rr","character":"r"}}', '2<|im_end|>'])

        def fake_generate(model, tokenizer, prompt, budget):
            adapter_states.append(model.adapter_enabled)
            return next(outputs), {"seconds": 0.01, "input_tokens": 10, "output_tokens": 5}

        with patch("scripts.evaluate_tool_feedback.generate", side_effect=fake_generate):
            row = evaluate_example(ex, model, FakeTokenizer(), "qwen", 1, 0, 192, 128, policy_config={})
        self.assertEqual(adapter_states, [True, False])
        self.assertTrue(model.adapter_enabled)
        self.assertTrue(row["phase1_exact"])
        self.assertTrue(row["final_exact"])

    def test_native_direct_answer_and_matched_routing_are_distinct(self):
        ex = next(row for row in self.rows if row.operation == "COUNT_CHAR")
        with patch("scripts.evaluate_tool_feedback.generate", return_value=(ex.expected + '<|im_end|>',
                {"seconds": 0.01, "input_tokens": 10, "output_tokens": 5})):
            row = evaluate_example(ex, object(), FakeTokenizer(), "qwen", 1, 0, 192, 128)
        self.assertTrue(row["final_exact"])
        self.assertFalse(row["matched_pipeline_exact"])
        self.assertFalse(row["executor_exact"])

    def test_summary_keeps_controls_and_task_denominators_separate(self):
        common = dict(operation="COUNT_CHAR", phase1_exact=True, no_call_correct=False,
                      tool_attempted=True, invalid_response=False, executor_exact=True,
                      matched_pipeline_exact=False)
        tasks = [{**common, "is_control": False, "final_exact": False, "system_success": False}]
        controls = [{**common, "operation": "NONE", "is_control": True, "final_exact": False,
                     "system_success": True, "no_call_correct": True, "tool_attempted": False}]
        metrics = summarize(tasks + controls)
        self.assertEqual(metrics["exact_match_accuracy"], 0)
        self.assertEqual(metrics["overall_system_success"], 0.5)
        self.assertEqual(metrics["control_no_call_accuracy"], 1)
        self.assertEqual(metrics["executor_direct_task_exact"], 1)
        self.assertEqual(metrics["example_count"], 1)
        self.assertEqual(metrics["evaluated_example_count"], 2)

    def test_sampling_is_deterministic_and_preserves_all_by_default(self):
        left = select_examples(self.rows, 2, 1, 20260919 + 91)
        right = select_examples(self.rows, 2, 1, 20260919 + 91)
        self.assertEqual(left, right)
        self.assertEqual(sum(not row.is_control for row in left), 24)
        self.assertEqual(len(select_examples(self.rows, -1, -1, 1)), len(self.rows))

    def test_paired_comparison_aligns_source_ids_and_rejects_changed_labels(self):
        reference = [{"example_id": "test:a", "source_example_id": "a", "operation": "COUNT_CHAR",
                      "expected": "3", "is_control": False, "final_exact": False}]
        candidate = [{"example_id": "a", "operation": "COUNT_CHAR", "expected": "3",
                      "is_control": False, "final_exact": True}]
        metrics = paired_comparison(reference, candidate)
        self.assertEqual(metrics["paired_task_count"], 1)
        self.assertEqual(metrics["candidate_only_correct"], 1)
        self.assertTrue(metrics["identical_task_sets"])
        with self.assertRaises(ValueError):
            paired_comparison(reference, [{**candidate[0], "expected": "4"}])


if __name__ == "__main__":
    unittest.main()
