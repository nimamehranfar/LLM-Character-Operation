"""Off-the-shelf function calling and text feedback; no learned result injector."""
from __future__ import annotations

from collections import defaultdict
import json
import random
import re
from typing import Any

from src.data.character_dataset import OPERATION_ARGUMENTS, SCALAR_RESULT_OPERATIONS
from src.data.tool_policy_dataset import build_tool_catalog

SYSTEM_PROMPT = (
    "Use one available tool only when the user's request exactly matches its operation. "
    "For closely related but unsupported requests, do not call a tool. "
    "Copy literal strings exactly, preserving case, punctuation, spaces and repetitions. "
    "All character and word positions are one-based. After receiving a tool result, "
    "answer with only the final value, with no explanation or quotation marks."
)
PROTOCOL_VERSION = 1
NUMERIC_ARGUMENTS = {"index", "word_index"}
ARGUMENT_DESCRIPTIONS = {
    "text": "The exact input text, preserving all characters and spaces.",
    "character": "Exactly one literal character; matching is case-sensitive.",
    "insertion": "The exact literal text or words to insert.",
    "index": "One-based character position, or insertion boundary (len+1 appends).",
    "word_index": "One-based word position, or insertion boundary (word count+1 appends).",
}


def build_tools(example_id: str, *, seed: int, mask_probability: float = 0.0):
    # Catalog construction uses only the example ID, never its answer or operation.
    entries, masked = build_tool_catalog(example_id, seed=seed, mask_probability=mask_probability)
    tools = []
    for entry in entries:
        arguments = OPERATION_ARGUMENTS[entry.operation]
        properties = {
            key: {"type": "integer" if key in NUMERIC_ARGUMENTS else "string",
                  "description": ARGUMENT_DESCRIPTIONS[key]}
            for key in arguments
        }
        tools.append({"type": "function", "function": {
            "name": entry.tool_id, "description": entry.description,
            "parameters": {"type": "object", "properties": properties,
                           "required": list(arguments), "additionalProperties": False},
        }})
    return tools, {entry.tool_id: entry.operation for entry in entries}, masked


def phi_tools(tools: list[dict]) -> list[dict]:
    """Microsoft's documented compact tool definitions, not OpenAI wrappers."""
    result = []
    for tool in tools:
        function = tool["function"]
        parameters = {
            key: {"type": "int" if value["type"] == "integer" else "str",
                  "description": value["description"]}
            for key, value in function["parameters"]["properties"].items()
        }
        result.append({"name": function["name"], "description": function["description"],
                       "parameters": parameters})
    return result


def initial_messages(prompt: str, tools: list[dict], interface: str) -> list[dict]:
    system: dict[str, Any] = {"role": "system", "content": SYSTEM_PROMPT}
    if interface == "phi":
        # Phi's supplied tokenizer reads tools from the system message itself.
        system["tools"] = json.dumps(phi_tools(tools), ensure_ascii=False)
    return [system, {"role": "user", "content": prompt}]


def render_native(tokenizer, messages: list[dict], tools: list[dict], interface: str) -> str:
    if interface not in {"qwen", "hermes", "phi"}:
        raise ValueError(f"Unsupported native interface: {interface}")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("The released tokenizer has no native chat template")
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if interface != "phi":
        kwargs["tools"] = tools
    if interface == "qwen":
        kwargs["enable_thinking"] = False
    rendered = tokenizer.apply_chat_template(messages, **kwargs)
    # Refuse silently dropping tools, which would turn this into an unaided baseline.
    for tool in tools:
        if tool["function"]["name"] not in rendered:
            raise ValueError("The tokenizer chat template omitted a tool definition")
    return rendered


def clean_response(text: str) -> str:
    text = text.strip()
    # Preserve function-call markers and literal argument content.
    endings = ("<|im_end|>", "<|endoftext|>", "<|end|>", "<|eot_id|>", "</s>")
    while any(text.endswith(ending) for ending in endings):
        for ending in endings:
            if text.endswith(ending):
                text = text[:-len(ending)].rstrip()
                break
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    return text


def parse_native_call(text: str, mapping: dict[str, str], interface: str) -> dict[str, Any]:
    """Parse one native call, without repairing arguments or consulting ground truth."""
    text = clean_response(text)
    base = {"decision": "INVALID", "operation": "NONE", "arguments": {}, "valid": False}
    if "<think>" in text:
        return {**base, "error": "Unfinished reasoning response"}
    if interface in {"qwen", "hermes"}:
        blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.S)
        attempted = "<tool_call" in text or "</tool_call>" in text
    elif interface == "phi":
        blocks = re.findall(r"<\|tool_call\|>\s*(.*?)\s*<\|/tool_call\|>", text, flags=re.S)
        attempted = "<|tool_call|>" in text or "<|/tool_call|>" in text
        if not blocks and text.startswith("["):
            blocks = [text]
            attempted = True
    else:
        raise ValueError(f"Unsupported native interface: {interface}")
    if not blocks:
        if attempted:
            return {**base, "error": "Malformed native tool-call marker"}
        return {"decision": "NO_CALL", "operation": "NONE", "arguments": {}, "valid": True}
    marker = "<|tool_call|>" if interface == "phi" else "<tool_call>"
    if len(blocks) != 1 or text.count(marker) > 1:
        return {**base, "decision": "CALL", "error": "Only one tool call is permitted"}
    try:
        obj = json.loads(blocks[0])
        if interface == "phi":
            if not isinstance(obj, list) or len(obj) != 1:
                raise ValueError("Expected one call in a JSON list")
            obj = obj[0]
        if not isinstance(obj, dict) or set(obj) != {"name", "arguments"}:
            raise ValueError("Expected name and arguments")
        name = obj["name"]
        operation = mapping[name]
        arguments = obj["arguments"]
        if not isinstance(arguments, dict) or set(arguments) != set(OPERATION_ARGUMENTS[operation]):
            raise ValueError("Wrong argument keys")
        schema_valid = True
        for key, value in arguments.items():
            if key in NUMERIC_ARGUMENTS:
                if type(value) is int:
                    continue
                # Match the existing policy's acceptance of integer strings, but
                # separately report whether native JSON argument types were correct.
                if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value):
                    schema_valid = False
                    continue
                raise ValueError(f"{key} must be an integer")
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a literal string")
        return {"decision": "CALL", "operation": operation,
                "arguments": {key: str(value) for key, value in arguments.items()},
                "valid": True, "schema_valid": schema_valid,
                "tool_name": name, "native_call": obj}
    except (ValueError, TypeError, KeyError) as exc:
        return {**base, "decision": "CALL", "error": str(exc)}


def feedback_messages(messages: list[dict], parsed: dict, result: Any, interface: str) -> list[dict]:
    call = parsed["native_call"]
    if interface == "phi":
        assistant = {"role": "assistant", "content": json.dumps([call], ensure_ascii=False)}
    else:
        # Both released Qwen/Hermes templates understand OpenAI-style tool_calls.
        assistant = {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_0", "type": "function", "function": call,
        }]}
    return [*messages, assistant, {"role": "tool", "tool_call_id": "call_0",
                                  "name": parsed["tool_name"], "content": str(result)}]


def select_examples(rows, per_operation: int, controls_per_category: int, seed: int):
    tasks = defaultdict(list)
    controls = defaultdict(list)
    for row in rows:
        (controls if row.is_control else tasks)[row.category if row.is_control else row.operation].append(row)
    selected = []
    for groups, count in ((tasks, per_operation), (controls, controls_per_category)):
        for key in sorted(groups):
            pool = groups[key]
            selected.extend(pool if count < 0 else random.Random(seed + sum(map(ord, key))).sample(pool, min(count, len(pool))))
    return selected


def summarize(rows: list[dict]) -> dict:
    tasks = [row for row in rows if not row["is_control"]]
    controls = [row for row in rows if row["is_control"]]
    scalar = [row for row in tasks if row["operation"] in SCALAR_RESULT_OPERATIONS]
    correct_calls = [row for row in scalar if row["phase1_exact"]]
    accuracy = lambda pool, key: sum(bool(row[key]) for row in pool) / len(pool) if pool else None
    groups = defaultdict(list)
    for row in rows:
        groups[row["operation"]].append(row)
    return {
        # Match unaided baseline exports: top-level exact_match_accuracy is task-only.
        "example_count": len(tasks), "evaluated_example_count": len(rows),
        "task_count": len(tasks), "control_count": len(controls),
        "exact_match_accuracy": accuracy(tasks, "final_exact"),
        "supported_task_system_exact": accuracy(tasks, "final_exact"),
        "overall_system_success": accuracy(rows, "system_success"),
        "phase1_execution_readiness": accuracy(rows, "phase1_exact"),
        "control_no_call_accuracy": accuracy(controls, "no_call_correct"),
        "control_tool_attempt_rate": accuracy(controls, "tool_attempted"),
        "invalid_response_rate": accuracy(rows, "invalid_response"),
        "executor_direct_task_exact": accuracy(tasks, "executor_exact"),
        "scalar_text_feedback_exact": accuracy(scalar, "final_exact"),
        "scalar_text_feedback_given_correct_phase1": accuracy(correct_calls, "final_exact"),
        "matched_pipeline_task_exact": accuracy(tasks, "matched_pipeline_exact"),
        "matched_pipeline_overall_success": (
            sum(bool(row["no_call_correct"] if row["is_control"] else row["matched_pipeline_exact"]) for row in rows)
            / len(rows) if rows else None),
        "by_operation": {key: {"count": len(pool),
            "phase1_exact": accuracy(pool, "phase1_exact"),
            "final_exact": accuracy(pool, "final_exact"),
            "system_success": accuracy(pool, "system_success"),
            "executor_direct_exact": accuracy(pool, "executor_exact"),
            "matched_pipeline_exact": accuracy(pool, "matched_pipeline_exact")}
            for key, pool in sorted(groups.items())},
        "per_example": rows,
    }
