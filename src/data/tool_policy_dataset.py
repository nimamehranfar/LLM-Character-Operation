from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Iterable

from src.data.character_dataset import CharacterExample, OPERATION_ARGUMENTS, SUPPORTED_OPERATIONS

_DESCRIPTION_BANK = {
    "COUNT_CHAR": ("Count exact occurrences of one literal character in the supplied text.", "Return literal frequency of one requested character."),
    "STRING_LENGTH": ("Return the total number of characters in the exact supplied text.", "Compute exact character length of the supplied string."),
    "FIND_CHAR": ("Return the first one-based position of a literal character, or zero if absent.", "Locate the earliest exact occurrence using one-based indexing; 0 means absent."),
    "WORD_COUNT": ("Return the number of whitespace-separated words in the supplied text.", "Count whitespace-delimited words in the text."),
    "WORD_LENGTH_AT": ("Return the character length of the specified one-based word.", "Count characters in one specified word of a whitespace-separated sentence."),
    "INSERT_TEXT": ("Insert the supplied text before a specified one-based character boundary; len+1 appends.", "Insert an exact literal substring at a one-based boundary in the string."),
    "INSERT_TEXT_IN_WORD": ("Insert literal text at a one-based boundary inside the specified one-based word.", "Modify one specified word by inserting an exact substring at the requested position."),
    "INSERT_WORDS": ("Insert one or more words before the specified one-based word boundary; n+1 appends.", "Insert exact whitespace-separated words at a one-based sentence boundary."),
    "CHAR_AT": ("Return the character at the specified one-based position in the text.", "Select exactly one character by one-based index."),
    "CHAR_AT_IN_WORD": ("Return the character at a one-based position inside a specified one-based word.", "Select one character from one specified word using one-based indices."),
    "REVERSE": ("Return the supplied text with its character order reversed exactly.", "Reverse all characters in the supplied string."),
    "REVERSE_WORD_AT": ("Reverse the characters of one specified one-based word while leaving other words unchanged.", "Reverse exactly one word in a whitespace-separated sentence."),
}


@dataclass(frozen=True)
class ToolCatalogEntry:
    tool_id: str
    operation: str
    description: str


@dataclass(frozen=True)
class RenderedToolPolicyExample:
    example_id: str
    prompt: str
    system_prompt: str
    user_prompt: str
    target: str
    operation: str
    is_control: bool
    tool_id_to_operation: dict[str, str]
    expected_arguments: dict[str, str]
    masked_names: bool


def _stable_seed(*parts: object) -> int:
    raw = "||".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


def build_tool_catalog(example_id: str, *, seed: int, epoch: int = 0, mask_probability: float = 0.5):
    rng = random.Random(_stable_seed(seed, epoch, example_id, "tool_catalog"))
    operations = list(SUPPORTED_OPERATIONS)
    rng.shuffle(operations)
    masked = rng.random() < float(mask_probability)
    entries = []
    for index, operation in enumerate(operations, start=1):
        entries.append(ToolCatalogEntry(
            tool_id=f"tool_{index}" if masked else operation,
            operation=operation,
            description=rng.choice(_DESCRIPTION_BANK[operation]),
        ))
    return entries, masked


def _catalog_text(entries: Iterable[ToolCatalogEntry]) -> str:
    return "\n".join(f"- {e.tool_id}: {e.description}" for e in entries)


def tool_policy_system_prompt(entries: Iterable[ToolCatalogEntry]) -> str:
    return (
        "You are an exact deterministic-tool controller. Use a tool only when the user's request exactly matches "
        "one available operation. Closely related but unsupported character, word, coding, math, or general-language "
        "requests must return NO_CALL. Copy every literal argument exactly and preserve case, punctuation, spaces, "
        "and repetitions. Numeric indices must be copied as integers.\n\nAvailable tools:\n"
        + _catalog_text(entries)
        + "\n\nOutput exactly one form:\nNO_CALL\nCALL {\"tool\":\"<tool-id>\",\"arguments\":{...}}"
    )


def _target_json(tool_id: str, operation: str, arguments: dict[str, object]) -> str:
    keys = OPERATION_ARGUMENTS[operation]
    args: dict[str, object] = {}
    for key in keys:
        value = arguments[key]
        args[key] = int(value) if key in {"index", "word_index"} else str(value)
    return json.dumps({"tool": tool_id, "arguments": args}, ensure_ascii=False, separators=(",", ":"))


def render_tool_policy_example(example: CharacterExample, *, seed: int, epoch: int = 0, mask_probability: float = 0.5):
    entries, masked = build_tool_catalog(example.example_id, seed=seed, epoch=epoch, mask_probability=mask_probability)
    mapping = {e.tool_id: e.operation for e in entries}
    system = tool_policy_system_prompt(entries)
    prompt = f"SYSTEM:\n{system}\n\nUSER:\n{example.prompt}\n\nASSISTANT:\n"
    if example.operation == "NONE":
        target = "NO_CALL"
        expected_args = {}
    else:
        tool_id = next(e.tool_id for e in entries if e.operation == example.operation)
        target = "CALL " + _target_json(tool_id, example.operation, example.arguments)
        expected_args = {str(k): str(v) for k, v in example.arguments.items()}
    return RenderedToolPolicyExample(
        example_id=example.example_id,
        prompt=prompt,
        system_prompt=system,
        user_prompt=example.prompt,
        target=target,
        operation=example.operation,
        is_control=example.operation == "NONE",
        tool_id_to_operation=mapping,
        expected_arguments=expected_args,
        masked_names=masked,
    )


def parse_policy_output(text: str, tool_id_to_operation: dict[str, str]) -> dict[str, object]:
    cleaned = text.strip()
    if cleaned.startswith("NO_CALL"):
        return {"decision": "NO_CALL", "operation": "NONE", "arguments": {}, "valid": True}
    if not cleaned.startswith("CALL"):
        return {"decision": "INVALID", "operation": "NONE", "arguments": {}, "valid": False}
    try:
        obj = json.loads(cleaned[len("CALL"):].strip())
        operation = tool_id_to_operation[str(obj["tool"])]
        raw_args = dict(obj.get("arguments", {}))
        required = set(OPERATION_ARGUMENTS[operation])
        if set(raw_args) != required:
            raise ValueError("wrong argument keys")
        args = {str(k): str(v) for k, v in raw_args.items()}
    except Exception:
        return {"decision": "CALL", "operation": "NONE", "arguments": {}, "valid": False}
    return {"decision": "CALL", "operation": operation, "arguments": args, "valid": True}


def call_is_exact(parsed: dict[str, object], example: CharacterExample) -> bool:
    if example.operation == "NONE":
        return parsed.get("decision") == "NO_CALL" and bool(parsed.get("valid"))
    if not bool(parsed.get("valid")) or parsed.get("decision") != "CALL":
        return False
    if parsed.get("operation") != example.operation:
        return False
    expected = {str(k): str(v) for k, v in example.arguments.items()}
    return dict(parsed.get("arguments", {})) == expected
