from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Iterable

from src.executor.operations import CharacterExecutor, CharacterOperation, ExecutionRequest

SUPPORTED_OPERATIONS = (
    "COUNT_CHAR",
    "STRING_LENGTH",
    "FIND_CHAR",
    "WORD_COUNT",
    "WORD_LENGTH_AT",
    "INSERT_TEXT",
    "INSERT_TEXT_IN_WORD",
    "INSERT_WORDS",
    "CHAR_AT",
    "CHAR_AT_IN_WORD",
    "REVERSE",
    "REVERSE_WORD_AT",
)
OPERATION_LABELS = ("NONE", *SUPPORTED_OPERATIONS)
LABEL_TO_ID = {label: i for i, label in enumerate(OPERATION_LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}

OPERATION_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "COUNT_CHAR": ("text", "character"),
    "STRING_LENGTH": ("text",),
    "FIND_CHAR": ("text", "character"),
    "WORD_COUNT": ("text",),
    "WORD_LENGTH_AT": ("text", "word_index"),
    "INSERT_TEXT": ("text", "insertion", "index"),
    "INSERT_TEXT_IN_WORD": ("text", "insertion", "word_index", "index"),
    "INSERT_WORDS": ("text", "insertion", "word_index"),
    "CHAR_AT": ("text", "index"),
    "CHAR_AT_IN_WORD": ("text", "word_index", "index"),
    "REVERSE": ("text",),
    "REVERSE_WORD_AT": ("text", "word_index"),
}

INTEGER_RESULT_OPERATIONS = {
    "COUNT_CHAR", "STRING_LENGTH", "FIND_CHAR", "WORD_COUNT", "WORD_LENGTH_AT"
}
CHARACTER_RESULT_OPERATIONS = {"CHAR_AT", "CHAR_AT_IN_WORD"}
STRING_RESULT_OPERATIONS = {
    "INSERT_TEXT", "INSERT_TEXT_IN_WORD", "INSERT_WORDS", "REVERSE", "REVERSE_WORD_AT"
}
SCALAR_RESULT_OPERATIONS = INTEGER_RESULT_OPERATIONS | CHARACTER_RESULT_OPERATIONS


@dataclass(frozen=True)
class CharacterExample:
    example_id: str
    split: str
    operation: str
    category: str
    prompt: str
    expected: str
    arguments: dict[str, Any]
    length_regime: str
    generation_style: str = "native"
    source_style: str = "control"
    template_family: str = ""

    @property
    def is_control(self) -> bool:
        return self.operation == "NONE"

    @property
    def result_kind(self) -> str:
        if self.operation in INTEGER_RESULT_OPERATIONS:
            return "integer"
        if self.operation in CHARACTER_RESULT_OPERATIONS:
            return "character"
        if self.operation in STRING_RESULT_OPERATIONS:
            return "string"
        return "control"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CharacterExample":
        operation = str(raw["operation"])
        if operation not in OPERATION_LABELS:
            raise ValueError(f"Unsupported operation label: {operation}")
        ex = cls(
            example_id=str(raw["example_id"]),
            split=str(raw["split"]),
            operation=operation,
            category=str(raw["category"]),
            prompt=str(raw["prompt"]),
            expected=str(raw.get("expected", "")),
            arguments=dict(raw.get("arguments", {})),
            length_regime=str(raw.get("length_regime", "control" if operation == "NONE" else "iid")),
            generation_style=str(raw.get("generation_style", "native")),
            source_style=str(raw.get("source_style", raw.get("category", "control"))),
            template_family=str(raw.get("template_family", "")),
        )
        ex.validate()
        return ex

    def validate(self) -> None:
        if not self.example_id or not self.prompt:
            raise ValueError("example_id and prompt are required")
        if self.operation == "NONE":
            if self.arguments:
                raise ValueError(f"Control must not contain executor arguments: {self.example_id}")
            return
        required = set(OPERATION_ARGUMENTS[self.operation])
        actual = set(self.arguments)
        if actual != required:
            raise ValueError(
                f"{self.operation} arguments must be exactly {sorted(required)}, got {sorted(actual)}"
            )

    def request(self) -> ExecutionRequest:
        if self.is_control:
            raise ValueError("Control examples are not executor calls")
        args = self.arguments
        return ExecutionRequest(
            operation=CharacterOperation(self.operation),
            text=str(args["text"]),
            character=None if "character" not in args else str(args["character"]),
            index=None if "index" not in args else int(args["index"]),
            word_index=None if "word_index" not in args else int(args["word_index"]),
            insertion=None if "insertion" not in args else str(args["insertion"]),
        )

    def execute(self, executor: CharacterExecutor) -> Any:
        result = executor.execute(self.request())
        if str(result) != self.expected:
            raise AssertionError(
                f"Deterministic ground truth mismatch for {self.example_id}: executor={result!r}, expected={self.expected!r}"
            )
        return result


def load_jsonl(path: str | Path) -> list[CharacterExample]:
    rows: list[CharacterExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(CharacterExample.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Invalid JSONL row {path}:{line_no}: {exc}") from exc
    return rows


def random_take(rows: Iterable[CharacterExample], count: int, seed: int) -> list[CharacterExample]:
    rows = list(rows)
    if count < 0 or count >= len(rows):
        return rows
    return random.Random(seed).sample(rows, count)
