from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class CharacterOperation(str, Enum):
    COUNT_CHAR = "COUNT_CHAR"
    STRING_LENGTH = "STRING_LENGTH"
    CHAR_AT = "CHAR_AT"
    REVERSE = "REVERSE"
    FIND_CHAR = "FIND_CHAR"


@dataclass(frozen=True)
class ExecutionRequest:
    operation: CharacterOperation
    text: str
    character: str | None = None
    index: int | None = None


class CharacterExecutor:
    """Deterministic executor for ASCII character/string operations.

    V1 semantics:
    - Strings are Python ``str`` values restricted by the benchmark to printable ASCII.
    - ``CHAR_AT`` uses one-based indexing.
    - ``FIND_CHAR`` returns the one-based index of the first occurrence, or 0 if absent.
    """

    @staticmethod
    def _validate_character(character: str | None) -> str:
        if character is None:
            raise ValueError("character is required for this operation")
        if len(character) != 1:
            raise ValueError("character must contain exactly one code point")
        return character

    def execute(self, request: ExecutionRequest) -> Any:
        op = request.operation
        text = request.text

        if op is CharacterOperation.COUNT_CHAR:
            character = self._validate_character(request.character)
            return text.count(character)

        if op is CharacterOperation.STRING_LENGTH:
            return len(text)

        if op is CharacterOperation.CHAR_AT:
            if request.index is None:
                raise ValueError("index is required for CHAR_AT")
            if request.index < 1 or request.index > len(text):
                raise IndexError(
                    f"one-based index {request.index} is outside string length {len(text)}"
                )
            return text[request.index - 1]

        if op is CharacterOperation.REVERSE:
            return text[::-1]

        if op is CharacterOperation.FIND_CHAR:
            character = self._validate_character(request.character)
            position = text.find(character)
            return 0 if position < 0 else position + 1

        raise ValueError(f"unsupported operation: {op}")
