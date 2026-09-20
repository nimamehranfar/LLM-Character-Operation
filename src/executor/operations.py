from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class CharacterOperation(str, Enum):
    # Original project operations.
    COUNT_CHAR = "COUNT_CHAR"
    STRING_LENGTH = "STRING_LENGTH"
    FIND_CHAR = "FIND_CHAR"

    # Additional character/word operations used by the benchmark suite.
    CHAR_AT = "CHAR_AT"
    REVERSE = "REVERSE"
    WORD_COUNT = "WORD_COUNT"
    WORD_LENGTH_AT = "WORD_LENGTH_AT"
    INSERT_TEXT = "INSERT_TEXT"
    INSERT_TEXT_IN_WORD = "INSERT_TEXT_IN_WORD"
    INSERT_WORDS = "INSERT_WORDS"
    CHAR_AT_IN_WORD = "CHAR_AT_IN_WORD"
    REVERSE_WORD_AT = "REVERSE_WORD_AT"


@dataclass(frozen=True)
class ExecutionRequest:
    operation: CharacterOperation
    text: str
    character: str | None = None
    index: int | None = None
    word_index: int | None = None
    insertion: str | None = None


class CharacterExecutor:
    """Deterministic executor for exact character/string operations.

    Semantics used throughout the final benchmark:
    - Character and word indices are one-based.
    - FIND_CHAR returns the first one-based position or 0 when absent.
    - INSERT_TEXT uses a one-based insertion *boundary*: 1 inserts before the
      first character and ``len(text)+1`` appends.
    - Sentence operations split on whitespace and rejoin with one ASCII space.
      The final dataset generates sentence inputs with this exact convention.
    """

    @staticmethod
    def _validate_character(character: str | None) -> str:
        if character is None:
            raise ValueError("character is required for this operation")
        if len(character) != 1:
            raise ValueError("character must contain exactly one code point")
        return character

    @staticmethod
    def _require_index(index: int | None, *, name: str, low: int, high: int) -> int:
        if index is None:
            raise ValueError(f"{name} is required for this operation")
        index = int(index)
        if index < low or index > high:
            raise IndexError(f"one-based {name} {index} is outside {low}..{high}")
        return index

    @staticmethod
    def _words(text: str) -> list[str]:
        words = text.split()
        if not words:
            raise ValueError("sentence operation requires at least one whitespace-separated word")
        return words

    def execute(self, request: ExecutionRequest) -> Any:
        op = request.operation
        text = request.text

        if op is CharacterOperation.COUNT_CHAR:
            return text.count(self._validate_character(request.character))

        if op is CharacterOperation.STRING_LENGTH:
            return len(text)

        if op is CharacterOperation.FIND_CHAR:
            character = self._validate_character(request.character)
            position = text.find(character)
            return 0 if position < 0 else position + 1

        if op is CharacterOperation.CHAR_AT:
            index = self._require_index(request.index, name="index", low=1, high=len(text))
            return text[index - 1]

        if op is CharacterOperation.REVERSE:
            return text[::-1]

        if op is CharacterOperation.WORD_COUNT:
            return len(text.split())

        if op is CharacterOperation.WORD_LENGTH_AT:
            words = self._words(text)
            word_index = self._require_index(request.word_index, name="word_index", low=1, high=len(words))
            return len(words[word_index - 1])

        if op is CharacterOperation.INSERT_TEXT:
            insertion = request.insertion
            if insertion is None:
                raise ValueError("insertion is required for INSERT_TEXT")
            index = self._require_index(request.index, name="index", low=1, high=len(text) + 1)
            return text[: index - 1] + insertion + text[index - 1 :]

        if op is CharacterOperation.INSERT_TEXT_IN_WORD:
            words = self._words(text)
            word_index = self._require_index(request.word_index, name="word_index", low=1, high=len(words))
            insertion = request.insertion
            if insertion is None:
                raise ValueError("insertion is required for INSERT_TEXT_IN_WORD")
            word = words[word_index - 1]
            index = self._require_index(request.index, name="index", low=1, high=len(word) + 1)
            words[word_index - 1] = word[: index - 1] + insertion + word[index - 1 :]
            return " ".join(words)

        if op is CharacterOperation.INSERT_WORDS:
            words = self._words(text)
            insertion = request.insertion
            if insertion is None or not insertion.split():
                raise ValueError("insertion must contain at least one word for INSERT_WORDS")
            word_index = self._require_index(request.word_index, name="word_index", low=1, high=len(words) + 1)
            inserted_words = insertion.split()
            return " ".join(words[: word_index - 1] + inserted_words + words[word_index - 1 :])

        if op is CharacterOperation.CHAR_AT_IN_WORD:
            words = self._words(text)
            word_index = self._require_index(request.word_index, name="word_index", low=1, high=len(words))
            word = words[word_index - 1]
            index = self._require_index(request.index, name="index", low=1, high=len(word))
            return word[index - 1]

        if op is CharacterOperation.REVERSE_WORD_AT:
            words = self._words(text)
            word_index = self._require_index(request.word_index, name="word_index", low=1, high=len(words))
            words[word_index - 1] = words[word_index - 1][::-1]
            return " ".join(words)

        raise ValueError(f"unsupported operation: {op}")
