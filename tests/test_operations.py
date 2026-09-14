import unittest

from src.executor.operations import (
    CharacterExecutor,
    CharacterOperation,
    ExecutionRequest,
)


class CharacterExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executor = CharacterExecutor()

    def run_op(self, operation, text, character=None, index=None):
        return self.executor.execute(
            ExecutionRequest(
                operation=operation,
                text=text,
                character=character,
                index=index,
            )
        )

    def test_count_char(self):
        self.assertEqual(
            self.run_op(CharacterOperation.COUNT_CHAR, "strawberry", "r"), 3
        )
        self.assertEqual(self.run_op(CharacterOperation.COUNT_CHAR, "", "r"), 0)
        self.assertEqual(self.run_op(CharacterOperation.COUNT_CHAR, "rrrr", "r"), 4)

    def test_string_length(self):
        self.assertEqual(self.run_op(CharacterOperation.STRING_LENGTH, "strawberry"), 10)
        self.assertEqual(self.run_op(CharacterOperation.STRING_LENGTH, ""), 0)

    def test_char_at_uses_one_based_indexing(self):
        self.assertEqual(self.run_op(CharacterOperation.CHAR_AT, "strawberry", index=1), "s")
        self.assertEqual(self.run_op(CharacterOperation.CHAR_AT, "strawberry", index=10), "y")

    def test_char_at_rejects_out_of_range_index(self):
        with self.assertRaises(IndexError):
            self.run_op(CharacterOperation.CHAR_AT, "abc", index=0)
        with self.assertRaises(IndexError):
            self.run_op(CharacterOperation.CHAR_AT, "abc", index=4)

    def test_reverse(self):
        self.assertEqual(self.run_op(CharacterOperation.REVERSE, "abc"), "cba")
        self.assertEqual(self.run_op(CharacterOperation.REVERSE, ""), "")

    def test_find_char(self):
        self.assertEqual(self.run_op(CharacterOperation.FIND_CHAR, "strawberry", "r"), 3)
        self.assertEqual(self.run_op(CharacterOperation.FIND_CHAR, "strawberry", "z"), 0)

    def test_character_must_be_single_code_point(self):
        with self.assertRaises(ValueError):
            self.run_op(CharacterOperation.COUNT_CHAR, "abc", "ab")


if __name__ == "__main__":
    unittest.main()
