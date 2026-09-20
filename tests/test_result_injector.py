from __future__ import annotations

import unittest

import torch

from src.model.result_injector import SymbolicResultMapper, result_to_symbol


class SymbolicResultMapperTests(unittest.TestCase):
    def test_integer_and_character_symbols(self) -> None:
        self.assertEqual(result_to_symbol(3, max_integer=32, ascii_vocab_size=128).kind, "integer")
        self.assertEqual(result_to_symbol("r", max_integer=32, ascii_vocab_size=128).value, ord("r"))

    def test_mapper_shape_and_gate(self) -> None:
        mapper = SymbolicResultMapper(
            hidden_size=16,
            result_dim=8,
            gate_dim=4,
            max_integer=32,
            ascii_vocab_size=128,
            gate_bias_init=-2.0,
        )
        hidden = torch.randn(1, 16)
        updated, gate = mapper(hidden, result_to_symbol(3, max_integer=32, ascii_vocab_size=128))
        self.assertEqual(tuple(updated.shape), (1, 16))
        self.assertEqual(tuple(gate.shape), (1, 1))
        self.assertTrue(0.0 <= float(gate.item()) <= 1.0)


if __name__ == "__main__":
    unittest.main()


def test_layered_symbolic_mapper_supports_all_candidate_layers():
    from src.model.result_injector import LayeredSymbolicResultMapper

    mapper = LayeredSymbolicResultMapper(
        hidden_size=16,
        candidate_layers=[1, 3, 5],
        result_dim=8,
        gate_dim=4,
        max_integer=128,
        ascii_vocab_size=128,
        gate_bias_init=-2.0,
    )
    hidden = torch.randn(1, 16)
    symbol = result_to_symbol("r", max_integer=128, ascii_vocab_size=128)
    for layer in (1, 3, 5):
        updated, gate = mapper(hidden, symbol, layer_index=layer)
        assert tuple(updated.shape) == (1, 16)
        assert tuple(gate.shape) == (1, 1)
        assert 0.0 <= float(gate.item()) <= 1.0
