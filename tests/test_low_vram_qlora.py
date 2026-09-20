from __future__ import annotations

import unittest

import torch

from src.training.low_vram_qlora import prepare_4bit_lora_base_low_vram


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.large_2d = torch.nn.Parameter(torch.ones((16, 8), dtype=torch.bfloat16))
        self.norm_1d = torch.nn.Parameter(torch.ones((8,), dtype=torch.bfloat16))
        self.gc_kwargs = None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gc_kwargs = gradient_checkpointing_kwargs


class _TinyNormHead(torch.nn.Module):
    """Minimal regression model for the Qwen final-norm -> bf16 lm_head path."""

    def __init__(self):
        super().__init__()
        self.norm_weight = torch.nn.Parameter(torch.ones((8,), dtype=torch.bfloat16))
        self.lm_head = torch.nn.Linear(8, 4, bias=False, dtype=torch.bfloat16)
        self.gc_kwargs = None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gc_kwargs = gradient_checkpointing_kwargs

    def forward(self, x):
        # A dtype-sensitive stand-in for a final normalization followed by the
        # frozen language-model head.
        hidden = x * self.norm_weight
        return self.lm_head(hidden)


class LowVramQloraPreparationTests(unittest.TestCase):
    def test_freezes_base_without_upcasting_any_base_tensor(self):
        model = _DummyModel()
        original_dtypes = {name: p.dtype for name, p in model.named_parameters()}
        prepare_4bit_lora_base_low_vram(model, use_gradient_checkpointing=True)
        self.assertFalse(model.large_2d.requires_grad)
        self.assertFalse(model.norm_1d.requires_grad)
        self.assertEqual(
            {name: p.dtype for name, p in model.named_parameters()},
            original_dtypes,
        )
        self.assertEqual(model.gc_kwargs, {"use_reentrant": False})

    def test_preserves_norm_to_lm_head_dtype_path(self):
        model = _TinyNormHead()
        prepare_4bit_lora_base_low_vram(model, use_gradient_checkpointing=True)
        self.assertEqual(model.norm_weight.dtype, torch.bfloat16)
        self.assertEqual(model.lm_head.weight.dtype, torch.bfloat16)
        x = torch.ones((2, 8), dtype=torch.bfloat16)
        out = model(x)
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_can_disable_gradient_checkpointing(self):
        model = _DummyModel()
        prepare_4bit_lora_base_low_vram(model, use_gradient_checkpointing=False)
        self.assertIsNone(model.gc_kwargs)


if __name__ == "__main__":
    unittest.main()
