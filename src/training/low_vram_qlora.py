from __future__ import annotations

from typing import Any


def _enable_input_require_grads_fallback(model: Any) -> None:
    """Support older Transformers gradient checkpointing implementations."""
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return

    embedding = model.get_input_embeddings()

    def _make_output_require_grad(_module, _inputs, output):
        output.requires_grad_(True)

    embedding.register_forward_hook(_make_output_require_grad)


def prepare_4bit_lora_base_low_vram(
    model: Any,
    *,
    use_gradient_checkpointing: bool = True,
) -> Any:
    """Prepare a bitsandbytes 4-bit causal LM for LoRA without dtype promotion.

    PEFT's ``prepare_model_for_kbit_training`` may promote non-quantized fp16/bf16
    base-model parameters to fp32.  On an 8 GiB GPU that can exceed the memory
    budget.  An earlier low-VRAM workaround promoted only 1-D norm parameters,
    but Qwen3's final RMSNorm then emitted fp32 hidden states while the frozen
    lm_head remained bf16, producing a dtype mismatch.

    For this experiment the pretrained Qwen weights are frozen.  We therefore
    preserve *every* base-model parameter in the dtype selected by Transformers /
    bitsandbytes, enable gradient checkpointing, and add LoRA afterwards.  This
    keeps both memory use and the model's forward dtype path consistent.
    """
    for param in model.parameters():
        param.requires_grad_(False)

    if use_gradient_checkpointing:
        # Non-reentrant checkpointing works with frozen input embeddings and
        # trainable LoRA parameters without forcing the embedding output to
        # require gradients.  Fall back for older Transformers releases.
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            _enable_input_require_grads_fallback(model)
            model.gradient_checkpointing_enable()

    return model
