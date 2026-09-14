from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class InjectionStats:
    layer_index: int
    scale: float
    result_text: str
    result_token_ids: tuple[int, ...]


class ResidualResultInjector:
    """Inject a deterministic result representation into one decoder layer.

    This is an oracle diagnostic, not the final trainable architecture.

    The deterministic result is encoded using the frozen LM's own input
    embeddings. The mean result embedding is RMS-matched to the hidden state
    at the final prompt position and added as a residual vector.

    During autoregressive generation the injection is applied only to the
    prefill pass (sequence length > 1). This prevents repeatedly adding the
    result on every one-token decoding step.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        layer_index: int,
        result: object,
        scale: float,
        prefill_only: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.layer_index = layer_index
        self.result_text = str(result)
        self.scale = float(scale)
        self.prefill_only = prefill_only
        self._handle = None

        layers = self._decoder_layers()
        if not 0 <= layer_index < len(layers):
            raise IndexError(
                f"layer_index={layer_index} is outside [0, {len(layers) - 1}]"
            )

        token_ids = tokenizer.encode(self.result_text, add_special_tokens=False)
        if not token_ids:
            raise ValueError("The deterministic result encoded to zero tokens.")
        self.result_token_ids = tuple(int(x) for x in token_ids)

    def _decoder_layers(self):
        base_model = getattr(self.model, "model", None)
        layers = getattr(base_model, "layers", None)
        if layers is None:
            raise TypeError(
                "Expected a Hugging Face causal LM exposing decoder layers at "
                "model.model.layers."
            )
        return layers

    @torch.no_grad()
    def _result_vector(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        embedding = self.model.get_input_embeddings()
        ids = torch.tensor(self.result_token_ids, device=device, dtype=torch.long)
        vectors = embedding(ids).to(dtype=dtype)
        return vectors.mean(dim=0)

    @staticmethod
    def _replace_hidden(output: Any, new_hidden: torch.Tensor) -> Any:
        if torch.is_tensor(output):
            return new_hidden
        if isinstance(output, tuple):
            return (new_hidden, *output[1:])
        if isinstance(output, list):
            return [new_hidden, *output[1:]]
        raise TypeError(f"Unsupported decoder-layer output type: {type(output)!r}")

    def _hook(self, module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            raise TypeError(
                "Expected decoder hidden state with shape [batch, seq, hidden]."
            )

        if self.prefill_only and hidden.shape[1] <= 1:
            return output

        result_vector = self._result_vector(hidden.device, hidden.dtype)
        last_hidden = hidden[:, -1, :]

        # Match the result-vector RMS to the current residual-stream RMS so
        # `scale` has a reasonably interpretable meaning across layers.
        eps = torch.finfo(torch.float32).eps
        hidden_rms = last_hidden.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
        vector_rms = result_vector.float().pow(2).mean().sqrt().clamp_min(eps)
        normalized_result = result_vector.float() / vector_rms
        residual = normalized_result.unsqueeze(0) * hidden_rms * self.scale

        new_hidden = hidden.clone()
        new_hidden[:, -1, :] = (
            last_hidden.float() + residual
        ).to(dtype=hidden.dtype)
        return self._replace_hidden(output, new_hidden)

    def __enter__(self) -> "ResidualResultInjector":
        if self._handle is not None:
            raise RuntimeError("Injector is already active.")
        layer = self._decoder_layers()[self.layer_index]
        self._handle = layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @property
    def stats(self) -> InjectionStats:
        return InjectionStats(
            layer_index=self.layer_index,
            scale=self.scale,
            result_text=self.result_text,
            result_token_ids=self.result_token_ids,
        )
