from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn
import torch.nn.functional as F

ResultKind = Literal["integer", "character"]


@dataclass(frozen=True)
class ResultSymbol:
    kind: ResultKind
    value: int


def result_to_symbol(result: object, *, max_integer: int, ascii_vocab_size: int) -> ResultSymbol:
    if isinstance(result, bool):
        raise TypeError("Boolean executor results are not supported in this experiment.")
    if isinstance(result, int):
        if not 0 <= result <= max_integer:
            raise ValueError(
                f"Integer result {result} is outside configured range 0..{max_integer}."
            )
        return ResultSymbol("integer", int(result))
    if isinstance(result, str) and len(result) == 1:
        code = ord(result)
        if not 0 <= code < ascii_vocab_size:
            raise ValueError(
                f"Character result {result!r} has code point {code}, outside 0..{ascii_vocab_size - 1}."
            )
        return ResultSymbol("character", code)
    raise TypeError(
        "This diagnostic supports integer and single-character executor results only."
    )


class SymbolicResultMapper(nn.Module):
    """Learn a hidden-state update from a symbolic deterministic result.

    The result is represented independently of the LM tokenizer. Integer and
    ASCII-character values are converted to deterministic binary/value features,
    then mapped into a small trainable result space. This avoids injecting the
    LM's own answer-token embedding and gives the mapper a compositional signal
    for values not seen during training. A learned scalar gate depends on both
    the current LM state and the symbolic result.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        result_dim: int,
        gate_dim: int,
        max_integer: int,
        ascii_vocab_size: int,
        gate_bias_init: float,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.max_integer = int(max_integer)
        self.ascii_vocab_size = int(ascii_vocab_size)

        self.kind_embedding = nn.Embedding(2, result_dim)
        self.integer_bits = max(1, int(max_integer).bit_length())
        self.character_bits = max(1, int(ascii_vocab_size - 1).bit_length())
        self.integer_feature_projection = nn.Linear(self.integer_bits + 1, result_dim)
        self.character_feature_projection = nn.Linear(self.character_bits + 1, result_dim)
        self.result_norm = nn.LayerNorm(result_dim)
        self.result_mlp = nn.Sequential(
            nn.Linear(result_dim, result_dim),
            nn.SiLU(),
            nn.Linear(result_dim, result_dim),
        )

        self.delta_projection = nn.Linear(result_dim, hidden_size, bias=False)
        self.gate_hidden_projection = nn.Linear(hidden_size, gate_dim, bias=False)
        self.gate_result_projection = nn.Linear(result_dim, gate_dim, bias=True)
        self.gate_output = nn.Linear(gate_dim, 1, bias=True)
        nn.init.constant_(self.gate_output.bias, float(gate_bias_init))

    @staticmethod
    def _binary_features(value: int, bits: int, maximum: int, device: torch.device) -> torch.Tensor:
        bit_values = [float((value >> bit) & 1) for bit in range(bits)]
        normalized = 0.0 if maximum <= 0 else float(value) / float(maximum)
        return torch.tensor([*bit_values, normalized], device=device, dtype=torch.float32)

    def _encode_symbol(self, symbol: ResultSymbol, device: torch.device) -> torch.Tensor:
        if symbol.kind == "integer":
            kind_id = 0
            features = self._binary_features(
                symbol.value, self.integer_bits, self.max_integer, device
            )
            value_embedding = self.integer_feature_projection(features)
        elif symbol.kind == "character":
            kind_id = 1
            features = self._binary_features(
                symbol.value, self.character_bits, self.ascii_vocab_size - 1, device
            )
            value_embedding = self.character_feature_projection(features)
        else:  # pragma: no cover - ResultKind constrains this
            raise ValueError(f"Unsupported result kind: {symbol.kind}")

        kind_embedding = self.kind_embedding(
            torch.tensor(kind_id, device=device, dtype=torch.long)
        )
        encoded = self.result_norm(kind_embedding + value_embedding)
        return encoded + self.result_mlp(encoded)

    def forward(
        self,
        hidden: torch.Tensor,
        symbol: ResultSymbol,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.ndim != 2:
            raise ValueError("hidden must have shape [batch, hidden_size]")
        if hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected hidden_size={self.hidden_size}, got {hidden.shape[-1]}."
            )

        # Batch size is intentionally 1 in this first diagnostic. Keeping the
        # interface explicit avoids silently broadcasting different oracle
        # results across examples.
        if hidden.shape[0] != 1:
            raise ValueError("This diagnostic currently supports batch_size=1 only.")

        work_hidden = hidden.float()
        result_repr = self._encode_symbol(symbol, hidden.device).float().unsqueeze(0)

        hidden_for_gate = F.layer_norm(work_hidden, (self.hidden_size,))
        gate_state = torch.tanh(
            self.gate_hidden_projection(hidden_for_gate)
            + self.gate_result_projection(result_repr)
        )
        gate = torch.sigmoid(self.gate_output(gate_state))  # [1, 1]

        delta = self.delta_projection(result_repr)
        eps = torch.finfo(torch.float32).eps
        hidden_rms = work_hidden.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
        delta_rms = delta.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
        normalized_delta = delta / delta_rms * hidden_rms

        updated = work_hidden + gate * normalized_delta
        return updated.to(dtype=hidden.dtype), gate


class TrainableOracleResultInjector:
    """Forward-hook wrapper for one trainable early/intermediate intervention."""

    def __init__(
        self,
        model: Any,
        mapper: SymbolicResultMapper,
        *,
        layer_index: int,
        symbol: ResultSymbol,
        position_index: int,
        prefill_only: bool,
    ) -> None:
        self.model = model
        self.mapper = mapper
        self.layer_index = int(layer_index)
        self.symbol = symbol
        self.position_index = int(position_index)
        self.prefill_only = bool(prefill_only)
        self.last_gate: float | None = None
        self._handle = None

        layers = self._layers()
        if not 0 <= self.layer_index < len(layers):
            raise IndexError(
                f"layer_index={self.layer_index} is outside [0, {len(layers) - 1}]"
            )

    def _layers(self):
        base_model = getattr(self.model, "model", None)
        layers = getattr(base_model, "layers", None)
        if layers is None:
            raise TypeError("Expected decoder layers at model.model.layers")
        return layers

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
            raise TypeError("Expected hidden state [batch, seq, hidden]")

        # During cached autoregressive decoding, sequence length is one. The
        # deterministic result is injected only during the prompt/prefill pass.
        if self.prefill_only and hidden.shape[1] <= 1:
            return output

        position = self.position_index
        if position < 0:
            position = hidden.shape[1] + position
        if not 0 <= position < hidden.shape[1]:
            raise IndexError(
                f"Injection position {self.position_index} is invalid for sequence length {hidden.shape[1]}."
            )

        current = hidden[:, position, :]
        updated, gate = self.mapper(current, self.symbol)
        self.last_gate = float(gate.detach().mean().item())

        new_hidden = hidden.clone()
        new_hidden[:, position, :] = updated
        return self._replace_hidden(output, new_hidden)

    def __enter__(self) -> "TrainableOracleResultInjector":
        if self._handle is not None:
            raise RuntimeError("Injector is already active.")
        self._handle = self._layers()[self.layer_index].register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class _LayerSymbolicAdapter(nn.Module):
    """Layer-specific residual translator for a shared symbolic result representation.

    This keeps the accepted symbolic-result encoding while restoring the original
    project's layer-specific injection capability. Each candidate layer has its
    own delta/gate projections because Qwen residual spaces are layer dependent.
    """

    def __init__(self, *, hidden_size: int, result_dim: int, gate_dim: int, gate_bias_init: float) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.delta_projection = nn.Linear(result_dim, hidden_size, bias=False)
        self.gate_hidden_projection = nn.Linear(hidden_size, gate_dim, bias=False)
        self.gate_result_projection = nn.Linear(result_dim, gate_dim, bias=True)
        self.gate_output = nn.Linear(gate_dim, 1, bias=True)
        nn.init.constant_(self.gate_output.bias, float(gate_bias_init))

    def forward(self, hidden: torch.Tensor, result_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        work_hidden = hidden.float()
        hidden_for_gate = F.layer_norm(work_hidden, (self.hidden_size,))
        gate_state = torch.tanh(
            self.gate_hidden_projection(hidden_for_gate)
            + self.gate_result_projection(result_repr)
        )
        gate = torch.sigmoid(self.gate_output(gate_state))
        delta = self.delta_projection(result_repr)
        eps = torch.finfo(torch.float32).eps
        hidden_rms = work_hidden.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
        delta_rms = delta.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
        normalized_delta = delta / delta_rms * hidden_rms
        updated = work_hidden + gate * normalized_delta
        return updated.to(dtype=hidden.dtype), gate


class LayeredSymbolicResultMapper(nn.Module):
    """Shared symbolic encoder with one accepted residual adapter per candidate layer.

    Integer and single-character results share one tokenizer-independent encoding.
    Only the hidden-space translation/gate is layer specific. This is the final
    release path for layer-wise analysis and preserves the original candidate-layer
    intervention concept without reintroducing the old Phase-1 classifier.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        candidate_layers: list[int] | tuple[int, ...],
        result_dim: int,
        gate_dim: int,
        max_integer: int,
        ascii_vocab_size: int,
        gate_bias_init: float,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.candidate_layers = tuple(int(x) for x in candidate_layers)
        if not self.candidate_layers or len(set(self.candidate_layers)) != len(self.candidate_layers):
            raise ValueError("candidate_layers must be a non-empty unique sequence")
        self.max_integer = int(max_integer)
        self.ascii_vocab_size = int(ascii_vocab_size)
        self.result_dim = int(result_dim)

        self.kind_embedding = nn.Embedding(2, result_dim)
        self.integer_bits = max(1, int(max_integer).bit_length())
        self.character_bits = max(1, int(ascii_vocab_size - 1).bit_length())
        self.integer_feature_projection = nn.Linear(self.integer_bits + 1, result_dim)
        self.character_feature_projection = nn.Linear(self.character_bits + 1, result_dim)
        self.result_norm = nn.LayerNorm(result_dim)
        self.result_mlp = nn.Sequential(
            nn.Linear(result_dim, result_dim),
            nn.SiLU(),
            nn.Linear(result_dim, result_dim),
        )
        self.adapters = nn.ModuleDict({
            str(layer): _LayerSymbolicAdapter(
                hidden_size=hidden_size,
                result_dim=result_dim,
                gate_dim=gate_dim,
                gate_bias_init=gate_bias_init,
            )
            for layer in self.candidate_layers
        })

    @staticmethod
    def _binary_features(value: int, bits: int, maximum: int, device: torch.device) -> torch.Tensor:
        bit_values = [float((value >> bit) & 1) for bit in range(bits)]
        normalized = 0.0 if maximum <= 0 else float(value) / float(maximum)
        return torch.tensor([*bit_values, normalized], device=device, dtype=torch.float32)

    def encode_symbol(self, symbol: ResultSymbol, device: torch.device) -> torch.Tensor:
        if symbol.kind == "integer":
            kind_id = 0
            features = self._binary_features(symbol.value, self.integer_bits, self.max_integer, device)
            value_embedding = self.integer_feature_projection(features)
        elif symbol.kind == "character":
            kind_id = 1
            features = self._binary_features(
                symbol.value, self.character_bits, self.ascii_vocab_size - 1, device
            )
            value_embedding = self.character_feature_projection(features)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported result kind: {symbol.kind}")
        kind_embedding = self.kind_embedding(torch.tensor(kind_id, device=device, dtype=torch.long))
        encoded = self.result_norm(kind_embedding + value_embedding)
        return encoded + self.result_mlp(encoded)

    def forward(
        self,
        hidden: torch.Tensor,
        symbol: ResultSymbol,
        *,
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_index = int(layer_index)
        if layer_index not in self.candidate_layers:
            raise ValueError(f"Layer {layer_index} is not in candidate_layers={self.candidate_layers}")
        if hidden.ndim != 2 or hidden.shape[-1] != self.hidden_size:
            raise ValueError(f"hidden must be [batch, {self.hidden_size}]")
        if hidden.shape[0] != 1:
            raise ValueError("Layered symbolic mapper currently supports batch_size=1 only")
        result_repr = self.encode_symbol(symbol, hidden.device).float().unsqueeze(0)
        return self.adapters[str(layer_index)](hidden, result_repr)


class TrainableLayeredResultInjector:
    """Inject a deterministic result through a LayeredSymbolicResultMapper."""

    def __init__(
        self,
        model: Any,
        mapper: LayeredSymbolicResultMapper,
        *,
        layer_index: int,
        symbol: ResultSymbol,
        position_index: int,
        prefill_only: bool,
    ) -> None:
        self.model = model
        self.mapper = mapper
        self.layer_index = int(layer_index)
        self.symbol = symbol
        self.position_index = int(position_index)
        self.prefill_only = bool(prefill_only)
        self.last_gate: float | None = None
        self._handle = None
        layers = self._layers()
        if not 0 <= self.layer_index < len(layers):
            raise IndexError(f"layer_index={self.layer_index} is outside [0, {len(layers)-1}]")
        if self.layer_index not in mapper.candidate_layers:
            raise ValueError(f"layer_index={self.layer_index} is not configured in the mapper")

    def _layers(self):
        base_model = getattr(self.model, "model", None)
        layers = getattr(base_model, "layers", None)
        if layers is None:
            raise TypeError("Expected decoder layers at model.model.layers")
        return layers

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
            raise TypeError("Expected hidden state [batch, seq, hidden]")
        if self.prefill_only and hidden.shape[1] <= 1:
            return output
        position = self.position_index
        if position < 0:
            position = hidden.shape[1] + position
        if not 0 <= position < hidden.shape[1]:
            raise IndexError(
                f"Injection position {self.position_index} is invalid for sequence length {hidden.shape[1]}."
            )
        current = hidden[:, position, :]
        updated, gate = self.mapper(current, self.symbol, layer_index=self.layer_index)
        self.last_gate = float(gate.detach().mean().item())
        new_hidden = hidden.clone()
        new_hidden[:, position, :] = updated
        return self._replace_hidden(output, new_hidden)

    def __enter__(self) -> "TrainableLayeredResultInjector":
        if self._handle is not None:
            raise RuntimeError("Injector is already active")
        self._handle = self._layers()[self.layer_index].register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
