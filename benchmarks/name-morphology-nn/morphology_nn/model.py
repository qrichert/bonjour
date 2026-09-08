from __future__ import annotations

import json
import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .data import BYTE_VOCABULARY_SIZE, PAD_ID

MAGIC = b"BNJRMNN1"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    byte_embedding: int
    conv_channels: int
    kernel_one: int
    kernel_two: int
    hidden: int
    country_embedding: int


CONFIGURATIONS = (
    ModelConfig("tiny_string", 8, 16, 3, 3, 16, 0),
    ModelConfig("small_string", 16, 32, 3, 5, 32, 0),
    ModelConfig("small_country", 16, 32, 3, 5, 32, 4),
)


class MorphologyModel(nn.Module):
    def __init__(self, config: ModelConfig, country_count: int) -> None:
        super().__init__()
        self.config = config
        self.byte_embedding = nn.Embedding(
            BYTE_VOCABULARY_SIZE,
            config.byte_embedding,
            padding_idx=PAD_ID,
        )
        self.conv_one = nn.Conv1d(
            config.byte_embedding,
            config.conv_channels,
            config.kernel_one,
            padding=config.kernel_one // 2,
        )
        self.conv_two = nn.Conv1d(
            config.conv_channels,
            config.conv_channels,
            config.kernel_two,
            padding=config.kernel_two // 2,
        )
        if config.country_embedding:
            self.country_embedding = nn.Embedding(
                country_count,
                config.country_embedding,
                padding_idx=0,
            )
        else:
            self.country_embedding = None
        pooled = config.conv_channels * 2 + config.country_embedding
        self.hidden = nn.Linear(pooled, config.hidden)
        self.output = nn.Linear(config.hidden, 1)

    def forward(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        countries: torch.Tensor,
    ) -> torch.Tensor:
        values = self.byte_embedding(ids).transpose(1, 2)
        values = torch.relu(self.conv_one(values))
        values = torch.relu(self.conv_two(values))
        expanded_mask = mask.unsqueeze(1)
        maximum = values.masked_fill(~expanded_mask, -torch.inf).amax(dim=2)
        mean = (values * expanded_mask).sum(dim=2) / expanded_mask.sum(dim=2).clamp_min(
            1
        )
        pooled = torch.cat((maximum, mean), dim=1)
        if self.country_embedding is not None:
            pooled = torch.cat((pooled, self.country_embedding(countries)), dim=1)
        return self.output(torch.relu(self.hidden(pooled))).squeeze(1)


def parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def copy_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def export_model(
    model: MorphologyModel,
    metadata: dict[str, Any],
    json_path: Path,
    binary_path: Path,
) -> None:
    complete = {
        **metadata,
        "format": "bonjour-byte-morphology-f32-v1",
        "config": asdict(model.config),
        "parameter_count": parameter_count(model),
        "tensor_order": list(model.state_dict()),
    }
    json_bytes = _json_bytes(complete)
    json_path.write_bytes(json_bytes)
    with binary_path.open("xb") as destination:
        destination.write(MAGIC)
        destination.write(struct.pack("<I", len(json_bytes)))
        destination.write(json_bytes)
        tensors = model.state_dict()
        destination.write(struct.pack("<I", len(tensors)))
        for name, tensor in tensors.items():
            contiguous = tensor.detach().cpu().to(torch.float32).contiguous()
            name_bytes = name.encode("utf-8")
            destination.write(struct.pack("<H", len(name_bytes)))
            destination.write(name_bytes)
            destination.write(struct.pack("<B", contiguous.ndim))
            for dimension in contiguous.shape:
                destination.write(struct.pack("<I", dimension))
            values = contiguous.reshape(-1).tolist()
            destination.write(struct.pack(f"<{len(values)}f", *values))


def load_export(path: Path) -> tuple[MorphologyModel, dict[str, Any]]:
    with path.open("rb") as source:
        if source.read(len(MAGIC)) != MAGIC:
            raise ValueError("invalid morphology model magic")
        metadata_size = _read_struct(source, "<I")[0]
        metadata = json.loads(source.read(metadata_size))
        config = ModelConfig(**metadata["config"])
        countries = tuple(metadata["countries"])
        model = MorphologyModel(config, len(countries))
        tensor_count = _read_struct(source, "<I")[0]
        tensors: dict[str, torch.Tensor] = {}
        for _ in range(tensor_count):
            name_size = _read_struct(source, "<H")[0]
            name = source.read(name_size).decode("utf-8")
            dimensions = _read_struct(source, "<B")[0]
            shape = tuple(_read_struct(source, "<I")[0] for _ in range(dimensions))
            count = math.prod(shape)
            values = _read_struct(source, f"<{count}f")
            tensors[name] = torch.tensor(values, dtype=torch.float32).reshape(shape)
        if source.read(1):
            raise ValueError("trailing bytes in morphology model")
    if list(tensors) != metadata["tensor_order"]:
        raise ValueError("morphology tensor order does not match metadata")
    model.load_state_dict(tensors, strict=True)
    model.eval()
    return model, metadata


def inference_operations(config: ModelConfig, length: int) -> dict[str, int]:
    conv_one = length * config.conv_channels * config.byte_embedding * config.kernel_one
    conv_two = length * config.conv_channels * config.conv_channels * config.kernel_two
    dense_input = config.conv_channels * 2 + config.country_embedding
    dense = dense_input * config.hidden + config.hidden
    pooling = config.conv_channels * max(0, length - 1) * 2
    return {
        "multiply_accumulates": conv_one + conv_two + dense,
        "pooling_additions_comparisons": pooling,
    }


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()


def _read_struct(source, specification: str) -> tuple:
    size = struct.calcsize(specification)
    value = source.read(size)
    if len(value) != size:
        raise ValueError("truncated morphology model")
    return struct.unpack(specification, value)
