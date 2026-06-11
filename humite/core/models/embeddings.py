from __future__ import annotations

from typing import Dict, List, Optional, Union, cast

import torch
from torch import nn


class TokenEnergyEmbedding(nn.Module):
    def __init__(
        self, token_vocab_size: int, embedding_dim: int, energy_vocab_size: Optional[int] = None
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.token_embedding = nn.Embedding(token_vocab_size, embedding_dim, padding_idx=0)
        self.energy_embedding = (
            nn.Embedding(energy_vocab_size, embedding_dim, padding_idx=0)
            if energy_vocab_size is not None
            else None
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        energy_ids: Optional[torch.Tensor] = None,
        *,
        axis_interleaved: bool = False,
    ) -> torch.Tensor:
        token_emb = self.token_embedding(token_ids)
        if self.energy_embedding is None or energy_ids is None:
            return token_emb
        energy_emb = self.energy_embedding(energy_ids)
        if axis_interleaved:
            stacked = torch.stack([token_emb, energy_emb], dim=2)
            merged = stacked.reshape(stacked.shape[0], stacked.shape[1] * 2, stacked.shape[3])
            return merged
        return token_emb + energy_emb


class TokenTypeEmbedding(nn.Module):
    def __init__(self, n_types: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_types, embedding_dim)

    def forward(self, type_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(type_ids)


class MultiStreamTokenEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        *,
        shared: bool,
        max_vocab_size: Optional[int] = None,
        per_stream_vocab_sizes: Optional[Dict[str, int]] = None,
        stream_names: Optional[List[str]] = None,
        add_stream_type_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.embedding_dim: int = embedding_dim
        self.shared: bool = shared
        self.add_stream_type_embeddings: bool = add_stream_type_embeddings
        self.shared_embedding: Optional[nn.Embedding] = None
        self.embeddings: Optional[nn.ModuleDict] = None
        self.type_embedding: Optional[nn.Embedding] = None
        self.stream_to_index: Dict[str, int] = {}
        if shared:
            assert max_vocab_size is not None
            assert stream_names is not None
            self.shared_embedding = nn.Embedding(max_vocab_size, embedding_dim, padding_idx=0)
            self.stream_to_index = {name: i for i, name in enumerate(stream_names)}
            self.type_embedding = (
                nn.Embedding(len(stream_names), embedding_dim)
                if add_stream_type_embeddings
                else None
            )
        else:
            assert per_stream_vocab_sizes is not None
            self.embeddings = nn.ModuleDict(
                {
                    k: nn.Embedding(v, embedding_dim, padding_idx=0)
                    for k, v in per_stream_vocab_sizes.items()
                }
            )
            if add_stream_type_embeddings:
                names = list(per_stream_vocab_sizes.keys())
                self.stream_to_index = {name: i for i, name in enumerate(names)}
                self.type_embedding = nn.Embedding(len(names), embedding_dim)

    def embed_one(self, name: str, token_ids: torch.Tensor) -> torch.Tensor:
        if self.shared:
            assert self.shared_embedding is not None
            base = self.shared_embedding(token_ids)
        else:
            assert self.embeddings is not None
            base = self.embeddings[name](token_ids)
        if self.type_embedding is None:
            return base
        type_index = torch.full_like(token_ids, fill_value=self.stream_to_index[name])
        type_emb = self.type_embedding(type_index)
        return base + type_emb

    def forward(
        self,
        streams: Union[Dict[str, torch.Tensor], List[Dict[str, Union[str, torch.Tensor]]]],
    ) -> Dict[str, torch.Tensor]:
        if isinstance(streams, list):
            tokens_by_name: Dict[str, torch.Tensor] = {
                cast(str, s["name"]): cast(torch.Tensor, s["input_ids"]) for s in streams
            }
        else:
            tokens_by_name = cast(Dict[str, torch.Tensor], streams)
        return {name: self.embed_one(name, ids) for name, ids in tokens_by_name.items()}


class InterleavedEmbedding(nn.Module):
    def __init__(
        self,
        token_vocab_size: int,
        embedding_dim: int,
        n_position_types: int,
        *,
        add_type_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(token_vocab_size, embedding_dim, padding_idx=0)
        self.type_embedding = (
            nn.Embedding(n_position_types, embedding_dim) if add_type_embedding else None
        )

    def forward(
        self, input_ids: torch.Tensor, type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        base = self.token_embedding(input_ids)
        if self.type_embedding is None or type_ids is None:
            return base
        return base + self.type_embedding(type_ids)


class PerAxisConcatEmbedding(nn.Module):
    def __init__(
        self,
        *,
        vocab_sizes: Dict[str, int],
        embedding_dims: Optional[Dict[str, int]] = None,
        pad_idx: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.axis_names = tuple(vocab_sizes.keys())
        if embedding_dims is None:
            embedding_dims = {axis: 64 for axis in self.axis_names}
        self.vocab_sizes = vocab_sizes
        self.embedding_dims = embedding_dims
        dropout_value = float(dropout)
        self.embeddings = nn.ModuleDict(
            {
                axis: nn.Embedding(vocab_sizes[axis], embedding_dims[axis])
                for axis in self.axis_names
            }
        )
        self.norms = nn.ModuleDict(
            {axis: nn.LayerNorm(embedding_dims[axis]) for axis in self.axis_names}
        )
        self.dropouts = nn.ModuleDict(
            {axis: nn.Dropout(dropout_value) for axis in self.axis_names}
        )

    @property
    def output_dim(self) -> int:
        return int(sum(self.embedding_dims[axis] for axis in self.axis_names))

    def _embed_axis(self, axis: str, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embeddings[axis](tokens)
        normalized = self.norms[axis](embedded)
        return self.dropouts[axis](normalized)

    def embed_axis_dict(self, axis_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {axis: self._embed_axis(axis, axis_tokens[axis]) for axis in self.axis_names}

    def forward(self, axis_tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        per_axis = self.embed_axis_dict(axis_tokens)
        ordered = [per_axis[name] for name in self.axis_names]
        return torch.cat(ordered, dim=-1)
