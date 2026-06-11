from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from .embeddings import PerAxisConcatEmbedding
from .gpt_decoder import AttentionCache, GPTDecoderStack, TransformerConfig
from .heads import (
    EnergyHead,
    MixtureOfGaussiansEnergyHead,
    TokenHead,
    mixture_expected_value,
)


class DecoderCore(nn.Module):
    def __init__(self, transformer: TransformerConfig) -> None:
        super().__init__()
        self.embedding_dim = transformer.embedding_dim
        self.decoder = GPTDecoderStack(transformer)

    def forward(
        self,
        input_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[AttentionCache]] = None,
        use_kv_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[AttentionCache]]]:  # type: ignore[override]
        return self.decoder(
            input_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_kv_cache=use_kv_cache,
        )


class BaseModel(nn.Module):
    def __init__(
        self,
        transformer: TransformerConfig,
        n_registers: int = 0,
        register_condition_incident_energy: bool = True,
        zero_init_register_incident: bool = True,
        register_condition_n_hits: bool = False,
    ) -> None:
        super().__init__()
        self.core = DecoderCore(transformer)
        self.enable_multi_axis_heads = False
        self._detach_energy_projection = False
        self.n_registers = int(max(0, n_registers))
        self.register_condition_incident_energy = bool(register_condition_incident_energy)
        self.register_condition_n_hits = bool(register_condition_n_hits)
        if self.n_registers > 0:
            # Match legacy backbone checkpoint shape: (n_registers, embedding_dim)
            self.register_embeddings = nn.Parameter(
                torch.randn(self.n_registers, self.embedding_dim)
            )
            self.register_incident_proj = (
                nn.Sequential(
                    nn.Linear(1, self.embedding_dim),
                    nn.SiLU(),
                    nn.Linear(self.embedding_dim, self.embedding_dim),
                )
                if self.register_condition_incident_energy
                else None
            )
            if self.register_incident_proj is not None and zero_init_register_incident:
                self.register_incident_proj[-1].weight.data.zero_()
                self.register_incident_proj[-1].bias.data.zero_()
        else:
            self.register_embeddings = None  # type: ignore[assignment]
            self.register_incident_proj = None  # type: ignore[assignment]
        if self.register_condition_n_hits and self.n_registers <= 0:
            raise ValueError("register_condition_n_hits requires at least one register")
        self.register_n_hits_proj = (
            nn.Sequential(
                nn.Linear(1, self.embedding_dim),
                nn.SiLU(),
                nn.Linear(self.embedding_dim, self.embedding_dim),
            )
            if self.register_condition_n_hits
            else None
        )

    @property
    def embedding_dim(self) -> int:
        return int(self.core.embedding_dim)

    def build_embeddings_and_mask(
        self,
        batch: Optional[dict],
        token_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def project_tokens(self, hidden: torch.Tensor, batch: Optional[dict] = None) -> Dict[str, Any]:
        return {}

    def project_energy(
        self,
        hidden: torch.Tensor,
        batch: Optional[dict] = None,
        token_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        return {}

    def get_targets(
        self, batch: Optional[dict], token_ids: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return {}

    def _graph_decode_core(
        self,
        chunk_embedding: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: List[AttentionCache],
    ) -> torch.Tensor:
        """Run transformer core only — capturable by torch.cuda.CUDAGraph.

        Embedding computation and token/energy projection must happen outside
        the graph boundary. The KV caches are updated in-place via filled_tensor.
        """
        hidden, _ = self.core(
            chunk_embedding,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_kv_cache=True,
        )
        return hidden

    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[override]
        batch: Optional[dict] = kwargs.get("batch")
        token_ids: Optional[torch.Tensor] = kwargs.get("token_ids")
        attention_mask: Optional[torch.Tensor] = kwargs.get("attention_mask")
        shower_energy: Optional[torch.Tensor] = kwargs.get("shower_energy")
        past_key_values: Optional[List[AttentionCache]] = kwargs.get("past_key_values")
        use_kv_cache = bool(kwargs.get("use_kv_cache", False))
        local_batch: dict = (
            dict(batch) if isinstance(batch, dict) else ({} if batch is None else batch)
        )

        # Route shower_energy (= x_shower from the generation call) into the batch so
        # that _fetch_incident_energy and _extract_incident_energy_tensor can find it.
        # Only inject when not already present — explicit batch values take precedence.
        if shower_energy is not None and "incident_energy_cond" not in local_batch:
            local_batch["incident_energy_cond"] = shower_energy

        embeddings, token_mask = self.build_embeddings_and_mask(
            local_batch, token_ids, attention_mask
        )

        core_embeddings = embeddings
        core_mask = token_mask
        if self.n_registers > 0:
            bsz = int(core_embeddings.size(0))
            assert self.register_embeddings is not None
            regs = self.register_embeddings.unsqueeze(0).expand(bsz, -1, -1)
            n_hits_tensor = self._extract_n_hits_tensor(
                local_batch, bsz, core_embeddings.device, core_embeddings.dtype
            )
            if n_hits_tensor is not None:
                if self.register_n_hits_proj is None:
                    raise RuntimeError("Missing register_n_hits_proj")
                regs = regs.clone()
                regs[:, 0, :] = regs[:, 0, :] + self.register_n_hits_proj(n_hits_tensor)

            incident_energy_tensor = self._extract_incident_energy_tensor(
                local_batch, bsz, core_embeddings.device, core_embeddings.dtype
            )
            if incident_energy_tensor is not None:
                if self.register_incident_proj is None:
                    raise RuntimeError(
                        "Missing register_incident_proj although incident_energy_tensor is present"
                    )

                proj_out = self.register_incident_proj(incident_energy_tensor)
                if getattr(self, "incident_energy_dropout", None) is not None:
                    proj_out = self.incident_energy_dropout(proj_out)

                regs = regs.clone()
                regs[:, 0, :] = regs[:, 0, :] + proj_out

            core_embeddings = torch.cat([regs, core_embeddings], dim=1)
            register_mask = torch.ones(
                (bsz, self.n_registers), dtype=core_mask.dtype, device=core_mask.device
            )
            core_mask = torch.cat([register_mask, core_mask], dim=1)

        if use_kv_cache and core_mask is not None:
            active_columns = core_mask.to(torch.bool).any(dim=0)
            if torch.any(active_columns):
                last_active = int(torch.nonzero(active_columns, as_tuple=False)[-1].item()) + 1
            else:
                last_active = 0
            trim_len = max(last_active, self.n_registers if self.n_registers > 0 else 1)
            trim_len = min(trim_len, core_embeddings.size(1))
            core_embeddings = core_embeddings[:, :trim_len, :]
            core_mask = core_mask[:, :trim_len]

        processed_tokens = 0
        if use_kv_cache and past_key_values:
            first_cache = past_key_values[0]
            processed_tokens = int(first_cache.offset + first_cache.key.size(2))
        chunk_embeddings = core_embeddings
        attention_mask_full = core_mask
        if use_kv_cache and processed_tokens > 0:
            new_tokens = int(core_embeddings.size(1) - processed_tokens)
            if new_tokens <= 0:
                raise RuntimeError("kv_cache has no room for new tokens")
            chunk_embeddings = core_embeddings[:, -new_tokens:, :]
        hidden_chunk, present = self.core(
            chunk_embeddings,
            attention_mask=attention_mask_full,
            past_key_values=past_key_values if use_kv_cache else None,
            use_kv_cache=use_kv_cache,
        )
        uses_cached_prefix = use_kv_cache and processed_tokens > 0
        if self.n_registers > 0 and not uses_cached_prefix:
            hidden_tokens = hidden_chunk[:, self.n_registers :, :]
            register_hidden = hidden_chunk[:, : self.n_registers, :]
        else:
            hidden_tokens = hidden_chunk
            register_hidden = None

        out: Dict[str, Any] = {"hidden_states": hidden_tokens, "mask": token_mask}
        out.update(self.get_targets(local_batch, token_ids))
        out.update(self.project_tokens(hidden_tokens, local_batch))
        out.update(self.project_energy(hidden_tokens, local_batch, token_ids))
        if register_hidden is not None:
            out["register_hidden"] = register_hidden
        if use_kv_cache:
            out["present_key_values"] = present
        return out

    def ensure_split_axes(self, batch: Optional[dict]) -> None:
        return

    def set_energy_projection_detach(self, should_detach: bool) -> None:
        self._detach_energy_projection = bool(should_detach)

    def _extract_n_hits_tensor(
        self,
        batch: Optional[dict],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self.register_condition_n_hits:
            return None
        if not isinstance(batch, dict):
            raise RuntimeError("n_hits conditioning requires batch dictionary")
        if "n_hits_cond" in batch and batch["n_hits_cond"] is not None:
            n_hits_raw = batch["n_hits_cond"]
        else:
            raise RuntimeError("Missing n_hits_cond for length conditioning")
        n_hits_tensor = n_hits_raw.to(device=device, dtype=dtype)
        if n_hits_tensor.dim() == 0:
            n_hits_tensor = n_hits_tensor.view(1, 1)
        elif n_hits_tensor.dim() == 1:
            n_hits_tensor = n_hits_tensor.view(-1, 1)
        else:
            n_hits_tensor = n_hits_tensor.view(n_hits_tensor.shape[0], -1)[:, :1]
        if n_hits_tensor.shape[0] == 1 and batch_size > 1:
            n_hits_tensor = n_hits_tensor.expand(batch_size, -1)
        if n_hits_tensor.shape[0] != batch_size:
            raise RuntimeError("n_hits batch dimension mismatch")
        return n_hits_tensor

    def _extract_incident_energy_tensor(
        self,
        batch: Optional[dict],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self.register_condition_incident_energy:
            return None
        if not isinstance(batch, dict):
            raise RuntimeError("incident_energy conditioning requires batch dictionary")
        if "incident_energy_cond" in batch and batch["incident_energy_cond"] is not None:
            incident_raw = batch["incident_energy_cond"]
        elif "incident_energy" in batch and batch["incident_energy"] is not None:
            incident_raw = batch["incident_energy"]
        else:
            raise RuntimeError("Missing incident_energy or incident_energy_cond for conditioning")

        incident_tensor = incident_raw.to(device=device, dtype=dtype)
        if incident_tensor.dim() == 0:
            incident_tensor = incident_tensor.view(1, 1)
        elif incident_tensor.dim() == 1:
            incident_tensor = incident_tensor.view(-1, 1)
        else:
            incident_tensor = incident_tensor.view(incident_tensor.shape[0], -1)[:, :1]

        if incident_tensor.shape[0] == 1 and batch_size > 1:
            incident_tensor = incident_tensor.expand(batch_size, -1)

        if incident_tensor.shape[0] != batch_size:
            raise RuntimeError("incident_energy batch dimension mismatch")
        return incident_tensor


class AutoregressiveBackbone(BaseModel):
    def __init__(
        self,
        *,
        transformer: TransformerConfig,
        n_registers: int = 0,
        register_condition_incident_energy: bool = True,
        zero_init_register_incident: bool = True,
        use_n_hits_register: bool = False,
        use_energy_tokenization: bool = False,
        energy_vocab_size: Optional[int] = None,
        energy_binning_mode: str = "uniform",
        energy_bin_centers: Optional[Any] = None,
        energy_bin_edges: Optional[Any] = None,
        hit_energy_dropout_p: float = 0.0,
        incident_energy_dropout_p: float = 0.0,
        use_hit_energy_feature: bool = False,
        hit_energy_feature_dim: int = 16,
        use_incident_energy_feature: bool = False,
        incident_energy_feature_dim: int = 32,
        use_n_hits_feature: bool = False,
        n_hits_feature_dim: int = 32,
        n_hits_dropout_p: float = 0.0,
        zero_init_n_hits_feature: bool = True,
        use_mog_energy: bool = True,
        mog_components: int = 5,
        energy_sample_min: float | None = -5.0,
        energy_sample_max: float | None = 5.0,
        energy_nan_replacement: Optional[Dict[str, float]] = None,
        energy_noise_rel_std: float = 0.0,
    ) -> None:
        self.use_n_hits_register = bool(use_n_hits_register)
        base_registers = int(max(0, n_registers))
        n_registers_effective = (
            max(base_registers, 1) if self.use_n_hits_register else base_registers
        )
        super().__init__(
            transformer=transformer,
            n_registers=n_registers_effective,
            register_condition_incident_energy=register_condition_incident_energy,
            zero_init_register_incident=zero_init_register_incident,
            register_condition_n_hits=self.use_n_hits_register,
        )
        self.dim = self.embedding_dim

        self.use_energy_tokenization = bool(use_energy_tokenization)
        if self.use_energy_tokenization:
            if energy_vocab_size is None:
                raise ValueError("use_energy_tokenization requires energy_vocab_size")
            energy_token_dim = 64
            self.energy_embedding = nn.Embedding(
                int(energy_vocab_size), energy_token_dim, padding_idx=0
            )
        else:
            self.energy_embedding = None
        self.energy_token_feature_dim = 64 if self.use_energy_tokenization else 0
        self.use_hit_energy_feature = (
            bool(use_hit_energy_feature) and not self.use_energy_tokenization
        )
        self.use_incident_energy_feature = bool(use_incident_energy_feature)
        self.use_n_hits_feature = bool(use_n_hits_feature)
        self.hit_energy_feature_dim = (
            int(hit_energy_feature_dim) if self.use_hit_energy_feature else 0
        )
        self.incident_energy_feature_dim = (
            int(incident_energy_feature_dim) if self.use_incident_energy_feature else 0
        )
        self.n_hits_feature_dim = int(n_hits_feature_dim) if self.use_n_hits_feature else 0
        self.require_hit_energy = self.use_hit_energy_feature
        self.require_incident_energy = self.use_incident_energy_feature
        self.require_n_hits_feature = self.use_n_hits_feature
        self.hit_energy_feature_proj = (
            nn.Sequential(
                nn.Linear(1, 16),
                nn.SiLU(),
                nn.Linear(16, self.hit_energy_feature_dim),
            )
            if self.hit_energy_feature_dim > 0
            else None
        )

        self.incident_energy_feature_proj = (
            nn.Sequential(
                nn.Linear(1, 16),
                nn.SiLU(),
                nn.Linear(16, self.incident_energy_feature_dim),
            )
            if self.incident_energy_feature_dim > 0
            else None
        )
        self.n_hits_feature_proj = (
            nn.Sequential(
                nn.Linear(1, 16),
                nn.SiLU(),
                nn.Linear(16, self.n_hits_feature_dim),
            )
            if self.n_hits_feature_dim > 0
            else None
        )
        # Zero-init the output layer so that freshly-added n_hits conditioning
        # starts as a no-op when loading from a checkpoint that was trained
        # without this feature. Training then gradually learns to exploit it.
        if self.n_hits_feature_proj is not None and bool(zero_init_n_hits_feature):
            nn.init.zeros_(self.n_hits_feature_proj[-1].weight)
            nn.init.zeros_(self.n_hits_feature_proj[-1].bias)
        self.mog_components = int(mog_components)
        self.mog_log_scale_min = -4.0
        self.mog_log_scale_max = 4.0
        self.mog_scale_epsilon = 0.0001
        if energy_sample_min is not None and energy_sample_max is not None:
            if float(energy_sample_min) >= float(energy_sample_max):
                raise ValueError("energy_sample_min must be less than energy_sample_max")
            self.energy_sample_clamp: Optional[Tuple[float, float]] = (
                float(energy_sample_min),
                float(energy_sample_max),
            )
        else:
            self.energy_sample_clamp = None
        if energy_nan_replacement is None:
            clamp_min = float(energy_sample_min) if energy_sample_min is not None else -10.0
            clamp_max = float(energy_sample_max) if energy_sample_max is not None else 10.0
            self.energy_nan_replacement = {
                "nan": 0.0,
                "posinf": clamp_max,
                "neginf": clamp_min,
            }
        else:
            self.energy_nan_replacement = {
                key: float(value) for key, value in energy_nan_replacement.items()
            }

        self.energy_noise_rel_std = float(max(0.0, energy_noise_rel_std))

        self.hit_energy_dropout = (
            nn.Dropout(hit_energy_dropout_p) if hit_energy_dropout_p > 0.0 else None
        )

        self.incident_energy_dropout = (
            nn.Dropout(incident_energy_dropout_p) if incident_energy_dropout_p > 0.0 else None
        )
        self.n_hits_dropout = nn.Dropout(n_hits_dropout_p) if n_hits_dropout_p > 0.0 else None

        # Generation-time cache for constant features (incident energy, n_hits).
        # Populated once via cache_generation_features(), cleared after generation.
        self._gen_cached_incident_features: Optional[torch.Tensor] = None
        self._gen_cached_n_hits_features: Optional[torch.Tensor] = None

        self.additional_dim = (
            self.hit_energy_feature_dim
            + self.incident_energy_feature_dim
            + self.n_hits_feature_dim
            + self.energy_token_feature_dim
        )
        self.energy_head_backend = "none"

        # Energy binning mode determines which head runs under tokenization:
        # "uniform"         -> existing MOG-per-bin head (CE + per-component NLL).
        # "hcal_resolution" -> pure classification head: Linear(dim, vocab_size)
        #                     emitting energy_logits for CE loss; continuous
        #                     decoding uses `_energy_bin_centers[token]`.
        self.energy_binning_mode = str(energy_binning_mode)
        if energy_bin_centers is not None:
            centers_tensor = torch.as_tensor(energy_bin_centers, dtype=torch.float32)
            self.register_buffer("_energy_bin_centers", centers_tensor, persistent=False)
        else:
            self._energy_bin_centers = None  # type: ignore[assignment]
        if energy_bin_edges is not None:
            edges_tensor = torch.as_tensor(energy_bin_edges, dtype=torch.float32)
            self.register_buffer("_energy_bin_edges", edges_tensor, persistent=False)
        else:
            self._energy_bin_edges = None  # type: ignore[assignment]

        if self.use_energy_tokenization:
            if energy_vocab_size is None:
                raise ValueError("use_energy_tokenization requires energy_vocab_size")
            if self.energy_binning_mode in ("hcal_resolution", "classification"):
                self.energy_head_type = "classification"
                self.energy_head_backend = "classification"
                self.energy_classification_head = TokenHead(self.dim, int(energy_vocab_size))
                self.mog_head = None
                self.energy_head = None
            else:
                if (
                    self.mog_log_scale_min is None
                    or self.mog_log_scale_max is None
                    or self.mog_scale_epsilon is None
                ):
                    raise ValueError(
                        "use_energy_tokenization requires mog_log_scale_min, mog_log_scale_max, and mog_scale_epsilon"
                    )
                self.energy_head_type = "mog"
                self.energy_head_backend = "mog"
                self.mog_head = MixtureOfGaussiansEnergyHead(
                    self.dim,
                    int(energy_vocab_size),
                    log_scale_min=self.mog_log_scale_min,
                    log_scale_max=self.mog_log_scale_max,
                )
                self.energy_head = None
                self.energy_classification_head = None
        elif use_mog_energy and int(mog_components) > 0:
            if (
                self.mog_log_scale_min is None
                or self.mog_log_scale_max is None
                or self.mog_scale_epsilon is None
            ):
                raise ValueError(
                    "use_mog_energy requires mog_log_scale_min, mog_log_scale_max, and mog_scale_epsilon"
                )
            self.energy_head_type = "mog"
            self.energy_head_backend = "mog"
            self.mog_head = MixtureOfGaussiansEnergyHead(
                self.dim,
                int(mog_components),
                log_scale_min=self.mog_log_scale_min,
                log_scale_max=self.mog_log_scale_max,
            )
            self.energy_head = None
            self.energy_classification_head = None
        else:
            self.energy_head_type = "linear"
            self.energy_head_backend = "linear"
            self.energy_head = EnergyHead(self.dim)
            self.mog_head = None
            self.energy_classification_head = None

    def sanitize_autoregressive_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Sanitize mask to prevent attending to tokens after the first invalid position."""
        mask_bool = mask.to(torch.bool)
        seen_true = torch.cumsum(mask_bool.to(torch.int64), dim=1) > 0
        false_after_true = (~mask_bool) & seen_true
        leading_valid = mask_bool & (~false_after_true)  # Simple validation
        # More complex trailing logic matching previous impl:
        trailing_false = torch.cumsum(false_after_true.to(torch.int64), dim=1) > 0
        sanitized = mask_bool & (~trailing_false)
        return sanitized

    def _resolve_target_n_hits(
        self,
        batch_size: int,
        max_len: int,
        device: torch.device,
        sampling_config: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        should_stop_on_length = bool(sampling_config.get("use_n_hits_stop", False))
        if (
            not self.use_n_hits_register
            and not self.use_n_hits_feature
            and not should_stop_on_length
        ):
            return None
        raw = sampling_config.get("n_hits")
        if raw is None:
            raise RuntimeError(
                "n_hits is required when use_n_hits_register or use_n_hits_feature is enabled"
            )
        if torch.is_tensor(raw):
            target = raw.to(device=device, dtype=torch.long)
        else:
            target = torch.as_tensor(raw, device=device, dtype=torch.long)
        if target.dim() == 0:
            target = target.view(1)
        if target.dim() == 1:
            if target.shape[0] == 1 and batch_size > 1:
                target = target.expand(batch_size)
        else:
            target = target.view(target.shape[0], -1)[:, 0]
        if target.shape[0] != batch_size:
            raise RuntimeError("n_hits shape mismatch for generation")
        target = target.clamp(min=1, max=max_len)
        return target

    def _resolve_n_hits_cond(
        self,
        batch_size: int,
        device: torch.device,
        sampling_config: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        raw = sampling_config.get("n_hits_cond")
        if raw is None:
            return None
        if torch.is_tensor(raw):
            n_hits = raw.to(device=device, dtype=torch.float32)
        else:
            n_hits = torch.as_tensor(raw, device=device, dtype=torch.float32)
        if n_hits.dim() == 0:
            n_hits = n_hits.view(1, 1)
        elif n_hits.dim() == 1:
            n_hits = n_hits.view(-1, 1)
        else:
            n_hits = n_hits.view(n_hits.shape[0], -1)[:, :1]
        if n_hits.shape[0] == 1 and batch_size > 1:
            n_hits = n_hits.expand(batch_size, -1)
        if n_hits.shape[0] != batch_size:
            raise RuntimeError("n_hits_cond batch dimension mismatch")
        return n_hits

    def _resolve_incident_energy_cond(
        self,
        batch_size: int,
        device: torch.device,
        sampling_config: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        raw = sampling_config.get("incident_energy_cond")
        if raw is None:
            return None
        if torch.is_tensor(raw):
            incident = raw.to(device=device, dtype=torch.float32)
        else:
            incident = torch.as_tensor(raw, device=device, dtype=torch.float32)
        if incident.dim() == 0:
            incident = incident.view(1, 1)
        elif incident.dim() == 1:
            incident = incident.view(-1, 1)
        else:
            incident = incident.view(incident.shape[0], -1)[:, :1]
        if incident.shape[0] == 1 and batch_size > 1:
            incident = incident.expand(batch_size, -1)
        if incident.shape[0] != batch_size:
            raise RuntimeError("incident_energy_cond batch dimension mismatch")
        return incident

    def _length_stop_mask(
        self, state: Dict[str, Any], loop_t: int, sampling_config: Dict[str, Any]
    ) -> Optional[torch.Tensor]:
        if not bool(sampling_config.get("use_n_hits_stop", False)):
            return None
        target = state.get("target_length")
        if target is None:
            return None
        attn_mask = state.get("attention_mask")
        device = attn_mask.device if isinstance(attn_mask, torch.Tensor) else target.device
        limit = (target.to(device=device, dtype=torch.long) - 1).clamp_min(0)
        step_tensor = torch.full_like(limit, int(loop_t))
        return limit.le(step_tensor)

    def cache_generation_features(self, batch: dict) -> None:
        """Pre-compute constant embedding features for the generation loop.

        Incident energy and n_hits projections are identical at every decode
        step. Computing them once and caching the result eliminates redundant
        Linear forward passes (~1700× for a typical sequence).
        """
        ref = next(self.parameters())
        bsz = batch.get("energy", ref).shape[0] if "energy" in batch else 1
        if self.incident_energy_feature_dim > 0:
            incident = self._fetch_incident_energy(batch, bsz, ref)
            projected = incident
            if self.incident_energy_feature_proj is not None:
                projected = self.incident_energy_feature_proj(projected)
            if self.incident_energy_dropout is not None:
                projected = self.incident_energy_dropout(projected)
            # [B, D] → [B, 1, D] — will be .expand()-ed to seq_len in the feature builder
            self._gen_cached_incident_features = projected.unsqueeze(1)
        if self.n_hits_feature_dim > 0:
            n_hits = self._fetch_n_hits_for_features(batch, bsz, ref)
            projected = n_hits
            if self.n_hits_feature_proj is not None:
                projected = self.n_hits_feature_proj(projected)
            if self.n_hits_dropout is not None:
                projected = self.n_hits_dropout(projected)
            self._gen_cached_n_hits_features = projected.unsqueeze(1)

    def clear_generation_cache(self) -> None:
        """Release cached generation features."""
        self._gen_cached_incident_features = None
        self._gen_cached_n_hits_features = None

    def preallocate_kv_cache(
        self,
        batch_size: int,
        max_len: int,
        device: torch.device,
        dtype: torch.dtype,
        cuda_graph: bool = False,
    ) -> List[AttentionCache]:
        """Delegate to the underlying GPTDecoderStack to pre-allocate KV buffers."""
        return self.core.decoder.preallocate_kv_cache(
            batch_size,
            max_len,
            device,
            dtype,
            cuda_graph=cuda_graph,
        )

    def init_generation_state(
        self,
        batch_size: int,
        max_len: int,
        pad_id: int,
        start_token_id: int,
        device: torch.device,
        sampling_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Initialize generation buffers and state.

        Args:
            batch_size: Number of sequences to generate
            max_len: Maximum length of sequences
            pad_id: ID for padding
            start_token_id: ID for start token
            device: Device to create tensors on
            sampling_config: Configuration dict containing generation settings.

        Returns:
            Dict containing initialized state (tokens, masks, energies, etc.)
        """
        raise NotImplementedError

    def compute_step_embedding(
        self,
        state: Dict[str, Any],
        t: int,
        incident_energy: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute a single-token embedding for generation position t.

        Used by the CUDA graph path to build the embedding outside the graph
        boundary. Returns a tensor of shape [B, 1, D].
        """
        raise NotImplementedError

    def decode_step(
        self,
        state: Dict[str, Any],
        loop_t: int,
        past_key_values: Optional[Any],
        use_kv_cache: bool,
        incident_energy: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        """Execute one forward pass for generation step t.

        Args:
            state: Current generation state dict
            loop_t: Current time step index
            past_key_values: KV cache from previous steps
            use_kv_cache: Whether to use KV caching

        Returns:
            Model output dict containing logits and other predictions
        """
        raise NotImplementedError

    def update_generation_state(
        self,
        state: Dict[str, Any],
        loop_t: int,
        new_tokens: Dict[str, torch.Tensor],
        new_energy: Optional[torch.Tensor],
        new_energy_token: Optional[torch.Tensor],
        active_mask: torch.Tensor,
    ) -> None:
        """Update generation buffers in-place.

        Args:
            state: Generation state dict to update
            loop_t: Current time step
            new_tokens: Dictionary of sampled tokens for this step
            new_energy: Sampled energy value
            new_energy_token: Sampled energy token ID
            active_mask: Boolean mask of sequences still generating
        """
        raise NotImplementedError

    def check_stopping_condition(
        self,
        state: Dict[str, Any],
        new_tokens: Dict[str, torch.Tensor],
        loop_t: int,
        sampling_config: Dict[str, Any],
    ) -> torch.Tensor:
        """Check which sequences should stop.

        Returns:
            Boolean tensor indicating which items in batch met stop condition
        """
        raise NotImplementedError

    def _build_energy_token_embeddings(
        self, reference: torch.Tensor, batch: Optional[dict]
    ) -> Optional[torch.Tensor]:
        if self.energy_embedding is None or not self.use_energy_tokenization:
            return None
        if batch is None:
            raise RuntimeError("use_energy_tokenization requires a batch with energy_token")
        energy_token = batch.get("energy_token")
        if energy_token is None:
            raise RuntimeError("Missing energy_token for energy tokenization")
        energy_token = energy_token.to(device=reference.device, dtype=torch.long)
        emb = self.energy_embedding(energy_token)
        if emb.shape[:2] != reference.shape[:2]:
            raise RuntimeError("energy token embeddings must match sequence shape")
        return emb

    def build_energy_feature_chunks(
        self,
        base_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor],
        batch: Optional[dict],
    ) -> List[torch.Tensor]:
        chunks: List[torch.Tensor] = [base_embeddings]
        energy_tokens = self._build_energy_token_embeddings(base_embeddings, batch)
        if energy_tokens is not None:
            chunks.append(energy_tokens)
        hit_features = self._build_hit_energy_features(base_embeddings, mask, batch)
        if hit_features is not None:
            chunks.append(hit_features)
        incident_features = self._build_incident_energy_features(base_embeddings, batch)
        if incident_features is not None:
            chunks.append(incident_features)
        n_hits_features = self._build_n_hits_features(base_embeddings, batch)
        if n_hits_features is not None:
            chunks.append(n_hits_features)
        return chunks

    def _build_hit_energy_features(
        self,
        base_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor],
        batch: Optional[dict],
    ) -> Optional[torch.Tensor]:
        if self.hit_energy_feature_dim <= 0:
            return None
        bsz, seq_len = base_embeddings.shape[:2]
        values = self._fetch_hit_energy_values(batch, bsz, seq_len, base_embeddings)
        features = values.unsqueeze(-1)

        if self.hit_energy_feature_proj is not None:
            features = self.hit_energy_feature_proj(features)
        if self.hit_energy_dropout is not None:
            features = self.hit_energy_dropout(features)
        return features

    def _build_incident_energy_features(
        self,
        base_embeddings: torch.Tensor,
        batch: Optional[dict],
    ) -> Optional[torch.Tensor]:
        if self.incident_energy_feature_dim <= 0:
            return None
        bsz, seq_len = base_embeddings.shape[:2]
        # Use cached projection when available (generation loop).
        if self._gen_cached_incident_features is not None:
            return self._gen_cached_incident_features.expand(-1, seq_len, -1)
        incident = self._fetch_incident_energy(batch, bsz, base_embeddings)
        projected = incident
        if self.incident_energy_feature_proj is not None:
            projected = self.incident_energy_feature_proj(projected)

        if self.incident_energy_dropout is not None:
            projected = self.incident_energy_dropout(projected)

        expanded = projected.unsqueeze(1).expand(-1, seq_len, -1)
        features = expanded
        return features

    def _build_n_hits_features(
        self,
        base_embeddings: torch.Tensor,
        batch: Optional[dict],
    ) -> Optional[torch.Tensor]:
        if self.n_hits_feature_dim <= 0:
            return None
        bsz, seq_len = base_embeddings.shape[:2]
        # Use cached projection when available (generation loop).
        if self._gen_cached_n_hits_features is not None:
            return self._gen_cached_n_hits_features.expand(-1, seq_len, -1)
        n_hits = self._fetch_n_hits_for_features(batch, bsz, base_embeddings)
        projected = n_hits
        if self.n_hits_feature_proj is not None:
            projected = self.n_hits_feature_proj(projected)
        expanded = projected.unsqueeze(1).expand(-1, seq_len, -1)
        features = expanded

        if self.n_hits_dropout is not None:
            features = self.n_hits_dropout(features)
        return features

    def _fetch_n_hits_for_features(
        self,
        batch: Optional[dict],
        batch_size: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(batch, dict):
            if self.require_n_hits_feature:
                raise RuntimeError("Missing n_hits_cond for n_hits features")
            return reference.new_zeros((batch_size, 1))
        value = None
        if "n_hits_cond" in batch and batch["n_hits_cond"] is not None:
            value = batch["n_hits_cond"]
        if value is None:
            if self.require_n_hits_feature:
                raise RuntimeError("Missing n_hits_cond for n_hits features")
            return reference.new_zeros((batch_size, 1))
        n_hits = value.to(reference.device).to(reference.dtype)
        if n_hits.dim() == 0:
            n_hits = n_hits.view(1, 1)
        elif n_hits.dim() == 1:
            n_hits = n_hits.view(-1, 1)
        else:
            n_hits = n_hits.view(n_hits.shape[0], -1)[:, :1]
        if n_hits.shape[0] == 1 and batch_size > 1:
            n_hits = n_hits.expand(batch_size, -1)
        if n_hits.shape[0] != batch_size:
            raise RuntimeError("n_hits batch dimension mismatch")
        return n_hits

    def _fetch_hit_energy_values(
        self,
        batch: Optional[dict],
        batch_size: int,
        seq_len: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Fetches and processes (apply noise) hit energy values from the batch."""

        energy = batch["energy"]  # type: ignore[index]
        if energy is None:
            if self.require_hit_energy:
                raise RuntimeError("Missing energy tensor for hit energy features")

        if energy.dim() == 3 and energy.size(-1) >= 1:
            energy = energy[..., 0]
        elif energy.dim() != 2:
            raise RuntimeError("energy tensor must be rank 2 or 3")
        energy = energy.to(reference.device).to(reference.dtype)
        if self.training and self.energy_noise_rel_std > 0.0:
            sigma = self.energy_noise_rel_std
            energy = energy + torch.randn_like(energy) * sigma
            if self.energy_sample_clamp is not None:
                energy = energy.clamp(
                    min=self.energy_sample_clamp[0], max=self.energy_sample_clamp[1]
                )
            energy = torch.nan_to_num(
                energy,
                nan=self.energy_nan_replacement.get("nan", 0.0),
                posinf=self.energy_nan_replacement.get("posinf", 0.0),
                neginf=self.energy_nan_replacement.get("neginf", 0.0),
            )
        if energy.shape[0] != batch_size:
            raise RuntimeError("energy batch dimension mismatch")
        if energy.shape[1] != seq_len:
            raise RuntimeError("energy sequence length mismatch")
        return energy

    def _fetch_incident_energy(
        self,
        batch: Optional[dict],
        batch_size: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        value = None
        if isinstance(batch, dict):
            if "incident_energy_cond" in batch and batch["incident_energy_cond"] is not None:
                value = batch["incident_energy_cond"]
        if value is None:
            if self.require_incident_energy:
                raise RuntimeError("Missing incident_energy_cond for incident features")
            return reference.new_zeros((batch_size, 1))
        incident = value.to(reference.device).to(reference.dtype)
        if incident.dim() == 1:
            incident = incident.view(-1, 1)
        elif incident.dim() == 2 and incident.size(1) != 1:
            incident = incident[:, :1]
        elif incident.dim() > 2:
            incident = incident.reshape(incident.shape[0], -1)[:, :1]
        if incident.shape[0] != batch_size:
            raise RuntimeError("incident energy batch dimension mismatch")
        return incident

    def get_energy_axis_embeddings(
        self, batch: Optional[dict], hidden: torch.Tensor
    ) -> Optional[torch.Tensor]:
        return None

    def project_energy(
        self,
        hidden: torch.Tensor,
        batch: Optional[dict] = None,
        token_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if getattr(self, "energy_head_type", "none") == "none":
            return {}
        energy_hidden = hidden.detach() if self._detach_energy_projection else hidden
        axis_embeddings = self.get_energy_axis_embeddings(batch, hidden)
        if axis_embeddings is not None:
            if self.energy_input_proj is None:
                raise RuntimeError(
                    "energy_input_proj must be defined when axis embeddings are used"
                )
            if axis_embeddings.shape[:2] != energy_hidden.shape[:2]:
                raise RuntimeError("Axis embeddings must match hidden state shape")
            energy_hidden = self.energy_input_proj(
                torch.cat([energy_hidden, axis_embeddings], dim=-1)
            )
        if self.energy_head_backend == "linear":
            assert self.energy_head is not None
            return {"raw_energy": self.energy_head(energy_hidden), "energy_head_type": "linear"}
        if self.energy_head_backend == "mog":
            assert self.mog_head is not None
            mog = self.mog_head(energy_hidden)
            expected = mixture_expected_value(mog["logits"], mog["means"])
            return {
                "raw_energy": expected,
                "mog_logits": mog["logits"],
                "mog_means": mog["means"],
                "mog_log_scales": mog["log_scales"],
                "energy_head_type": "mog",
            }
        if self.energy_head_backend == "classification":
            assert self.energy_classification_head is not None
            logits = self.energy_classification_head(energy_hidden)
            return {
                "energy_logits": logits,
                "energy_head_type": "classification",
            }
        return {}


@dataclass(frozen=True)
class GenerationSchedule:
    axis_shifts: Dict[str, int]
    pad_id: int
    missing_ids: Dict[str, int]
    axis_names: Tuple[str, ...]
    energy_is_tokenized: bool


class ARCombinedAxis(AutoregressiveBackbone):
    # Standard Hierarchical Schedule for Combined Axis
    SCHEDULE_SHIFTS = {"token_ids": 1, "energy": 2}

    def __init__(
        self,
        *,
        transformer: TransformerConfig,
        token_vocab_size: int,
        token_pad_id: int = 0,
        energy_vocab_size: Optional[int] = None,
        use_energy_tokenization: bool = False,
        energy_binning_mode: str = "uniform",
        energy_bin_centers: Optional[Any] = None,
        energy_bin_edges: Optional[Any] = None,
        hierarchical_conditioning: bool = False,
        n_registers: int = 0,
        register_condition_incident_energy: bool = True,
        zero_init_register_incident: bool = True,
        use_n_hits_register: bool = False,
        hit_energy_dropout_p: float = 0.0,
        incident_energy_dropout_p: float = 0.0,
        use_hit_energy_feature: bool = False,
        hit_energy_feature_dim: int = 16,
        use_incident_energy_feature: bool = False,
        incident_energy_feature_dim: int = 32,
        use_n_hits_feature: bool = False,
        n_hits_feature_dim: int = 32,
        n_hits_dropout_p: float = 0.0,
        zero_init_n_hits_feature: bool = True,
        use_mog_energy: bool = True,
        mog_components: int = 5,
        energy_sample_min: float | None = -5.0,
        energy_sample_max: float | None = 5.0,
        energy_nan_replacement: Optional[Dict[str, float]] = None,
        energy_noise_rel_std: float = 0.0,
    ) -> None:
        super().__init__(
            transformer=transformer,
            n_registers=n_registers,
            register_condition_incident_energy=register_condition_incident_energy,
            zero_init_register_incident=zero_init_register_incident,
            use_n_hits_register=use_n_hits_register,
            use_energy_tokenization=use_energy_tokenization,
            energy_vocab_size=energy_vocab_size,
            energy_binning_mode=energy_binning_mode,
            energy_bin_centers=energy_bin_centers,
            energy_bin_edges=energy_bin_edges,
            hit_energy_dropout_p=hit_energy_dropout_p,
            incident_energy_dropout_p=incident_energy_dropout_p,
            use_hit_energy_feature=use_hit_energy_feature,
            hit_energy_feature_dim=hit_energy_feature_dim,
            use_incident_energy_feature=use_incident_energy_feature,
            incident_energy_feature_dim=incident_energy_feature_dim,
            use_n_hits_feature=use_n_hits_feature,
            n_hits_feature_dim=n_hits_feature_dim,
            n_hits_dropout_p=n_hits_dropout_p,
            zero_init_n_hits_feature=zero_init_n_hits_feature,
            use_mog_energy=use_mog_energy,
            mog_components=mog_components,
            energy_sample_min=energy_sample_min,
            energy_sample_max=energy_sample_max,
            energy_nan_replacement=energy_nan_replacement,
            energy_noise_rel_std=energy_noise_rel_std,
        )
        self.hierarchical_conditioning = hierarchical_conditioning
        self.energy_vocab_size = energy_vocab_size
        self.token_pad_id = int(token_pad_id)
        self.token_embedding = nn.Embedding(
            int(token_vocab_size), self.dim, padding_idx=self.token_pad_id
        )

        feature_in_dim = self.dim + self.additional_dim
        self.token_feature_proj = (
            nn.Linear(feature_in_dim, self.dim) if self.additional_dim > 0 else None
        )
        self.token_head = TokenHead(self.dim, int(token_vocab_size))
        self.enable_multi_axis_heads = False

    def build_embeddings_and_mask(
        self,
        batch: Optional[dict],
        token_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tok = token_ids
        mask = attention_mask
        if tok is None and batch is not None:
            tok = batch["token_ids"]
            mask = batch.get("attention_mask")
        if tok is None:
            raise RuntimeError("Missing token_ids for ARSharedTokens")
        tok = tok.long()
        if mask is None:
            mask = tok.ne(self.token_pad_id)
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)
        emb = self.token_embedding(tok)
        feature_chunks = self.build_energy_feature_chunks(emb, mask, batch)
        merged_features = torch.cat(feature_chunks, dim=-1)
        if self.token_feature_proj is not None:
            emb = self.token_feature_proj(merged_features)
        else:
            emb = merged_features
        return emb, mask

    def project_tokens(self, hidden: torch.Tensor, batch: Optional[dict] = None) -> Dict[str, Any]:
        out = {"token_logits": self.token_head(hidden)}
        return out

    def get_targets(
        self, batch: Optional[dict], token_ids: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        tok = token_ids
        if tok is None and batch is not None:
            if "token_ids" in batch:
                tok = batch["token_ids"]
            elif "interleaved_token" in batch:
                tok = batch["interleaved_token"]
        if tok is None:
            return {}
        return {"targets": tok.long()}

    def _build_schedule(self, pad_id: int) -> Optional[GenerationSchedule]:
        # Determine missing IDs
        # For combined token_ids, assumes missing is last token (vocab - 1)
        token_vocab = self.token_embedding.num_embeddings
        missing_id_token = int(token_vocab - 1)

        bs_missing = {"token_ids_token": missing_id_token}

        if self.use_energy_tokenization:
            e_vocab = self.energy_vocab_size or 1000
            bs_missing["energy_token"] = int(e_vocab - 1)

        shifts = dict(self.SCHEDULE_SHIFTS)

        return GenerationSchedule(
            axis_shifts=shifts,
            pad_id=pad_id,
            missing_ids=bs_missing,
            axis_names=("token_ids",),
            energy_is_tokenized=self.use_energy_tokenization,
        )

    def init_generation_state(
        self,
        batch_size: int,
        max_len: int,
        pad_id: int,
        start_token_id: int,
        device: torch.device,
        sampling_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Initialize generation state for single-stream generation."""
        schedule = self._build_schedule(pad_id)

        tokens = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
        attn_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
        energy = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)

        if self.use_energy_tokenization:
            energy_token = torch.full(
                (batch_size, max_len), pad_id, dtype=torch.long, device=device
            )
        else:
            energy_token = None

        if not schedule:
            # Set start token (Standard Mode)
            tokens[:, 0] = start_token_id
            attn_mask[:, 0] = True
        else:
            # Hierarchical Init via Schedule
            # 1. Token Prefix
            k = schedule.axis_shifts.get("token_ids", 0)
            miss = schedule.missing_ids["token_ids_token"]
            tokens[:, :k] = miss

            # 2. Energy Prefix
            if energy_token is not None:
                k_e = schedule.axis_shifts.get("energy", 0)
                miss_e = schedule.missing_ids.get("energy_token", pad_id)
                energy_token[:, :k_e] = miss_e

            # 3. Mask: Visibility: ONLY t=0 is visible initially
            attn_mask[:, 0] = True

        state = {
            "token_ids": tokens,
            "attention_mask": attn_mask,
            "energy": energy,
            "schedule": schedule,
        }

        if energy_token is not None:
            state["energy_token"] = energy_token

        target_hits = self._resolve_target_n_hits(batch_size, max_len, device, sampling_config)
        if target_hits is not None:
            state["target_length"] = target_hits
            state["n_hits"] = target_hits
        n_hits_cond = self._resolve_n_hits_cond(batch_size, device, sampling_config)
        if n_hits_cond is not None:
            state["n_hits_cond"] = n_hits_cond
        incident_energy_cond = self._resolve_incident_energy_cond(
            batch_size, device, sampling_config
        )
        if incident_energy_cond is not None:
            state["incident_energy_cond"] = incident_energy_cond

        return state

    def _build_decode_batch(
        self,
        state: Dict[str, Any],
        incident_energy: Optional[torch.Tensor],
    ) -> dict:
        """Build the batch dict used by decode_step and compute_step_embedding."""
        batch: dict = {"energy": state.get("energy")}
        if "energy_token" in state:
            batch["energy_token"] = state["energy_token"]
        if "target_length" in state:
            batch["n_hits"] = state["target_length"]
        if "n_hits_cond" in state:
            batch["n_hits_cond"] = state["n_hits_cond"]
        if "incident_energy_cond" in state:
            batch["incident_energy_cond"] = state["incident_energy_cond"]
        if incident_energy is not None and "incident_energy_cond" not in batch:
            batch["incident_energy_cond"] = incident_energy
        return batch

    def compute_step_embedding(
        self,
        state: Dict[str, Any],
        t: int,
        incident_energy: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute single-token embedding for position t (CUDA graph path)."""
        batch = self._build_decode_batch(state, incident_energy)
        # Slice positional fields to a single position so build_energy_feature_chunks
        # sees matching shapes (seq_len=1).
        if batch.get("energy") is not None:
            energy = batch["energy"]
            if energy.dim() >= 2 and energy.size(1) > 1:
                batch["energy"] = (
                    energy[:, t : t + 1] if energy.dim() == 2 else energy[:, t : t + 1, :]
                )
        if "energy_token" in batch and batch["energy_token"].dim() >= 2:
            batch["energy_token"] = batch["energy_token"][:, t : t + 1]

        tok = state["token_ids"][:, t : t + 1].long()
        emb = self.token_embedding(tok)
        feature_chunks = self.build_energy_feature_chunks(emb, None, batch)
        merged = torch.cat(feature_chunks, dim=-1)
        if self.token_feature_proj is not None:
            emb = self.token_feature_proj(merged)
        else:
            emb = merged
        return emb

    def decode_step(
        self,
        state: Dict[str, Any],
        loop_t: int,
        past_key_values: Optional[Any],
        use_kv_cache: bool,
        incident_energy: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        """Execute forward pass for ARSharedTokens."""
        batch = self._build_decode_batch(state, incident_energy)

        return self.forward(
            token_ids=state["token_ids"],
            attention_mask=state["attention_mask"],
            shower_energy=incident_energy,
            mode="gen",
            past_key_values=past_key_values,
            use_kv_cache=use_kv_cache,
            batch=batch,
        )

    def update_generation_state(
        self,
        state: Dict[str, Any],
        loop_t: int,
        new_tokens: Dict[str, torch.Tensor],
        new_energy: Optional[torch.Tensor],
        new_energy_token: Optional[torch.Tensor],
        active_mask: torch.Tensor,
    ) -> None:
        """Update buffers for single stream."""
        schedule: Optional[GenerationSchedule] = state.get("schedule")

        next_token = new_tokens.get("token_ids")

        # Enforce Schedule for Tokens
        if schedule:
            shift = schedule.axis_shifts.get("token_ids", 0)
            if loop_t < shift:
                missing_id = schedule.missing_ids["token_ids_token"]
                next_token = torch.full_like(next_token, missing_id)

        # Update Tokens
        token_ids = state["token_ids"]
        # In-place update masked
        token_ids[:, loop_t] = torch.where(active_mask, next_token, token_ids[:, loop_t])
        state["attention_mask"][:, loop_t] = state["attention_mask"][:, loop_t] | active_mask

        # Update Energy
        energy_shift = 0
        if schedule:
            energy_shift = schedule.axis_shifts.get("energy", 0)

        if new_energy is not None:
            # Flatten energy to (B,) if coming as (B, 1)
            e_val = new_energy.squeeze(-1) if new_energy.dim() > 1 else new_energy

            if schedule and loop_t < energy_shift:
                e_val = torch.zeros_like(e_val)

            # Ensure it matches state["energy"] dimensionality
            state["energy"][:, loop_t] = torch.where(
                active_mask, e_val, state["energy"][:, loop_t]
            )

        # Update Energy Token
        if "energy_token" in state and new_energy_token is not None:
            e_tok_val = new_energy_token
            if schedule and loop_t < energy_shift:
                miss_e = schedule.missing_ids.get("energy_token", 0)
                e_tok_val = torch.full_like(e_tok_val, miss_e)

            state["energy_token"][:, loop_t] = torch.where(
                active_mask, e_tok_val, state["energy_token"][:, loop_t]
            )

    def check_stopping_condition(
        self,
        state: Dict[str, Any],
        new_tokens: Dict[str, torch.Tensor],
        loop_t: int,
        sampling_config: Dict[str, Any],
    ) -> torch.Tensor:
        """Single-stream stop check."""
        length_stop = self._length_stop_mask(state, loop_t, sampling_config)
        if length_stop is not None:
            return length_stop
        attn = state.get("attention_mask")
        batch_size = int(attn.shape[0]) if torch.is_tensor(attn) else 1
        return torch.zeros(
            batch_size, dtype=torch.bool, device=attn.device if torch.is_tensor(attn) else None
        )


class ARSplitAxis(AutoregressiveBackbone):
    def __init__(
        self,
        *,
        transformer: TransformerConfig,
        per_axis_vocab_sizes: Dict[str, int],
        per_axis_embedding_dims: Optional[Dict[str, int]] = None,
        per_axis_embedding_dropout: float = 0.0,
        hierarchical_conditioning: bool = False,
        hierarchical_axis_order: Optional[List[str]] = None,
        n_registers: int = 0,
        register_condition_incident_energy: bool = True,
        zero_init_register_incident: bool = True,
        use_n_hits_register: bool = False,
        use_energy_tokenization: bool = False,
        energy_vocab_size: Optional[int] = None,
        energy_binning_mode: str = "uniform",
        energy_bin_centers: Optional[Any] = None,
        energy_bin_edges: Optional[Any] = None,
        hit_energy_dropout_p: float = 0.0,
        incident_energy_dropout_p: float = 0.0,
        use_hit_energy_feature: bool = False,
        hit_energy_feature_dim: int = 16,
        use_incident_energy_feature: bool = False,
        incident_energy_feature_dim: int = 32,
        use_n_hits_feature: bool = False,
        n_hits_feature_dim: int = 32,
        n_hits_dropout_p: float = 0.0,
        zero_init_n_hits_feature: bool = True,
        use_mog_energy: bool = True,
        mog_components: int = 5,
        energy_sample_min: float | None = -5.0,
        energy_sample_max: float | None = 5.0,
        energy_nan_replacement: Optional[Dict[str, float]] = None,
        energy_noise_rel_std: float = 0.0,
        use_spatial_axis_sinusoid: bool = False,
        spatial_axis_sinusoid_scale: float = 1.0,
        spatial_axis_sinusoid_axes: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            transformer=transformer,
            n_registers=n_registers,
            register_condition_incident_energy=register_condition_incident_energy,
            zero_init_register_incident=zero_init_register_incident,
            use_n_hits_register=use_n_hits_register,
            use_energy_tokenization=use_energy_tokenization,
            energy_vocab_size=energy_vocab_size,
            energy_binning_mode=energy_binning_mode,
            energy_bin_centers=energy_bin_centers,
            energy_bin_edges=energy_bin_edges,
            hit_energy_dropout_p=hit_energy_dropout_p,
            incident_energy_dropout_p=incident_energy_dropout_p,
            use_hit_energy_feature=use_hit_energy_feature,
            hit_energy_feature_dim=hit_energy_feature_dim,
            use_incident_energy_feature=use_incident_energy_feature,
            incident_energy_feature_dim=incident_energy_feature_dim,
            use_n_hits_feature=use_n_hits_feature,
            n_hits_feature_dim=n_hits_feature_dim,
            n_hits_dropout_p=n_hits_dropout_p,
            zero_init_n_hits_feature=zero_init_n_hits_feature,
            use_mog_energy=use_mog_energy,
            mog_components=mog_components,
            energy_sample_min=energy_sample_min,
            energy_sample_max=energy_sample_max,
            energy_nan_replacement=energy_nan_replacement,
            energy_noise_rel_std=energy_noise_rel_std,
        )
        self.energy_vocab_size = energy_vocab_size
        self.axis_names = tuple(per_axis_vocab_sizes.keys())
        self.use_spatial_axis_sinusoid = bool(use_spatial_axis_sinusoid)
        self.spatial_axis_sinusoid_scale = float(spatial_axis_sinusoid_scale)
        requested_spatial_axes = (
            ["x", "y", "z"] if spatial_axis_sinusoid_axes is None else spatial_axis_sinusoid_axes
        )
        self.spatial_axis_sinusoid_axes = tuple(
            axis for axis in requested_spatial_axes if axis in self.axis_names
        )
        resolved_dims: Dict[str, int] = {}
        if per_axis_embedding_dims is None:
            resolved_dims = {axis: 64 for axis in self.axis_names}
        else:
            for axis in self.axis_names:
                resolved_dims[axis] = int(per_axis_embedding_dims.get(axis, 64))
        self.per_axis_embed = PerAxisConcatEmbedding(
            vocab_sizes=per_axis_vocab_sizes,
            embedding_dims=resolved_dims,
            dropout=float(per_axis_embedding_dropout),
        )
        self.axis_embedding_dim = self.per_axis_embed.output_dim
        concat_in_dim = self.axis_embedding_dim + self.additional_dim
        self.axis_concat_proj = nn.Linear(concat_in_dim, self.dim)
        self.use_hierarchical_conditioning = bool(hierarchical_conditioning)
        if hierarchical_axis_order is None:
            preferred = [axis for axis in ("z", "x", "y") if axis in self.axis_names]
            remaining = [axis for axis in self.axis_names if axis not in preferred]
            resolved_order = preferred + remaining
        else:
            resolved_order = [str(axis) for axis in hierarchical_axis_order]
        if set(resolved_order) != set(self.axis_names):
            extra = [axis for axis in resolved_order if axis not in self.axis_names]
            missing = [axis for axis in self.axis_names if axis not in resolved_order]
            if extra:
                raise ValueError(f"hierarchical_axis_order contains unknown axes: {extra}")
            raise ValueError(f"hierarchical_axis_order is missing axes: {missing}")
        if len(resolved_order) != len(self.axis_names):
            raise ValueError("hierarchical_axis_order must include each axis exactly once")
        self.hierarchical_axis_order = tuple(resolved_order)
        self.hierarchical_axis_projections = nn.ModuleDict()
        total_axis_dim = sum(
            self.per_axis_embed.embedding_dims[axis] for axis in self.hierarchical_axis_order
        )
        if self.use_hierarchical_conditioning:
            accumulated: List[str] = []
            for axis in self.hierarchical_axis_order:
                if accumulated:
                    concat_dim = self.dim + sum(
                        self.per_axis_embed.embedding_dims[name] for name in accumulated
                    )
                    self.hierarchical_axis_projections[axis] = nn.Linear(concat_dim, self.dim)
                accumulated.append(axis)
            self.energy_input_proj = nn.Linear(self.dim + total_axis_dim, self.dim)
        else:
            self.energy_input_proj = None
        self._cached_axis_embeddings: Optional[Dict[str, torch.Tensor]] = None

        self.axis_token_heads = nn.ModuleDict(
            {
                axis: TokenHead(self.dim, int(per_axis_vocab_sizes[axis]))
                for axis in self.axis_names
            }
        )

        self.enable_multi_axis_heads = True

    def _build_axis_value_sinusoid(
        self,
        axis_name: str,
        axis_tokens: torch.Tensor,
        embedding_dim: int,
        embedding_dtype: torch.dtype,
    ) -> torch.Tensor:
        vocab_size = int(self.per_axis_embed.vocab_sizes[axis_name])
        pad_token = 0
        missing_token = vocab_size - 1
        valid_bin_mask = axis_tokens.ne(pad_token) & axis_tokens.ne(missing_token)
        if not torch.any(valid_bin_mask):
            return torch.zeros(
                (*axis_tokens.shape, embedding_dim),
                device=axis_tokens.device,
                dtype=embedding_dtype,
            )

        coordinate_index = (axis_tokens - 1).clamp_min(0).to(dtype=embedding_dtype)
        half_dim = (embedding_dim + 1) // 2
        frequency_index = torch.arange(
            half_dim,
            device=axis_tokens.device,
            dtype=embedding_dtype,
        )
        inverse_frequency = torch.exp(
            -torch.log(torch.tensor(10000.0, device=axis_tokens.device, dtype=embedding_dtype))
            * (2.0 * frequency_index / float(embedding_dim))
        )
        sinusoid_phase = coordinate_index.unsqueeze(-1) * inverse_frequency.view(1, 1, -1)
        sinusoid_embedding = torch.zeros(
            (*axis_tokens.shape, embedding_dim),
            device=axis_tokens.device,
            dtype=embedding_dtype,
        )
        sinusoid_embedding[..., 0::2] = torch.sin(
            sinusoid_phase[..., : sinusoid_embedding[..., 0::2].shape[-1]]
        )
        sinusoid_embedding[..., 1::2] = torch.cos(
            sinusoid_phase[..., : sinusoid_embedding[..., 1::2].shape[-1]]
        )
        sinusoid_embedding = sinusoid_embedding * valid_bin_mask.unsqueeze(-1).to(embedding_dtype)
        return sinusoid_embedding * self.spatial_axis_sinusoid_scale

    def build_embeddings_and_mask(
        self,
        batch: Optional[dict],
        token_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if batch is None:
            raise RuntimeError("ARPerAxis requires a batch with per-axis tokens")
        axis_tokens: Dict[str, torch.Tensor] = {}
        for axis in self.axis_names:
            key = f"{axis}_token"
            if key not in batch:
                raise RuntimeError(f"Missing {key} for ARPerAxis")
            axis_tokens[axis] = batch[key].long()
        sample_axis = self.axis_names[0]
        token_example = axis_tokens[sample_axis]
        mask = attention_mask
        if mask is not None and mask.shape[-1] != token_example.shape[-1]:
            hit_mask = batch.get("hit_mask")
            if (
                isinstance(hit_mask, torch.Tensor)
                and hit_mask.shape[-1] == token_example.shape[-1]
            ):
                mask = hit_mask
            else:
                mask = None
        if mask is None:
            hit_mask = batch.get("hit_mask")
            if isinstance(hit_mask, torch.Tensor):
                mask = hit_mask
        if mask is None:
            raise RuntimeError("Missing attention mask for ARPerAxis")
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)
        per_axis_embeddings = self.per_axis_embed.embed_axis_dict(axis_tokens)
        if self.use_spatial_axis_sinusoid:
            per_axis_embeddings = {
                axis: (
                    embedding
                    + self._build_axis_value_sinusoid(
                        axis_name=axis,
                        axis_tokens=axis_tokens[axis],
                        embedding_dim=int(embedding.shape[-1]),
                        embedding_dtype=embedding.dtype,
                    )
                    if axis in self.spatial_axis_sinusoid_axes
                    else embedding
                )
                for axis, embedding in per_axis_embeddings.items()
            }

        # Gating removed: Rely on adapter-provided masks and deterministic padding/missing tokens.
        # See Constraint B in instructions.

        if self.use_hierarchical_conditioning:
            self._cached_axis_embeddings = per_axis_embeddings
        else:
            self._cached_axis_embeddings = None
        emb = torch.cat([per_axis_embeddings[name] for name in self.axis_names], dim=-1)
        feature_chunks = self.build_energy_feature_chunks(emb, mask, batch)
        merged_features = (
            feature_chunks[0] if len(feature_chunks) == 1 else torch.cat(feature_chunks, dim=-1)
        )
        emb = self.axis_concat_proj(merged_features)
        return emb, mask

    def project_tokens(self, hidden: torch.Tensor, batch: Optional[dict] = None) -> Dict[str, Any]:
        out = {}
        if not self.use_hierarchical_conditioning:
            out = {
                f"token_logits_{axis}": head(hidden)
                for axis, head in self.axis_token_heads.items()
            }
        else:
            per_axis_embeddings = self._align_cached_axis_embeddings(hidden)
            if per_axis_embeddings is None:
                out = {
                    f"token_logits_{axis}": head(hidden)
                    for axis, head in self.axis_token_heads.items()
                }
            else:
                logits: Dict[str, torch.Tensor] = {}
                conditioned_axes: List[str] = []
                for axis in self.hierarchical_axis_order:
                    axis_hidden = hidden
                    if conditioned_axes and axis in self.hierarchical_axis_projections:
                        concat_embeds = torch.cat(
                            [per_axis_embeddings[name] for name in conditioned_axes], dim=-1
                        )
                        axis_hidden = torch.cat([hidden, concat_embeds], dim=-1)
                        axis_hidden = torch.relu(
                            self.hierarchical_axis_projections[axis](axis_hidden)
                        )
                    logits[axis] = self.axis_token_heads[axis](axis_hidden)
                    conditioned_axes.append(axis)
                out = {
                    f"token_logits_{axis}": logits.get(axis, self.axis_token_heads[axis](hidden))
                    for axis in self.axis_names
                }

        return out

    def get_targets(
        self, batch: Optional[dict], token_ids: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if batch is None:
            return {}
        targets: Dict[str, torch.Tensor] = {}
        for axis in self.axis_names:
            key = f"{axis}_token"
            if key not in batch:
                return {}
            targets[f"targets_{axis}"] = batch[key].long()
        return targets

    def get_energy_axis_embeddings(
        self,
        batch: Optional[dict],
        hidden: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not self.use_hierarchical_conditioning:
            return None
        per_axis_embeddings = self._align_cached_axis_embeddings(hidden)
        if per_axis_embeddings is None:
            return None
        ordered = [per_axis_embeddings[axis] for axis in self.hierarchical_axis_order]
        return torch.cat(ordered, dim=-1)

    def _align_cached_axis_embeddings(
        self, reference: torch.Tensor
    ) -> Optional[Dict[str, torch.Tensor]]:
        if self._cached_axis_embeddings is None:
            return None
        batch_size, seq_len = reference.shape[0], reference.shape[1]
        aligned: Dict[str, torch.Tensor] = {}
        for axis in self.axis_names:
            emb = self._cached_axis_embeddings.get(axis)
            if emb is None:
                return None
            if emb.shape[0] != batch_size:
                if emb.shape[0] == 1 and batch_size > 1:
                    emb = emb.expand(batch_size, -1, -1)
                elif batch_size == 1:
                    emb = emb[:1]
                else:
                    return None
            axis_seq = emb.shape[1]
            if axis_seq != seq_len:
                if axis_seq >= seq_len:
                    emb = emb[:, :seq_len, :]
                else:
                    pad = seq_len - axis_seq
                    emb = torch.nn.functional.pad(emb, (0, 0, 0, pad))
            aligned[axis] = emb
        return aligned

    # Standard Hierarchical Schedule
    # Hardcoded to match adapter's training contract
    SCHEDULE_SHIFTS = {"z": 1, "x": 2, "y": 3, "energy": 4}

    def _build_schedule(self, pad_id: int) -> GenerationSchedule:
        axis_names = list(self.axis_names) if self.axis_names else ["x", "y", "z"]

        # Missing ID assumed to be vocab - 1 (adapter contract)
        bs_missing = {}
        for ax in axis_names:
            vocab = self.per_axis_embed.vocab_sizes[ax]
            bs_missing[f"{ax}_token"] = int(vocab - 1)

        # Energy Missing ID
        if self.use_energy_tokenization:
            e_vocab = getattr(self, "energy_vocab_size", 1000) or 1000
            bs_missing["energy_token"] = int(e_vocab - 1)

        shifts = {}
        for k in axis_names:
            if k not in self.SCHEDULE_SHIFTS:
                raise ValueError(f"No hierarchical shift defined for axis={k}")
            shifts[f"{k}_token"] = self.SCHEDULE_SHIFTS[k]

        shifts["energy"] = self.SCHEDULE_SHIFTS["energy"]

        return GenerationSchedule(
            axis_shifts=shifts,
            pad_id=pad_id,
            missing_ids=bs_missing,
            axis_names=tuple(axis_names),
            energy_is_tokenized=self.use_energy_tokenization,
        )

    def init_generation_state(
        self,
        batch_size: int,
        max_len: int,
        pad_id: int,
        start_token_id: int,
        device: torch.device,
        sampling_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Initialize generation state with structural prefix filled but hidden."""
        self.ensure_split_axes(None)

        schedule = self._build_schedule(pad_id)
        if schedule is None:
            raise RuntimeError("ARSplitAxis requires a valid schedule for generation.")
        axis_names = list(schedule.axis_names)

        axis_tokens = {
            axis: torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
            for axis in axis_names
        }
        attn_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
        energy = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)

        energy_token = None
        if schedule.energy_is_tokenized:
            energy_token = torch.full(
                (batch_size, max_len), pad_id, dtype=torch.long, device=device
            )

        # 1) Prefill MISSING prefix exactly like training expects
        for axis in axis_names:
            k = schedule.axis_shifts.get(f"{axis}_token", 0)
            miss = schedule.missing_ids[f"{axis}_token"]
            if k > 0:
                axis_tokens[axis][:, :k] = miss

        if energy_token is not None:
            k_e = schedule.axis_shifts.get("energy", 0)
            miss_e = schedule.missing_ids.get("energy_token", pad_id)
            if k_e > 0:
                energy_token[:, :k_e] = miss_e

        # 2) Visibility: ONLY t=0 is visible initially
        attn_mask[:, 0] = True

        state = {
            "axis_tokens": axis_tokens,
            "attention_mask": attn_mask,
            "energy": energy,
            "axis_names": axis_names,
            "reference_axis": axis_names[-1],
            "schedule": schedule,
        }
        if energy_token is not None:
            state["energy_token"] = energy_token

        for axis, tokens in axis_tokens.items():
            state[f"{axis}_token"] = tokens

        target_hits = self._resolve_target_n_hits(batch_size, max_len, device, sampling_config)
        if target_hits is not None:
            state["target_length"] = target_hits
            state["n_hits"] = target_hits
        n_hits_cond = self._resolve_n_hits_cond(batch_size, device, sampling_config)
        if n_hits_cond is not None:
            state["n_hits_cond"] = n_hits_cond
        incident_energy_cond = self._resolve_incident_energy_cond(
            batch_size, device, sampling_config
        )
        if incident_energy_cond is not None:
            state["incident_energy_cond"] = incident_energy_cond

        return state

    def _build_decode_batch(
        self,
        state: Dict[str, Any],
        incident_energy: Optional[torch.Tensor],
    ) -> dict:
        """Build batch dict for decode_step / compute_step_embedding."""
        batch: dict = {
            "energy": state["energy"],
            "hit_mask": state["attention_mask"],
        }
        for axis in state["axis_names"]:
            batch[f"{axis}_token"] = state[f"{axis}_token"]
        if "energy_token" in state:
            batch["energy_token"] = state["energy_token"]
        if "target_length" in state:
            batch["n_hits"] = state["target_length"]
        if "n_hits_cond" in state:
            batch["n_hits_cond"] = state["n_hits_cond"]
        if "incident_energy_cond" in state:
            batch["incident_energy_cond"] = state["incident_energy_cond"]
        if incident_energy is not None and "incident_energy_cond" not in batch:
            batch["incident_energy_cond"] = incident_energy
        return batch

    def compute_step_embedding(
        self,
        state: Dict[str, Any],
        t: int,
        incident_energy: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute single-token embedding for position t (CUDA graph path)."""
        batch = self._build_decode_batch(state, incident_energy)
        # Slice positional fields to seq_len=1 for shape consistency.
        if batch.get("energy") is not None:
            energy = batch["energy"]
            if energy.dim() >= 2 and energy.size(1) > 1:
                batch["energy"] = (
                    energy[:, t : t + 1] if energy.dim() == 2 else energy[:, t : t + 1, :]
                )
        if "energy_token" in batch and batch["energy_token"].dim() >= 2:
            batch["energy_token"] = batch["energy_token"][:, t : t + 1]

        axis_tokens = {
            axis: state[f"{axis}_token"][:, t : t + 1].long() for axis in self.axis_names
        }
        per_axis_embeddings = self.per_axis_embed.embed_axis_dict(axis_tokens)
        if self.use_spatial_axis_sinusoid:
            per_axis_embeddings = {
                axis: (
                    embedding
                    + self._build_axis_value_sinusoid(
                        axis_name=axis,
                        axis_tokens=axis_tokens[axis],
                        embedding_dim=int(embedding.shape[-1]),
                        embedding_dtype=embedding.dtype,
                    )
                    if axis in self.spatial_axis_sinusoid_axes
                    else embedding
                )
                for axis, embedding in per_axis_embeddings.items()
            }
        if self.use_hierarchical_conditioning:
            self._cached_axis_embeddings = per_axis_embeddings
        else:
            self._cached_axis_embeddings = None
        emb = torch.cat([per_axis_embeddings[name] for name in self.axis_names], dim=-1)
        feature_chunks = self.build_energy_feature_chunks(emb, None, batch)
        merged = (
            feature_chunks[0] if len(feature_chunks) == 1 else torch.cat(feature_chunks, dim=-1)
        )
        emb = self.axis_concat_proj(merged)
        return emb

    def decode_step(
        self,
        state: Dict[str, Any],
        loop_t: int,
        past_key_values: Optional[Any],
        use_kv_cache: bool,
        incident_energy: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        """Forward pass for split-axis."""
        batch = self._build_decode_batch(state, incident_energy)

        return self.forward(
            batch=batch,
            attention_mask=state["attention_mask"],
            shower_energy=incident_energy,
            mode="gen",
            past_key_values=past_key_values,
            use_kv_cache=use_kv_cache,
        )

    def update_generation_state(
        self,
        state: Dict[str, Any],
        loop_t: int,
        new_tokens: Dict[str, torch.Tensor],
        new_energy: Optional[torch.Tensor],
        new_energy_token: Optional[torch.Tensor],
        active_mask: torch.Tensor,
    ) -> None:
        """Update buffers for split-axis."""
        # --- Schedule Enforcement Logic ---
        schedule: Optional[GenerationSchedule] = state.get("schedule")

        axis_names = state["axis_names"]
        axis_tokens = state["axis_tokens"]

        # 1. Update Energy (Continuous)
        # Enforce schedule: Zero out energy before its shift
        energy_shift = schedule.axis_shifts["energy"] if schedule else 4

        if new_energy is not None:
            e_val = new_energy.squeeze(-1) if new_energy.dim() > 1 else new_energy

            # If we are in the structural prefix for energy, force 0.0
            if loop_t < energy_shift:
                e_val = torch.zeros_like(e_val)

            state["energy"][:, loop_t] = torch.where(
                active_mask, e_val, state["energy"][:, loop_t]
            )

        if "energy_token" in state and new_energy_token is not None:
            # If using tokenized energy, enforce missing ID in prefix
            e_token_val = new_energy_token
            if schedule and loop_t < energy_shift:
                miss_e = schedule.missing_ids.get("energy_token", 0)
                e_token_val = torch.full_like(new_energy_token, miss_e)

            state["energy_token"][:, loop_t] = torch.where(
                active_mask, e_token_val, state["energy_token"][:, loop_t]
            )

        # 2. Update Axis Tokens
        for axis in axis_names:
            if axis in new_tokens:
                token_val = new_tokens[axis]

                # Enforce Schedule: if current step is before this axis starts, use MISSING_ID
                if schedule:
                    shift = schedule.axis_shifts.get(f"{axis}_token", 0)
                    if loop_t < shift:
                        missing_id = schedule.missing_ids[f"{axis}_token"]
                        token_val = torch.full_like(token_val, missing_id)

                # Update buffer
                axis_tokens[axis][:, loop_t] = torch.where(
                    active_mask, token_val, axis_tokens[axis][:, loop_t]
                )
                # Keep flat dict in sync for next forward logic if it accesses state directly
                state[f"{axis}_token"] = axis_tokens[axis]

        state["attention_mask"][:, loop_t] = state["attention_mask"][:, loop_t] | active_mask

    def check_stopping_condition(
        self,
        state: Dict[str, Any],
        new_tokens: Dict[str, torch.Tensor],
        loop_t: int,
        sampling_config: Dict[str, Any],
    ) -> torch.Tensor:
        """Vectorized stopping policy driven by stop head predictions."""
        length_stop = self._length_stop_mask(state, loop_t, sampling_config)
        if length_stop is not None:
            return length_stop
        attn = state.get("attention_mask")
        batch_size = int(attn.shape[0]) if torch.is_tensor(attn) else 1
        return torch.zeros(
            batch_size, dtype=torch.bool, device=attn.device if torch.is_tensor(attn) else None
        )
