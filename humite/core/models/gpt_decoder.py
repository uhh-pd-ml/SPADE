from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except Exception:  # pragma: no cover
    flex_attention = None  # type: ignore[assignment]
    create_block_mask = None  # type: ignore[assignment]


@dataclass
class TransformerConfig:
    embedding_dim: int
    num_layers: int
    num_heads: int
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attn_dropout: float = 0.0
    max_seq_len: int = 2048
    bias: bool = True
    use_flash_attention: bool = True  # requires PyTorch 2.1+
    use_rope: bool = True
    use_parallel_residual: bool = False
    use_RMSNorm: bool = False
    use_multi_query_attention: bool = True
    attention_window: Optional[int] = None
    num_registers: int = 0


@dataclass
class AttentionCache:
    key: torch.Tensor
    value: torch.Tensor
    offset: int
    # Optional static pre-allocated buffers. When present, the static path is used
    # (in-place write + slice) instead of torch.cat, eliminating per-step allocations.
    k_buf: Optional[torch.Tensor] = None
    v_buf: Optional[torch.Tensor] = None
    filled: int = 0  # number of tokens already written into k_buf / v_buf
    # GPU scalar tracking filled count for CUDA-graph-compatible path.
    # When present, scatter_ + full-buffer reads are used instead of
    # Python-int slicing, making the forward capturable by torch.cuda.CUDAGraph.
    filled_tensor: Optional[torch.Tensor] = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def apply_rope(q, k, rope_cache):
    # q, k: [B, T, H, D]
    sin, cos = rope_cache
    # Expand sin/cos to full head dimension for broadcasting over even-odd pairs
    sin_full = torch.repeat_interleave(sin, 2, dim=-1)
    cos_full = torch.repeat_interleave(cos, 2, dim=-1)
    q_rot = (q * cos_full) + (rotate_half(q) * sin_full)
    k_rot = (k * cos_full) + (rotate_half(k) * sin_full)
    return q_rot, k_rot


def rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat((-x2, x1), dim=-1)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device=None,
    dtype=None,
    start: int = 0,
):
    device = device or torch.device("cpu")
    dtype = dtype or torch.get_default_dtype()
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim)
    )
    t = torch.arange(start, start + seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    sin, cos = freqs.sin()[None, :, None, :], freqs.cos()[None, :, None, :]
    return sin, cos


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))


class MultiQueryAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_flash: bool = True,
        multi_query: bool = True,
        attention_window: Optional[int] = None,
        num_registers: int = 0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.use_flash = use_flash
        self.multi_query = bool(multi_query)
        self.attention_window = attention_window
        self.num_registers = max(int(num_registers), 0)
        self.kv_heads = 1 if self.multi_query else self.num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        if self.multi_query:
            self.k_proj = nn.Linear(dim, self.head_dim, bias=False)
            self.v_proj = nn.Linear(dim, self.head_dim, bias=False)
        else:
            self.k_proj = nn.Linear(dim, dim, bias=False)
            self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        try:
            from ..utils.pylogger import get_pylogger  # type: ignore

            self._logger = get_pylogger(__name__)
        except Exception:  # pragma: no cover
            self._logger = None
        try:
            self._debug_steps_left = int(os.getenv("HUMITE_DEBUG_STEPS", "5"))
        except Exception:
            self._debug_steps_left = 5
        self._debug_layer0_only = False
        self._flex_ready = flex_attention is not None and create_block_mask is not None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[AttentionCache] = None,
        use_kv_cache: bool = False,
        block_mask=None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[AttentionCache]]:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        if self.multi_query:
            k = self.k_proj(x).view(B, T, 1, self.head_dim)
            v = self.v_proj(x).view(B, T, 1, self.head_dim)
        else:
            k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim)
            v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)
        if rope_cache is not None:
            q, k = apply_rope(q, k, rope_cache)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        key_offset = 0
        new_filled: int = 0  # updated by static path; only read back in present creation
        _used_graph_path = False
        if past_key_value is not None and past_key_value.k_buf is not None:
            buf_k = past_key_value.k_buf
            buf_v = past_key_value.v_buf
            ft = past_key_value.filled_tensor
            if ft is not None and T == 1:
                # CUDA-graph-compatible path (single-token decode only).
                # scatter_ with a GPU scalar index + full-buffer read keeps all shapes
                # static, making this forward capturable by torch.cuda.CUDAGraph.
                idx = ft.long().view(1, 1, 1, 1).expand(k.shape[0], k.shape[1], 1, self.head_dim)
                buf_k.scatter_(2, idx, k)
                buf_v.scatter_(2, idx, v)
                ft.add_(1)
                k = buf_k
                v = buf_v
                _used_graph_path = True
            else:
                # Standard static path: in-place write, then a cheap slice view.
                # Used for both non-graph single-token steps and all multi-token prefills.
                filled = int(ft.item()) if ft is not None else past_key_value.filled
                buf_k[:, :, filled : filled + T, :] = k
                buf_v[:, :, filled : filled + T, :] = v
                new_filled = filled + T
                k = buf_k[:, :, :new_filled, :]
                v = buf_v[:, :, :new_filled, :]
                if ft is not None:
                    ft.fill_(new_filled)
            # key_offset stays 0: no trimming in the static path
        elif past_key_value is not None:
            # Dynamic (legacy) path — used for windowed layers or when prealloc is off
            k = torch.cat([past_key_value.key, k], dim=2)
            v = torch.cat([past_key_value.value, v], dim=2)
            key_offset = int(past_key_value.offset)
        if use_kv_cache and self.attention_window is not None:
            window = max(int(self.attention_window), 1)
            register_tokens = min(max(self.num_registers, 0), k.size(2))
            total_tokens = k.size(2)
            if total_tokens > register_tokens:
                normal_tokens = total_tokens - register_tokens
                if normal_tokens > window:
                    trim_normal = normal_tokens - window
                    start_keep = register_tokens + trim_normal
                    if register_tokens > 0:
                        keep_registers = k[:, :, :register_tokens, :]
                        keep_values = v[:, :, :register_tokens, :]
                        keep_recent_k = k[:, :, start_keep:, :]
                        keep_recent_v = v[:, :, start_keep:, :]
                        k = torch.cat([keep_registers, keep_recent_k], dim=2)
                        v = torch.cat([keep_values, keep_recent_v], dim=2)
                    else:
                        k = k[:, :, trim_normal:, :]
                        v = v[:, :, trim_normal:, :]
                    key_offset += trim_normal
        target_len = q.shape[2]
        out = self._compute_attention(
            q,
            k,
            v,
            mask,
            block_mask,
            use_kv_cache,
            position_offset,
            key_offset,
        )
        if use_kv_cache:
            if past_key_value is not None and past_key_value.k_buf is not None:
                present: Optional[AttentionCache] = AttentionCache(
                    key=k,
                    value=v,
                    offset=0,
                    k_buf=past_key_value.k_buf,
                    v_buf=past_key_value.v_buf,
                    filled=new_filled if not _used_graph_path else past_key_value.filled,
                    filled_tensor=past_key_value.filled_tensor,
                )
            else:
                present = AttentionCache(key=k, value=v, offset=key_offset)
        else:
            present = None
        out = out.transpose(1, 2).contiguous().view(B, target_len, self.num_heads * self.head_dim)
        return self.o_proj(out), present

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        block_mask,
        use_kv_cache: bool,
        position_offset: int,
        key_offset: int,
    ) -> torch.Tensor:
        use_flex = (
            block_mask is not None
            and self._flex_ready
            and not use_kv_cache
            and self.dropout.p == 0.0
        )
        if use_flex:
            flex_fn = flex_attention
            if flex_fn is None:  # pragma: no cover
                raise RuntimeError("flex_attention is unavailable")
            out = flex_fn(
                q,
                k,
                v,
                block_mask=block_mask,
                scale=self.scale,
                enable_gqa=self.multi_query,
                return_lse=False,
            )
            if isinstance(out, tuple):
                out = out[0]
            return out
        return self._standard_attention(
            q,
            k,
            v,
            mask,
            position_offset,
            key_offset,
            use_kv_cache,
        )

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        position_offset: int,
        key_offset: int,
        use_kv_cache: bool,
    ) -> torch.Tensor:
        B = q.shape[0]
        T_q = q.shape[2]
        T_k = k.shape[2]
        _sdpa = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        # Fast path: single-query decode step with SDPA.
        # T_q=1 means there is no causal constraint between query positions — the
        # single query simply attends to all valid KV positions.  We still apply
        # the padding mask so that unfilled (zero-padded) KV positions are ignored,
        # but we never need the full causal matrix.
        #
        # Important: no bool()/item() calls here — those trigger D2H synchronisation
        # which is illegal inside torch.cuda.graph() capture.  We always build the
        # attn_bias tensor; when the mask is all-True the bias is all-zero (no-op).
        if use_kv_cache and T_q == 1 and self.use_flash and _sdpa:
            k_for = k.expand(-1, self.num_heads, -1, -1) if self.multi_query else k
            v_for = v.expand(-1, self.num_heads, -1, -1) if self.multi_query else v
            attn_bias: Optional[torch.Tensor] = None
            if mask is not None:
                if mask.dim() == 2:
                    mask = mask[:, None, None, :]
                mask_bool = mask.to(torch.bool)
                start_index = int(max(key_offset, 0))
                mask_len = mask_bool.shape[-1]
                if start_index < mask_len:
                    mask_slice = mask_bool[..., start_index : start_index + T_k]
                else:
                    mask_slice = mask_bool.new_zeros(B, 1, 1, T_k)
                # Pad to T_k if the mask is shorter (e.g. mask covers fewer positions)
                if mask_slice.shape[-1] < T_k:
                    pad = mask_slice.new_zeros(*mask_slice.shape[:-1], T_k - mask_slice.shape[-1])
                    mask_slice = torch.cat([mask_slice, pad], dim=-1)
                # Build float bias: 0 where valid, -inf where masked.
                # All-zero (mask all-True) is a no-op for SDPA.
                attn_bias = q.new_zeros(B, 1, 1, T_k).masked_fill_(
                    ~mask_slice.expand(B, 1, 1, T_k), float("-inf")
                )
            return torch.nn.functional.scaled_dot_product_attention(
                q, k_for, v_for, attn_mask=attn_bias, dropout_p=0.0
            )

        # General path: build full causal + padding mask.
        base = self._build_base_mask(T_q, T_k, q.device, position_offset, key_offset)
        base = base.unsqueeze(0).unsqueeze(0)
        allowed = base
        if mask is not None:
            if mask.dim() == 2:
                mask = mask[:, None, None, :]
            mask_bool = mask.to(torch.bool)
            mask_len = mask_bool.shape[-1]
            start_index = int(max(key_offset, 0))
            if start_index >= mask_len:
                mask_slice = torch.zeros(
                    (*mask_bool.shape[:-1], T_k), dtype=torch.bool, device=mask_bool.device
                )
            else:
                mask_slice = mask_bool[..., start_index:]
                if mask_slice.shape[-1] > T_k:
                    mask_slice = mask_slice[..., :T_k]
            if mask_slice.shape[-1] < T_k:
                pad_width = T_k - mask_slice.shape[-1]
                pad_values = torch.zeros(
                    (*mask_slice.shape[:-1], pad_width), dtype=torch.bool, device=mask_slice.device
                )
                mask_slice = torch.cat([mask_slice, pad_values], dim=-1)
            allowed = mask_slice.expand(B, 1, T_q, T_k) & base
        use_flash_sdpa = self.use_flash and _sdpa
        k_for = k
        v_for = v
        if self.multi_query and use_flash_sdpa:
            k_for = k.expand(-1, self.num_heads, -1, -1)
            v_for = v.expand(-1, self.num_heads, -1, -1)
        if use_flash_sdpa:
            if bool(allowed.all()):
                return torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k_for,
                    v_for,
                    attn_mask=None,
                    dropout_p=self.dropout.p,
                    is_causal=True,
                )
            attn_mask = torch.zeros((B, 1, T_q, T_k), device=q.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(~allowed, float("-inf"))
            return torch.nn.functional.scaled_dot_product_attention(
                q,
                k_for,
                v_for,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p,
                is_causal=False,
            )
        scores = torch.matmul(q, k_for.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~allowed, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        return torch.matmul(probs, v_for)

    def _build_base_mask(
        self,
        q_len: int,
        k_len: int,
        device: torch.device,
        position_offset,
        key_offset: int,
    ) -> torch.Tensor:
        # position_offset may be a GPU scalar tensor (CUDA-graph path) or int.
        if isinstance(position_offset, torch.Tensor) and q_len == 1:
            query_pos = position_offset.to(device=device).view(1)
        else:
            query_pos = torch.arange(position_offset, position_offset + q_len, device=device)
        key_pos = torch.arange(key_offset, key_offset + k_len, device=device)
        causal = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
        register_tokens = max(self.num_registers, 0)
        register_mask = None
        if register_tokens > 0:
            register_mask = key_pos.unsqueeze(0) < register_tokens
        if self.attention_window is not None:
            window = max(int(self.attention_window), 1)
            within = key_pos.unsqueeze(0) >= (query_pos.unsqueeze(1) - (window - 1))
            if register_mask is not None:
                within = within | register_mask
            causal = causal & within
        return causal


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        dropout=0.0,
        attn_dropout=0.0,
        use_flash=True,
        use_parallel_residual: bool = False,
        use_RMSNorm: bool = False,
        multi_query: bool = True,
        attention_window: Optional[int] = None,
        num_registers: int = 0,
    ):
        super().__init__()
        if use_RMSNorm:
            self.norm1 = RMSNorm(dim)
            self.norm2 = RMSNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        self.attn = MultiQueryAttention(
            dim,
            num_heads,
            attn_dropout,
            use_flash,
            multi_query,
            attention_window,
            num_registers,
        )
        self.ffn = SwiGLU(dim, int(mlp_ratio * dim))
        self.dropout = nn.Dropout(dropout)
        self.use_parallel_residual = use_parallel_residual

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope_cache=None,
        past_key_value: Optional[AttentionCache] = None,
        use_kv_cache: bool = False,
        block_mask=None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[AttentionCache]]:
        if self.use_parallel_residual:
            normed = self.norm1(x)
            attn_out, present = self.attn(
                normed,
                mask,
                rope_cache,
                past_key_value=past_key_value,
                use_kv_cache=use_kv_cache,
                block_mask=block_mask,
                position_offset=position_offset,
            )
            ffn_out = self.ffn(normed)
            x = x + self.dropout(attn_out + ffn_out)
            return x, present
        else:
            normed = self.norm1(x)
            attn_out, present = self.attn(
                normed,
                mask,
                rope_cache,
                past_key_value=past_key_value,
                use_kv_cache=use_kv_cache,
                block_mask=block_mask,
                position_offset=position_offset,
            )
            x = x + self.dropout(attn_out)
            normed = self.norm2(x)
            x = x + self.dropout(self.ffn(normed))
            return x, present


class GPTDecoderStack(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.rope_cache = None
        self.register_tokens = max(int(cfg.num_registers), 0)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim=cfg.embedding_dim,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                    attn_dropout=cfg.attn_dropout,
                    use_flash=cfg.use_flash_attention,
                    use_parallel_residual=cfg.use_parallel_residual,
                    multi_query=cfg.use_multi_query_attention,
                    attention_window=cfg.attention_window,
                    num_registers=cfg.num_registers,
                    use_RMSNorm=cfg.use_RMSNorm,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        if cfg.use_RMSNorm:
            self.final_norm = RMSNorm(cfg.embedding_dim)
        else:
            self.final_norm = nn.LayerNorm(cfg.embedding_dim)
        self._full_rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[AttentionCache]] = None,
        use_kv_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[AttentionCache]]]:
        B, T, _ = x.shape
        mask = None
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].to(torch.bool)
        head_dim = self.cfg.embedding_dim // self.cfg.num_heads
        position_offset = 0
        if past_key_values is not None and len(past_key_values) > 0:
            first_cache = past_key_values[0]
            if first_cache.filled_tensor is not None and T == 1:
                # Single-token decode: keep as tensor for graph-capturable ops
                position_offset = first_cache.filled_tensor
            elif first_cache.filled_tensor is not None:
                # Multi-token prefill: convert to int (not graph-captured)
                position_offset = int(first_cache.filled_tensor.item())
            else:
                position_offset = int(first_cache.offset + first_cache.key.size(2))
        rope_cache = None
        if self.cfg.use_rope:
            dtype = x.dtype if x.dtype.is_floating_point else torch.get_default_dtype()
            if use_kv_cache:
                # Always index into a pre-computed full RoPE table when KV caching.
                # This avoids rebuilding (arange → einsum → sin/cos) on every step.
                # Both graph (tensor offset) and non-graph (int offset) paths use it.
                required_seq_len = int(self.cfg.max_seq_len)
                if past_key_values is not None and len(past_key_values) > 0:
                    _fc = past_key_values[0]
                    if _fc.k_buf is not None:
                        # .shape[N] returns a plain Python int for static shapes,
                        # avoiding SymInt materialisation (int(SymInt) overflows in
                        # PyTorch ≤2.2 when tracing inside torch.compile).
                        required_seq_len = max(required_seq_len, _fc.k_buf.shape[2])
                if not isinstance(position_offset, torch.Tensor):
                    required_seq_len = max(required_seq_len, int(position_offset) + int(T))
                if (
                    self._full_rope_cache is None
                    or self._full_rope_cache[0].device != x.device
                    or self._full_rope_cache[0].dtype != dtype
                    or self._full_rope_cache[0].shape[1] < required_seq_len
                ):
                    self._full_rope_cache = build_rope_cache(
                        seq_len=required_seq_len,
                        head_dim=head_dim,
                        device=x.device,
                        dtype=dtype,
                    )
                full_sin, full_cos = self._full_rope_cache
                if isinstance(position_offset, torch.Tensor):
                    # Graph-compatible: index with GPU tensor (no Python int)
                    pos_idx = position_offset.long().view(1)
                    rope_cache = (
                        full_sin.index_select(1, pos_idx),
                        full_cos.index_select(1, pos_idx),
                    )
                else:
                    # Standard path: slice by Python int, works for any T
                    rope_cache = (
                        full_sin[:, position_offset : position_offset + T, :, :],
                        full_cos[:, position_offset : position_offset + T, :, :],
                    )
            else:
                if (
                    self.rope_cache is None
                    or self.rope_cache[0].size(1) < T
                    or self.rope_cache[0].device != x.device
                ):
                    self.rope_cache = build_rope_cache(
                        seq_len=T,
                        head_dim=head_dim,
                        device=x.device,
                        dtype=dtype,
                    )
                elif self.rope_cache[0].size(1) > T:
                    sin, cos = self.rope_cache
                    self.rope_cache = (sin[:, :T, :, :], cos[:, :T, :, :])
                rope_cache = self.rope_cache
        block_mask = None
        if (
            not use_kv_cache
            and self.cfg.attention_window is not None
            and flex_attention is not None
            and create_block_mask is not None
        ):
            block_mask = self._build_block_mask(attention_mask, T, x.device)
        new_states: List[AttentionCache] = []
        for idx, layer in enumerate(self.layers):
            layer_past = None
            if past_key_values is not None and idx < len(past_key_values):
                layer_past = past_key_values[idx]
            x, present = layer(
                x,
                mask=mask,
                rope_cache=rope_cache,
                past_key_value=layer_past,
                use_kv_cache=use_kv_cache,
                block_mask=block_mask,
                position_offset=position_offset,
            )
            if use_kv_cache:
                if present is None:  # pragma: no cover
                    raise RuntimeError("KV cache missing present state")
                new_states.append(present)
        hidden = self.final_norm(x)
        return hidden, (new_states if use_kv_cache else None)

    def _build_block_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        seq_len: int,
        device: torch.device,
    ):
        if self.cfg.attention_window is None:
            return None
        base_window = max(int(self.cfg.attention_window), 1)
        mask_tensor = None
        kv_limit = seq_len
        register_tokens = max(self.register_tokens, 0)
        if attention_mask is not None:
            mask_tensor = attention_mask.to(torch.bool).to(device)
            token_count = int(mask_tensor.shape[1])
            if token_count < seq_len:
                pad_width = seq_len - token_count
                pad_values = torch.ones(
                    (mask_tensor.shape[0], pad_width), dtype=mask_tensor.dtype, device=device
                )
                mask_tensor = torch.cat([mask_tensor, pad_values], dim=1)
            elif token_count > seq_len:
                mask_tensor = mask_tensor[:, :seq_len]
            kv_limit = int(mask_tensor.shape[1])
        batch_size = None if mask_tensor is None else int(mask_tensor.shape[0])

        def mask_mod(b, h, q_idx, kv_idx):
            causal = q_idx >= kv_idx
            window_condition = (q_idx - kv_idx) < base_window
            if register_tokens > 0:
                register_condition = kv_idx < register_tokens
                window_condition = window_condition | register_condition
            if mask_tensor is not None:
                kv_idx_tensor = torch.as_tensor(kv_idx, device=mask_tensor.device)
                q_idx_tensor = torch.as_tensor(q_idx, device=mask_tensor.device)
                kv_idx_clamped = torch.clamp(kv_idx_tensor, 0, kv_limit - 1)
                q_idx_clamped = torch.clamp(q_idx_tensor, 0, kv_limit - 1)
                key_valid = mask_tensor[b, kv_idx_clamped]
                query_valid = mask_tensor[b, q_idx_clamped]
                key_valid = key_valid & (kv_idx_tensor >= 0) & (kv_idx_tensor < kv_limit)
                query_valid = query_valid & (q_idx_tensor >= 0) & (q_idx_tensor < kv_limit)
                return causal & window_condition & key_valid & query_valid
            return causal & window_condition

        builder = create_block_mask
        if builder is None:  # pragma: no cover
            raise RuntimeError("create_block_mask is unavailable")
        return builder(
            mask_mod,
            B=batch_size,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=str(device),
        )

    def preallocate_kv_cache(
        self,
        batch_size: int,
        max_len: int,
        device: torch.device,
        dtype: torch.dtype,
        cuda_graph: bool = False,
    ) -> List[AttentionCache]:
        """Pre-allocate static KV-cache buffers for all layers.

        Layers with attention_window fall back to the dynamic (torch.cat) path and
        receive an empty-but-valid AttentionCache with no static buffer attached.

        The returned list is passed as the initial past_key_values before the
        generation loop.  Because filled=0 and key.size(2)==0, the backbone's
        processed_tokens computation sees 0 at step t=0 and passes the full
        embedding — identical to the None behaviour, but without a None check.

        When cuda_graph=True, buffers are zero-initialized (not empty) to prevent
        NaN contamination from reading the full buffer, and each layer gets a
        filled_tensor (GPU scalar) for graph-capturable position tracking.
        """
        head_dim = self.cfg.embedding_dim // self.cfg.num_heads
        alloc_fn = torch.zeros if cuda_graph else torch.empty
        caches: List[AttentionCache] = []
        for layer in self.layers:
            attn = layer.attn
            kv_heads: int = attn.kv_heads
            if attn.attention_window is not None:
                # Sliding-window layer: keep dynamic path; empty sentinel cache
                caches.append(
                    AttentionCache(
                        key=torch.empty(
                            batch_size, kv_heads, 0, head_dim, device=device, dtype=dtype
                        ),
                        value=torch.empty(
                            batch_size, kv_heads, 0, head_dim, device=device, dtype=dtype
                        ),
                        offset=0,
                    )
                )
            else:
                k_buf = alloc_fn(
                    batch_size, kv_heads, max_len, head_dim, device=device, dtype=dtype
                )
                v_buf = alloc_fn(
                    batch_size, kv_heads, max_len, head_dim, device=device, dtype=dtype
                )
                ft = torch.zeros(1, dtype=torch.long, device=device) if cuda_graph else None
                caches.append(
                    AttentionCache(
                        key=torch.empty(
                            batch_size, kv_heads, 0, head_dim, device=device, dtype=dtype
                        ),
                        value=torch.empty(
                            batch_size, kv_heads, 0, head_dim, device=device, dtype=dtype
                        ),
                        offset=0,
                        k_buf=k_buf,
                        v_buf=v_buf,
                        filled=0,
                        filled_tensor=ft,
                    )
                )
        return caches
