"""Enhance-A-Video runtime helpers for Kandinsky-5 Lite."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


class _EnhanceState:
    __slots__ = (
        "enabled",
        "weight",
        "num_frames",
        "max_tokens",
        "skip_due_to_error",
        "diag_cache",
        "warned_bad_shape",
        "warned_disabled_frames",
    )

    def __init__(self) -> None:
        self.enabled: bool = False
        self.weight: float = 3.4
        self.num_frames: Optional[int] = None
        self.max_tokens: Optional[int] = None
        self.skip_due_to_error: bool = False
        self.diag_cache: dict[Tuple[str, int], torch.Tensor] = {}
        self.warned_bad_shape: bool = False
        self.warned_disabled_frames: bool = False


_STATE = _EnhanceState()


def configure_enhance(enable: bool, weight: float, num_frames: Optional[int], max_tokens: Optional[int]) -> None:
    """Configure runtime Enhance-A-Video behaviour for the next generation pass."""
    _STATE.enabled = bool(enable)
    _STATE.weight = float(weight) if weight is not None else 3.4
    _STATE.num_frames = int(num_frames) if num_frames and num_frames > 0 else None
    _STATE.max_tokens = int(max_tokens) if max_tokens and max_tokens > 0 else None
    _STATE.skip_due_to_error = False
    _STATE.warned_bad_shape = False
    _STATE.warned_disabled_frames = False


def clear_enhance() -> None:
    """Reset runtime state after generation."""
    _STATE.enabled = False
    _STATE.num_frames = None
    _STATE.skip_due_to_error = False


def is_enhance_enabled() -> bool:
    if not _STATE.enabled:
        return False
    if _STATE.num_frames is None or _STATE.num_frames < 2:
        if not _STATE.warned_disabled_frames and _STATE.enabled:
            print("Enhance-A-Video skipped: need >=2 frames (enable only for video runs).")
            _STATE.warned_disabled_frames = True
        return False
    if _STATE.skip_due_to_error:
        return False
    return True


def _diag_mask(device: torch.device, num_frames: int) -> torch.Tensor:
    key = (f"{device.type}:{device.index}", num_frames)
    mask = _STATE.diag_cache.get(key)
    if mask is None:
        mask = torch.eye(num_frames, device=device, dtype=torch.bool)
        _STATE.diag_cache[key] = mask
    return mask


def compute_enhance_multiplier(query: torch.Tensor, key: torch.Tensor) -> Optional[torch.Tensor]:
    """Return a scalar multiplier (tensor) or ``None`` when no enhancement is applied."""
    if not is_enhance_enabled():
        return None

    num_frames = _STATE.num_frames
    assert num_frames is not None

    if query.dim() == 3:
        query = query.unsqueeze(0)
        key = key.unsqueeze(0)
    if query.dim() != 4 or key.dim() != 4:
        if not _STATE.warned_bad_shape:
            print(f"Enhance-A-Video skipped: unexpected tensor rank {query.dim()} (expected 3 or 4).")
            _STATE.warned_bad_shape = True
        return None

    try:
        batch, seq_len, num_heads, head_dim = query.shape
    except ValueError as exc:
        if not _STATE.warned_bad_shape:
            print(f"Enhance-A-Video skipped: unexpected tensor shape {query.shape}: {exc}")
            _STATE.warned_bad_shape = True
        return None

    total_tokens = seq_len
    if total_tokens % num_frames != 0:
        if not _STATE.warned_bad_shape:
            print(
                "Enhance-A-Video skipped: token count not divisible by frame count ("
                f"tokens={total_tokens}, frames={num_frames})."
            )
            _STATE.warned_bad_shape = True
        return None

    spatial_tokens = total_tokens // num_frames
    if spatial_tokens == 0:
        return None

    try:
        query_heads_first = query.permute(0, 2, 1, 3)
        key_heads_first = key.permute(0, 2, 1, 3)

        query_reshaped = query_heads_first.reshape(batch, num_heads, num_frames, spatial_tokens, head_dim)
        key_reshaped = key_heads_first.reshape(batch, num_heads, num_frames, spatial_tokens, head_dim)

        max_tokens = _STATE.max_tokens
        if max_tokens is not None and spatial_tokens > max_tokens:
            stride = math.ceil(spatial_tokens / max_tokens)
            query_reshaped = query_reshaped[:, :, :, ::stride, :]
            key_reshaped = key_reshaped[:, :, :, ::stride, :]

        query_latents = query_reshaped.permute(0, 3, 1, 2, 4).reshape(-1, num_heads, num_frames, head_dim)
        key_latents = key_reshaped.permute(0, 3, 1, 2, 4).reshape(-1, num_heads, num_frames, head_dim)

        if query_latents.numel() == 0:
            return None

        scale = head_dim ** -0.5
        query_scaled = query_latents * scale
        attn_scores = torch.matmul(query_scaled, key_latents.transpose(-2, -1))
        attn_scores = attn_scores.to(torch.float32)
        attn_scores = attn_scores.softmax(dim=-1)

        diag_mask = _diag_mask(attn_scores.device, num_frames)
        attn_wo_diag = attn_scores.masked_fill(diag_mask, 0.0)
        num_off_diag = num_frames * num_frames - num_frames
        if num_off_diag <= 0:
            return None
        mean_scores = attn_wo_diag.sum(dim=(-2, -1)) / num_off_diag
        enhance_value = mean_scores.mean() * (num_frames + _STATE.weight)
        enhance_value = torch.clamp(enhance_value, min=1.0)
        return enhance_value.to(query.dtype)
    except RuntimeError as exc:
        if not _STATE.skip_due_to_error:
            print(f"Enhance-A-Video disabled for this run due to runtime error: {exc}")
        _STATE.skip_due_to_error = True
        return None


__all__ = [
    "configure_enhance",
    "clear_enhance",
    "is_enhance_enabled",
    "compute_enhance_multiplier",
]
