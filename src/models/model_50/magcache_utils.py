# This source code is licensed under the Apache License found in the
# LICENSE file in the current directory.
"""
The code has been adopted from Kandinsky-5
(https://github.com/ai-forever/Kandinsky-5/blob/main/kandinsky/magcache_utils.py)
"""

# This is an adaptation of Magcache from https://github.com/Zehong-Ma/MagCache/

import numpy as np
import torch

from .models.nn import kd5_compile


def nearest_interp(src_array, target_length):
    src_length = len(src_array)
    if target_length == 1:
        return np.array([src_array[-1]])

    scale = (src_length - 1) / (target_length - 1)
    mapped_indices = np.round(np.arange(target_length) * scale).astype(int)
    return src_array[mapped_indices]


def set_magcache_params(dit, mag_ratios, num_steps, no_cfg, calibrate=False):
    """
    Setup magcache on the DIT model.

    Args:
        dit: DIT model
        mag_ratios: List of magnitude ratios (ignored if calibrate=True)
        num_steps: Number of denoising steps
        no_cfg: Whether this is a no-CFG model
        calibrate: If True, run calibration mode to compute mag_ratios
    """
    if calibrate:
        print(f"🔬 Initializing Magcache CALIBRATION mode")
    else:
        print(f"🚀 Initializing Magcache")
    print(f"   → Mode: {'no_cfg (counter +2)' if no_cfg else 'cfg (counter +1)'}")
    print(f"   → Num steps: {num_steps}")
    print(f"   → Total steps: {num_steps * 2}")

    # Store original forward method if not already stored
    if not hasattr(dit.__class__, "_original_forward"):
        dit.__class__._original_forward = dit.__class__.forward

    # Choose forward method based on calibrate flag
    if calibrate:
        dit.__class__.forward = magcache_calibration
        # Initialize calibration tracking lists
        dit.norm_ratio = []
        dit.norm_std = []
        dit.cos_dis = []
        print(f"   → Using CALIBRATION forward (will compute mag_ratios)")
    else:
        dit.__class__.forward = magcache_forward
        dit.accumulated_err = [0.0, 0.0]
        dit.accumulated_steps = [0, 0]
        dit.accumulated_ratio = [1.0, 1.0]
        dit.magcache_thresh = 0.12
        dit.K = 2
        dit.retention_ratio = 0.2
        dit.mag_ratios = np.array([1.0] * 2 + mag_ratios)
        dit._magcache_enabled = True
        dit._skip_count = 0
        dit._compute_count = 0

        if len(dit.mag_ratios) != num_steps * 2:
            print(
                f"   → Interpolating mag_ratios: {len(dit.mag_ratios)} -> {num_steps * 2}"
            )
            mag_ratio_con = nearest_interp(dit.mag_ratios[0::2], num_steps)
            mag_ratio_ucon = nearest_interp(dit.mag_ratios[1::2], num_steps)
            interpolated_mag_ratios = np.concatenate(
                [mag_ratio_con.reshape(-1, 1), mag_ratio_ucon.reshape(-1, 1)], axis=1
            ).reshape(-1)
            dit.mag_ratios = interpolated_mag_ratios

    dit.cnt = 0
    dit.num_steps = num_steps * 2
    dit.residual_cache = [None, None]
    dit.no_cfg = no_cfg

    if calibrate:
        print(f"✓ Calibration mode initialized - will save results to JSON files")
    else:
        print(f"✓ Magcache initialized successfully")


def disable_magcache(dit):
    """Disable magcache and restore original forward method."""
    if hasattr(dit.__class__, "_original_forward"):
        dit.__class__.forward = dit.__class__._original_forward
        dit._magcache_enabled = False
        print("✓ Magcache disabled, restored original forward method")


def reset_magcache_state(dit):
    """Reset magcache state counters - call this before each generation."""
    if hasattr(dit, "_magcache_enabled") and dit._magcache_enabled:
        old_cnt = dit.cnt
        dit.cnt = 0
        dit.accumulated_ratio = [1.0, 1.0]
        dit.accumulated_err = [0.0, 0.0]
        dit.accumulated_steps = [0, 0]
        dit.residual_cache = [None, None]
        if hasattr(dit, "_skip_count"):
            dit._skip_count = 0
        if hasattr(dit, "_compute_count"):
            dit._compute_count = 0
        print(f"🔄 Magcache state reset for new generation (cnt: {old_cnt} -> {dit.cnt})")


@kd5_compile(mode="max-autotune-no-cudagraphs")
def magcache_forward(
    self,
    x,
    text_embed,
    pooled_text_embed,
    time,
    visual_rope_pos,
    text_rope_pos,
    scale_factor=(1.0, 1.0, 1.0),
    sparse_params=None,
):
    if not getattr(self, "_magcache_enabled", False) or not hasattr(self, "cnt"):
        if hasattr(self.__class__, "_original_forward"):
            return self.__class__._original_forward(
                self,
                x,
                text_embed,
                pooled_text_embed,
                time,
                visual_rope_pos,
                text_rope_pos,
                scale_factor,
                sparse_params,
            )

    text_embed, time_embed, text_rope, visual_embed = (
        self.before_text_transformer_blocks(
            text_embed, time, pooled_text_embed, x, text_rope_pos
        )
    )

    for text_transformer_block in self.text_transformer_blocks:
        text_embed = text_transformer_block(text_embed, time_embed, text_rope)

    visual_embed, visual_shape, to_fractal, visual_rope = (
        self.before_visual_transformer_blocks(
            visual_embed, visual_rope_pos, scale_factor, sparse_params
        )
    )

    skip_forward = False
    ori_visual_embed = visual_embed

    # Debug: Log first forward pass to verify reset worked
    if self.cnt == 0:
        print(f"🎬 Magcache: Starting new generation (cnt={self.cnt})")

    retention_start = int(self.num_steps * self.retention_ratio)
    if self.cnt >= retention_start:
        cur_mag_ratio = self.mag_ratios[self.cnt]
        self.accumulated_ratio[self.cnt % 2] = (
            self.accumulated_ratio[self.cnt % 2] * cur_mag_ratio
        )
        self.accumulated_steps[self.cnt % 2] += 1
        cur_skip_err = np.abs(1 - self.accumulated_ratio[self.cnt % 2])
        self.accumulated_err[self.cnt % 2] += cur_skip_err

        can_skip = (
            self.accumulated_err[self.cnt % 2] < self.magcache_thresh
            and self.accumulated_steps[self.cnt % 2] <= self.K
        )

        if can_skip:
            skip_forward = True
            residual_visual_embed = self.residual_cache[self.cnt % 2]
        else:
            # Log why we're NOT skipping
            if self.cnt == retention_start:
                print(
                    f"🔍 Magcache: Starting from forward pass {retention_start} (retention_ratio={self.retention_ratio})"
                )
            self.accumulated_err[self.cnt % 2] = 0
            self.accumulated_steps[self.cnt % 2] = 0
            self.accumulated_ratio[self.cnt % 2] = 1.0

    if skip_forward:
        visual_embed = visual_embed + residual_visual_embed
        self._skip_count += 1
        print(
            f"⚡ Magcache: Forward pass {self.cnt}/{self.num_steps} SKIPPED (err: {self.accumulated_err[self.cnt % 2]:.4f}, consecutive: {self.accumulated_steps[self.cnt % 2]})"
        )
    else:
        self._compute_count += 1
        for visual_transformer_block in self.visual_transformer_blocks:
            visual_embed = visual_transformer_block(
                visual_embed, text_embed, time_embed, visual_rope, sparse_params
            )
        residual_visual_embed = visual_embed - ori_visual_embed

    self.residual_cache[self.cnt % 2] = residual_visual_embed

    x = self.after_blocks(
        visual_embed, visual_shape, to_fractal, text_embed, time_embed
    )

    # Increment counter: +2 for no_cfg models, +1 for cfg models
    # Use getattr for backward compatibility with models that don't have no_cfg set
    if getattr(self, "no_cfg", False):
        self.cnt += 2
    else:
        self.cnt += 1

    if self.cnt >= self.num_steps:
        total_processed = self._skip_count + self._compute_count
        print(f"")
        print(f"✓ Magcache Summary:")
        print(f"   → Total forward passes: {self.num_steps}")
        print(f"   → Passes processed: {total_processed}")
        print(f"   → Passes computed: {self._compute_count}")
        print(f"   → Passes skipped: {self._skip_count}")
        if total_processed > 0:
            print(f"   → Skip ratio: {self._skip_count/total_processed*100:.1f}%")
            print(f"   → Performance gain: {self._skip_count/self.num_steps*100:.1f}%")
        print(f"")
        self.cnt = 0
        self.accumulated_ratio = [1.0, 1.0]
        self.accumulated_err = [0.0, 0.0]
        self.accumulated_steps = [0, 0]
        self._skip_count = 0
        self._compute_count = 0
    return x


def magcache_calibration(
    self,
    x,
    text_embed,
    pooled_text_embed,
    time,
    visual_rope_pos,
    text_rope_pos,
    scale_factor=(1.0, 1.0, 1.0),
    sparse_params=None,
):
    """
    Calibration mode for magcache - computes magnitude ratios between consecutive forward passes.

    To use:
    1. In set_magcache_params(), comment out: dit.__class__.forward = magcache_forward
    2. Uncomment: dit.__class__.forward = magcache_calibration
    3. Initialize calibration lists in set_magcache_params()
    4. Run generation - it will save mag_ratios to JSON files
    """
    text_embed, time_embed, text_rope, visual_embed = (
        self.before_text_transformer_blocks(
            text_embed, time, pooled_text_embed, x, text_rope_pos
        )
    )

    for text_transformer_block in self.text_transformer_blocks:
        text_embed = text_transformer_block(text_embed, time_embed, text_rope)

    visual_embed, visual_shape, to_fractal, visual_rope = (
        self.before_visual_transformer_blocks(
            visual_embed, visual_rope_pos, scale_factor, sparse_params
        )
    )

    ori_visual_embed = visual_embed
    for visual_transformer_block in self.visual_transformer_blocks:
        visual_embed = visual_transformer_block(
            visual_embed, text_embed, time_embed, visual_rope, sparse_params
        )
    residual_visual_embed = visual_embed - ori_visual_embed

    if self.cnt >= 2:
        norm_ratio = (
            (
                residual_visual_embed.norm(dim=-1)
                / self.residual_cache[self.cnt % 2].norm(dim=-1)
            )
            .mean()
            .item()
        )
        norm_std = (
            residual_visual_embed.norm(dim=-1)
            / self.residual_cache[self.cnt % 2].norm(dim=-1)
        ).std().item()
        cos_dis = (
            1
            - torch.nn.functional.cosine_similarity(
                residual_visual_embed,
                self.residual_cache[self.cnt % 2],
                dim=-1,
                eps=1e-8,
            )
        ).mean().item()
        self.norm_ratio.append(round(norm_ratio, 5))
        self.norm_std.append(round(norm_std, 5))
        self.cos_dis.append(round(cos_dis, 5))
        print(
            f"📊 Calibration step {self.cnt}: norm_ratio={norm_ratio:.5f}, norm_std={norm_std:.5f}, cos_dis={cos_dis:.5f}"
        )

    self.residual_cache[self.cnt % 2] = residual_visual_embed

    x = self.after_blocks(
        visual_embed, visual_shape, to_fractal, text_embed, time_embed
    )

    if getattr(self, "no_cfg", False):
        self.cnt += 2
    else:
        self.cnt += 1

    if self.cnt >= self.num_steps:
        self.cnt = 0
        print("\n" + "=" * 70)
        print("✓ MAGCACHE CALIBRATION COMPLETE!")
        print("=" * 70)
        print("\n📋 Norm Ratios (use these as mag_ratios in your config):")
        print(self.norm_ratio)
        print("\n📊 Norm Std:")
        print(self.norm_std)
        print("\n📐 Cosine Distance:")
        print(self.cos_dis)

        import json

        with open("kandy_mag_ratio.json", "w") as f:
            json.dump(self.norm_ratio, f, indent=2)
        with open("kandy_mag_std.json", "w") as f:
            json.dump(self.norm_std, f, indent=2)
        with open("kandy_cos_dis.json", "w") as f:
            json.dump(self.cos_dis, f, indent=2)

        print("\n✓ Results saved to:")
        print("  → kandy_mag_ratio.json (copy these values to your config!)")
        print("  → kandy_mag_std.json")
        print("  → kandy_cos_dis.json")
        print("=" * 70 + "\n")

    return x
