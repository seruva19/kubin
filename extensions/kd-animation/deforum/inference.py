# This source code is licensed under the MIT License found in the
# LICENSE file in the current directory.
"""
The code has been adopted from Deforum-Kandinsky
(https://github.com/ai-forever/deforum-kandinsky/blob/main/deforum_kandinsky/inference.py)
"""

import subprocess, time, gc, os, sys, time

sub_p_res = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free",
        "--format=csv,noheader",
    ],
    stdout=subprocess.PIPE,
).stdout.decode("utf-8")

import subprocess
import time
import gc
import os
import sys
import random
import clip
import torch
from types import SimpleNamespace
from deforum_kandinsky.configs import animations, update_anim_args
from deforum_kandinsky.helpers.prompts import Prompts
from deforum_kandinsky.helpers.render import (
    render_animation,
    render_input_video,
    render_image_batch,
    render_interpolation,
)
from deforum_kandinsky.helpers.aesthetics import load_aesthetics_model
from deforum_kandinsky.helpers.script import Script


def get_gpu_info():
    sub_p_res = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader",
        ],
        stdout=subprocess.PIPE,
    ).stdout.decode("utf-8")
    return sub_p_res[:-1]


class Models:
    def __init__(self, prior, decoder_img2img, device="cpu"):
        self.device = device
        self.prior = prior.to(device)
        self.decoder_img2img = decoder_img2img.to(device)


class DeforumKandinsky:
    def __init__(self, prior, decoder_img2img, device="cpu"):
        self.root = self.setup_models(prior, decoder_img2img, device)
        self.cond_prompts, self.uncond_prompts = None, None
        self.args, self.anim_args = None, None

    def setup_models(self, prior, decoder_img2img, device="cpu"):
        root = SimpleNamespace()
        root.models_path = "models"
        root.configs_path = "configs"
        root.output_path = "output"
        root.map_location = device
        root.device = torch.device(root.map_location)
        root.model = Models(prior, decoder_img2img, root.device)
        return root

    def prepare_configs(self, animations, durations, accelerations, fps, **kwargs):
        args_dict, anim_args_dict = Script(
            animations, durations, accelerations, **kwargs
        ).args

        if "max_frames" not in kwargs:
            anim_args_dict["max_frames"] = int(sum(durations) * fps) + 1

        args = SimpleNamespace(**args_dict)
        anim_args = SimpleNamespace(**anim_args_dict)

        args.timestring = time.strftime("%Y%m%d%H%M%S")

        # Load clip model if using clip guidance
        if (args.clip_scale > 0) or (args.aesthetics_scale > 0):
            self.root.clip_model = (
                clip.load(args.clip_name, jit=False)[0]
                .eval()
                .requires_grad_(False)
                .to(self.root.device)
            )
            if args.aesthetics_scale > 0:
                self.root.aesthetics_model = load_aesthetics_model(args, self.root)

        if args.seed == -1:
            args.seed = random.randint(0, 2**32 - 1)
        if not args.use_init:
            args.init_image = None

        return args, anim_args

    def prepare_prompts(self, prompts, negative_prompts, durations, fps):
        def set_prompt_timings(prompts, durations, fps):
            seconds_elapsed = 0
            prompts_dict = dict()
            for index, (prompt, duration) in enumerate(zip(prompts, durations)):
                prompts_dict[str(int(seconds_elapsed * fps))] = prompt
                seconds_elapsed += duration
            return prompts_dict

        if isinstance(prompts, list):
            prompts = set_prompt_timings(prompts, durations, fps)
        if isinstance(negative_prompts, list):
            negative_prompts = set_prompt_timings(negative_prompts, durations, fps)

        cond, uncond = Prompts(prompt=prompts, neg_prompt=negative_prompts).as_dict()
        return cond, uncond

    def __len__(self):
        return 0 if self.anim_args.max_frames is None else self.anim_args.max_frames - 1

    def __call__(
        self,
        prompts,
        animations,
        prompt_durations,
        negative_prompts=None,
        accelerations=None,
        fps=24,
        **kwargs
    ):
        if negative_prompts is None:
            negative_prompts = ["low quility, bad image, cropped, out of frame"] * len(
                prompts
            )
        if accelerations is None:
            accelerations = [1.0] * len(prompts)

        args = [prompts, animations, prompt_durations, negative_prompts, accelerations]
        assert len(prompts) > 0, "you should pass at least one prompt"
        assert all(
            map(lambda a: isinstance(a, list), args)
        ), "all params should be of type 'list'"
        assert len(set(map(len, (args)))) == 1, "all params should be the same length"
        assert all(
            map(lambda a: a > 0, prompt_durations)
        ), "all durations should be positive float/int values"

        self.args, self.anim_args = self.prepare_configs(
            animations, prompt_durations, accelerations, fps, **kwargs
        )

        self.cond_prompts, self.uncond_prompts = self.prepare_prompts(
            prompts, negative_prompts, prompt_durations, fps
        )

        return render_animation(
            self.root, self.anim_args, self.args, self.cond_prompts, self.uncond_prompts
        )
