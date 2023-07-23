import argparse
import os
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
import pandas as pd
import math
from packaging import version

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from torchvision import transforms
import transformers
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from transformers.utils import ContextManagers

import diffusers
from diffusers import DDPMScheduler, UNet2DConditionModel, VQModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.models.attention_processor import (
    LoRAAttnProcessor,
    LoRAAttnAddedKVProcessor,
)
from diffusers.loaders import AttnProcsLayers

if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")

default_lora_decoder_config_path = (
    "extensions/kd-training/train_modules/train_configs/config_22_lora_decoder.yaml"
)


def add_default_values(lora_decoder_config):
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if (
        env_local_rank != -1
        and env_local_rank != lora_decoder_config["training"]["local_rank"]
    ):
        lora_decoder_config["training"]["local_rank"] = env_local_rank

    return lora_decoder_config


import time


def fix_lora_decoder_config(config):
    if config["dataset"]["train_image_folder"] == "":
        config["dataset"]["train_image_folder"] = None

    if config["dataset"]["val_image_folder"] == "":
        config["dataset"]["val_image_folder"] = None

    if config["dataset"]["val_images_paths_csv"] == "":
        config["dataset"]["val_images_paths_csv"] = None

    if config["training"]["snr_gamma"] == "":
        config["training"]["snr_gamma"] = None
    if config["training"]["snr_gamma"] is not None:
        config["training"]["snr_gamma"] = int(config["training"]["snr_gamma"])

    if config["training"]["resume_from_checkpoint"] == "":
        config["training"]["resume_from_checkpoint"] = None

    if config["training"]["checkpoints_total_limit"] == "":
        config["training"]["checkpoints_total_limit"] = None
    if config["training"]["checkpoints_total_limit"] is not None:
        config["training"]["checkpoints_total_limit"] = int(
            config["training"]["checkpoints_total_limit"]
        )

    if config["training"]["report_to"] == "none":
        config["training"]["report_to"] = None

    if config["training"]["seed"] == "":
        config["training"]["seed"] = None
    if config["training"]["seed"] is not None:
        config["training"]["seed"] = int(config["training"]["seed"])

    return config


def launch_lora_decoder_training(kubin, config, progress):
    cache_dir = kubin.params("general", "cache_dir")

    config = fix_lora_decoder_config(config)
    print(f"launching training of LoRA decoder model with params: {config}")

    logging_dir = os.path.join(
        config["training"]["output_dir"], config["training"]["logging_dir"]
    )
    accelerator_project_config = ProjectConfiguration(
        total_limit=config["training"]["checkpoints_total_limit"],
        project_dir=config["training"]["output_dir"],
        logging_dir=logging_dir,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        mixed_precision=config["training"]["mixed_precision"],
        log_with=config["training"]["report_to"],
        project_config=accelerator_project_config,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if config["training"]["seed"] is not None:
        set_seed(config["training"]["seed"])

    if accelerator.is_main_process:
        if config["training"]["output_dir"] is not None:
            os.makedirs(config["training"]["output_dir"], exist_ok=True)

    noise_scheduler = DDPMScheduler.from_pretrained(
        config["paths"]["scheduler_path"], subfolder="scheduler", cache_dir=cache_dir
    )
    image_processor = CLIPImageProcessor.from_pretrained(
        config["paths"]["image_processor_path"],
        subfolder="image_processor",
        cache_dir=cache_dir,
    )
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae = VQModel.from_pretrained(
        config["paths"]["pretrained_vae_path"],
        subfolder="movq",
        torch_dtype=weight_dtype,
        cache_dir=cache_dir,
    ).eval()

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        config["paths"]["pretrained_image_encoder"],
        subfolder="image_encoder",
        torch_dtype=weight_dtype,
        cache_dir=cache_dir,
    ).eval()

    unet = UNet2DConditionModel.from_pretrained(
        config["paths"]["pretrained_kandinsky_path"],
        subfolder="unet",
        torch_dtype=weight_dtype,
        cache_dir=cache_dir,
    )

    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    image_encoder.to(accelerator.device)
    vae.to(accelerator.device)
    unet.to(accelerator.device)
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        lora_attn_procs[name] = LoRAAttnAddedKVProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=config["training"]["rank"],
        )

    unet.set_attn_processor(lora_attn_procs)

    def compute_snr(timesteps):
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
            timesteps
        ].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
            device=timesteps.device
        )[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        snr = (alpha / sigma) ** 2
        return snr

    lora_layers = AttnProcsLayers(unet.attn_processors)

    if config["training"]["allow_tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True

    if config["training"]["use_8bit_adam"]:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        lora_layers.parameters(),
        lr=config["training"]["lr"],
        betas=(config["training"]["adam_beta1"], config["training"]["adam_beta2"]),
        weight_decay=config["training"]["weight_decay"],
        eps=config["training"]["adam_epsilon"],
    )

    train_dataset = LoRAImageDatasetForDecoder(
        image_folder=config["dataset"]["train_image_folder"],
        images_paths_csv=config["dataset"]["train_images_paths_csv"],
        image_processor=image_processor,
        img_size=config["decoder"]["image_resolution"],
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["training"]["train_batch_size"],
        num_workers=config["training"]["dataloader_num_workers"],
    )
    if (
        config["dataset"]["val_image_folder"] is not None
        or config["dataset"]["val_images_paths_csv"] is not None
    ):
        do_val = True
        val_dataset = LoRAImageDatasetForDecoder(
            image_folder=config["dataset"]["val_image_folder"],
            images_paths_csv=config["dataset"]["val_images_paths_csv"],
            image_processor=image_processor,
            img_size=config["decoder"]["image_resolution"],
        )
        val_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config["training"]["val_batch_size"],
            num_workers=config["training"]["dataloader_num_workers"],
        )
    else:
        do_val = False
    override_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config["training"]["gradient_accumulation_steps"]
    )
    if config["training"]["max_train_steps"] is None:
        config["training"]["max_train_steps"] = (
            config["training"]["num_train_epochs"] * num_update_steps_per_epoch
        )
        override_max_train_steps = True

    lr_scheduler = get_scheduler(
        config["training"]["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=config["training"]["lr_warmup_steps"]
        * config["training"]["gradient_accumulation_steps"],
        num_training_steps=config["training"]["max_train_steps"]
        * config["training"]["gradient_accumulation_steps"],
    )
    lora_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_layers, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config["training"]["gradient_accumulation_steps"]
    )
    if override_max_train_steps:
        config["training"]["max_train_steps"] = (
            config["training"]["num_train_epochs"] * num_update_steps_per_epoch
        )
    config["training"]["num_train_epochs"] = math.ceil(
        config["training"]["max_train_steps"] / num_update_steps_per_epoch
    )

    if accelerator.is_main_process:
        tracker_config = dict(vars(config.to_container()))
        accelerator.init_trackers("kubin-lora", tracker_config)

    total_batch_size = (
        config["training"]["train_batch_size"]
        * accelerator.num_processes
        * config["training"]["gradient_accumulation_steps"]
    )

    logger.info("running training")
    logger.info(f"number of dataset samples: {len(train_dataset)}")
    logger.info(f"number of epochs: {config['training']['num_train_epochs']}")
    logger.info(
        f"instantaneous batch size per device: {config['training']['train_batch_size']}"
    )
    logger.info(
        f"total train batch size (w. parallel, distributed & accumulation): {total_batch_size}"
    )
    logger.info(
        f"gradient accumulation steps: {config['training']['gradient_accumulation_steps']}"
    )
    logger.info(f"total optimization steps: {config['training']['max_train_steps']}")

    global_step = 0
    first_epoch = 0

    if config["training"]["resume_from_checkpoint"] is not None:
        if config["training"]["resume_from_checkpoint"] != "latest":
            path = os.path.basename(config["training"]["resume_from_checkpoint"])
        else:
            dirs = os.listdir(config["training"]["output_dir"])
            dirs = [d for d in dirs if d.startswith(config["decoder"]["output_name"])]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{config['training']['resume_from_checkpoint']}' does not exist. Starting a new training run."
            )
            config["training"]["resume_from_checkpoint"] = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(config["training"]["output_dir"], path))
            global_step = int(path.split("-")[1])

            resume_global_step = (
                global_step * config["training"]["gradient_accumulation_steps"]
            )
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch
                * config["training"]["gradient_accumulation_steps"]
            )

    progress_bar = tqdm(
        range(global_step, config["training"]["max_train_steps"]),
        disable=not accelerator.is_local_main_process,
    )

    progress_bar.set_description("lora decoder training progress")
    for epoch in range(first_epoch, config["training"]["num_train_epochs"]):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if (
                config["training"]["resume_from_checkpoint"]
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % config["training"]["gradient_accumulation_steps"] == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                images, clip_images = batch
                images, clip_images = images.to(weight_dtype), clip_images.to(
                    weight_dtype
                )
                latents = vae.encode(images).latents
                image_embeds = image_encoder(clip_images).image_embeds

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                target = noise

                added_cond_kwargs = {"image_embeds": image_embeds}

                model_pred = unet(
                    noisy_latents, timesteps, None, added_cond_kwargs=added_cond_kwargs
                ).sample[:, :4]

                if config["training"]["snr_gamma"] is None:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    snr = compute_snr(timesteps)
                    mse_loss_weights = (
                        torch.stack(
                            [
                                snr,
                                config["training"]["snr_gamma"]
                                * torch.ones_like(timesteps),
                            ],
                            dim=1,
                        ).min(dim=1)[0]
                        / snr
                    )

                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()

                avg_loss = accelerator.gather(
                    loss.repeat(config["training"]["train_batch_size"])
                ).mean()
                train_loss += (
                    avg_loss.item() / config["training"]["gradient_accumulation_steps"]
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers.parameters()
                    accelerator.clip_grad_norm_(
                        params_to_clip, config["training"]["max_grad_norm"]
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % config["training"]["checkpointing_steps"] == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(
                            config["training"]["output_dir"],
                            f"{config['decoder']['output_name']}-{global_step}",
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"saved state to {save_path}")

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            progress(
                progress=(global_step, config["training"]["max_train_steps"]),
                unit="steps",
                desc="Training LoRA",
            )

            if global_step >= config["training"]["max_train_steps"]:
                break
    accelerator.end_training()


def center_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))


class LoRAImageDatasetForDecoder(torch.utils.data.Dataset):
    def __init__(
        self,
        image_folder=None,
        images_paths_csv=None,
        image_processor=None,
        img_size=512,
    ):
        assert image_folder is None or images_paths_csv is None
        self.image_processor = image_processor
        self.img_size = img_size
        if images_paths_csv is not None:
            self.paths = pd.read_csv(images_paths_csv)["image_name"].values
        else:
            self.paths = [
                os.path.join(image_folder, path)
                for path in os.listdir(image_folder)
                if ".jpg" in path.lower() or ".png" in path.lower()
            ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i])
        clip_image = self.image_processor(img)
        img = center_crop(img)
        img = img.resize(
            (self.img_size, self.img_size), resample=Image.BICUBIC, reducing_gap=1
        )
        img = np.array(img.convert("RGB"))
        img = img.astype(np.float32) / 127.5 - 1
        return np.transpose(img, [2, 0, 1]), clip_image.pixel_values[0]
