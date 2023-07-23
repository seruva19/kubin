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
from transformers import (
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)
from transformers.utils import ContextManagers

import diffusers
from diffusers import PriorTransformer, UnCLIPScheduler, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers

if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")

default_lora_prior_config_path = (
    "extensions/kd-training/train_modules/train_configs/config_22_lora_prior.yaml"
)


def add_default_values(lora_prior_config):
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if (
        env_local_rank != -1
        and env_local_rank != lora_prior_config["training"]["local_rank"]
    ):
        lora_prior_config["training"]["local_rank"] = env_local_rank

    return lora_prior_config


def fix_lora_prior_config(config):
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


def launch_lora_prior_training(kubin, config, progress):
    cache_dir = kubin.params("general", "cache_dir")

    config = fix_lora_prior_config(config)
    print(f"launching training of LoRA prior model with params: {config}")

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

    noise_scheduler = DDPMScheduler(
        beta_schedule="squaredcos_cap_v2", prediction_type="sample"
    )
    image_processor = CLIPImageProcessor.from_pretrained(
        config["paths"]["image_processor_path"],
        subfolder="image_processor",
        cache_dir=cache_dir,
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        config["paths"]["tokenizer_path"], subfolder="tokenizer", cache_dir=cache_dir
    )

    def deepspeed_zero_init_disabled_context_manager():
        deepspeed_plugin = (
            AcceleratorState().deepspeed_plugin
            if accelerate.state.is_initialized()
            else None
        )
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            config["paths"]["pretrained_image_encoder"],
            subfolder="image_encoder",
            torch_dtype=weight_dtype,
            cache_dir=cache_dir,
        ).eval()

        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            config["paths"]["text_encoder_path"],
            subfolder="text_encoder",
            torch_dtype=weight_dtype,
            cache_dir=cache_dir,
        ).eval()

    print("pretrained_prior_path =", config["paths"]["pretrained_prior_path"])
    prior = PriorTransformer.from_pretrained(
        config["paths"]["pretrained_prior_path"], subfolder="prior", cache_dir=cache_dir
    )

    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)

    if config["prior"]["use_ema"]:
        ema_prior = PriorTransformer.from_pretrained(
            config["paths"]["pretrained_prior_path"],
            subfolder="prior",
            cache_dir=cache_dir,
        )
        ema_prior = EMAModel(
            ema_prior.parameters(),
            model_cls=PriorTransformer,
            model_config=ema_prior.config,
        )
        ema_prior.to(accelerator.device)

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

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            if config["prior"]["use_ema"]:
                ema_prior.save_pretrained(os.path.join(output_dir, "prior_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "prior"))
                weights.pop()

        def load_model_hook(models, input_dir):
            if config["prior"]["use_ema"]:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "prior_ema"), PriorTransformer
                )
                ema_prior.load_state_dict(load_model.state_dict())
                ema_prior.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                model = models.pop()

                load_model = PriorTransformer.from_pretrained(
                    input_dir, subfolder="prior"
                )
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

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
        prior.parameters(),
        lr=config["training"]["lr"],
        betas=(config["training"]["adam_beta1"], config["training"]["adam_beta2"]),
        weight_decay=config["training"]["weight_decay"],
        eps=config["training"]["adam_epsilon"],
    )

    train_dataset = LoRAImageDatasetForPrior(
        images_paths_csv=config["dataset"]["train_images_paths_csv"],
        image_processor=image_processor,
        tokenizer=tokenizer,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["training"]["train_batch_size"],
        num_workers=config["training"]["dataloader_num_workers"],
    )
    if config["dataset"]["val_images_paths_csv"] is not None:
        do_val = True
        val_dataset = LoRAImageDatasetForPrior(
            images_paths_csv=config["dataset"]["val_images_paths_csv"],
            image_processor=image_processor,
            tokenizer=tokenizer,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
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
    clip_mean = prior.clip_mean
    clip_std = prior.clip_std
    prior.clip_mean = None
    prior.clip_std = None
    prior, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        prior, optimizer, train_dataloader, lr_scheduler
    )

    image_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

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
            dirs = [d for d in dirs if d.startswith(config["prior"]["output_name"])]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{config['training']['resume_from_checkpoint']}' does not exist. Starting a new training run."
            )
            config["training"]["resume_from_checkpoint"] = None
        else:
            accelerator.print(f"resuming from checkpoint: {path}")
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

    progress_bar.set_description("lora prior training progress")
    clip_mean = clip_mean.to(weight_dtype).to(accelerator.device)
    clip_std = clip_std.to(weight_dtype).to(accelerator.device)
    for epoch in range(first_epoch, config["training"]["num_train_epochs"]):
        prior.train()
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

            with accelerator.accumulate(prior):
                text_input_ids, text_mask, clip_images = batch
                text_input_ids, text_mask, clip_images = (
                    text_input_ids,
                    text_mask,
                    clip_images.to(weight_dtype),
                )
                with torch.no_grad():
                    text_encoder_output = text_encoder(text_input_ids)
                    prompt_embeds = text_encoder_output.text_embeds
                    text_encoder_hidden_states = text_encoder_output.last_hidden_state

                    image_embeds = image_encoder(clip_images).image_embeds
                    noise = torch.randn_like(image_embeds)
                    bsz = image_embeds.shape[0]
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=image_embeds.device,
                    )
                    timesteps = timesteps.long()
                    image_embeds = (image_embeds - clip_mean) / clip_std
                    noisy_latents = noise_scheduler.add_noise(
                        image_embeds, noise, timesteps
                    )

                    target = image_embeds

                model_pred = prior(
                    noisy_latents,
                    timestep=timesteps,
                    proj_embedding=prompt_embeds,
                    encoder_hidden_states=text_encoder_hidden_states,
                    attention_mask=text_mask,
                ).predicted_image_embedding

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
                    accelerator.clip_grad_norm_(
                        prior.parameters(), config["training"]["max_grad_norm"]
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if config["prior"]["use_ema"]:
                    ema_prior.step(prior.parameters())
                progress_bar.update(1)
                global_step += 1

                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % config["training"]["checkpointing_steps"] == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(
                            config["training"]["output_dir"],
                            f"{config['prior']['output_name']}-{global_step}",
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


class LoRAImageDatasetForPrior(torch.utils.data.Dataset):
    def __init__(self, images_paths_csv=None, image_processor=None, tokenizer=None):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        df = pd.read_csv(images_paths_csv)
        self.paths = df["image_name"].values
        self.captions = df["caption"].values

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i])
        clip_image = self.image_processor(img)
        text_inputs = self.tokenizer(
            self.captions[i],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids[0]
        text_mask = text_inputs.attention_mask.bool()[0]

        return text_input_ids, text_mask, clip_image.pixel_values[0]
