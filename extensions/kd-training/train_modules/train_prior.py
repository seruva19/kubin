import torch
from huggingface_hub import hf_hub_url, cached_download
import pytorch_lightning as pl
import clip

from kandinsky2.model.model_creation import create_model, create_gaussian_diffusion
from kandinsky2.model.utils import get_obj_from_str
from kandinsky2.model.resample import UniformSampler
from kandinsky2.model.prior import PriorDiffusionModel, CustomizedTokenizer
from train_modules.train_utils.data.dataset_prior import create_loader
from train_modules.train_utils.train_module_pl2_1 import Decoder
from train_modules.train_utils.trainer_prior import train_prior

default_prior_config_path = (
    "extensions/kd-training/train_modules/train_configs/config_prior.yaml"
)


def add_default_values(cache_dir, config_prior):
    config_prior["params_path"] = f"{cache_dir}/2_1/prior_fp16.ckpt"
    config_prior["clip_mean_std_path"] = f"{cache_dir}/2_1/ViT-L-14_stats.th"
    config_prior["data"]["train"]["df_path"] = "train/dataset.csv"
    config_prior["num_epochs"] = 2
    config_prior["save_path"] = "train/checkpoint"
    config_prior["save_name"] = "prior_ckpt"
    return config_prior


def get_prior_model(kubin):
    cache_dir = f"{kubin.root}/{kubin.options.cache_dir}/2_1"

    prior_name = "prior_fp16.ckpt"
    config_file_url = hf_hub_url(
        repo_id="sberbank-ai/Kandinsky_2.1", filename=prior_name
    )
    cached_download(config_file_url, cache_dir=cache_dir, force_filename=prior_name)

    config_file_url = hf_hub_url(
        repo_id="sberbank-ai/Kandinsky_2.1", filename="ViT-L-14_stats.th"
    )
    cached_download(
        config_file_url, cache_dir=cache_dir, force_filename="ViT-L-14_stats.th"
    )


def process_prior_config(config):
    return config


def start_prior_training(kubin, config):
    print(f"launching training of prior model with params: {config}")
    config = process_prior_config(config)

    get_prior_model(kubin)

    device = config["device"]
    clip_mean, clip_std = torch.load(config["clip_mean_std_path"], map_location="cpu")
    tokenizer = CustomizedTokenizer()
    model = PriorDiffusionModel(
        config["model_config"],
        tokenizer,
        clip_mean,
        clip_std,
    )

    diffusion = model.create_prior_diffusion()
    print("start loading")
    if config["params_path"] != "":
        model.load_state_dict(torch.load(config["params_path"]))

    model = model.to(device)
    train_loader = create_loader(**config["data"]["train"])
    schedule_sampler = UniformSampler(diffusion)
    optimizer = get_obj_from_str(config["optim_params"]["name"])(
        model.parameters(), **config["optim_params"]["params"]
    )

    if "scheduler_params" in config:
        lr_scheduler = get_obj_from_str(config["scheduler_params"]["name"])(
            optimizer, **config["scheduler_params"]["params"]
        )
    else:
        lr_scheduler = None

    clip_model, _ = clip.load(config["clip_name"], device="cpu", jit=False)
    clip_model = clip_model.eval().to(device)
    train_prior(
        model=model,
        diffusion=diffusion,
        clip_model=clip_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        schedule_sampler=schedule_sampler,
        train_loader=train_loader,
        val_loader=None,
        num_epochs=config["num_epochs"],
        save_every=config["save_every"],
        save_name=config["save_name"],
        save_path=config["save_path"],
        device=device,
    )
