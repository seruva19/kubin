import os
from huggingface_hub import hf_hub_url, cached_download
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
import clip

from kandinsky2.model.model_creation import create_model, create_gaussian_diffusion
from kandinsky2.model.text_encoders import TextEncoder
from kandinsky2.model.utils import get_obj_from_str
from kandinsky2.vqgan.autoencoder import VQModelInterface, AutoencoderKL, MOVQ
from kandinsky2.model.resample import UniformSampler

from train_modules.train_utils.train_module_pl2_1 import Decoder
from train_modules.train_utils.trainer_2_1_uclip import train_unclip
from train_modules.train_utils.data.dataset_unclip_2_1 import create_loader
from train_modules.train_utils.utils import freeze_decoder

default_unclip_config_path = (
    "extensions/kd-training/train_modules/train_configs/config_unclip_2_1.yaml"
)


def add_default_values(cache_dir, config_unclip):
    config_unclip["params_path"] = f"{cache_dir}/2_1/decoder_fp16.ckpt"
    config_unclip["image_enc_params"]["ckpt_path"] = f"{cache_dir}/2_1/movq_final.ckpt"
    config_unclip["text_enc_params"]["model_path"] = f"{cache_dir}/2_1/text_encoder"
    config_unclip["data"]["train"]["tokenizer_name"] = f"{cache_dir}/2_1/text_encoder"
    config_unclip["data"]["train"]["df_path"] = "train/dataset.csv"
    config_unclip["num_epochs"] = 2
    config_unclip["save_path"] = "train/checkpoint"
    config_unclip["save_name"] = "unclip_ckpt"
    return config_unclip


def drop_first_layer(path):
    d = {}
    state_dict = torch.load(path)
    for key in state_dict.keys():
        if key != "input_blocks.0.0.weight":
            d[key] = state_dict[key]
    return d


def get_unclip_model(kubin, inpainting):
    cache_dir = f"{kubin.root}/{kubin.options.cache_dir}/2_1"

    if not inpainting:
        model_name = "decoder_fp16.ckpt"
        config_file_url = hf_hub_url(
            repo_id="sberbank-ai/Kandinsky_2.1", filename=model_name
        )
    else:
        model_name = "inpainting_fp16.ckpt"
        config_file_url = hf_hub_url(
            repo_id="sberbank-ai/Kandinsky_2.1", filename=model_name
        )

    cached_download(
        config_file_url,
        cache_dir=cache_dir,
        force_filename=model_name,
        use_auth_token=None,  # type: ignore
    )

    cache_dir_text_en = os.path.join(cache_dir, "text_encoder")
    for name in [
        "config.json",
        "pytorch_model.bin",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]:
        config_file_url = hf_hub_url(
            repo_id="sberbank-ai/Kandinsky_2.1", filename=f"text_encoder/{name}"
        )
        cached_download(
            config_file_url,
            cache_dir=cache_dir_text_en,
            force_filename=name,
            use_auth_token=None,  # type: ignore
        )

    config_file_url = hf_hub_url(
        repo_id="sberbank-ai/Kandinsky_2.1", filename="movq_final.ckpt"
    )
    cached_download(
        config_file_url,
        cache_dir=cache_dir,
        force_filename="movq_final.ckpt",
        use_auth_token=None,  # type: ignore
    )


def process_unclip_config(config):
    return config


def start_unclip_training(kubin, config):
    print(f"launching training of unclip model with params: {config}")
    config = process_unclip_config(config)

    get_unclip_model(kubin, config["inpainting"])

    device = config["device"]
    model = create_model(**config["model_config"])
    diffusion = create_gaussian_diffusion(**config["diffusion_config"])

    print("start loading")

    if config["params_path"] != "":
        if config["drop_first_layer"]:
            model.load_state_dict(drop_first_layer(config["params_path"]), strict=False)
        else:
            model.load_state_dict(torch.load(config["params_path"]))

    model = freeze_decoder(model, **config["freeze"]).to(device)
    train_loader = create_loader(**config["data"]["train"])

    image_encoder = MOVQ(**config["image_enc_params"]["params"]).half()
    image_encoder.load_state_dict(torch.load(config["image_enc_params"]["ckpt_path"]))
    image_encoder = image_encoder.eval().to(device)

    schedule_sampler = UniformSampler(diffusion)
    text_encoder = TextEncoder(**config["text_enc_params"]).eval().half().to(device)
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
    clip_model.transformer = None  # type: ignore
    clip_model.positional_embedding = None  # type: ignore
    clip_model.ln_final = None  # type: ignore
    clip_model.token_embedding = None  # type: ignore
    clip_model.text_projection = None  # type: ignore
    clip_model = clip_model.eval().to(device)

    train_unclip(
        unet=model,
        diffusion=diffusion,
        image_encoder=image_encoder,
        clip_model=clip_model,
        text_encoder=text_encoder,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        schedule_sampler=schedule_sampler,
        train_loader=train_loader,
        val_loader=None,
        scale=config["image_enc_params"]["scale"],
        num_epochs=config["num_epochs"],
        save_every=config["save_every"],
        save_name=config["save_name"],
        save_path=config["save_path"],
        inpainting=config["inpainting"],
        device=device,
    )
