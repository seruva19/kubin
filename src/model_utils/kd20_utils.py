import os
from huggingface_hub import hf_hub_download
from copy import deepcopy

from kandinsky2.configs import CONFIG_2_0
from kandinsky2.kandinsky2_model import Kandinsky2
import torch


def patch_ae():
    def patched_init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        self.load_state_dict(sd, strict=False)

    from kandinsky2.vqgan.autoencoder import AutoencoderKL

    AutoencoderKL.init_from_ckpt = patched_init_from_ckpt
    print(f"ae was patched")


def get_kandinsky2_0(
    device,
    task_type="text2img",
    cache_dir="/tmp/kandinsky2",
    use_auth_token=None,
):
    cache_dir = os.path.join(cache_dir, "2_0")

    config = deepcopy(CONFIG_2_0)
    if task_type == "inpainting":
        model_name = "Kandinsky-2-0-inpainting.pt"
    elif task_type == "text2img":
        model_name = "Kandinsky-2-0.pt"
    else:
        raise ValueError("Only text2img, img2img and inpainting is available")

    hf_hub_download(
        repo_id="ai-forever/Kandinsky_2.0",
        filename=model_name,
        local_dir=cache_dir,
        force_filename=model_name,
        use_auth_token=use_auth_token,
    )

    for name in [
        "config.json",
        "pytorch_model.bin",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]:
        hf_hub_download(
            repo_id="ai-forever/Kandinsky_2.0",
            local_dir=cache_dir,
            filename=f"text_encoder1/{name}",
            force_filename=name,
            use_auth_token=use_auth_token,
        )

    for name in [
        "config.json",
        "pytorch_model.bin",
        "spiece.model",
        "special_tokens_map.json",
        "tokenizer_config.json",
    ]:
        hf_hub_download(
            repo_id="ai-forever/Kandinsky_2.0",
            filename=f"text_encoder2/{name}",
            local_dir=cache_dir,
            force_filename=name,
            use_auth_token=use_auth_token,
        )

    hf_hub_download(
        repo_id="ai-forever/Kandinsky_2.0",
        filename=f"vae.ckpt",
        local_dir=cache_dir,
        force_filename="vae.ckpt",
        use_auth_token=use_auth_token,
    )

    config["text_enc_params1"]["model_path"] = os.path.join(cache_dir, "text_encoder1")
    config["text_enc_params2"]["model_path"] = os.path.join(cache_dir, "text_encoder2")
    config["tokenizer_name1"] = os.path.join(cache_dir, "text_encoder1")
    config["tokenizer_name2"] = os.path.join(cache_dir, "text_encoder2")
    config["image_enc_params"]["params"]["ckpt_path"] = os.path.join(
        cache_dir, "vae.ckpt"
    )
    unet_path = os.path.join(cache_dir, model_name)

    patch_ae()
    model = Kandinsky2(config, unet_path, device, task_type)
    return model
