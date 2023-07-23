from dataclasses import dataclass, fields
import os
from huggingface_hub import hf_hub_url, cached_download
from copy import deepcopy
from omegaconf.dictconfig import DictConfig


@dataclass
class KandinskyCheckpoint:
    prior_model_dir: str = "2_1"
    prior_model_name: str = "prior_fp16.ckpt"
    decoder_model_dir: str = "2_1"
    decoder_model_name: str = "decoder_fp16.ckpt"
    inpaint_model_dir: str = "2_1"
    inpaint_model_name: str = "inpainting_fp16.ckpt"

    def base_checkpoints_path(self, cache_dir):
        return [
            os.path.normpath(
                os.path.join(
                    cache_dir,
                    self.prior_model_dir,
                    self.prior_model_name,
                )
            ),
            os.path.normpath(
                os.path.join(
                    cache_dir,
                    self.decoder_model_dir,
                    self.decoder_model_name,
                )
            ),
            os.path.normpath(
                os.path.join(
                    cache_dir,
                    self.inpaint_model_dir,
                    self.inpaint_model_name,
                )
            ),
        ]

    def is_base_prior(self) -> bool:
        default_ckpt = KandinskyCheckpoint()
        return (
            self.prior_model_dir == default_ckpt.prior_model_dir
            and self.prior_model_name == default_ckpt.prior_model_name
        )

    def is_base_decoder(self) -> bool:
        default_ckpt = KandinskyCheckpoint()
        return (
            self.decoder_model_dir == default_ckpt.decoder_model_dir
            and self.decoder_model_name == default_ckpt.decoder_model_name
        )

    def is_base_inpaint(self) -> bool:
        default_ckpt = KandinskyCheckpoint()
        return (
            self.inpaint_model_dir == default_ckpt.inpaint_model_dir
            and self.inpaint_model_name == default_ckpt.inpaint_model_name
        )


def get_checkpoint(
    device,
    task_type="text2img",
    cache_dir="/tmp/kandinsky2",
    use_auth_token=None,
    use_flash_attention=False,
    checkpoint_info=KandinskyCheckpoint(),
):
    from kandinsky2.configs import CONFIG_2_0, CONFIG_2_1
    from kandinsky2.kandinsky2_model import Kandinsky2
    from kandinsky2.kandinsky2_1_model import Kandinsky2_1

    cache_dir = os.path.join(cache_dir, "2_1")

    config = DictConfig(deepcopy(CONFIG_2_1))
    config["model_config"]["use_flash_attention"] = use_flash_attention
    config["model_config"]["use_fp16"] = device != "cpu"

    if task_type == "text2img":
        model_name = checkpoint_info.decoder_model_name
        if checkpoint_info.is_base_decoder():
            print("loading default decoder model")
            cache_model_name = os.path.join(cache_dir, model_name)
            config_file_url = hf_hub_url(
                repo_id="sberbank-ai/Kandinsky_2.1", filename=model_name
            )
            cached_download(
                config_file_url,
                cache_dir=cache_dir,
                force_filename=model_name,
                use_auth_token=use_auth_token,
            )

        else:
            print(
                f"loading custom decoder model '{checkpoint_info.decoder_model_name}' at '{checkpoint_info.decoder_model_dir}'"
            )
            cache_model_name = os.path.join(
                checkpoint_info.decoder_model_dir, checkpoint_info.decoder_model_name
            )

    elif task_type == "inpainting":
        model_name = checkpoint_info.inpaint_model_name
        if checkpoint_info.is_base_inpaint():
            print("loading default inpaint decoder model")
            cache_model_name = os.path.join(cache_dir, model_name)
            config_file_url = hf_hub_url(
                repo_id="sberbank-ai/Kandinsky_2.1", filename=model_name
            )

            cached_download(
                config_file_url,
                cache_dir=cache_dir,
                force_filename=model_name,
                use_auth_token=use_auth_token,
            )
        else:
            print(
                f"loading custom inpaint decoder model '{checkpoint_info.inpaint_model_name}' at '{checkpoint_info.inpaint_model_dir}'"
            )

            cache_model_name = os.path.join(
                checkpoint_info.inpaint_model_dir, checkpoint_info.inpaint_model_name
            )

    prior_name = checkpoint_info.prior_model_name
    if checkpoint_info.is_base_prior():
        print("loading default prior model")
        cache_prior_name = os.path.join(cache_dir, prior_name)
        config_file_url = hf_hub_url(
            repo_id="sberbank-ai/Kandinsky_2.1", filename=prior_name
        )
        cached_download(
            config_file_url,
            cache_dir=cache_dir,
            force_filename=prior_name,
            use_auth_token=use_auth_token,
        )
    else:
        print(
            f"loading custom prior model '{checkpoint_info.prior_model_name}' at '{checkpoint_info.prior_model_dir}'"
        )
        cache_prior_name = os.path.join(
            checkpoint_info.prior_model_dir, checkpoint_info.prior_model_name
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
            use_auth_token=use_auth_token,
        )

    config_file_url = hf_hub_url(
        repo_id="sberbank-ai/Kandinsky_2.1", filename="movq_final.ckpt"
    )
    cached_download(
        config_file_url,
        cache_dir=cache_dir,
        force_filename="movq_final.ckpt",
        use_auth_token=use_auth_token,
    )

    config_file_url = hf_hub_url(
        repo_id="sberbank-ai/Kandinsky_2.1", filename="ViT-L-14_stats.th"
    )
    cached_download(
        config_file_url,
        cache_dir=cache_dir,
        force_filename="ViT-L-14_stats.th",
        use_auth_token=use_auth_token,
    )

    config["tokenizer_name"] = cache_dir_text_en
    config["text_enc_params"]["model_path"] = cache_dir_text_en
    config["prior"]["clip_mean_std_path"] = os.path.join(cache_dir, "ViT-L-14_stats.th")
    config["image_enc_params"]["ckpt_path"] = os.path.join(cache_dir, "movq_final.ckpt")

    # import gc
    # gc.collect()

    model = Kandinsky2_1(
        config, cache_model_name, cache_prior_name, device, task_type=task_type
    )

    return model
