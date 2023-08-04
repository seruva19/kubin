from file_tools import calculate_file_hash, load_model_from_path
import torch
import os
from diffusers.models.attention_processor import (
    LoRAAttnProcessor,
    LoRAAttnAddedKVProcessor,
    AttnProcessor,
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)


def bind_unbind_networks(kubin, model_config, prior, decoder, params, networks_info):
    bind_unbind_lora(kubin, model_config, prior, decoder, params, networks_info["lora"])


def bind_unbind_lora(kubin, model_config, prior, decoder, params, loras):
    params_session = params[".session"]
    lora_info = loras.get(params_session, None)
    lora_binded = model_config.get(".lora", None)

    if lora_info is not None:
        lora_enabled = lora_info["enabled"]
        lora_prior_path = lora_info["prior"]
        lora_decoder_path = lora_info["decoder"]

        if lora_enabled:
            if lora_binded:
                b_prior, b_decoder = lora_binded

                if b_prior != lora_prior_path:
                    remove_lora_from_prior(prior)
                    print("removing previous LoRA attention layers from prior")

                if b_decoder != lora_decoder_path:
                    remove_lora_from_decoder(decoder)
                    print("removing previous LoRA attention layers from decoder")

            else:
                if lora_prior_path is not None:
                    apply_lora_to_prior(kubin, lora_prior_path, prior)
                    params[
                        "lora_prior"
                    ] = f"{os.path.basename(lora_prior_path)} [{calculate_file_hash(lora_prior_path)}]"
                    print(
                        f"applying prior LoRA attention layers from {lora_prior_path}"
                    )
                else:
                    print(f"no prior LoRA path declared")

                if lora_decoder_path is not None:
                    apply_lora_to_decoder(kubin, lora_decoder_path, decoder)
                    params[
                        "lora_decoder"
                    ] = f"{os.path.basename(lora_decoder_path)} [{calculate_file_hash(lora_decoder_path)}]"
                    print(
                        f"applying decoder LoRA attention layers from {lora_decoder_path}"
                    )
                else:
                    print(f"no decoder LoRA path declared")

                model_config[".lora"] = lora_prior_path, lora_decoder_path

        else:
            if lora_binded:
                b_prior, b_decoder = lora_binded
                if b_prior is not None:
                    remove_lora_from_prior(prior)
                    print("discarding LoRA attention layers from prior")
                if b_decoder is not None:
                    remove_lora_from_decoder(decoder)
                    print("discarding LoRA attention layers from decoder")
                model_config.pop(".lora")


def apply_lora_to_prior(kubin, lora_prior_path, prior):
    device = (
        "cpu"
        if kubin.params("diffusers", "run_prior_on_cpu")
        else kubin.params("general", "device")
    )

    lora_model = load_model_from_path(lora_prior_path)
    rank, hidden_size = get_rank_and_hidden_size(lora_model)
    lora_attn_procs = {}

    for name in prior.prior.attn_processors.keys():
        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size, rank=rank
        ).to(device)

    prior.prior.set_attn_processor(lora_attn_procs)
    prior.prior.load_state_dict(lora_model, strict=False)


def remove_lora_from_prior(prior):
    prior.prior.set_default_attn_processor()


def get_rank_and_hidden_size(lora_model):
    # there should be a better way to do it, but anyway
    return tuple(lora_model[list(lora_model.keys())[0]].size())


def apply_lora_to_decoder(kubin, lora_decoder_path, decoder):
    device = kubin.params("general", "device")

    lora_attn_procs = {}
    lora_model = torch.load(lora_decoder_path)
    rank, _ = get_rank_and_hidden_size(lora_model)

    for name in decoder.unet.attn_processors.keys():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else decoder.unet.config.cross_attention_dim
        )

        if name.startswith("mid_block"):
            hidden_size = decoder.unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(decoder.unet.config.block_out_channels))[
                block_id
            ]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = decoder.unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnAddedKVProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank,
        ).to(device)

    decoder.unet.set_attn_processor(lora_attn_procs)
    decoder.unet.load_state_dict(lora_model, strict=False)


def remove_lora_from_decoder(decoder):
    # decoder.unet.set_default_attn_processor()
    unet_attention_classes = {
        type(processor) for _, processor in decoder.unet.attn_processors.items()
    }
    LORA_ATTENTION_PROCESSORS = (
        LoRAAttnProcessor,
        LoRAAttnProcessor2_0,
        LoRAXFormersAttnProcessor,
        LoRAAttnAddedKVProcessor,
    )

    if unet_attention_classes.issubset((LORA_ATTENTION_PROCESSORS)):
        if (
            len(unet_attention_classes) > 1
            or LoRAAttnAddedKVProcessor in unet_attention_classes
        ):
            decoder.unet.set_default_attn_processor()
        else:
            regular_attention_classes = {
                LoRAAttnProcessor: AttnProcessor,
                LoRAAttnProcessor2_0: AttnProcessor2_0,
                LoRAXFormersAttnProcessor: XFormersAttnProcessor,
            }
            [attention_proc_class] = unet_attention_classes
            decoder.unet.set_attn_processor(
                regular_attention_classes[attention_proc_class]()
            )

        for _, module in decoder.unet.named_modules():
            if hasattr(module, "set_lora_layer"):
                module.set_lora_layer(None)
