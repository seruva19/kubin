from typing import List, Optional, Union
import torch

from diffusers import (
    KandinskyV22PriorPipeline,
    KandinskyV22PriorEmb2EmbPipeline,
    KandinskyV22Pipeline,
    KandinskyV22Img2ImgPipeline,
    KandinskyV22InpaintPipeline,
    KandinskyV22ControlnetPipeline,
    KandinskyV22ControlnetImg2ImgPipeline,
)

from diffusers.utils import logging
from utils.logging import k_log

logger = logging.get_logger(__name__)


def patch_pipelines():
    k_log("patching default diffusers pipelines")
    replace_encode_prompt_fn()


def replace_encode_prompt_fn():
    KandinskyV22PriorPipeline._encode_prompt = patched_encode_prompt


def patched_encode_prompt(
    self,
    prompt,
    device,
    num_images_per_prompt,
    do_classifier_free_guidance,
    negative_prompt=None,
):
    batch_size = len(prompt) if isinstance(prompt, list) else 1

    text_inputs = self.tokenizer(
        prompt,
        padding="max_length",
        max_length=self.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    text_mask = text_inputs.attention_mask.bool().to(device)

    untruncated_ids = self.tokenizer(
        prompt, padding="longest", return_tensors="pt"
    ).input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
        text_input_ids, untruncated_ids
    ):
        removed_text = self.tokenizer.batch_decode(
            untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
        )
        logger.warning(
            "The following part of your input was truncated because CLIP can only handle sequences up to"
            f" {self.tokenizer.model_max_length} tokens: {removed_text}"
        )
        text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]

    text_encoder_output = self.text_encoder(text_input_ids.to(device))

    prompt_embeds = text_encoder_output.text_embeds
    text_encoder_hidden_states = text_encoder_output.last_hidden_state

    prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
    text_encoder_hidden_states = text_encoder_hidden_states.repeat_interleave(
        num_images_per_prompt, dim=0
    )
    text_mask = text_mask.repeat_interleave(num_images_per_prompt, dim=0)

    if do_classifier_free_guidance:
        uncond_tokens: List[str]
        if negative_prompt is None:
            uncond_tokens = [""] * batch_size
        elif type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = negative_prompt

        uncond_input = self.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_text_mask = uncond_input.attention_mask.bool().to(device)
        negative_prompt_embeds_text_encoder_output = self.text_encoder(
            uncond_input.input_ids.to(device)
        )

        negative_prompt_embeds = negative_prompt_embeds_text_encoder_output.text_embeds
        uncond_text_encoder_hidden_states = (
            negative_prompt_embeds_text_encoder_output.last_hidden_state
        )

        seq_len = negative_prompt_embeds.shape[1]
        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt)
        negative_prompt_embeds = negative_prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len
        )

        seq_len = uncond_text_encoder_hidden_states.shape[1]
        uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.repeat(
            1, num_images_per_prompt, 1
        )
        uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )
        uncond_text_mask = uncond_text_mask.repeat_interleave(
            num_images_per_prompt, dim=0
        )

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        text_encoder_hidden_states = torch.cat(
            [uncond_text_encoder_hidden_states, text_encoder_hidden_states]
        )

        text_mask = torch.cat([uncond_text_mask, text_mask])

    return prompt_embeds, text_encoder_hidden_states, text_mask
