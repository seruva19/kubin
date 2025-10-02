# This source code is licensed under the Apache License found in the
# LICENSE file in the current directory's parent directory.
"""
The code has been adopted from Kandinsky-5
(https://github.com/ai-forever/Kandinsky-5/blob/main/kandinsky/models/text_embedders.py)
"""

import os
from typing import List, Sequence

import torch
from transformers import (
    AutoProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

from .utils import freeze


def _ensure_module_dtype(module: torch.nn.Module, dtype: torch.dtype) -> List[str]:
    """Force all floating point parameters/buffers on the module to the given dtype."""
    module.to(dtype=dtype)
    converted: List[str] = []

    for name, param in module.named_parameters():
        if param.dtype != dtype:
            param.data = param.data.to(dtype)  # type: ignore[assignment]
            converted.append(name)

    for name, buffer in module.named_buffers():
        if torch.is_floating_point(buffer) and buffer.dtype != dtype:
            module._buffers[name] = buffer.to(dtype)  # type: ignore[index]
            converted.append(name)

    return converted


def _patch_sageattention(target_dtype: torch.dtype, source: str) -> None:
    import torch.nn.functional as F
    from sageattention import sageattn

    cast_dtype = target_dtype if target_dtype in (torch.float16, torch.bfloat16) else torch.float16

    def _sage_with_cast(q, k, v, *args, **kwargs):
        original_dtype = q.dtype
        if original_dtype not in (torch.float16, torch.bfloat16):
            q_cast = q.to(cast_dtype)
            k_cast = k.to(cast_dtype)
            v_cast = v.to(cast_dtype)
            result = sageattn(q_cast, k_cast, v_cast, *args, **kwargs)
            if isinstance(result, tuple):
                attn_output, *rest = result
                attn_output = attn_output.to(original_dtype)
                result = (attn_output, *rest)
            else:
                result = result.to(original_dtype)
            if not getattr(_sage_with_cast, "_logged", False):
                print(
                    f"[warn] {source} cast SageAttention inputs from {original_dtype} to {cast_dtype}"
                )
                _sage_with_cast._logged = True
            return result
        return sageattn(q, k, v, *args, **kwargs)

    _sage_with_cast._logged = False
    F.scaled_dot_product_attention = _sage_with_cast  # type: ignore[assignment]


class ClipTextEmbedder:
    def __init__(self, conf, device: str):
        attn_mode = os.environ.get("KD5_ATTENTION_MODE", "flash").lower()
        target_dtype = torch.float16 if device != "cpu" else torch.float32
        use_flash = os.environ.get("KD5_USE_FLASH_ATTENTION", "1") == "1"

        torch_dtype = target_dtype if target_dtype in (torch.float16, torch.bfloat16) else None
        load_kwargs = {}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype

        if use_flash and device != "cpu":
            try:
                self.model = CLIPTextModel.from_pretrained(
                    conf.checkpoint_path,
                    attn_implementation="flash_attention_2",
                    **load_kwargs,
                ).to(device)
                print("[info] CLIP text embedder using Flash Attention 2")
            except (ImportError, ValueError) as exc:
                print(
                    f"[warn] Flash Attention 2 unavailable for CLIP text embedder: {exc}\n[warn] Falling back to eager attention"
                )
                self.model = CLIPTextModel.from_pretrained(
                    conf.checkpoint_path,
                    **load_kwargs,
                ).to(device)
        else:
            self.model = CLIPTextModel.from_pretrained(
                conf.checkpoint_path,
                **load_kwargs,
            ).to(device)

        if attn_mode == "sage":
            try:
                _patch_sageattention(torch.float16, "CLIP text embedder")
                converted = []
                if device != "cpu":
                    converted = _ensure_module_dtype(self.model, torch.float16)
                if converted:
                    preview = ", ".join(converted[:5])
                    if len(converted) > 5:
                        preview += ", ..."
                    print(
                        f"[warn] SageAttention forced float16 for CLIP tensors: {preview}"
                    )
                else:
                    print("[info] CLIP model confirmed in float16 for Sage Attention")
            except ImportError:
                print(
                    "[warn] Sage Attention not available for CLIP text embedder; using default attention"
                )

        self.model = freeze(self.model)
        self.tokenizer = CLIPTokenizer.from_pretrained(conf.checkpoint_path)
        self.max_length = conf.max_length

    def __call__(self, texts: Sequence[str]):
        inputs = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self.model.device)

        model_dtype = next(self.model.parameters()).dtype
        for key in inputs:
            tensor = inputs[key]
            if tensor.dtype not in (torch.int32, torch.int64):
                inputs[key] = tensor.to(model_dtype)

        with torch.no_grad():
            pooled_embed = self.model(**inputs)["pooler_output"]
        return pooled_embed


class Qwen2_5_VLTextEmbedder:
    PROMPT_TEMPLATE = {
        "template": {
            "video": (
                "<|im_start|>system\nYou are a promt engineer. Describe the video in detail.",
                "Describe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.",
                "Describe the location of the video, main characters or objects and their action.",
                "Describe the dynamism of the video and presented actions.",
                "Name the visual style of the video: whether it is a professional footage, user generated content, some kind of animation, video game or scren content.",
                "Describe the visual effects, postprocessing and transitions if they are presented in the video.",
                "Pay attention to the order of key actions shown in the scene.<|im_end|>",
                "<|im_start|>user\n{}<|im_end|>",
            ),
            "image": (
                "<|im_start|>system\nYou are a promt engineer. Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>",
                "<|im_start|>user\n{}<|im_end|>",
            ),
        },
        "crop_start": {"video": 129, "image": 41},
    }

    def __init__(self, conf, device: str, use_torch_compile: bool = True):
        cache_dir = os.environ.get("KD50_CACHE_DIR", "./weights/")
        cache_dir = os.path.abspath(os.path.normpath(cache_dir))
        print(f"[info] Using KD50 cache directory: {cache_dir}")

        if "/" in conf.checkpoint_path and not conf.checkpoint_path.startswith("./"):
            from huggingface_hub import snapshot_download
            import yaml

            local_path = snapshot_download(
                repo_id=conf.checkpoint_path,
                cache_dir=cache_dir,
            )
            checkpoint_path = local_path
            print(f"[info] Models downloaded to: {local_path}")
            print(f"[info] Loading Qwen2.5-VL from: {checkpoint_path}")
        else:
            checkpoint_path = os.path.abspath(conf.checkpoint_path)

        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir

        attn_mode = os.environ.get("KD5_ATTENTION_MODE", "flash").lower()
        use_fa = os.environ.get("KD5_USE_FLASH_ATTENTION", "1") == "1"

        is_awq = "awq" in checkpoint_path.lower()
        dtype = torch.float16 if is_awq else torch.bfloat16

        if is_awq and device == "cpu":
            print("[warn] AWQ text embedder needs CUDA; forcing device to cuda:0")
            device = "cuda:0"

        load_kwargs = dict(torch_dtype=dtype, device_map=device, cache_dir=cache_dir)

        if use_fa and device != "cpu":
            try:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    checkpoint_path,
                    attn_implementation="flash_attention_2",
                    **load_kwargs,
                )
                print("[info] Qwen2.5-VL text embedder using Flash Attention 2")
            except (ImportError, ValueError) as exc:
                print(
                    f"[warn] Flash Attention 2 unavailable for Qwen2.5-VL: {exc}\n[warn] Falling back to eager attention"
                )
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    checkpoint_path,
                    attn_implementation="eager",
                    **load_kwargs,
                )
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                checkpoint_path,
                attn_implementation="eager",
                **load_kwargs,
            )

        if attn_mode == "sage":
            try:
                _patch_sageattention(dtype, "Qwen2.5-VL text embedder")
                converted = []
                if device != "cpu":
                    converted = _ensure_module_dtype(self.model, dtype)
                if converted:
                    preview = ", ".join(converted[:5])
                    if len(converted) > 5:
                        preview += ", ..."
                    print(
                        f"[warn] SageAttention forced dtype for Qwen tensors: {preview}"
                    )
                else:
                    print("[info] Qwen2.5-VL tensors already in expected dtype for Sage Attention")
            except ImportError:
                print(
                    "[warn] Sage Attention not available for Qwen2.5-VL; keeping default attention"
                )

        self.model = freeze(self.model)
        if use_torch_compile:
            self.model = torch.compile(self.model, dynamic=True)

        self.processor = AutoProcessor.from_pretrained(
            checkpoint_path,
            use_fast=True,
            cache_dir=cache_dir,
        )
        self.max_length = conf.max_length

    def __call__(self, texts: Sequence[str], type_of_content: str = "video"):
        template = self.PROMPT_TEMPLATE["template"][type_of_content]
        crop_start = self.PROMPT_TEMPLATE["crop_start"][type_of_content]
        prompt_template = "\n".join(template)
        full_texts = [prompt_template.format(text) for text in texts]

        max_length = self.max_length + crop_start
        inputs = self.processor(
            text=full_texts,
            images=None,
            videos=None,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        model_dtype = next(self.model.parameters()).dtype
        for key in inputs:
            tensor = inputs[key]
            if tensor.dtype not in (torch.int32, torch.int64):
                inputs[key] = tensor.to(model_dtype)

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                return_dict=True,
                output_hidden_states=True,
            )
        hidden = outputs["hidden_states"][-1][:, crop_start:]
        attention_mask = inputs["attention_mask"][:, crop_start:]
        hidden = hidden[attention_mask.bool()]
        cu_seqlens = torch.cumsum(attention_mask.sum(1), dim=0)
        cu_seqlens = torch.cat(
            [torch.zeros_like(cu_seqlens)[:1], cu_seqlens]
        ).to(dtype=torch.int32)
        return hidden, cu_seqlens


class Kandinsky5TextEmbedder:
    def __init__(self, conf, device: str = "cpu", use_torch_compile: bool = True):
        self.embedder = Qwen2_5_VLTextEmbedder(conf.qwen, device, use_torch_compile)
        self.clip_embedder = ClipTextEmbedder(conf.clip, device)
        self.conf = conf

    def encode(self, texts: Sequence[str], type_of_content: str = "image"):
        text_embeds, cu_seqlens = self.embedder(texts, type_of_content=type_of_content)
        pooled_embed = self.clip_embedder(texts)
        return {"text_embeds": text_embeds, "pooled_embed": pooled_embed}, cu_seqlens

    def to(self, device: str):
        self.embedder.model = self.embedder.model.to(device)
        self.clip_embedder.model = self.clip_embedder.model.to(device)
        return self


def get_text_embedder(conf, device: str = "cpu", use_torch_compile: bool = True):
    return Kandinsky5TextEmbedder(conf, device, use_torch_compile)
