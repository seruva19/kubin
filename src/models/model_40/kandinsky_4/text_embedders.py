# This source code is licensed under the Apache License found in the
# LICENSE file in the current directory.
"""
The code has been adopted from Kandinsky-4
(https://github.com/ai-forever/Kandinsky-4/blob/main/kandinsky/model/text_embedders.py)
"""


import torch
import numpy as np
import sys
import os

from .utils import freeze


class BaseEmbedder:
    def __init__(self, conf):
        self.checkpoint_path = conf.text_embedder.params.checkpoint_path
        self.tokenizer_path = conf.text_embedder.params.tokenizer_path
        self.max_length = conf.text_embedder.tokens_length
        self.llm = None

    def to(self, device="cpu", dtype=torch.float32):
        self.llm = self.llm.to(device=device, dtype=dtype)
        return self

    def freeze(self):
        self.llm = freeze(self.llm)
        return self

    def compile(self):
        self.llm = torch.compile(self.llm)
        return self


class EmbedderWithTokenizer(BaseEmbedder):

    def __init__(self, conf):
        super().__init__(conf)
        self.tokenizer = None

    def tokenize(self, text):
        model_input = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
            padding="max_length",
            return_tensors="pt",
        )
        return model_input.input_ids.to(self.llm.device)

    def __call__(self, text):
        return self.llm(self.tokenize(text), output_hidden_states=True)[0]


class T5TextEmbedder(EmbedderWithTokenizer):

    def __init__(self, conf):
        from transformers import T5EncoderModel, T5Tokenizer

        super().__init__(conf)

        self.llm = T5EncoderModel.from_pretrained(self.checkpoint_path)
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.tokenizer_path, clean_up_tokenization_spaces=False
        )


def get_text_embedder(conf):
    return T5TextEmbedder(conf)
