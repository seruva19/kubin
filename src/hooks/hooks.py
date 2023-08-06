import subprocess
import sys
import os
import importlib.util
import sys
import yaml
from enum import Enum


class HOOK(Enum):
    BEFORE_PREPARE_MODEL = "BEFORE_PREPARE_MODEL"
    BEFORE_PREPARE_PARAMS = "BEFORE_PREPARE_PARAMS"
    BEFORE_PREPARE_EMBEDS = "BEFORE_PREPARE_EMBEDS"
    BEFORE_PREPARE_DECODER = "BEFORE_PREPARE_DECODER"
    BEFORE_BATCH_SAVE = "BEFORE_BATCH_SAVE"
    BEFORE_TASK_QUIT = "BEFORE_TASK_QUIT"

    BEFORE_FLUSH_MODEL = "BEFORE_FLUSH_MODEL"
    AFTER_FLUSH_MODEL = "AFTER_FLUSH_MODEL"


class HookStore:
    def __init__(self, params):
        self.params = params
        self.params.HOOK = HOOK

        self.registered_hooks = {".core": lambda t, **k: None}

    def register_hook(self, consumer, hook_fn):
        self.registered_hooks[consumer] = hook_fn

    def call(self, hook_type, **hook_info):
        for consumer, hook in self.registered_hooks.items():
            hook(hook_type, **hook_info)
