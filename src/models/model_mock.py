import base64
import os
from time import sleep
from PIL import Image, ImageOps
from io import BytesIO
import numpy as np

from hooks.hooks import HOOK
from params import KubinParams
from progress import report_progress
from utils.image import create_inpaint_targets
from utils.logging import k_log

# used for testing


class Model_Mock:
    def __init__(self, params: KubinParams):
        self.params = params
        k_log("activating pipeline: mock")

        self.prior = type("obj", (object,), {"model": "prior"})
        self.decoder = type("obj", (object,), {"model": "decoder"})

        self.config = {}

        self.params.hook_store.register_hook(
            ".mock",
            lambda t, **k: k_log(f"mock model received hook event: {t}"),
        )

    def prepare_model(self, task):
        assert task in [
            "text2img",
            "text2img_cnet",
            "img2img",
            "img2img_cnet",
            "mix",
            "mix_cnet",
            "inpainting",
            "outpainting",
        ]

        k_log(f"preparing mock for {task}")
        return self.prior, self.decoder

    def flush(self, target=None):
        self.params.hook_store.call(
            HOOK.BEFORE_FLUSH_MODEL,
            **{"model": self, "target": target},
        )
        k_log(f"mock memory freed")
        self.params.hook_store.call(
            HOOK.AFTER_FLUSH_MODEL,
            **{"model": self, "target": target},
        )

        self.config = {}

    def prepare_params(self, params):
        k_log(params)
        k_log("mock seed generated")

    def t2i(self, params):
        task = "text2img"
        params[".ui-task"] = task

        self.params.hook_store.call(
            HOOK.BEFORE_PREPARE_MODEL,
            **{"model": self, "params": params, "task": task},
        )

        prior, decoder = self.prepare_model(task)

        self.params.hook_store.call(
            HOOK.BEFORE_PREPARE_PARAMS,
            **{
                "model": self,
                "params": params,
                "task": task,
                "prior": prior,
                "decoder": decoder,
            },
        )

        self.prepare_params(params)
        k_log("mock t2i executed")

        dummy_images = self.dummyImages()

        for count, _ in enumerate(dummy_images):
            report_progress(
                task, "prior", len(dummy_images), count + 1, count + 1, None
            )
            sleep(1)

        for count, _ in enumerate(dummy_images):
            report_progress(
                task, "decoder", len(dummy_images), count + 1, count + 1, None
            )
            sleep(1)

        return dummy_images

    def i2i(self, params):
        task = "img2img"
        params[".ui-task"] = task

        self.params.hook_store.call(
            HOOK.BEFORE_PREPARE_MODEL,
            **{"model": self, "params": params, "task": task},
        )

        prior, decoder = self.prepare_model(task)

        self.params.hook_store.call(
            HOOK.BEFORE_PREPARE_PARAMS,
            **{
                "model": self,
                "params": params,
                "task": task,
                "prior": prior,
                "decoder": decoder,
            },
        )

        self.prepare_params(params)
        k_log("mock i2i executed")
        return self.dummyImages()

    def mix(self, params):
        task = "mix"
        params[".ui-task"] = task

        self.params.hook_store.call(
            HOOK.BEFORE_PREPARE_MODEL,
            **{"model": self, "params": params, "task": task},
        )

        prior, decoder = self.prepare_model(task)

        self.params.hook_store.call(
            HOOK.BEFORE_PREPARE_PARAMS,
            **{
                "model": self,
                "params": params,
                "task": task,
                "prior": prior,
                "decoder": decoder,
            },
        )

        self.prepare_params(params)
        k_log("mock mix executed")
        return self.dummyImages()

    def inpaint(self, params):
        task = "inpainting"
        params[".ui-task"] = task

        self.params.hook_store.call(
            HOOK.BEFORE_PREPARE_MODEL,
            **{"model": self, "params": params, "task": task},
        )

        prior, decoder = self.prepare_model(task)

        self.params.hook_store.call(
            HOOK.BEFORE_PREPARE_PARAMS,
            **{
                "model": self,
                "params": params,
                "task": task,
                "prior": prior,
                "decoder": decoder,
            },
        )

        self.prepare_params(params)

        image_with_mask = params["image_mask"]
        image = image_with_mask["image"]
        width, height = (
            image.width if params["infer_size"] else params["w"],
            image.height if params["infer_size"] else params["h"],
        )

        output_size = (width, height)

        image = image.resize(output_size, resample=Image.LANCZOS)
        image = image.convert("RGB")

        mask = image_with_mask["mask"]
        mask = mask.resize(output_size, resample=Image.LANCZOS)
        mask = mask.convert("L")

        k_log("mock inpaint executed")
        return [image, mask]

    def outpaint(self, params):
        task = "outpainting"
        params[".ui-task"] = task

        self.params.hook_store.call(
            HOOK.BEFORE_PREPARE_MODEL,
            **{"model": self, "params": params, "task": task},
        )

        prior, decoder = self.prepare_model(task)

        self.params.hook_store.call(
            HOOK.BEFORE_PREPARE_PARAMS,
            **{
                "model": self,
                "params": params,
                "task": task,
                "prior": prior,
                "decoder": decoder,
            },
        )

        self.prepare_params(params)

        image = params["image"]
        image_w, image_h = image.size

        offset = params["offset"]

        if offset is not None:
            top, right, bottom, left = offset
            inferred_mask_size = tuple(
                a + b for a, b in zip(image.size, (left + right, top + bottom))  # type: ignore
            )[::-1]
            mask = np.zeros(inferred_mask_size, dtype=np.float32)  # type: ignore
            mask[top : image_h + top, left : image_w + left] = 1
            image = ImageOps.expand(image, border=(left, top, right, bottom), fill=0)

        else:
            x1, y1, x2, y2 = image.getbbox()
            mask = np.ones((image_h, image_w), dtype=np.float32)
            mask[0:y1, :] = 0
            mask[:, 0:x1] = 0
            mask[y2:image_h, :] = 0
            mask[:, x2:image_w] = 0

        infer_size = params["infer_size"]
        if infer_size:
            height, width = mask.shape[:2]
        else:
            width = params["w"]
            height = params["h"]

        mask_img = Image.fromarray(np.uint8(mask * 255)).resize(
            (width, height), resample=Image.LANCZOS
        )
        image = image.resize(mask_img.size, resample=Image.LANCZOS)

        k_log("mock outpaint executed")
        return [image, mask_img]

    def dummyImages(self):
        screenshots_as_dummies = [
            entry.name for entry in os.scandir("sshots") if entry.is_file()
        ]
        return [
            Image.open(f"sshots/{screenshot}") for screenshot in screenshots_as_dummies
        ]
