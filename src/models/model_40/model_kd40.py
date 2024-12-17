import gc
import re
import torch
import torch.backends
import torch
import os

from params import KubinParams
from utils.file_system import save_output
from utils.logging import k_log

from models.model_40.kandinsky_4.pipelines import get_T2V_pipeline
from models.model_40.kandinsky_4.t2v_pipeline import Kandinsky4T2VPipeline
from models.model_40.model_kd40_env import Model_KD40_Environment
from utils.file_system import save_output


class Model_KD40:
    def __init__(self, params: KubinParams):
        k_log("using pipeline: native (4.0)")

        self.kparams = params

        self.use_flash_t2v_pipeline = False
        self.t2v_pipe: Kandinsky4T2VPipeline | None = None

        self.i2v_pipe = None
        self.v2a_pipe = None

    def prepare_model(self, task, use_t2v_flash, kd40_conf):
        k_log(f"task queued: {task}")
        assert task in ["text2video"]

        cache_dir = os.path.join(self.kparams("general", "cache_dir"), "kandinsky-4")
        device = self.kparams("general", "device")

        use_flash_t4v_pipeline_before = self.use_flash_t2v_pipeline
        self.use_flash_t2v_pipeline = use_t2v_flash

        environment = Model_KD40_Environment().from_config(self.kparams)
        environment.set_conf(kd40_conf)

        device_map = {
            "dit": torch.device(device),
            "vae": torch.device(device),
            "text_embedder": torch.device(device),
        }

        if task == "text2video":
            if (
                self.t2v_pipe is None
                or use_flash_t4v_pipeline_before != self.use_flash_t2v_pipeline
            ):
                self.flush(task)

                if not self.use_flash_t2v_pipeline:
                    print(f"Only flash pipeline currently available!")
                    self.use_flash_t2v_pipeline = True

                if self.use_flash_t2v_pipeline:
                    k_log(f"preparing K4.0-T2V-F pipeline")

                    self.t2v_pipe = get_T2V_pipeline(
                        device_map=device_map,
                        resolution=512,
                        cache_dir=cache_dir,
                        dit_path=None,
                        text_encoder_path=None,
                        tokenizer_path=None,
                        vae_path=None,
                        scheduler_path=None,
                        conf_path=None,
                        environment=environment,
                    )

    def t2v(self, params):
        task = "text2video"

        use_t2v_flash = params["pipeline_args"]["use_flash"]
        kd40_conf = params["pipeline_args"]["kd40_conf"]

        prompt = params["prompt"]
        time_length = params["time_length"]

        resolution = params["resolution"]
        match = re.search(r"\[(\d+)x(\d+)\]", resolution)
        width = int(match.group(1))
        height = int(match.group(2))

        seed = params["seed"]
        generate_image = params["generate_image"]

        if generate_image:
            time_length = 0

        self.prepare_model(task, use_t2v_flash, kd40_conf)

        save_image_path = None
        save_video_path = os.path.join(
            params.get(
                ".output_dir",
                os.path.join(self.kparams("general", "output_dir"), task),
            ),
            f"k4v{'-f-' if use_t2v_flash else '-'}{'_'.join(prompt.split()[:5])}.mp4",
        )

        os.makedirs(os.path.dirname(save_video_path), exist_ok=True)
        result = self.t2v_pipe(
            text=prompt,
            save_path=save_video_path,
            bs=1,
            time_length=time_length,
            width=width,
            height=height,
            seed=None if seed == -1 else seed,
            return_frames=generate_image,
        )

        if generate_image:
            save_image_path = save_images(self.kparams, params, task, result)
            save_video_path = None
        k_log("text2video task: done")
        return save_video_path, save_image_path

    def i2v(self, params):
        task = "img2video"
        return None

    def v2a(self, params):
        task = "video2audio"
        return None

    def flush(self, task=None):
        cleared = False

        def _flush_pipe(pipe_name):
            pipe = getattr(self, pipe_name, None)
            if pipe is not None:
                k_log(f"{pipe_name} -> cpu")
                for component in ["text_embedder", "dit", "vae"]:
                    comp = getattr(pipe, component, None)
                    if comp is not None:
                        comp.to("cpu")
                setattr(self, pipe_name, None)
                return True
            return False

        task_map = {
            "image2video": ["t2v_pipe", "v2a_pipe"],
            "video2audio": ["t2v_pipe", "i2v_pipe"],
            "text2video": ["i2v_pipe", "v2a_pipe"],
        }

        pipes_to_flush = task_map.get(task, ["t2v_pipe", "i2v_pipe", "v2a_pipe"])
        for pipe_name in pipes_to_flush:
            cleared |= _flush_pipe(pipe_name)

        if cleared:
            gc.collect()
            device = self.kparams("general", "device")
            if device.startswith("cuda") and torch.cuda.is_available():
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()


def save_images(kparams, params, task, batch):
    if params.get("task", None) is None:
        params["task"] = task

    output_dir = params.get(
        ".output_dir",
        os.path.join(kparams("general", "output_dir"), task),
    )

    saved_batch = save_output(output_dir, batch, params)
    return saved_batch
