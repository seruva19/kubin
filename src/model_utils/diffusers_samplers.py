import numpy as np
import torch
from typing import Optional, Tuple, Union
from diffusers.schedulers.scheduling_utils import SchedulerOutput

from utils.logging import k_log
from params import KubinParams

from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DEISMultistepScheduler,
    DPMSolverSDEScheduler,
    DPMSolverSinglestepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)


def use_sampler(pipeline, sampler, task):
    k_log(f"using sampler: {sampler}")

    if sampler == "DDPM":
        # pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

        pipeline.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            beta_start=0.00085,
            beta_end=0.012,
            clip_sample=task
            not in ["inpainting", "outpainting", "cnet_inpainting", "cnet_outpainting"],
            steps_offset=1,
            prediction_type="epsilon",
            thresholding=False,
            trained_betas=None,
            variance_type="fixed_small",
            dynamic_thresholding_ratio=0.995,
            clip_sample_range=1,
            sample_max_value=1,
            timestep_spacing="leading",
        )

    elif sampler == "DDIM":
        pipeline.scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            beta_start=0.001,
            beta_end=0.1,
            trained_betas=None,
            clip_sample=False,
            set_alpha_to_one=True,
            steps_offset=1,
            prediction_type="v_prediction",
            thresholding=False,
            dynamic_thresholding_ratio=0.995,
            clip_sample_range=1,
            sample_max_value=1,
            timestep_spacing="trailing",
            rescale_betas_zero_snr=True,
        )

    elif sampler == "DPMSM":
        pipeline.scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            beta_start=0.00085,
            beta_end=0.012,
            trained_betas=None,
            solver_order=2,
            prediction_type="epsilon",
            thresholding=False,
            dynamic_thresholding_ratio=0.995,
            sample_max_value=1,
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            lower_order_final=True,
            use_karras_sigmas=False,
            lambda_min_clipped=-float("inf"),
            variance_type=None,
            timestep_spacing="linspace",
            steps_offset=1,
        )

    elif sampler == "DEISM":
        pipeline.scheduler = DEISMultistepScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            beta_start=0.00085,
            beta_end=0.012,
            trained_betas=None,
            solver_order=2,
            prediction_type="epsilon",
            thresholding=False,
            dynamic_thresholding_ratio=0.995,
            sample_max_value=1,
            algorithm_type="deis",
            solver_type="logrho",
            lower_order_final=True,
            use_karras_sigmas=False,
            timestep_spacing="linspace",
            steps_offset=1,
        )

    elif sampler == "DPMSSDE":
        pipeline.scheduler = DPMSolverSDEScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            beta_start=0.00085,
            beta_end=0.012,
            trained_betas=None,
            prediction_type="epsilon",
            use_karras_sigmas=False,
            noise_sampler_seed=None,
            timestep_spacing="linspace",
            steps_offset=1,
        )

    elif sampler == "DPMSS":
        pipeline.scheduler = DPMSolverSinglestepScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            beta_start=0.00085,
            beta_end=0.012,
            trained_betas=None,
            solver_order=2,
            prediction_type="epsilon",
            thresholding=False,
            dynamic_thresholding_ratio=0.995,
            sample_max_value=1,
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            lower_order_final=True,
            use_karras_sigmas=False,
            lambda_min_clipped=-float("inf"),
            variance_type="fixed_small",
        )

    elif sampler == "Euler":
        pipeline.scheduler = EulerDiscreteScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            beta_start=0.0001,
            beta_end=0.02,
            trained_betas=None,
            prediction_type="v_prediction",
            interpolation_type="linear",
            use_karras_sigmas=True,
            timestep_spacing="linspace",
            steps_offset=1,
        )

    elif sampler == "EulerA":
        pipeline.scheduler = EulerAncestralDiscreteScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            beta_start=0.00085,
            beta_end=0.012,
            trained_betas=None,
            prediction_type="epsilon",
            timestep_spacing="linspace",
            steps_offset=1,
        )

    elif sampler == "Heun":
        pipeline.scheduler = HeunDiscreteScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            beta_start=0.00085,
            beta_end=0.012,
            trained_betas=None,
            prediction_type="epsilon",
            timestep_spacing="linspace",
            steps_offset=1,
        )

    elif sampler == "LMS":
        pipeline.scheduler = LMSDiscreteScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            beta_start=0.0001,
            beta_end=0.02,
            trained_betas=None,
            use_karras_sigmas=True,
            prediction_type="epsilon",
            timestep_spacing="linspace",
            steps_offset=1,
        )

    elif sampler == "KDPM2":
        pipeline.scheduler = KDPM2DiscreteScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            beta_start=0.00085,
            beta_end=0.012,
            trained_betas=None,
            prediction_type="epsilon",
            timestep_spacing="linspace",
            steps_offset=1,
        )

    elif sampler == "KDPM2A":
        pipeline.scheduler = KDPM2AncestralDiscreteScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            beta_start=0.00085,
            beta_end=0.012,
            trained_betas=None,
            prediction_type="epsilon",
            timestep_spacing="linspace",
            steps_offset=1,
        )

    elif sampler == "PNDM":
        pipeline.scheduler = PNDMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            beta_start=0.00085,
            beta_end=0.012,
            trained_betas=None,
            skip_prk_steps=False,
            set_alpha_to_one=False,
            prediction_type="epsilon",
            timestep_spacing="leading",
            steps_offset=1,
        )

    elif sampler == "UniPC":
        pipeline.scheduler = UniPCMultistepScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            beta_start=0.00085,
            beta_end=0.012,
            trained_betas=None,
            solver_order=2,
            prediction_type="epsilon",
            thresholding=False,
            dynamic_thresholding_ratio=0.995,
            sample_max_value=1,
            predict_x0=True,
            solver_type="bh2",
            lower_order_final=True,
            disable_corrector=[],
            solver_p=None,
            use_karras_sigmas=False,
            timestep_spacing="linspace",
            steps_offset=1,
        )


deis_step = DEISMultistepScheduler.step
dpmss_step = DPMSolverSinglestepScheduler.step
euler_step = EulerDiscreteScheduler.step
eulera_step = EulerAncestralDiscreteScheduler.step
heun_step = HeunDiscreteScheduler.step
lms_step = LMSDiscreteScheduler.step
kdpm2_step = KDPM2DiscreteScheduler.step
pndm_step = PNDMScheduler.step
unipc_step = UniPCMultistepScheduler.step


def patched_deis_step(
    self,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    return_dict: bool = True,
    generator=None,
) -> Union[SchedulerOutput, Tuple]:
    return deis_step(self, model_output, timestep, sample, return_dict)


def patched_dpmss_step(
    self,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    return_dict: bool = True,
    generator=None,
) -> Union[SchedulerOutput, Tuple]:
    return dpmss_step(self, model_output, timestep, sample, return_dict)


def patched_euler_step(
    self,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    s_churn: float = 0.0,
    s_tmin: float = 0.0,
    s_tmax: float = float("inf"),
    s_noise: float = 1.0,
    generator: Optional[torch.Generator] = None,
    return_dict: bool = True,
) -> Union[SchedulerOutput, Tuple]:
    sample = EulerDiscreteScheduler.scale_model_input(self, sample, timestep)
    return euler_step(
        self,
        model_output,
        timestep,
        sample,
        s_churn,
        s_tmin,
        s_tmax,
        s_noise,
        generator,
        return_dict,
    )


def patched_eulera_step(
    self,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    generator: Optional[torch.Generator] = None,
    return_dict: bool = True,
) -> Union[SchedulerOutput, Tuple]:
    sample = EulerAncestralDiscreteScheduler.scale_model_input(self, sample, timestep)
    return eulera_step(self, model_output, timestep, sample, generator, return_dict)


def patched_heun_step(
    self,
    model_output: Union[torch.FloatTensor, np.ndarray],
    timestep: Union[float, torch.FloatTensor],
    sample: Union[torch.FloatTensor, np.ndarray],
    return_dict: bool = True,
    generator=None,
) -> Union[SchedulerOutput, Tuple]:
    return heun_step(self, model_output, timestep, sample, return_dict)


def patched_lms_step(
    self,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    order: int = 4,
    return_dict: bool = True,
    generator=None,
) -> Union[SchedulerOutput, Tuple]:
    sample = LMSDiscreteScheduler.scale_model_input(self, sample, timestep)
    return lms_step(self, model_output, timestep, sample, order, return_dict)


def patched_kdpm2_step(
    self,
    model_output: Union[torch.FloatTensor, np.ndarray],
    timestep: Union[float, torch.FloatTensor],
    sample: Union[torch.FloatTensor, np.ndarray],
    return_dict: bool = True,
    generator=None,
) -> Union[SchedulerOutput, Tuple]:
    return kdpm2_step(self, model_output, timestep, sample, return_dict)


def patched_pndm_step(
    self,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    return_dict: bool = True,
    generator=None,
) -> Union[SchedulerOutput, Tuple]:
    return pndm_step(self, model_output, timestep, sample, return_dict)


def patched_unipc_step(
    self,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    return_dict: bool = True,
    generator=None,
) -> Union[SchedulerOutput, Tuple]:
    return unipc_step(self, model_output, timestep, sample, return_dict)


DEISMultistepScheduler.step = patched_deis_step
DPMSolverSinglestepScheduler.step = patched_dpmss_step
EulerDiscreteScheduler.step = patched_euler_step
EulerAncestralDiscreteScheduler.step = patched_eulera_step
HeunDiscreteScheduler.step = patched_heun_step
LMSDiscreteScheduler.step = patched_lms_step
KDPM2DiscreteScheduler.step = patched_kdpm2_step
PNDMScheduler.step = patched_pndm_step
UniPCMultistepScheduler.step = patched_unipc_step
