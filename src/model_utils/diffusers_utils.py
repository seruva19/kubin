from utils.logging import k_log
from params import KubinParams


def use_scheduler(pipeline, sampler):
    from diffusers import (
        DDPMScheduler,
        DDIMScheduler,
        DDIMInverseScheduler,
        DPMSolverMultistepScheduler,
        DPMSolverMultistepInverseScheduler,
        DPMSolverSDEScheduler,
        DPMSolverSinglestepScheduler,
        EulerDiscreteScheduler,
        EulerAncestralDiscreteScheduler,
        HeunDiscreteScheduler,
        KarrasVeScheduler,
        LMSDiscreteScheduler,
        KDPM2DiscreteScheduler,
        KDPM2AncestralDiscreteScheduler,
        PNDMScheduler,
        UniPCMultistepScheduler,
    )

    k_log(f"using scheduler: {sampler}")

    if sampler == "ddpm_sampler":
        pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

    elif sampler == "ddim_sampler":
        pipeline.scheduler = DDIMScheduler()

    elif sampler == "ddim_inverse_sampler":
        pipeline.scheduler = DDIMInverseScheduler()

    elif sampler == "dpms_m_sampler":
        pipeline.scheduler = DPMSolverMultistepScheduler()

    elif sampler == "dpms_m_inverse_sampler":
        pipeline.scheduler = DPMSolverMultistepInverseScheduler()

    elif sampler == "dpms_sde_sampler":
        pipeline.scheduler = DPMSolverSDEScheduler()

    elif sampler == "dpms_ss_sampler":
        pipeline.scheduler = DPMSolverSinglestepScheduler()

    elif sampler == "euler_sampler":
        pipeline.scheduler = EulerDiscreteScheduler()

    elif sampler == "euler_a_sampler":
        pipeline.scheduler = EulerAncestralDiscreteScheduler()

    elif sampler == "heun_sampler":
        pipeline.scheduler = HeunDiscreteScheduler()

    elif sampler == "karras_sampler":
        pipeline.scheduler = KarrasVeScheduler()

    elif sampler == "lms_sampler":
        pipeline.scheduler = LMSDiscreteScheduler()

    elif sampler == "kdpm2_sampler":
        pipeline.scheduler = KDPM2DiscreteScheduler()

    elif sampler == "kdpm2_a_sampler":
        pipeline.scheduler = KDPM2AncestralDiscreteScheduler()

    elif sampler == "pndm_sampler":
        pipeline.scheduler = PNDMScheduler()

    elif sampler == "unipc_sampler":
        pipeline.scheduler = UniPCMultistepScheduler()
