try:
    from diffusers import (
        DDIMInverseScheduler,
        DDIMScheduler,
        DDPMScheduler,
        DPMSolverMultistepInverseScheduler,
        DPMSolverMultistepScheduler,
        DPMSolverSDEScheduler,
        DPMSolverSinglestepScheduler,
        EulerDiscreteScheduler,
        EulerAncestralDiscreteScheduler,
        HeunDiscreteScheduler,
        KarrasVeScheduler,
        KDPM2AncestralDiscreteScheduler,
        KDPM2DiscreteScheduler,
        LMSDiscreteScheduler,
        PNDMScheduler,
        UniPCMultistepScheduler,
    )
    from diffusers import DiffusionPipeline, KandinskyPipeline
    from params import KubinParams

    def use_scheduler(pipeline: KandinskyPipeline, sampler):
        if sampler == "ddim_inverse_sampler":
            pipeline.scheduler = DDIMInverseScheduler.from_config(
                pipeline.scheduler.config
            )
        elif sampler == "ddpm_sampler":
            pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
        elif pipeline.scheduler == "dpms_m_sampler":
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                pipeline.scheduler.config
            )
        elif sampler == "ddim_sampler":
            pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

except:
    None
