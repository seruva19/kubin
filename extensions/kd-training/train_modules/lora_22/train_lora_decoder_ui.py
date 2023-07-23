import gradio as gr
import os
from train_modules.train_tools import (
    save_config_to_path,
    load_config_from_path,
    text_tip,
)
from train_modules.lora_22.train_lora_decoder import (
    default_lora_decoder_config_path,
    add_default_values,
    launch_lora_decoder_training,
)


def train_lora_decoder_ui(kubin, tabs):
    default_lora_config = load_config_from_path(default_lora_decoder_config_path)
    default_lora_config = add_default_values(default_lora_config)

    with gr.Row() as train_lora_decoder_block:
        current_lora_config = gr.State(default_lora_config)

        with gr.Column(scale=3):
            with gr.Accordion("Paths", open=True):
                with gr.Row():
                    pretrained_kandinsky_path = gr.Textbox(
                        value=default_lora_config["paths"]["pretrained_kandinsky_path"],
                        label="Pretrained Kandinsky path",
                        info=text_tip(
                            "Path to pretrained model or model identifier from huggingface.co/models"
                        ),
                    )
                    pretrained_vae_path = gr.Textbox(
                        value=default_lora_config["paths"]["pretrained_vae_path"],
                        label="Pretrained VAE path",
                        info=text_tip("Path to pretrained VAE"),
                    )
                    pretrained_image_encoder = gr.Textbox(
                        value=default_lora_config["paths"]["pretrained_image_encoder"],
                        label="Pretrained image encoder",
                        info=text_tip("Path to pretrained image encoder"),
                    )
                with gr.Row():
                    scheduler_path = gr.Textbox(
                        value=default_lora_config["paths"]["scheduler_path"],
                        label="Scheduler path",
                        info=text_tip("Path to scheduler"),
                    )
                    image_processor_path = gr.Textbox(
                        value=default_lora_config["paths"]["image_processor_path"],
                        label="Image processor path",
                        info=text_tip("Path to image_processor"),
                    )

            with gr.Accordion("Dataset", open=True):
                with gr.Row():
                    train_image_folder = gr.Textbox(
                        value=default_lora_config["dataset"]["train_image_folder"],
                        label="Train images folder",
                        info=text_tip("Path to train image folder"),
                    )
                    train_images_paths_csv = gr.Textbox(
                        value=default_lora_config["dataset"]["train_images_paths_csv"],
                        label="Train images path (CSV)",
                        info=text_tip("Path to csv with train images paths"),
                    )

                    open_tools = gr.Button("📷 Dataset preparation", size="sm", scale=0)
                    open_tools.click(
                        lambda: gr.Tabs.update(selected="training-dataset"),
                        outputs=tabs,
                    )

                    val_image_folder = gr.Textbox(
                        value=default_lora_config["dataset"]["val_image_folder"],
                        label="Validation image folder",
                        info=text_tip("Path to validation image folder"),
                        interactive=False,
                    )
                    val_images_paths_csv = gr.Textbox(
                        value=default_lora_config["dataset"]["val_images_paths_csv"],
                        label="Validation images path (CSV)",
                        info=text_tip(
                            "Path to csv with validation images paths with column paths"
                        ),
                        interactive=False,
                    )

            with gr.Accordion("Training", open=True):
                with gr.Row():
                    train_batch_size = gr.Number(
                        value=default_lora_config["training"]["train_batch_size"],
                        label="Train batch size",
                        info=text_tip("train batch size"),
                        precision=0,
                    )

                    max_train_steps = gr.Number(
                        value=default_lora_config["training"]["max_train_steps"],
                        label="Max train steps",
                        info=text_tip(
                            "Total number of training steps to perform. If provided, overrides number of epochs parameter"
                        ),
                        precision=0,
                    )

                    checkpointing_steps = gr.Number(
                        value=default_lora_config["training"]["checkpointing_steps"],
                        label="Checkpointing steps",
                        info=text_tip(
                            "Save a checkpoint of the training state every N updates"
                        ),
                        precision=0,
                    )

                    lr = gr.Number(
                        value=default_lora_config["training"]["lr"],
                        label="Learning rate",
                        info=text_tip("Learning rate"),
                    )

                    rank = gr.Number(
                        value=default_lora_config["training"]["rank"],
                        label="LORA rank",
                        info=text_tip("A rank of LORA"),
                        precision=0,
                    )

                with gr.Row():
                    output_dir = gr.Textbox(
                        value=default_lora_config["training"]["output_dir"],
                        label="Output dir",
                        info=text_tip(
                            "The output directory where the model predictions and checkpoints will be written"
                        ),
                    )

                    lr_scheduler = gr.Dropdown(
                        choices=[
                            "linear",
                            "cosine",
                            "cosine_with_restarts",
                            "polynomial",
                            "constant",
                            "constant_with_warmup",
                        ],
                        value=default_lora_config["training"]["lr_scheduler"],
                        label="Lr scheduler",
                        info=text_tip("The scheduler type to use"),
                    )
                    weight_decay = gr.Number(
                        value=default_lora_config["training"]["weight_decay"],
                        label="Weight decay",
                        info=text_tip("Weight decay"),
                    )
                    mixed_precision = gr.Dropdown(
                        choices=["no", "fp16", "bf16"],
                        value=default_lora_config["training"]["mixed_precision"],
                        label="Use mixed precision",
                        info=text_tip("Whether to use mixed precision"),
                    )

                with gr.Row():
                    lr_warmup_steps = gr.Number(
                        value=default_lora_config["training"]["lr_warmup_steps"],
                        label="Learning rate warmup steps",
                        info=text_tip(
                            "Number of steps for the warmup in the lr scheduler"
                        ),
                        precision=0,
                    )
                    snr_gamma = gr.Textbox(
                        value=default_lora_config["training"]["snr_gamma"],
                        label="SNR gamma",
                        info=text_tip(
                            "SNR weighting gamma to be used if rebalancing the loss"
                        ),
                    )
                    dataloader_num_workers = gr.Number(
                        value=default_lora_config["training"]["dataloader_num_workers"],
                        label="Dataloader number of workers",
                        info=text_tip(
                            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process"
                        ),
                        precision=0,
                    )
                    logging_dir = gr.Textbox(
                        value=default_lora_config["training"]["logging_dir"],
                        label="Logging dir",
                        info=text_tip("Log directory, relative to output dir"),
                    )

                with gr.Row():
                    resume_from_checkpoint = gr.Textbox(
                        value=default_lora_config["training"]["resume_from_checkpoint"],
                        label="Path of checkpoint to resume from",
                        info=text_tip(
                            "Whether training should be resumed from a previous checkpoint"
                        ),
                    )

                    report_to = gr.Dropdown(
                        choices=["none", "all", "tensorboard", "wandb", "comet_ml"],
                        value=default_lora_config["training"]["report_to"],
                        label="Report to",
                        info=text_tip(
                            "The integration to report the results and logs to"
                        ),
                    )
                    local_rank = gr.Number(
                        value=default_lora_config["training"]["local_rank"],
                        label="Local rank",
                        info=text_tip("Rank for distributed training"),
                        precision=0,
                    )

                    seed = gr.Textbox(
                        value=default_lora_config["training"]["seed"],
                        label="Seed",
                        info=text_tip("A seed for reproducible training"),
                    )

                with gr.Row():
                    checkpoints_total_limit = gr.Textbox(
                        value=default_lora_config["training"][
                            "checkpoints_total_limit"
                        ],
                        label="Checkpoints total limit",
                        info=text_tip("Max number of checkpoints to store"),
                    )

                    num_epochs = gr.Number(
                        value=default_lora_config["training"]["num_epochs"],
                        label="Number of epochs",
                        info=text_tip("Number of epochs"),
                        interactive=False,
                        precision=0,
                    )

                    val_batch_size = gr.Number(
                        value=default_lora_config["training"]["val_batch_size"],
                        label="Validation batch size",
                        info=text_tip("Validation batch size"),
                        interactive=False,
                        precision=0,
                    )

                    gradient_accumulation_steps = gr.Number(
                        value=default_lora_config["training"][
                            "gradient_accumulation_steps"
                        ],
                        label="Gradient accumulation steps",
                        info=text_tip(
                            "Number of updates steps to accumulate before performing a backward/update pass"
                        ),
                        precision=0,
                    )

                    max_grad_norm = gr.Number(
                        value=default_lora_config["training"]["max_grad_norm"],
                        label="Max gradient norm",
                        info=text_tip("Max gradient norm"),
                    )

                with gr.Row():
                    with gr.Column(scale=1):
                        gradient_checkpointing = gr.Checkbox(
                            value=default_lora_config["training"][
                                "gradient_checkpointing"
                            ],
                            label="Use gradient checkpointing",
                            info=text_tip(
                                "Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass"
                            ),
                            interactive=False,
                        )
                        allow_tf32 = gr.Checkbox(
                            value=default_lora_config["training"]["allow_tf32"],
                            label="Allow TF32",
                            info=text_tip(
                                "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training"
                            ),
                        )

                    with gr.Column(scale=3):
                        with gr.Row():
                            use_8bit_adam = gr.Checkbox(
                                value=default_lora_config["training"]["use_8bit_adam"],
                                label="Use 8bit Adam",
                                info=text_tip(
                                    "Whether or not to use 8-bit Adam from bitsandbytes"
                                ),
                            )
                            adam_beta1 = gr.Number(
                                value=default_lora_config["training"]["adam_beta1"],
                                label="Adam beta1",
                                info=text_tip(
                                    "The beta1 parameter for the Adam optimizer"
                                ),
                            )
                            adam_beta2 = gr.Number(
                                value=default_lora_config["training"]["adam_beta2"],
                                label="Adam beta2",
                                info=text_tip(
                                    "The beta2 parameter for the Adam optimizer"
                                ),
                            )
                            adam_epsilon = gr.Number(
                                value=default_lora_config["training"]["adam_epsilon"],
                                label="Adam epsilon",
                                info=text_tip("Epsilon value for the Adam optimizer"),
                            )

                with gr.Row():
                    copy_training_params = gr.Button(
                        "📑 Copy training params from prior tab",
                        size="sm",
                        interactive=False,
                    )

            with gr.Accordion("Other", open=True):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_resolution = gr.Number(
                            value=default_lora_config["decoder"]["image_resolution"],
                            label="Image resolution",
                            info=text_tip("Image resolution"),
                            precision=0,
                        )
                    with gr.Column(scale=2):
                        output_name = gr.Textbox(
                            value=default_lora_config["decoder"]["output_name"],
                            label="Output name",
                            info=text_tip(
                                "Name of the LoRA decoder checkpoint in output directory"
                            ),
                        )

            config_params = {
                current_lora_config,
                pretrained_kandinsky_path,
                pretrained_vae_path,
                pretrained_image_encoder,
                scheduler_path,
                image_processor_path,
                train_image_folder,
                train_images_paths_csv,
                val_image_folder,
                val_images_paths_csv,
                train_batch_size,
                val_batch_size,
                lr,
                weight_decay,
                num_epochs,
                output_dir,
                lr_scheduler,
                max_train_steps,
                gradient_accumulation_steps,
                gradient_checkpointing,
                lr_warmup_steps,
                snr_gamma,
                use_8bit_adam,
                resume_from_checkpoint,
                allow_tf32,
                dataloader_num_workers,
                logging_dir,
                mixed_precision,
                report_to,
                local_rank,
                checkpointing_steps,
                checkpoints_total_limit,
                rank,
                seed,
                adam_beta1,
                adam_beta2,
                adam_epsilon,
                max_grad_norm,
                image_resolution,
                output_name,
            }

            def insert_values_to_ui(current_config):
                return {
                    pretrained_kandinsky_path: current_config["paths"][
                        "pretrained_kandinsky_path"
                    ],
                    pretrained_vae_path: current_config["paths"]["pretrained_vae_path"],
                    pretrained_image_encoder: current_config["paths"][
                        "pretrained_image_encoder"
                    ],
                    scheduler_path: current_config["paths"]["scheduler_path"],
                    image_processor_path: current_config["paths"][
                        "image_processor_path"
                    ],
                    train_image_folder: current_config["dataset"]["train_image_folder"],
                    train_images_paths_csv: current_config["dataset"][
                        "train_images_paths_csv"
                    ],
                    val_image_folder: current_config["dataset"]["val_image_folder"],
                    val_images_paths_csv: current_config["dataset"][
                        "val_images_paths_csv"
                    ],
                    train_batch_size: current_config["training"]["train_batch_size"],
                    val_batch_size: current_config["training"]["val_batch_size"],
                    lr: current_config["training"]["lr"],
                    weight_decay: current_config["training"]["weight_decay"],
                    num_epochs: current_config["training"]["num_epochs"],
                    output_dir: current_config["training"]["output_dir"],
                    lr_scheduler: current_config["training"]["lr_scheduler"],
                    max_train_steps: current_config["training"]["max_train_steps"],
                    gradient_accumulation_steps: current_config["training"][
                        "gradient_accumulation_steps"
                    ],
                    gradient_checkpointing: current_config["training"][
                        "gradient_checkpointing"
                    ],
                    lr_warmup_steps: current_config["training"]["lr_warmup_steps"],
                    snr_gamma: current_config["training"]["snr_gamma"],
                    use_8bit_adam: current_config["training"]["use_8bit_adam"],
                    resume_from_checkpoint: current_config["training"][
                        "resume_from_checkpoint"
                    ],
                    allow_tf32: current_config["training"]["allow_tf32"],
                    dataloader_num_workers: current_config["training"][
                        "dataloader_num_workers"
                    ],
                    logging_dir: current_config["training"]["logging_dir"],
                    mixed_precision: current_config["training"]["mixed_precision"],
                    report_to: current_config["training"]["report_to"],
                    local_rank: current_config["training"]["local_rank"],
                    checkpointing_steps: current_config["training"][
                        "checkpointing_steps"
                    ],
                    checkpoints_total_limit: current_config["training"][
                        "checkpoints_total_limit"
                    ],
                    rank: current_config["training"]["rank"],
                    seed: current_config["training"]["seed"],
                    adam_beta1: current_config["training"]["adam_beta1"],
                    adam_beta2: current_config["training"]["adam_beta2"],
                    adam_epsilon: current_config["training"]["adam_epsilon"],
                    max_grad_norm: current_config["training"]["max_grad_norm"],
                    image_resolution: current_config["decoder"]["image_resolution"],
                    output_name: current_config["decoder"]["output_name"],
                }

            def update_config_from_ui(params):
                updated_config = default_lora_config.copy()

                updated_config["paths"]["pretrained_kandinsky_path"] = params[
                    pretrained_kandinsky_path
                ]
                updated_config["paths"]["pretrained_vae_path"] = params[
                    pretrained_vae_path
                ]
                updated_config["paths"]["pretrained_image_encoder"] = params[
                    pretrained_image_encoder
                ]
                updated_config["paths"]["scheduler_path"] = params[scheduler_path]
                updated_config["paths"]["image_processor_path"] = params[
                    image_processor_path
                ]
                updated_config["dataset"]["train_image_folder"] = params[
                    train_image_folder
                ]
                updated_config["dataset"]["train_images_paths_csv"] = params[
                    train_images_paths_csv
                ]
                updated_config["dataset"]["val_image_folder"] = params[val_image_folder]
                updated_config["dataset"]["val_images_paths_csv"] = params[
                    val_images_paths_csv
                ]
                updated_config["training"]["train_batch_size"] = params[
                    train_batch_size
                ]
                updated_config["training"]["val_batch_size"] = params[val_batch_size]
                updated_config["training"]["lr"] = params[lr]
                updated_config["training"]["weight_decay"] = params[weight_decay]
                updated_config["training"]["num_epochs"] = params[num_epochs]
                updated_config["training"]["output_dir"] = params[output_dir]
                updated_config["training"]["lr_scheduler"] = params[lr_scheduler]
                updated_config["training"]["max_train_steps"] = params[max_train_steps]
                updated_config["training"]["gradient_accumulation_steps"] = params[
                    gradient_accumulation_steps
                ]
                updated_config["training"]["gradient_checkpointing"] = params[
                    gradient_checkpointing
                ]
                updated_config["training"]["lr_warmup_steps"] = params[lr_warmup_steps]
                updated_config["training"]["snr_gamma"] = params[snr_gamma]
                updated_config["training"]["use_8bit_adam"] = params[use_8bit_adam]
                updated_config["training"]["resume_from_checkpoint"] = params[
                    resume_from_checkpoint
                ]
                updated_config["training"]["allow_tf32"] = params[allow_tf32]
                updated_config["training"]["dataloader_num_workers"] = params[
                    dataloader_num_workers
                ]
                updated_config["training"]["logging_dir"] = params[logging_dir]
                updated_config["training"]["mixed_precision"] = params[mixed_precision]
                updated_config["training"]["report_to"] = params[report_to]
                updated_config["training"]["local_rank"] = params[local_rank]
                updated_config["training"]["checkpointing_steps"] = params[
                    checkpointing_steps
                ]
                updated_config["training"]["checkpoints_total_limit"] = params[
                    checkpoints_total_limit
                ]
                updated_config["training"]["rank"] = params[rank]
                updated_config["training"]["seed"] = params[seed]
                updated_config["training"]["adam_beta1"] = params[adam_beta1]
                updated_config["training"]["adam_beta2"] = params[adam_beta2]
                updated_config["training"]["adam_epsilon"] = params[adam_epsilon]
                updated_config["training"]["max_grad_norm"] = params[max_grad_norm]

                updated_config["decoder"]["image_resolution"] = params[image_resolution]
                updated_config["decoder"]["output_name"] = params[output_name]

                return updated_config

            def load_config_values(path, current_config):
                return load_config_values_from_path(path, current_config)

            def load_config_values_from_path(path, current_config):
                if os.path.exists(path):
                    config_from_path = load_config_from_path(path)
                    return config_from_path, False
                else:
                    print("config path not found")
                    return current_config, True

            def append_recommended_values(current_config):
                current_config = add_default_values(current_config)
                return current_config

            def save_config_values(path, current_config):
                if os.path.exists(path):
                    print("existing lora decoder config file found, overwriting")

                save_config_to_path(current_config, path)
                return False

        with gr.Column(scale=1):
            ready_to_train = gr.State(False)
            start_lora_decoder_training = gr.Button(
                "Start LoRA decoder training", variant="primary"
            )
            lora_decoder_training_info = gr.HTML(
                "Training not started", elem_classes=["lora-decoder-progress"]
            )

            def check_training_params(config):
                return True, ""

            def launch_training(success, lora_training_config, progress=gr.Progress()):
                if not success:
                    return

                path = lora_training_config["training"]["output_dir"]

                if not os.path.exists(path):
                    print(f"creating output path {path}")
                    os.mkdir(path)

                launch_lora_decoder_training(kubin, lora_training_config, progress)
                print("\ntraining of LoRA decoder model completed")
                return "Training finished"

            training_config = gr.State(default_lora_config)

            start_lora_decoder_training.click(
                fn=lambda: gr.update(interactive=False),
                queue=False,
                outputs=start_lora_decoder_training,
            ).then(
                fn=update_config_from_ui,
                inputs=config_params,
                outputs=[training_config],
                queue=False,
            ).then(
                fn=check_training_params,
                inputs=[training_config],
                outputs=[ready_to_train, lora_decoder_training_info],
                queue=False,
                show_progress=False,
            ).then(
                fn=launch_training,
                inputs=[ready_to_train, training_config],
                outputs=[lora_decoder_training_info],
            ).then(
                fn=lambda: gr.update(interactive=True),
                queue=False,
                outputs=start_lora_decoder_training,
            )

            with gr.Accordion("Miscellaneous", open=True) as misc_params:
                with gr.Row():
                    config_path = gr.Textbox(
                        "train/train_lora_decoder_config.yaml", label="Config path"
                    )
                    load_config = gr.Button("📂 Load parameters from file", size="sm")
                    save_config = gr.Button("💾 Save parameters to file", size="sm")
                    reset_config = gr.Button(
                        "🔁 Reset parameters to default values", size="sm"
                    )

            config_error = gr.Checkbox(False, visible=False)

            load_config.click(
                fn=load_config_values,
                inputs=[config_path, current_lora_config],
                outputs=[current_lora_config, config_error],
                queue=False,
            ).then(
                fn=insert_values_to_ui,
                inputs=current_lora_config,
                show_progress=False,
                outputs=config_params,
            ).then(
                fn=None,
                inputs=[config_error],
                outputs=[config_error],
                show_progress=False,
                _js='(e) => !e ? kubin.notify.success("Parameters loaded from file") : kubin.notify.error("Error loading config")',
            )

            save_config.click(
                fn=update_config_from_ui,
                inputs=config_params,
                outputs=[current_lora_config],
                queue=False,
            ).then(
                fn=save_config_values,
                inputs=[config_path, current_lora_config],
                outputs=[config_error],
                queue=False,
            ).then(
                fn=None,
                inputs=[config_error],
                outputs=[config_error],
                show_progress=False,
                _js='(e) => !e ? kubin.notify.success("Parameters saved to file") : kubin.notify.error("Error loading config")',
            )

            reset_config.click(
                fn=load_config_values_from_path,
                inputs=[
                    gr.State(default_lora_decoder_config_path),
                    current_lora_config,
                ],
                outputs=[current_lora_config, config_error],
                queue=False,
            ).then(
                fn=append_recommended_values,
                inputs=[current_lora_config],
                outputs=[current_lora_config],
                queue=False,
            ).then(
                fn=insert_values_to_ui,
                inputs=current_lora_config,
                outputs=config_params,
                queue=False,
            ).then(
                fn=None,
                inputs=[config_error],
                outputs=[config_error],
                show_progress=False,
                _js='() => kubin.notify.success("Parameters were reset to default values")',
            )

            misc_params.elem_classes = ["training-misc-params"]

    return train_lora_decoder_block
