import gradio as gr
import os
import numpy as np
from train_modules.train_tools import save_config_to_path, load_config_from_path
from train_modules.train_21.train_unclip import (
    default_unclip_config_path,
    add_default_values,
    start_unclip_training,
)


def array_to_str(value):
    return str(value).strip("[]")


def train_unclip_ui(kubin, tabs):
    default_config_from_path = load_config_from_path(default_unclip_config_path)
    default_config_from_path = add_default_values(default_config_from_path)

    with gr.Row() as train_unclip_block:
        current_config = gr.State(default_config_from_path)

        with gr.Column(scale=3):
            with gr.Accordion("General params", open=True):
                with gr.Row():
                    params_path = gr.Textbox(
                        value=default_config_from_path["params_path"],
                        label="Params path",
                        interactive=True,
                    )
                    clip_name = gr.Textbox(
                        value=default_config_from_path["clip_name"],
                        label="Clip Name",
                        interactive=True,
                    )
                    save_path = gr.Textbox(
                        value=default_config_from_path["save_path"],
                        label="Save path",
                        interactive=True,
                    )
                    save_name = gr.Textbox(
                        value=default_config_from_path["save_name"],
                        label="Save name",
                        interactive=True,
                    )
                with gr.Row():
                    with gr.Column():
                        num_epochs = gr.Number(
                            value=default_config_from_path["num_epochs"],
                            label="Number of epochs",
                            interactive=True,
                        )
                        with gr.Row():
                            save_every = gr.Number(
                                value=default_config_from_path["save_every"],
                                label="Save every N steps",
                                interactive=True,
                            )
                            save_epoch = gr.Number(
                                value=default_config_from_path["kubin"]["save_epoch"],
                                label="Save after N epochs",
                                interactive=True,
                            )
                    with gr.Column():
                        with gr.Row():
                            device = gr.Textbox(
                                value=default_config_from_path["device"],
                                label="Device",
                                interactive=True,
                            )
                        with gr.Row():
                            num_workers = gr.Number(
                                value=default_config_from_path["data"]["train"][
                                    "num_workers"
                                ],
                                label="Number of workers",
                                interactive=True,
                            )
                    with gr.Column():
                        with gr.Row():
                            image_size = gr.Textbox(
                                value=default_config_from_path["data"]["train"][
                                    "image_size"
                                ],
                                label="Image Size",
                                interactive=True,
                            )
                        with gr.Row():
                            tokenizer_name = gr.Textbox(
                                value=default_config_from_path["data"]["train"][
                                    "tokenizer_name"
                                ],
                                label="Tokenizer Name",
                                interactive=True,
                            )

                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            df_path = gr.Textbox(
                                value=default_config_from_path["data"]["train"][
                                    "df_path"
                                ],
                                label="Dataset path",
                                interactive=True,
                            )
                            open_tools = gr.Button(
                                "📷 Dataset preparation", size="sm", scale=0
                            )
                            open_tools.click(
                                lambda: gr.Tabs.update(selected="training-dataset"),
                                outputs=tabs,
                            )
                            with gr.Column():
                                clip_image_size = gr.Number(
                                    value=default_config_from_path["data"]["train"][
                                        "clip_image_size"
                                    ],
                                    label="Clip image size",
                                    interactive=True,
                                )
                                batch_size = gr.Number(
                                    value=default_config_from_path["data"]["train"][
                                        "batch_size"
                                    ],
                                    label="Batch size",
                                    interactive=True,
                                )

                    with gr.Column():
                        seq_len = gr.Number(
                            value=default_config_from_path["data"]["train"]["seq_len"],
                            label="Sequence Length",
                            interactive=True,
                        )
                        with gr.Column():
                            drop_text_prob = gr.Number(
                                value=default_config_from_path["data"]["train"][
                                    "drop_text_prob"
                                ],
                                label="Dropout text probability",
                                interactive=True,
                            )
                            drop_image_prob = gr.Number(
                                value=default_config_from_path["data"]["train"][
                                    "drop_image_prob"
                                ],
                                label="Dropout image probability",
                                interactive=True,
                            )
                    with gr.Column():
                        inpainting = gr.Checkbox(
                            value=default_config_from_path["inpainting"],
                            label="Inpainting",
                            interactive=True,
                        )
                        shuffle = gr.Checkbox(
                            value=default_config_from_path["data"]["train"]["shuffle"],
                            label="Shuffle",
                            interactive=True,
                        )
                        drop_first_layer = gr.Checkbox(
                            value=default_config_from_path["drop_first_layer"],
                            label="Drop first layer",
                            interactive=True,
                        )
                        freeze_resblocks = gr.Checkbox(
                            value=default_config_from_path["freeze"][
                                "freeze_resblocks"
                            ],
                            label="Freeze Residual Blocks",
                            interactive=True,
                        )
                        freeze_attention = gr.Checkbox(
                            value=default_config_from_path["freeze"][
                                "freeze_attention"
                            ],
                            label="Freeze Attention",
                            interactive=True,
                        )

            with gr.Accordion("Optimizer params", open=True):
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row():
                            optimizer_name = gr.Textbox(
                                value=default_config_from_path["optim_params"]["name"],
                                label="Optimizer name",
                                interactive=True,
                            )
                            lr = gr.Number(
                                value=default_config_from_path["optim_params"][
                                    "params"
                                ]["lr"],
                                label="Learning rate",
                                interactive=True,
                            )
                            weight_decay = gr.Number(
                                value=default_config_from_path["optim_params"][
                                    "params"
                                ]["weight_decay"],
                                label="Weight decay",
                                interactive=True,
                            )
                    with gr.Column(scale=1):
                        scale_parameter = gr.Checkbox(
                            value=default_config_from_path["optim_params"]["params"][
                                "scale_parameter"
                            ],
                            label="Scale parameter",
                            interactive=True,
                        )
                        relative_step = gr.Checkbox(
                            value=default_config_from_path["optim_params"]["params"][
                                "relative_step"
                            ],
                            label="Relative step",
                            interactive=True,
                        )

            with gr.Accordion("Image encoder params", open=True):
                with gr.Row():
                    scale = gr.Number(
                        value=default_config_from_path["image_enc_params"]["scale"],
                        label="Scale",
                        interactive=True,
                    )
                    ckpt_path = gr.Textbox(
                        value=default_config_from_path["image_enc_params"]["ckpt_path"],
                        label="Checkpoint Path",
                        interactive=True,
                    )
                    embed_dim = gr.Number(
                        value=default_config_from_path["image_enc_params"]["params"][
                            "embed_dim"
                        ],
                        label="Embedding Dimension",
                        interactive=True,
                    )
                    n_embed = gr.Number(
                        value=default_config_from_path["image_enc_params"]["params"][
                            "n_embed"
                        ],
                        label="Number of Embeddings",
                        interactive=True,
                    )
                with gr.Row():
                    double_z = gr.Checkbox(
                        value=default_config_from_path["image_enc_params"]["params"][
                            "ddconfig"
                        ]["double_z"],
                        label="Double Z",
                        interactive=True,
                    )
                    z_channels = gr.Number(
                        value=default_config_from_path["image_enc_params"]["params"][
                            "ddconfig"
                        ]["z_channels"],
                        label="Z Channels",
                        interactive=True,
                    )
                    resolution = gr.Number(
                        value=default_config_from_path["image_enc_params"]["params"][
                            "ddconfig"
                        ]["resolution"],
                        label="Resolution",
                        interactive=True,
                    )
                    in_channels = gr.Number(
                        value=default_config_from_path["image_enc_params"]["params"][
                            "ddconfig"
                        ]["in_channels"],
                        label="Input Channels",
                        interactive=True,
                    )
                    out_ch = gr.Number(
                        value=default_config_from_path["image_enc_params"]["params"][
                            "ddconfig"
                        ]["out_ch"],
                        label="Output Channels",
                        interactive=True,
                    )
                    ch = gr.Number(
                        value=default_config_from_path["image_enc_params"]["params"][
                            "ddconfig"
                        ]["ch"],
                        label="Channels",
                        interactive=True,
                    )
                    ch_mult = gr.Textbox(
                        value=array_to_str(
                            default_config_from_path["image_enc_params"]["params"][
                                "ddconfig"
                            ]["ch_mult"]
                        ),
                        label="Channel Multiplier",
                        interactive=True,
                    )
                with gr.Row():
                    num_res_blocks = gr.Number(
                        value=default_config_from_path["image_enc_params"]["params"][
                            "ddconfig"
                        ]["num_res_blocks"],
                        label="Number of Residual Blocks",
                        interactive=True,
                    )
                    attn_resolutions = gr.Textbox(value=array_to_str(default_config_from_path["image_enc_params"]["params"]["ddconfig"]["attn_resolutions"]), label="Attention Resolutions", interactive=True)  # type: ignore
                    dropout = gr.Number(
                        value=default_config_from_path["image_enc_params"]["params"][
                            "ddconfig"
                        ]["dropout"],
                        label="Dropout",
                        interactive=True,
                    )

            with gr.Accordion("Text encoder params", open=True):
                with gr.Row():
                    model_path = gr.Textbox(
                        value=default_config_from_path["text_enc_params"]["model_path"],
                        label="Model Path",
                        interactive=True,
                    )
                    model_name = gr.Textbox(
                        value=default_config_from_path["text_enc_params"]["model_name"],
                        label="Model Name",
                        interactive=True,
                    )
                    in_features = gr.Number(
                        value=default_config_from_path["text_enc_params"][
                            "in_features"
                        ],
                        label="Input Features",
                        interactive=True,
                    )
                    out_features = gr.Number(
                        value=default_config_from_path["text_enc_params"][
                            "out_features"
                        ],
                        label="Output Features",
                        interactive=True,
                    )

            config_params = {
                current_config,
                params_path,
                drop_first_layer,
                clip_name,
                num_epochs,
                save_every,
                save_epoch,
                save_name,
                save_path,
                device,
                num_workers,
                inpainting,
                shuffle,
                freeze_resblocks,
                freeze_attention,
                df_path,
                image_size,
                tokenizer_name,
                clip_image_size,
                drop_text_prob,
                drop_image_prob,
                seq_len,
                batch_size,
                optimizer_name,
                lr,
                weight_decay,
                scale_parameter,
                relative_step,
                scale,
                ckpt_path,
                embed_dim,
                n_embed,
                double_z,
                z_channels,
                resolution,
                in_channels,
                out_ch,
                ch,
                ch_mult,
                num_res_blocks,
                attn_resolutions,
                dropout,
                model_path,
                model_name,
                in_features,
                out_features,
            }

            def insert_values_to_ui(current_config):
                return {
                    params_path: current_config["params_path"],
                    drop_first_layer: current_config["drop_first_layer"],
                    clip_name: current_config["clip_name"],
                    num_epochs: current_config["num_epochs"],
                    save_every: current_config["save_every"],
                    save_epoch: current_config["kubin"]["save_epoch"],
                    save_name: current_config["save_name"],
                    save_path: current_config["save_path"],
                    device: current_config["device"],
                    num_workers: current_config["data"]["train"]["num_workers"],
                    inpainting: current_config["inpainting"],
                    shuffle: current_config["data"]["train"]["shuffle"],
                    freeze_resblocks: current_config["freeze"]["freeze_resblocks"],
                    freeze_attention: current_config["freeze"]["freeze_attention"],
                    df_path: current_config["data"]["train"]["df_path"],
                    image_size: current_config["data"]["train"]["image_size"],
                    tokenizer_name: current_config["data"]["train"]["tokenizer_name"],
                    clip_image_size: current_config["data"]["train"]["clip_image_size"],
                    drop_text_prob: current_config["data"]["train"]["drop_text_prob"],
                    drop_image_prob: current_config["data"]["train"]["drop_image_prob"],
                    seq_len: current_config["data"]["train"]["seq_len"],
                    batch_size: current_config["data"]["train"]["batch_size"],
                    optimizer_name: current_config["optim_params"]["name"],
                    lr: current_config["optim_params"]["params"]["lr"],
                    weight_decay: current_config["optim_params"]["params"][
                        "weight_decay"
                    ],
                    scale_parameter: current_config["optim_params"]["params"][
                        "scale_parameter"
                    ],
                    relative_step: current_config["optim_params"]["params"][
                        "relative_step"
                    ],
                    scale: current_config["image_enc_params"]["scale"],
                    ckpt_path: current_config["image_enc_params"]["ckpt_path"],
                    embed_dim: current_config["image_enc_params"]["params"][
                        "embed_dim"
                    ],
                    n_embed: current_config["image_enc_params"]["params"]["n_embed"],
                    double_z: current_config["image_enc_params"]["params"]["ddconfig"][
                        "double_z"
                    ],
                    z_channels: current_config["image_enc_params"]["params"][
                        "ddconfig"
                    ]["z_channels"],
                    resolution: current_config["image_enc_params"]["params"][
                        "ddconfig"
                    ]["resolution"],
                    in_channels: current_config["image_enc_params"]["params"][
                        "ddconfig"
                    ]["in_channels"],
                    out_ch: current_config["image_enc_params"]["params"]["ddconfig"][
                        "out_ch"
                    ],
                    ch: current_config["image_enc_params"]["params"]["ddconfig"]["ch"],
                    ch_mult: array_to_str(
                        current_config["image_enc_params"]["params"]["ddconfig"][
                            "ch_mult"
                        ]
                    ),
                    num_res_blocks: current_config["image_enc_params"]["params"][
                        "ddconfig"
                    ]["num_res_blocks"],
                    attn_resolutions: array_to_str(
                        current_config["image_enc_params"]["params"]["ddconfig"][
                            "attn_resolutions"
                        ]
                    ),
                    dropout: current_config["image_enc_params"]["params"]["ddconfig"][
                        "dropout"
                    ],
                    model_path: current_config["text_enc_params"]["model_path"],
                    model_name: current_config["text_enc_params"]["model_name"],
                    in_features: current_config["text_enc_params"]["in_features"],
                    out_features: current_config["text_enc_params"]["out_features"],
                }

            def update_config_from_ui(params):
                def str_to_int_array(text):
                    return [int(value) for value in text.split(",")]

                updated_config = default_config_from_path.copy()

                updated_config["params_path"] = params[params_path]
                updated_config["drop_first_layer"] = params[drop_first_layer]
                updated_config["clip_name"] = params[clip_name]
                updated_config["num_epochs"] = int(params[num_epochs])
                updated_config["save_every"] = int(params[save_every])
                updated_config["save_name"] = params[save_name]
                updated_config["save_path"] = params[save_path]
                updated_config["device"] = params[device]
                updated_config["data"]["train"]["num_workers"] = int(
                    params[num_workers]
                )
                updated_config["inpainting"] = params[inpainting]
                updated_config["data"]["train"]["shuffle"] = params[shuffle]
                updated_config["freeze"]["freeze_resblocks"] = params[freeze_resblocks]
                updated_config["freeze"]["freeze_attention"] = params[freeze_attention]
                updated_config["data"]["train"]["df_path"] = params[df_path]
                updated_config["data"]["train"]["image_size"] = int(params[image_size])
                updated_config["data"]["train"]["tokenizer_name"] = params[
                    tokenizer_name
                ]
                updated_config["data"]["train"]["clip_image_size"] = int(
                    params[clip_image_size]
                )
                updated_config["data"]["train"]["drop_text_prob"] = params[
                    drop_text_prob
                ]
                updated_config["data"]["train"]["drop_image_prob"] = params[
                    drop_image_prob
                ]
                updated_config["data"]["train"]["seq_len"] = int(params[seq_len])
                updated_config["data"]["train"]["batch_size"] = int(params[batch_size])
                updated_config["optim_params"]["name"] = params[optimizer_name]
                updated_config["optim_params"]["params"]["lr"] = params[lr]
                updated_config["optim_params"]["params"]["weight_decay"] = int(
                    params[weight_decay]
                )
                updated_config["optim_params"]["params"]["scale_parameter"] = params[
                    scale_parameter
                ]
                updated_config["optim_params"]["params"]["relative_step"] = params[
                    relative_step
                ]
                updated_config["image_enc_params"]["scale"] = int(params[scale])
                updated_config["image_enc_params"]["ckpt_path"] = params[ckpt_path]
                updated_config["image_enc_params"]["params"]["embed_dim"] = int(
                    params[embed_dim]
                )
                updated_config["image_enc_params"]["params"]["n_embed"] = int(
                    params[n_embed]
                )
                updated_config["image_enc_params"]["params"]["ddconfig"][
                    "double_z"
                ] = params[double_z]
                updated_config["image_enc_params"]["params"]["ddconfig"][
                    "z_channels"
                ] = int(params[z_channels])
                updated_config["image_enc_params"]["params"]["ddconfig"][
                    "resolution"
                ] = int(params[resolution])
                updated_config["image_enc_params"]["params"]["ddconfig"][
                    "in_channels"
                ] = int(params[in_channels])
                updated_config["image_enc_params"]["params"]["ddconfig"][
                    "out_ch"
                ] = int(params[out_ch])
                updated_config["image_enc_params"]["params"]["ddconfig"]["ch"] = int(
                    params[ch]
                )
                updated_config["image_enc_params"]["params"]["ddconfig"][
                    "ch_mult"
                ] = str_to_int_array(params[ch_mult])
                updated_config["image_enc_params"]["params"]["ddconfig"][
                    "num_res_blocks"
                ] = int(params[num_res_blocks])
                updated_config["image_enc_params"]["params"]["ddconfig"][
                    "attn_resolutions"
                ] = str_to_int_array(params[attn_resolutions])
                updated_config["image_enc_params"]["params"]["ddconfig"][
                    "dropout"
                ] = params[dropout]
                updated_config["text_enc_params"]["model_path"] = params[model_path]
                updated_config["text_enc_params"]["model_name"] = params[model_name]
                updated_config["text_enc_params"]["in_features"] = int(
                    params[in_features]
                )
                updated_config["text_enc_params"]["out_features"] = int(
                    params[out_features]
                )
                updated_config["kubin"]["save_epoch"] = int(params[save_epoch])

                return updated_config

            def load_config_values(path, current_config):
                return load_config_values_from_path(path, current_config)

            def load_config_values_from_path(path, current_config):
                if os.path.exists(path):
                    config_from_path = load_config_from_path(path)
                    return config_from_path, False
                else:
                    print("path not found")
                    return current_config, True

            def append_recommended_values(current_config):
                current_config = add_default_values(current_config)
                return current_config

            def save_config_values(path, current_config):
                if os.path.exists(path):
                    print("existing unclip config file found, overwriting")

                save_config_to_path(current_config, path)
                return False

        with gr.Column(scale=1):
            ready_to_train = gr.State(False)
            start_training = gr.Button("Start training", variant="primary")
            unclip_training_info = gr.HTML("Training not started")

            def check_training_params(config):
                return True, ""

            def launch_training(success, training_config):
                if not success:
                    return

                path = training_config["save_path"]

                if not os.path.exists(path):
                    print(f"creating output path {path}")
                    os.mkdir(path)

                start_unclip_training(kubin, training_config)
                print("finetuning of unclip model completed")
                return "Training finished"

            training_config = gr.State(default_config_from_path)
            start_training.click(
                fn=update_config_from_ui,
                inputs=config_params,
                outputs=[training_config],
                queue=False,
            ).then(
                fn=check_training_params,
                inputs=[training_config],
                outputs=[ready_to_train, unclip_training_info],
                queue=False,
                show_progress=False,
            ).then(
                fn=launch_training,
                inputs=[ready_to_train, training_config],
                outputs=[unclip_training_info],
            )

            with gr.Accordion("Miscellaneous", open=True) as misc_params:
                with gr.Row():
                    config_path = gr.Textbox(
                        "train/train_unclip_config.yaml", label="Config path"
                    )
                    load_config = gr.Button("📂 Load parameters from file", size="sm")
                    save_config = gr.Button("💾 Save parameters to file", size="sm")
                    reset_config = gr.Button(
                        "🔁 Reset parameters to default values", size="sm"
                    )

            config_error = gr.Checkbox(False, visible=False)

            load_config.click(
                fn=load_config_values,
                inputs=[config_path, current_config],
                outputs=[current_config, config_error],
                queue=False,
            ).then(
                fn=insert_values_to_ui,
                inputs=current_config,
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
                outputs=[current_config],
                queue=False,
            ).then(
                fn=save_config_values,
                inputs=[config_path, current_config],
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
                inputs=[gr.State(default_unclip_config_path), current_config],
                outputs=[current_config, config_error],
                queue=False,
            ).then(
                fn=append_recommended_values,
                inputs=[current_config],
                outputs=[current_config],
                queue=False,
            ).then(
                fn=insert_values_to_ui,
                inputs=current_config,
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

    return train_unclip_block
