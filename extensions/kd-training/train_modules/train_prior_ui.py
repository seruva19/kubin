import gradio as gr
import os
from train_modules.train_tools import save_config_to_path, load_config_from_path
from train_modules.train_prior import (
    default_prior_config_path,
    add_default_values,
    start_prior_training,
)


def train_prior_ui(kubin, tabs):
    default_config_from_path = load_config_from_path(default_prior_config_path)
    default_config_from_path = add_default_values(default_config_from_path)

    with gr.Row() as train_prior_block:
        current_config = gr.State(default_config_from_path)

        with gr.Column(scale=3):
            with gr.Accordion("General params", open=True):
                with gr.Row():
                    params_path = gr.Textbox(value=default_config_from_path["params_path"], label="Params path", interactive=True)  # type: ignore
                    clip_mean_std_path = gr.Textbox(value=default_config_from_path["clip_mean_std_path"], label="Clip mean standard path", interactive=True)  # type: ignore
                    clip_name = gr.Textbox(value=default_config_from_path["clip_name"], label="Clip Name", interactive=True)  # type: ignore
                with gr.Row().style(equal_height=True):
                    with gr.Column():
                        num_epochs = gr.Number(value=default_config_from_path["num_epochs"], label="Number of epochs", interactive=True)  # type: ignore
                        with gr.Row():
                            save_every = gr.Number(value=default_config_from_path["save_every"], label="Save after steps", interactive=True)  # type: ignore
                            save_epoch = gr.Number(value=default_config_from_path["kubin"]["save_epoch"], label="Save after epochs", interactive=True)  # type: ignore
                    with gr.Column():
                        save_name = gr.Textbox(value=default_config_from_path["save_name"], label="Save name", interactive=True)  # type: ignore
                        save_path = gr.Textbox(value=default_config_from_path["save_path"], label="Save path", interactive=True)  # type: ignore
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row():
                            device = gr.Textbox(value=default_config_from_path["device"], label="Device", interactive=True)  # type: ignore
                            num_workers = gr.Number(value=default_config_from_path["data"]["train"]["num_workers"], label="Number of workers", interactive=True)  # type: ignore" interactive=True) # type: ignore
                    with gr.Column(scale=1):
                        inpainting = gr.Checkbox(value=default_config_from_path["inpainting"], label="Inpainting", interactive=True)  # type: ignore
                        shuffle = gr.Checkbox(value=default_config_from_path["data"]["train"]["shuffle"], label="Shuffle", interactive=True)  # type: ignore
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            df_path = gr.Textbox(value=default_config_from_path["data"]["train"]["df_path"], label="Dataset path", interactive=True)  # type: ignore
                            open_tools = gr.Button("Dataset preparation").style(
                                size="sm", full_width=False
                            )
                            open_tools.click(
                                lambda: gr.Tabs.update(selected="training-tools"),
                                outputs=tabs,
                            )
                    with gr.Column(scale=2):
                        with gr.Row():
                            clip_image_size = gr.Number(value=default_config_from_path["data"]["train"]["clip_image_size"], label="Clip image size", interactive=True)  # type: ignore
                            drop_text_prob = gr.Number(value=default_config_from_path["data"]["train"]["drop_text_prob"], label="Dropout text probability", interactive=True)  # type: ignore
                            batch_size = gr.Number(value=default_config_from_path["data"]["train"]["batch_size"], label="Batch size", interactive=True)  # type: ignore

            with gr.Accordion("Optimizer params", open=True):
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row():
                            optimizer_name = gr.Textbox(value=default_config_from_path["optim_params"]["name"], label="Optimizer name", interactive=True)  # type: ignore
                            lr = gr.Number(value=default_config_from_path["optim_params"]["params"]["lr"], label="Learning rate", interactive=True)  # type: ignore
                            weight_decay = gr.Number(value=default_config_from_path["optim_params"]["params"]["weight_decay"], label="Weight decay", interactive=True)  # type: ignore
                    with gr.Column(scale=1):
                        scale_parameter = gr.Checkbox(value=default_config_from_path["optim_params"]["params"]["scale_parameter"], label="Scale parameter", interactive=True)  # type: ignore
                        relative_step = gr.Checkbox(value=default_config_from_path["optim_params"]["params"]["relative_step"], label="Relative step", interactive=True)  # type: ignore

            with gr.Accordion("Miscellaneous", open=True):
                with gr.Row():
                    config_path = gr.Textbox(
                        "train/train_prior_config.yaml", label="Config path"
                    )
                    load_config = gr.Button("Load parameters from file").style(
                        size="sm", full_width=False
                    )
                    save_config = gr.Button("Save parameters to file").style(
                        size="sm", full_width=False
                    )
                    reset_config = gr.Button(
                        "Reset parameters to default values"
                    ).style(size="sm", full_width=False)

            config_params = {
                current_config,
                params_path,
                clip_mean_std_path,
                clip_name,
                num_epochs,
                save_every,
                save_epoch,
                save_name,
                save_path,
                inpainting,
                device,
                df_path,
                clip_image_size,
                drop_text_prob,
                batch_size,
                shuffle,
                num_workers,
                optimizer_name,
                lr,
                weight_decay,
                scale_parameter,
                relative_step,
            }

            def insert_values_to_ui(current_config):
                return {
                    params_path: current_config["params_path"],
                    clip_mean_std_path: current_config["clip_mean_std_path"],
                    clip_name: current_config["clip_name"],
                    num_epochs: current_config["num_epochs"],
                    save_every: current_config["save_every"],
                    save_epoch: current_config["kubin"]["save_epoch"],
                    save_name: current_config["save_name"],
                    save_path: current_config["save_path"],
                    inpainting: current_config["inpainting"],
                    device: current_config["device"],
                    df_path: current_config["data"]["train"]["df_path"],
                    clip_image_size: current_config["data"]["train"]["clip_image_size"],
                    drop_text_prob: current_config["data"]["train"]["drop_text_prob"],
                    batch_size: current_config["data"]["train"]["batch_size"],
                    shuffle: current_config["data"]["train"]["shuffle"],
                    num_workers: current_config["data"]["train"]["num_workers"],
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
                }

            def update_config_from_ui(params):
                updated_config = default_config_from_path.copy()

                updated_config["params_path"] = params[params_path]  # type: ignore
                updated_config["clip_mean_std_path"] = params[clip_mean_std_path]  # type: ignore
                updated_config["clip_name"] = params[clip_name]  # type: ignore
                updated_config["num_epochs"] = int(params[num_epochs])  # type: ignore
                updated_config["save_every"] = int(params[save_every])  # type: ignore
                updated_config["save_name"] = params[save_name]  # type: ignore
                updated_config["save_path"] = params[save_path]  # type: ignore
                updated_config["inpainting"] = params[inpainting]  # type: ignore
                updated_config["device"] = params[device]  # type: ignore
                updated_config["data"]["train"]["df_path"] = params[df_path]  # type: ignore
                updated_config["data"]["train"]["clip_image_size"] = int(params[clip_image_size])  # type: ignore
                updated_config["data"]["train"]["drop_text_prob"] = params[drop_text_prob]  # type: ignore
                updated_config["data"]["train"]["batch_size"] = int(params[batch_size])  # type: ignore
                updated_config["data"]["train"]["shuffle"] = params[shuffle]  # type: ignore
                updated_config["data"]["train"]["num_workers"] = int(params[num_workers])  # type: ignore
                updated_config["optim_params"]["name"] = params[optimizer_name]  # type: ignore
                updated_config["optim_params"]["params"]["lr"] = params[lr]  # type: ignore
                updated_config["optim_params"]["params"]["weight_decay"] = int(params[weight_decay])  # type: ignore
                updated_config["optim_params"]["params"]["scale_parameter"] = params[scale_parameter]  # type: ignore
                updated_config["optim_params"]["params"]["relative_step"] = params[relative_step]  # type: ignore
                updated_config["kubin"]["save_epoch"] = int(params[save_epoch])  # type: ignore

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
                    print("existing prior config file found, overwriting")

                save_config_to_path(current_config, path)
                return False

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
                outputs=config_params,  # type: ignore
            ).then(
                fn=None,
                inputs=[config_error],
                outputs=[config_error],
                show_progress=False,
                _js='(e) => !e ? kubin.notify.success("Parameters loaded from file") : kubin.notify.error("Error loading config")',
            )

            save_config.click(fn=update_config_from_ui, inputs=config_params, outputs=[current_config], queue=False).then(  # type: ignore
                fn=save_config_values,
                inputs=[config_path, current_config],
                outputs=[config_error],
                queue=False,
            ).then(  # type: ignore
                fn=None,
                inputs=[config_error],
                outputs=[config_error],
                show_progress=False,
                _js='(e) => !e ? kubin.notify.success("Parameters saved to file") : kubin.notify.error("Error loading config")',
            )

            reset_config.click(
                fn=load_config_values_from_path,
                inputs=[gr.State(default_prior_config_path), current_config],
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
                outputs=config_params,  # type: ignore
                queue=False,
            ).then(
                fn=None,
                inputs=[config_error],
                outputs=[config_error],
                show_progress=False,
                _js='() => kubin.notify.success("Parameters were reset to default values")',
            )

        with gr.Column(scale=1):
            ready_to_train = gr.State(False)
            start_training = gr.Button("Start training", variant="primary")
            prior_training_info = gr.HTML("Training not started")

            def check_training_params(config):
                return True, ""

            def launch_training(success, training_config):
                if not success:
                    return

                path = training_config["save_path"]

                if not os.path.exists(path):
                    print(f"creating output path {path}")
                    os.mkdir(path)

                start_prior_training(kubin, training_config)
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
                outputs=[ready_to_train, prior_training_info],
                queue=False,
                show_progress=False,
            ).then(
                fn=launch_training,
                inputs=[ready_to_train, training_config],
                outputs=[prior_training_info],
            )

    return train_prior_block
