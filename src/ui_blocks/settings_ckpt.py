import gradio as gr
import fnmatch
import os
from engine.kandinsky import KandinskyCheckpoint
from env import Kubin

prior_base_path_placeholder = "🏠 base prior checkpoint"
decoder_base_path_placeholder = "🏠 base decoder checkpoint"
inpaint_decoder_base_path_placeholder = "🏠 base inpainting decoder checkpoint"


def ckpt_selector(kubin: Kubin):
    scanned_directories = "models;checkpoints;train/checkpoints"
    prior_filename_pattern = "*prior*.ckpt"
    decoder_filename_pattern = "*decoder*.ckpt"
    inpaint_filename_pattern = "*inpainting*.ckpt"

    (
        default_prior_path,
        default_decoder_path,
        default_inpaint_decoder_path,
    ) = kubin.params.checkpoint.base_checkpoints_path(kubin.params.cache_dir)

    with gr.Row() as selector:
        with gr.Column():
            current_checkpoints = gr.HTML(
                show_current_checkpoints(kubin), elem_classes=["block-info"]
            )

            prior_select = gr.Dropdown(
                choices=scan_checkpoints(
                    scanned_directories,
                    prior_filename_pattern,
                    default_prior_path,
                    prior_base_path_placeholder,
                ),
                value=prior_base_path_placeholder,
                label="Select prior checkpoint",
                interactive=True,
            )
            prior_select.select(
                lambda path, kubin=kubin: select_prior_checkpoint(path, kubin),
                inputs=[prior_select],
                outputs=[current_checkpoints],
                show_progress=False,
            )
            decoder_select = gr.Dropdown(
                choices=scan_checkpoints(
                    scanned_directories,
                    decoder_filename_pattern,
                    default_decoder_path,
                    decoder_base_path_placeholder,
                ),
                value=decoder_base_path_placeholder,
                label="Select decoder checkpoint",
                interactive=True,
            )
            decoder_select.select(
                lambda path, kubin=kubin: select_decoder_checkpoint(path, kubin),
                inputs=[decoder_select],
                outputs=[current_checkpoints],
                show_progress=False,
            )
            inpaint_select = gr.Dropdown(
                choices=scan_checkpoints(
                    scanned_directories,
                    inpaint_filename_pattern,
                    default_inpaint_decoder_path,
                    inpaint_decoder_base_path_placeholder,
                ),
                value=inpaint_decoder_base_path_placeholder,
                label="Select inpaint decoder checkpoint",
                interactive=True,
            )
            inpaint_select.select(
                lambda path, kubin=kubin: select_inpaint_checkpoint(path, kubin),
                inputs=[inpaint_select],
                outputs=[current_checkpoints],
                show_progress=False,
            )

        with gr.Accordion("Checkpoint location settings", open=True):
            rescan_btn = gr.Button(value="🔄 Rescan checkpoints").style(full_width=False)

            directories = gr.Textbox(
                value=scanned_directories, label="Directories to scan"
            )
            prior_pattern = gr.Textbox(
                value=prior_filename_pattern, label="Prior filename pattern"
            )
            decoder_pattern = gr.Textbox(
                value=decoder_filename_pattern, label="Decoder filename pattern"
            )
            inpaint_pattern = gr.Textbox(
                value=inpaint_filename_pattern, label="Inpaint decoder filename pattern"
            )

            rescan_btn.click(
                lambda directories, prior_pattern, decoder_pattern, inpaint_pattern, kubin=kubin: rescan_checkpoints(
                    kubin,
                    directories,
                    prior_pattern,
                    decoder_pattern,
                    inpaint_pattern,
                    default_prior_path,
                    default_decoder_path,
                    default_inpaint_decoder_path,
                ),
                inputs=[directories, prior_pattern, decoder_pattern, inpaint_pattern],
                outputs=[prior_select, decoder_select, inpaint_select],
            )

    return selector


def select_prior_checkpoint(prior_path, kubin: Kubin):
    checkpoint_info = kubin.params.checkpoint
    if prior_path == prior_base_path_placeholder:
        default_checkpoint = KandinskyCheckpoint()
        checkpoint_info.prior_model_dir, checkpoint_info.prior_model_name = (
            default_checkpoint.prior_model_dir,
            default_checkpoint.prior_model_name,
        )
    else:
        (
            checkpoint_info.prior_model_dir,
            checkpoint_info.prior_model_name,
        ) = extract_checkpoint_path(prior_path)

    return show_current_checkpoints(kubin)


def select_decoder_checkpoint(decoder_path, kubin: Kubin):
    checkpoint_info = kubin.params.checkpoint
    if decoder_path == decoder_base_path_placeholder:
        default_checkpoint = KandinskyCheckpoint()
        checkpoint_info.decoder_model_dir, checkpoint_info.decoder_model_name = (
            default_checkpoint.decoder_model_dir,
            default_checkpoint.decoder_model_name,
        )
    else:
        (
            checkpoint_info.decoder_model_dir,
            checkpoint_info.decoder_model_name,
        ) = extract_checkpoint_path(decoder_path)

    return show_current_checkpoints(kubin)


def select_inpaint_checkpoint(inpaint_decoder_path, kubin: Kubin):
    checkpoint_info = kubin.params.checkpoint
    if inpaint_decoder_path == inpaint_decoder_base_path_placeholder:
        default_checkpoint = KandinskyCheckpoint()
        checkpoint_info.inpaint_model_dir, checkpoint_info.inpaint_model_name = (
            default_checkpoint.inpaint_model_dir,
            default_checkpoint.inpaint_model_name,
        )
    else:
        (
            checkpoint_info.inpaint_model_dir,
            checkpoint_info.inpaint_model_name,
        ) = extract_checkpoint_path(inpaint_decoder_path)

    return show_current_checkpoints(kubin)


def show_current_checkpoints(kubin: Kubin):
    checkpoint = kubin.params.checkpoint
    default_checkpoint = KandinskyCheckpoint()
    (
        default_prior_path,
        default_decoder_path,
        default_inpaint_decoder_path,
    ) = checkpoint.base_checkpoints_path(kubin.params.cache_dir)

    if (
        checkpoint.prior_model_dir == default_checkpoint.prior_model_dir
        and checkpoint.prior_model_name == default_checkpoint.prior_model_name
    ):
        prior_path = default_prior_path
    else:
        prior_path = os.path.normpath(
            os.path.join(checkpoint.prior_model_dir, checkpoint.prior_model_name)
        )

    if (
        checkpoint.decoder_model_dir == default_checkpoint.decoder_model_dir
        and checkpoint.decoder_model_name == default_checkpoint.decoder_model_name
    ):
        decoder_path = default_decoder_path
    else:
        decoder_path = os.path.normpath(
            os.path.join(checkpoint.decoder_model_dir, checkpoint.decoder_model_name)
        )

    if (
        checkpoint.inpaint_model_dir == default_checkpoint.inpaint_model_dir
        and checkpoint.inpaint_model_name == default_checkpoint.inpaint_model_name
    ):
        inpaint_path = default_inpaint_decoder_path
    else:
        inpaint_path = os.path.normpath(
            os.path.join(checkpoint.inpaint_model_dir, checkpoint.inpaint_model_name)
        )

    return "<br />".join(
        [
            f"Current prior checkpoint path: <b>{prior_path}</b>",
            f"Current decoder checkpoint path: <b>{decoder_path}</b>",
            f"Current inpaint decoder checkpoint path: <b>{inpaint_path}</b>",
        ]
    )


def extract_checkpoint_path(checkpoint_file: str):
    directory = os.path.dirname(checkpoint_file)
    filename = os.path.basename(checkpoint_file)

    return directory, filename


def scan_checkpoints(directories, pattern, default_checkpoint, default_text):
    matching_files = []

    for directory in directories.split(";"):
        for root, dirnames, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, pattern):
                checkpoint_file = os.path.normpath(os.path.join(root, filename))
                matching_files.append(
                    default_text
                    if os.path.exists(default_checkpoint)
                    and os.path.samefile(checkpoint_file, default_checkpoint)
                    else checkpoint_file
                )

    if not default_text in matching_files:
        matching_files = [default_text] + matching_files

    return matching_files


def rescan_checkpoints(
    kubin,
    directories,
    prior_pattern,
    decoder_pattern,
    inpaint_pattern,
    default_prior,
    default_decoder,
    default_inpaint,
):
    prior_checkpoints = scan_checkpoints(
        directories, prior_pattern, default_prior, prior_base_path_placeholder
    )
    decoder_checkpoints = scan_checkpoints(
        directories, decoder_pattern, default_decoder, decoder_base_path_placeholder
    )
    inpaint_checkpoints = scan_checkpoints(
        directories,
        inpaint_pattern,
        default_inpaint,
        inpaint_decoder_base_path_placeholder,
    )
    return [
        gr.update(choices=prior_checkpoints),
        gr.update(choices=decoder_checkpoints),
        gr.update(choices=inpaint_checkpoints),
    ]
