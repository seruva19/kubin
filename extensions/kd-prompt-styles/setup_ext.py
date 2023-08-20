import gradio as gr
from pathlib import Path
import yaml
import uuid
import os


def merge_styles(*styles, only_active=False):
    merged_dict = {}
    for style in styles:
        for item in style:
            name = item["name"]
            if name in merged_dict:
                merged_dict[name].update(item)
            else:
                merged_dict[name] = item

    merged_array = list(merged_dict.values())
    if only_active:
        merged_array = [item for item in merged_array if item.get("active", True)]
    return merged_array


dir = Path(__file__).parent.absolute()
default_styles_path = f"{dir}/styles.default.yaml"
user_styles_path = f"{dir}/styles.user.yaml"


def read_default_styles():
    with open(default_styles_path, "r") as stream:
        default_data = yaml.safe_load(stream)
        return default_data["styles"]


def read_user_styles():
    if os.path.exists(user_styles_path):
        with open(user_styles_path, "r") as stream:
            user_data = yaml.safe_load(stream)
            return user_data["styles"]
    return []


def get_styles():
    return merge_styles(
        [
            {
                "name": "none",
                "prompt": None,
                "negative": None,
                "active": True,
                "source": None,
            }
        ],
        read_default_styles(),
        read_user_styles(),
        only_active=True,
    )


def write_user_styles(updated_styles):
    user_styles = read_user_styles()
    with open(user_styles_path, "w") as stream:
        data = {"styles": merge_styles(user_styles, updated_styles)}
        yaml.safe_dump(
            data,
            stream,
            default_flow_style=False,
            indent=2,
            allow_unicode=True,
            width=1000,
        )


def append_style(target, params, current_style, default_style):
    style_not_chosen = current_style["name"] == default_style["name"]
    style_prompt = current_style["prompt"]
    style_negative_prompt = current_style["negative"]

    if "prompt" in params:
        if style_not_chosen or style_prompt is None:
            None
        else:
            params["prompt"] = style_prompt.replace("{prompt}", params["prompt"])

    if "negative_prompt" in params:
        if style_not_chosen or style_negative_prompt is None:
            None
        else:
            params["negative_prompt"] = style_negative_prompt.replace(
                "{negative_prompt}", params["negative_prompt"]
            )


def setup(kubin):
    targets = ["t2i", "i2i", "mix", "inpaint", "outpaint"]

    yaml_config = kubin.yaml_utils.YamlConfig(Path(__file__).parent.absolute())
    config = yaml_config.read()

    def load_styles():
        initial_styles = get_styles()

        return (
            initial_styles,
            initial_styles[0],
            initial_styles[0],
            gr.update(
                choices=[style["name"] for style in initial_styles],
                value=initial_styles[0]["name"],
            ),
            gr.update(value=""),
        )

    def select_style(target, selected_style_name, available):
        selected_style = next(
            filter(lambda x: x["name"] == selected_style_name, available)
        )

        selected_modifier = selected_style["prompt"]
        selected_negative_modifier = selected_style["negative"]
        selected_source = selected_style["source"]

        return (
            "<br />".join(
                filter(
                    lambda x: x is not None,
                    [
                        None
                        if selected_modifier is None
                        else f"<span style='font-weight: bold'>prompt template: </span> {selected_modifier}".replace(
                            "{prompt}", "<span style='color: blue'>{prompt}</span>"
                        ),
                        None
                        if selected_negative_modifier is None
                        else f"<span style='font-weight: bold'>negative prompt template: </span> {selected_negative_modifier}".replace(
                            "{negative_prompt}",
                            "<span style='color: red'>{negative_prompt}</span>",
                        ),
                        None
                        if selected_source is None
                        else f"<br /><span>source:  {selected_source}</span>",
                    ],
                )
            ),
            gr.update(visible=selected_style["name"] != "none"),
            selected_style,
        )

    def add_style(chosen_style):
        return (
            gr.update(visible=True),
            f"User style {uuid.uuid4()}"
            if chosen_style is None
            else chosen_style["name"],
            "{prompt}" if chosen_style is None else chosen_style["prompt"],
            "{negative_prompt}" if chosen_style is None else chosen_style["negative"],
            "" if chosen_style is None else chosen_style["source"],
            gr.update(visible=False),
        )

    def save_style(name, prompt, negative_prompt, source, active):
        write_user_styles(
            [
                {
                    "name": name,
                    "prompt": prompt,
                    "negative": negative_prompt,
                    "source": source,
                    "active": active,
                }
            ]
        )

        return (gr.update(visible=True), gr.update(visible=False))

    def remove_style(name):
        user_styles = read_user_styles()
        found_style = None

        for style in user_styles:
            if style["name"] == name:
                found_style = style
                break

        if found_style is not None:
            found_style["active"] = False
        else:
            user_styles.append({"name": name, "active": False})

        write_user_styles(user_styles)

        return (
            gr.update(visible=True),
            gr.update(visible=False),
        )

    def style_select_ui(target):
        target = gr.State(value=target)

        initial_styles = get_styles()
        available_styles = gr.State(value=initial_styles)
        default_style = gr.State(value=initial_styles[0])
        current_style = gr.State(value=initial_styles[0])

        with gr.Column() as style_selector_block:
            style_search = gr.Textbox(
                "",
                label="Filter by name",
                visible=config["use_radiobutton_list"],
                interactive=True,
                elem_classes=["kd-styles-search-box"],
            )
            style_variant = (
                gr.Radio(
                    [style["name"] for style in initial_styles],
                    value=initial_styles[0]["name"],
                    show_label=False,
                    interactive=True,
                    elem_classes=["kd-styles-radiobutton-list"],
                )
                if config["use_radiobutton_list"]
                else gr.Dropdown(
                    [style["name"] for style in initial_styles],
                    value=initial_styles[0]["name"],
                    show_label=False,
                    interactive=True,
                )
            )

            style_info = gr.HTML(value="", elem_classes="block-info")

            with gr.Row() as style_edit_elements:
                add_style_btn = gr.Button("➕ Add style", size="sm")
                edit_style_btn = gr.Button("✏️ Edit style", visible=False, size="sm")
                refresh_styles_btn = gr.Button("🔄 Reload all styles", size="sm")

            with gr.Column(visible=False) as edit_prompt_elements:
                style_name = gr.Textbox(
                    label="Style name", value="", lines=1, interactive=True
                )
                style_prompt = gr.Textbox(
                    label="Style prompt", value="", lines=4, interactive=True
                )
                style_negative_prompt = gr.Textbox(
                    label="Style negative prompt", value="", lines=4, interactive=True
                )
                style_source = gr.Textbox(
                    label="Style source", value="", lines=1, interactive=True
                )

                with gr.Row():
                    save_style_btn = gr.Button("💾 Save style", size="sm")
                    cancel_style_btn = gr.Button("❌ Cancel editing", size="sm")
                    remove_style_btn = gr.Button("🗑️ Remove style", size="sm")
                gr.HTML(
                    "To apply changes after adding or editing a style, you need to press 'Refresh' button, otherwise changes won't be reflected in list."
                )

            style_variant.change(
                fn=select_style,
                inputs=[target, style_variant, available_styles],
                outputs=[
                    style_info,
                    edit_style_btn,
                    current_style,
                ],
                show_progress=False,
            )

            refresh_styles_btn.click(
                fn=load_styles,
                inputs=[],
                outputs=[
                    available_styles,
                    default_style,
                    current_style,
                    style_variant,
                    style_info,
                ],
                show_progress=False,
            )

            add_style_btn.click(
                fn=add_style,
                inputs=[gr.State(None)],
                outputs=[
                    edit_prompt_elements,
                    style_name,
                    style_prompt,
                    style_negative_prompt,
                    style_source,
                    style_edit_elements,
                ],
            )

            edit_style_btn.click(
                fn=add_style,
                inputs=[current_style],
                outputs=[
                    edit_prompt_elements,
                    style_name,
                    style_prompt,
                    style_negative_prompt,
                    style_source,
                    style_edit_elements,
                ],
            )

            save_style_btn.click(
                fn=save_style,
                inputs=[
                    style_name,
                    style_prompt,
                    style_negative_prompt,
                    style_source,
                    gr.State(True),
                ],
                outputs=[style_edit_elements, edit_prompt_elements],
            )

            cancel_style_btn.click(
                fn=lambda: [gr.update(visible=True), gr.update(visible=False)],
                inputs=[],
                outputs=[style_edit_elements, edit_prompt_elements],
            )

            remove_style_btn.click(
                fn=remove_style,
                inputs=[style_name],
                outputs=[style_edit_elements, edit_prompt_elements],
            )

        style_selector_block.elem_classes = ["kd-prompt-styles-selector"]
        return style_selector_block, current_style, default_style

    def settings_ui():
        def save_changes(use_radiobutton_list):
            config["use_radiobutton_list"] = use_radiobutton_list
            yaml_config.write(config)

        with gr.Column() as settings_block:
            use_radiobutton_list = gr.Checkbox(
                lambda: config["use_radiobutton_list"],
                label="Use list of radiobuttons for styles",
                scale=0,
            )

            save_btn = gr.Button("Save settings", size="sm", scale=0)
            save_btn.click(
                save_changes, inputs=[use_radiobutton_list], outputs=[], queue=False
            ).then(fn=None, _js=("(x) => kubin.notify.success('Settings saved')"))

        settings_block.elem_classes = ["k-form"]
        return settings_block

    return {
        "title": "Style",
        "inject_ui": lambda target: style_select_ui(target),
        "settings_ui": settings_ui,
        "inject_fn": lambda target, params, augmentations: append_style(
            target, params, augmentations[0], augmentations[1]
        ),
        "inject_position": "before_generate",
        "targets": targets,
    }
