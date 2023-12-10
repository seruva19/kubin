from env import Kubin
from utils.image import image_path_to_pil
import gradio as gr
from collections.abc import Iterable


class SharedUI:
    def __init__(self, kubin: Kubin, extension_targets, injected_exts):
        self.general_params = lambda a: kubin.params("general", a)
        self.ui_params = lambda a: kubin.params("ui", a)
        self.get_ext_property = lambda a, b: a.get(b, None)

        self.input_cnet_t2i_image = gr.Image(
            type="pil",
            elem_classes=["input_cnet_t2i_image", "full-height"],
            label="Reference image",
        )
        self.input_cnet_i2i_image = gr.Image(
            type="pil",
            elem_classes=["input_cnet_i2i_image", "full-height"],
            label="Reference image",
            visible=False,
        )
        self.input_i2i_image = gr.Image(
            type="pil", elem_classes=["i2i_image", "full-height"]
        )
        self.input_mix_image_1 = gr.Image(
            type="pil", elem_classes=["mix_1_image", "full-height"]
        )
        self.input_mix_image_2 = gr.Image(
            type="pil", elem_classes=["mix_2_image", "full-height"]
        )
        self.input_cnet_mix_image = gr.Image(
            type="pil",
            elem_classes=["input_cnet_mix_image", "full-height"],
            label="Reference image",
        )
        self.input_inpaint_image = gr.ImageMask(
            type="pil", elem_classes=["inpaint_image"], height=600
        )
        self.input_outpaint_image = gr.ImageMask(
            type="pil", tool="editor", elem_classes=["outpaint_image"]
        )

        self.extensions_images_targets = extension_targets
        self.extensions_augment = injected_exts

    def select_theme(self, theme):
        themes = {
            "base": lambda: gr.themes.Base(),
            "default": lambda: gr.themes.Default(),
            "glass": lambda: gr.themes.Glass(),
            "monochrome": lambda: gr.themes.Monochrome(),
            "soft": lambda: gr.themes.Soft(),
        }

        return themes.get(theme, themes["default"])()

    def open_another_tab(self, tab_index):
        return gr.Tabs.update(selected=tab_index)

    def send_gallery_image_to_another_tab(self, gallery, gallery_selected_index):
        gallery_selected_index = int(gallery_selected_index, 10)
        image_data = gallery[gallery_selected_index]
        image_url = image_data["data"]

        img = image_path_to_pil(image_url)
        return gr.update(value=img)

    def create_base_send_targets(self, output, sender, tabs):
        with gr.Row() as base_targets:
            sender_index = gr.Textbox("-1", visible=False)

            send_i2i_btn = gr.Button(
                "📸 Send to Img2Img", variant="secondary", size="sm"
            )

            send_i2i_btn.click(
                fn=self.open_another_tab,
                inputs=[gr.State(1)],
                outputs=tabs,
                queue=False,
            ).then(
                fn=self.send_gallery_image_to_another_tab,
                _js=f"(o, i) => kubin.UI.getImageIndex(o, i, '{sender}')",
                inputs=[output, sender_index],
                outputs=[self.input_i2i_image],
            )

            send_mix_1_btn = gr.Button(
                "🎨 Send to Mix (1)", variant="secondary", size="sm"
            )
            send_mix_1_btn.click(
                fn=self.open_another_tab,
                inputs=[gr.State(2)],
                outputs=tabs,
                queue=False,
            ).then(
                fn=self.send_gallery_image_to_another_tab,
                _js=f"(o, i) => kubin.UI.getImageIndex(o, i, '{sender}')",
                inputs=[output, sender_index],
                outputs=[self.input_mix_image_1],
            )

            send_mix_2_btn = gr.Button(
                "🎨 Send to Mix (2)", variant="secondary", size="sm"
            )
            send_mix_2_btn.click(
                fn=self.open_another_tab,
                inputs=[gr.State(2)],
                outputs=tabs,
                queue=False,
            ).then(
                fn=self.send_gallery_image_to_another_tab,
                _js=f"(o, i) => kubin.UI.getImageIndex(o, i, '{sender}')",
                inputs=[output, sender_index],
                outputs=[self.input_mix_image_2],
            )

            send_mix_1_btn.elem_classes = ["unsupported_20"]
            send_mix_2_btn.elem_classes = ["unsupported_20"]

            send_inpaint_btn = gr.Button(
                "🖌️ Send to Inpaint", variant="secondary", size="sm"
            )
            send_inpaint_btn.click(
                fn=self.open_another_tab,
                inputs=[gr.State(3)],
                outputs=tabs,
                queue=False,
            ).then(
                fn=self.send_gallery_image_to_another_tab,
                _js=f"(o, i) => kubin.UI.getImageIndex(o, i, '{sender}')",
                inputs=[output, sender_index],
                outputs=[self.input_inpaint_image],
            )

            send_outpaint_btn = gr.Button(
                "🖋️ Send to Outpaint",
                variant="secondary",
                size="sm",
                elem_classes=["unsupported_20"],
            )
            send_outpaint_btn.click(
                fn=self.open_another_tab,
                inputs=[gr.State(4)],
                outputs=tabs,
                queue=False,
            ).then(
                fn=self.send_gallery_image_to_another_tab,
                _js=f"(o, i) => kubin.UI.getImageIndex(o, i, '{sender}')",
                inputs=[output, sender_index],
                outputs=[self.input_outpaint_image],
            )
        base_targets.elem_classes = ["send-targets", "send-targets-base"]

        with gr.Row() as cnet_targets:
            send_cnet_t2i_btn = gr.Button(
                "🖼️ Send to T2I ControlNet",
                variant="secondary",
                elem_classes=["diffusers-kd22-control"],
                size="sm",
            )
            send_cnet_t2i_btn.click(
                fn=self.open_another_tab,
                inputs=[gr.State(0)],
                outputs=tabs,
                queue=False,
            ).then(
                fn=self.send_gallery_image_to_another_tab,
                _js=f"(o, i) => kubin.UI.getImageIndex(o, i, '{sender}')",
                inputs=[output, sender_index],
                outputs=[self.input_cnet_t2i_image],
            )

            send_cnet_i2i_btn = gr.Button(
                "🖼️ Send to I2I ControlNet",
                variant="secondary",
                elem_classes=["diffusers-kd22-control"],
                size="sm",
            )
            send_cnet_i2i_btn.click(
                fn=self.open_another_tab,
                inputs=[gr.State(1)],
                outputs=tabs,
                queue=False,
            ).then(
                fn=self.send_gallery_image_to_another_tab,
                _js=f"(o, i) => kubin.UI.getImageIndex(o, i, '{sender}')",
                inputs=[output, sender_index],
                outputs=[self.input_cnet_i2i_image],
            )

            send_cnet_mix_btn = gr.Button(
                "🖼️ Send to Mix ControlNet",
                variant="secondary",
                elem_classes=["diffusers-kd22-control"],
                size="sm",
            )
            send_cnet_mix_btn.click(
                fn=self.open_another_tab,
                inputs=[gr.State(2)],
                outputs=tabs,
                queue=False,
            ).then(
                fn=self.send_gallery_image_to_another_tab,
                _js=f"(o, i) => kubin.UI.getImageIndex(o, i, '{sender}')",
                inputs=[output, sender_index],
                outputs=[self.input_cnet_mix_image],
            )
        cnet_targets.elem_classes = ["send-targets", "send-targets-cnet"]

    def create_ext_send_targets(self, output, sender, tabs):
        with gr.Row() as send_targets:
            sender_index = gr.Textbox("-1", visible=False)

            ext_image_targets = []
            for ext in self.extensions_images_targets:
                send_target_title = (
                    self.get_ext_property(ext[0], "send_to")
                    or f"Send to {ext[0]['title']}"
                )

                send_toext_btn = gr.Button(
                    send_target_title,
                    variant="secondary",
                    size="sm",
                    scale=0,
                )
                send_toext_btn.click(
                    fn=self.open_another_tab,
                    inputs=[gr.State(ext[2])],
                    outputs=tabs,
                    queue=False,
                ).then(
                    fn=self.send_gallery_image_to_another_tab,
                    _js=f"(o, i) => kubin.UI.getImageIndex(o, i, '{sender}')",
                    inputs=[output, sender_index],
                    outputs=[ext[1]],
                )

                ext_image_targets.append(send_toext_btn)
        send_targets.elem_classes = ["send-targets", "send-targets-extensions"]

    def create_ext_augment_blocks(self, target):
        ext_exec = {}
        ext_injections = []

        def augment(t, p, a, fn):
            fn(t, p, a)
            return p

        def create_block(position):
            for ext_augment in self.extensions_augment:
                name = ext_augment["_name"]
                ext_position = ext_augment.get("inject_position", "after_params")
                if position == ext_position:
                    if target in ext_augment["targets"]:
                        current_ext = ext_exec[name] = {
                            "fn": ext_augment.get("inject_fn", lambda t, p, a: None)
                        }

                        with gr.Row() as row:
                            title = ext_augment.get(
                                "inject_title", ext_augment["title"]
                            )
                            with gr.Accordion(
                                title,
                                open=ext_augment.get("opened", lambda o: False)(target),
                            ) as ext_container:
                                ext_container.elem_classes = [
                                    "extension-container",
                                    "kubin-accordion",
                                    f"{name}",
                                    *self.availability_classes(ext_augment),
                                ]

                                ext_info = ext_augment["inject_ui"](target)
                                if isinstance(ext_info, Iterable):
                                    current_ext["input_size"] = (
                                        len(ext_injections),
                                        len(ext_injections) + len(ext_info[1:]),
                                    )
                                    for ext_injection in ext_info[1:]:
                                        ext_injections.append(ext_injection)
                                else:
                                    ext_injections.append(gr.State(None))
                                    current_ext["input_size"] = (
                                        len(ext_injections),
                                        len(ext_injections) + 1,
                                    )
                else:
                    None

        def augment_params(target, params, injections):
            for _, data in ext_exec.items():
                ext_fn = data["fn"]
                size = data["input_size"]
                params = augment(target, params, injections[size[0] : size[1]], ext_fn)
            return params

        return {
            "ui": lambda: create_block("after_params"),
            "ui_before_prompt": lambda: create_block("before_prompt"),
            "ui_before_cnet": lambda: create_block("before_cnet"),
            "ui_before_params": lambda: create_block("before_params"),
            "ui_before_generate": lambda: create_block("before_generate"),
            "ui_after_generate": lambda: create_block("after_generate"),
            "exec": lambda p, a: augment_params(target, p, a),
            "injections": ext_injections,
        }

    def info(self, text):
        return text if self.ui_params("show_help_text") else None

    def select_sampler(self, sampler20, sampler21, sampler_diffusers):
        model = self.general_params("model_name")
        pipeline = self.general_params("pipeline")

        if model == "kd20":
            return sampler20
        if model == "kd21" and pipeline == "native":
            return sampler21
        else:
            return sampler_diffusers

    def availability_classes(self, ext_augment):
        classes = []
        supports_pipeline_model = ext_augment.get(
            "supports",
            [
                "diffusers-kd30",
                "native-kd30",
                "diffusers-kd22",
                "diffusers-kd21",
                "native-kd21",
                "native-kd20",
            ],
        )
        if "native-kd20" not in supports_pipeline_model:
            classes.append("unsupported_20")

        if "native-kd21" not in supports_pipeline_model:
            classes.append("unsupported_21")

        if "diffusers-kd21" not in supports_pipeline_model:
            classes.append("unsupported_d21")

        if "diffusers-kd22" not in supports_pipeline_model:
            classes.append("unsupported_d21")

        if "native-kd30" not in supports_pipeline_model:
            classes.append("unsupported_30")

        if "diffusers-kd30" not in supports_pipeline_model:
            classes.append("unsupported_d30")

        return classes
