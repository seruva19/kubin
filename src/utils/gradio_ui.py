import gradio as gr


def click_and_disable(element, fn, inputs=None, outputs=None, js=(None, None)):
    return (
        element.click(
            fn=lambda: gr.update(interactive=False),
            _js=js[0],
            queue=False,
            outputs=element,
        )
        .then(fn=fn, inputs=inputs, outputs=outputs)
        .then(
            _js=js[1],
            fn=lambda: gr.update(interactive=True),
            queue=False,
            outputs=element,
        )
    )


def info_message(ui_params, tooltip):
    return tooltip if ui_params("show_help_text") else None
