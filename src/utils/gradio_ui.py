import gradio as gr


def click_and_disable(element, fn, inputs, outputs):
    element.click(
        fn=lambda: gr.update(interactive=False),
        queue=False,
        outputs=element,
    ).then(fn=fn, inputs=inputs, outputs=outputs).then(
        fn=lambda: gr.update(interactive=True),
        queue=False,
        outputs=element,
    )
