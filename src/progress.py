import gradio as gr

task_progress = {"progress": {}, "poll_interval": -1, "cancel": False}


def report_progress(task, stage, total_steps, step, timestep, latents):
    task_progress["progress"] = {
        "task": task,
        "stage": stage,
        "total_steps": total_steps,
        "step": step,
        "timestep": timestep,
        "latents": latents,
    }


def progress_api(kubin):
    task_progress["poll_interval"] = kubin.params("ui", "progress_poll_interval")

    def output_progress():
        return task_progress

    with gr.Row(visible=False):
        dummy_progress_btn = gr.Button()
        results_json = gr.JSON()

        dummy_progress_btn.click(
            output_progress,
            inputs=None,
            outputs=results_json,
            api_name="progress",
        )
