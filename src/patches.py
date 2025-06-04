import os

check_executed = False


def get_current_env_packages():
    import sysconfig

    return sysconfig.get_path("purelib")


def patch():
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TRITON_CACHE_DIR"] = os.path.join(os.getcwd(), "__triton__")
    import gradio.analytics

    def custom_version_check():
        global check_executed
        if not check_executed:
            check_executed = True
            print(
                "fyi: kubin uses an old version of Gradio (3.50.2), which is now considered deprecated for security reasons.\nhowever, the author is too stubborn to upgrade (https://github.com/seruva19/kubin/blob/main/DOCS.md#gradio-4)."
            )

    gradio.analytics.version_check = custom_version_check

    try:
        kandinsky21_file_path = os.path.join(
            get_current_env_packages(), "kandinsky2", "__init__.py"
        )

        with open(kandinsky21_file_path, "r") as f:
            content = f.read()

        if "cached_download" in content:
            modified_content = content.replace("cached_download", "hf_hub_download")

            with open(kandinsky21_file_path, "w") as f:
                f.write(modified_content)

            print(
                "successfully replaced all occurrences of 'cached_download' with 'hf_hub_download' in kandinsky2/__init__.py"
            )
    except Exception as e:
        print(f"Cannot patch kandinsky2: {str(e)}")
        print(
            "You should manually replace all calls to 'cached_download' with 'hf_hub_download'"
        )

    try:
        pytorchvideo_file_path = os.path.join(
            get_current_env_packages(),
            "pytorchvideo",
            "transforms",
            "augmentations.py",
        )

        old_import = "import torchvision.transforms.functional_tensor as F_t"
        new_import = "import torchvision.transforms._functional_tensor as F_t"

        with open(pytorchvideo_file_path, "r") as f:
            content = f.read()

        if "import torchvision.transforms.functional_tensor as F_t" in content:
            replace_in_file(pytorchvideo_file_path, old_import, new_import)
            print(
                "patched 'torchvision.transforms.functional_tensor' import in pytorchvideo"
            )
    except:
        print("cannot patch pytorchvideo, v2a might be not functional")

    try:
        transformers_file_path = os.path.join(
            get_current_env_packages(),
            "transformers",
            "generation",
            "utils.py",
        )

        old_line = "def _extract_past_from_model_output(self, outputs: ModelOutput):"
        new_line = "def _extract_past_from_model_output(self, outputs: ModelOutput, standardize_cache_format=None):"

        with open(transformers_file_path, "r") as f:
            content = f.read()

        if (
            "def _extract_past_from_model_output(self, outputs: ModelOutput):"
            in content
        ):
            replace_in_file(transformers_file_path, old_line, new_line)
            print("patched '_extract_past_from_model_output' signature in transformers")
    except:
        print("cannot patch transformers, some modules might be not functional")


def replace_in_file(file_path, old_text, new_text):
    with open(file_path, "r") as f:
        content = f.read()

    new_content = content.replace(old_text, new_text)

    with open(file_path, "w") as f:
        f.write(new_content)
