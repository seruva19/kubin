### Disclaimer

There is no comprehensive documentation (yet).
If you have questions, please feel free to open an issue or start a discussion.

### Alternative solutions

Other apps I am aware of that can be used to run Kandinsky locally:

* AI Runner https://github.com/Capsize-Games/airunner
* aiNodes Engine https://github.com/XmYx/ainodes-engine
* biniou https://github.com/Woolverine94/biniou
* Kandinsky plugin for ComfyUI https://github.com/vsevolod-oparin/comfyui-kandinsky22
* Kandinsky extension for Automatic1111 https://github.com/MMqd/kandinsky-for-automatic1111
* SD.Next https://github.com/vladmandic/automatic

### Installation and usage (scripts)
(With Python3.10 and Git installed)
* clone core repository:

```git clone https://github.com/seruva19/kubin```
* (optional) clone repository with extensions:
```
mkdir extensions
git clone https://github.com/seruva19/kubin-extensions.git extensions
```
* go to 'kubin' folder and run following scripts:

<table>
<tr>
<td></td>
<td><b>Windows</b></td>
<td><b>Linux</b></td>
</tr><tr></tr>
<tr>
<td>to install</td>
<td>install.bat</td>
<td>install.sh</td>
</tr><tr></tr>
<tr>
<td>to update</td>
<td>update.bat</td>
<td>update.sh</td>
</tr><tr></tr>
<tr>
<td>to launch</td>
<td>start.bat</td>
<td>start.sh</td>
</tr><tr></tr>
<tr>
<td>to install pytorch (optional)</td>
<td>install-torch.bat</td>
<td>install-torch.sh</td>
</tr>
</table>

* to force extensions update on application run, go to 'kubin/extensions' folder and run `update.bat` (or `update.sh` on Linux):

### Installation and usage (manual)
(Windows 10, Python 3.10, PowerShell)
```
git clone https://github.com/seruva19/kubin
cd kubin
mkdir extensions
git clone https://github.com/seruva19/kubin-extensions.git extensions
python -m venv venv
./venv/Scripts/Activate.ps1 # for powershell
call venv\Scripts\activate.bat # for command prompt
pip install -r requirements.txt
python src/kubin.py
```
GUI then should be available at `http://127.0.0.1:7860/`
<br>
To update to latest version, use:
```
git pull
./venv/Scripts/Activate.ps1 # for powershell
call venv\Scripts\activate.bat # for command prompt
pip install -r requirements.txt
```
Running on Metal GPUs (Apple) [instructions here](https://github.com/seruva19/kubin/pull/62)

### Kandinsky 3.*

Its [text encoder](https://huggingface.co/google/flan-ul2) is large, so "out-of-the-box" inference without CUDA OOM error is not possible even for GPUs with 24 Gb of VRAM.

There are two options to overcome this:
1) Run original ('native') pipeline with some optimizations that were borrowed from @SLAPaper's [work](https://github.com/seruva19/kubin/discussions/166). This pipeline is enabled by adding `kd30_low_vram` (or `kd31_low_vram` accordingly) string into 'Optimization flags' field ('Options' -> 'Native'). The optimizations  have been enabled by default, so choosing the 'kd30+native' (or 'kd31+native') pipeline will automatically reduce VRAM usage (to about 11 Gb for 3.0 and 17 Gb for 3.1). 

2) (currently only for 3.0) Run 🤗 diffusers-based pipeline, which offer a [sequential model offloading option](https://huggingface.co/docs/diffusers/optimization/memory#cpu-offloading). It should be turned on manually, go to "Settings" -> "Diffusers" tab and check the box for "Enable sequential CPU offload". 

Perhaps MPS users also might be able to run Kandinsky 3 thanks to unified memory (see https://github.com/huggingface/diffusers/issues/6028), but I haven't been able to try it out and confirm this. 

### Diffusers

In latest version of the app diffusers pipeline is activated by default.
To change it, launch app with command argument --pipeline="native" or change pipeline in 'Settings/Options' tab.

### System requirements

At default settings, full 2.2 model does not fit into 8 Gb. If you have a low-end GPU, this is what you can try:

* Go to "Settings" - "Diffusers" tab 
* Make sure the following checkboxes are turned on:
  * Enable half precision weights
  * Enable sliced attention
  * Enable sequential CPU offload 
* Another option is to turn on 'Enable prior generation on CPU' 
* Save settings and restart the app

That should decrease VRAM consumption to somewhere around 2 Gb for 512x512 image (3 Gb for 1024x1024). 
Depending of your computing power, you may try turn on/off specific flags until optimal VRAM/speed ratio will be met. 
Note that these optimizations are implemented only for 2.2 model, and not applicable to earlier models (2.0 and 2.1) or Kandinsky 3.* (which has its own optimizations, read section above).  

### FlashAttention

[FlashAttention](https://github.com/HazyResearch/flash-attention) may only be used for 2.1 'native' pipeline, and won't be used with diffusers. 
Enabling it should speed up inference. Use:
```
./venv/Scripts/Activate.ps1 # for powershell
call venv\Scripts\activate.bat # for command prompt
pip install flash-attn
python src/kubin.py --flash-attention='use'
```

Building flash-attn from source might take a while (20-40 mins) and requires installation of CUDA Toolkit and VS Build Tools. Besides that, it should be as simple as typing `pip wheel flash-attn -w /target_folder` command. 
In colab precompiled wheel is used.

### xFormers

I haven't tested it extensively. In theory, `pip install xformers` and turning on 'Enable xformers memory efficient attention' flag in Settings should enable it. But since torch 2.0 is default now and has its own internal optimizations, I don't think using xFormers is justified, even though some [extensions](https://github.com/seruva19/kubin-extensions) use xFormers.

### Theme

[Gradio theme](https://gradio.app/theming-guide/) can be changed by setting 'theme' CLI argument (or through GUI, in 'Settings' tab).  
The default theme is 'default' (yep). 4 other are 'base', 'glass' , 'monochrome' and 'soft'.  
Dark mode can be forced (as in any other gradio app) by launching URL `http://127.0.0.1:7860/?__theme=dark`  

### Developing extensions

A brief tutorial might be published after 1.0 release, but currently there are no plans for writing any kind of docs. API for extensions is not very consistent and is still a subject to change.

### Changelog

You may want to check [closed pull requests](https://github.com/seruva19/kubin/issues?q=is%3Apr+is%3Aclosed) to track features that are merged from dev branch to main.  
(Upd. 25/02/2024: nevermind, most features are now published directly to main, because why not)

### Credits and links

* Web interface: https://gradio.app/
* Kandinsky model weights: https://huggingface.co/ai-forever, https://huggingface.co/kandinsky-community
* Default styles list from: https://fusionbrain.ai/, https://github.com/Douleb/SDXL-A1111-Styles
* Diffusers Kandinsky pipeline: https://huggingface.co/docs/diffusers/main/en/api/pipelines/kandinsky
* Scripts for t2i, i2i, mix, inpaint, fine-tuning: https://github.com/ai-forever/Kandinsky-2, https://github.com/ai-forever/Kandinsky-3 
* Upscaling: https://github.com/xinntao/Real-ESRGAN, https://github.com/ai-forever/Real-ESRGAN
* 3D model generation: https://github.com/openai/shap-e
* Mask extraction: https://github.com/facebookresearch/segment-anything
* Deforum-Kandinsky: https://github.com/ai-forever/deforum-kandinsky
* Rembg: https://github.com/danielgatis/rembg
* VideoCrafter: https://github.com/AILab-CVC/VideoCrafter 
* Zero123++: https://github.com/SUDO-AI-3D/zero123plus
* Kandinsky Video: https://github.com/ai-forever/KandinskyVideo/
* Prompt interrogation: https://github.com/pharmapsychotic/clip-interrogator
* JS libraries: https://github.com/caroso1222/notyf, https://github.com/andreknieriem/simplelightbox, https://github.com/scaleflex/filerobot-image-editor
