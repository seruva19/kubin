
# Kubin: Web-GUI for [Kandinsky 2.1](https://github.com/ai-forever/Kandinsky-2/)

## Disclaimer

WIP - DO NOT USE 🛑

## Features

Currently only basic functions implemented (nothing new compared to [official notebooks](https://github.com/ai-forever/Kandinsky-2/tree/main/notebooks)):
* txt2img
* img2img
* mixing
* inpainting 


## Screenshots (outdated)
<details> 
<summary>Expand</summary>

### txt2img
	
![img](/sshots/t2i.png)
	
<br>

### img2img
	
![img](/sshots/i2i.png)

<br>

### mixing
	
![img](/sshots/mix.png)

<br>
	
### inpainting
    
![img](/sshots/inpaint.png)
	
</details>

## Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lx4lQS61hYb02BSoAoJUAVwPr7PhhkJt)
<br>

## Roadmap xD

* Upscaling
* Outpainting
* Extension support 
* Fine-tuning ([textual inversion](https://github.com/TheDenk/Kandinsky-2-textual-inversion) 👀)
* Advanced prompt syntax
* Interrogation
* More samplers
* SAM/Grounded SAM 🤩
* Animation
* ControlNet 🙏
* Inference optimization: memory 📉 + speed 📈
* TODO: insert another features I will never get done


## Local installation (Windows 10, Python 3.10, PowerShell)

```
git clone https://github.com/seruva19/kubin
cd kubin
py -3.10 -m venv venv
./venv/Scripts/Activate.ps1
pip install -r requirements.txt
py src/kubin.py
```
GUI then should be available at http://127.0.0.1:7860/
<br>
To update to latest version, use:
```
git pull
pip install -r requirements.txt
```

## FlashAttention

To enable [FlashAttention](https://github.com/HazyResearch/flash-attention), which should speed up inference and (theoretically) lower VRAM consumption, you may try:
```
pip install flash-attn
py src/kubin.py --use-flash-attention
```

Building flash-attn from source might take a while (20-40 mins) and requires installation of CUDA Toolkit and VS Build Tools.

In colab precompiled wheels are used.

I made some rough measurements (768x768, 100 steps, p_sampler):
|                 |T4              |T4+f/a           |V100             |V100+f/a         |RTX5000          |RTX5000+f/a      |A4000            |A4000+f/a     | 
|:----------------|:--------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|
| it/s            | 1.71           |3.24             |12.67            |15.97            |4.41             |6.38             |4.61             |6.61             |
| VRAM usage (Gb) | 8.6            |9.4              |9.5              |9.7              |11.36            |9.76             |12.48            |10.56            |

I haven't tested it on a 8 Gb GPU (FlashAttention does not support Pascal, and I have GTX 1070). But memory usage is still too high (not sure if this is expected behaviour) to run 512x512 inference. Need to investigate deeper 🙄

## Documentation

Maybe later 🤷
