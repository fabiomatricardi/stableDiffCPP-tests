# PoorGPUguy use StableDiffusion on CPU
### AKA super powers of stable-diffusion-cpp-python

this is the best I can do... starting from 0.<br>
But it is not bad right?<br>
<img src='https://github.com/fabiomatricardi/stableDiffCPP-tests/raw/main/BEST-image-DD8UD.png' width=400>

## Installation
The only requirement is `stable-diffusion-cpp-python`

Create a Virtual environment
```
python -m venv venv
venv\Scripts\activate
```

Install with PIP
```
pip install stable-diffusion-cpp-python
```
Download SD1.5 [from here](https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors?download=true) and put it into `model` directory


- For GPU users follow instructions in [official GitHub Repo](https://github.com/leejet/stable-diffusion.cpp)
- you can use it as `text2image`
- you can use it as `image2image`
- you can use it as `upscaler`

>  Loras support is included (the picture above is using `retrowave_0.12` and `zeekars`<br>
>  Read Medium article for full step by step guide

For the image above I used the following
```
'ImageName': 'image-DD8UD.png'<br>
'PositivePrompt': 'futuristic black on orange CAR ZEEKARS, glowing blu from the lights, through city highway, surrounded by skyscrapers, hyper realistic RETROWAVE <lora:retrowave_0.12:0.8>, hyperdetailed, masterpiece, insane details, intricate details, ultra quality, unity 8k, best quality <lora:zeekars:0.6>'<br>
'NegativePrompt': 'bad, ugly, deformed, out of frame, watermark'<br>
'SDMODEL': 'StableDiffusion 1.5 pruned-emaonly'<br>
'LoraSettings': '<lora:retrowave_0.12:0.8>, <lora:zeekars:0.6>'<br>
'SampleMethod': '6 - DPMPP2Mv2'<br>
'CfgScale': '10'<br>
'SampleSteps': '20'<br>
```
> Image generated in  '0:22:50.471408'


---

> stable-diffusion-cpp-python can also be used with LCM
### LCM - Latent Consistency Models
https://latent-consistency-models.github.io/
<br>
We propose Latent Consistency Models (LCMs) to overcome the slow iterative sampling process of Latent Diffusion models (LDMs), enabling fast inference with minimal steps on any pre-trained LDMs (e.g Stable Diffusion).
Viewing the guided reverse diffusion process as solving an augmented probability flow ODE (PF-ODE) , LCMs predict its solution directly in latent space, achieving super fast inference with few steps.
A high-quality 768x768 LCM, distilled from Stable Diffusion, requires only 32 A100 GPU training hours (8 node for only 4 hours) for 2~4-step inference.

Here few generated images...<br>
<img src='https://github.com/fabiomatricardi/stableDiffCPP-tests/blob/main/lcm-4steps.png' width=450><br>
<img src='https://github.com/fabiomatricardi/stableDiffCPP-tests/blob/main/lcm-2-1-steps.png' width=450><br>


For example
```
Disty0/LCM_SoteMix -  https://huggingface.co/Disty0/LCM_SoteMix
A Stable Diffusion 1.5 anime model with LCM that allows image generation with only 4 steps.
More details on LCM: https://latent-consistency-models.github.io/

Original SoteMix is available here:
https://huggingface.co/Disty0/SoteMix

Coverted to LCM with this script:
https://github.com/vladmandic/automatic/blob/master/cli/lcm-convert.py
```

#### Additional resources on LCM
- [ LCM on GitHub](https://github.com/luosiallen/latent-consistency-model)
- [Official Diffusers LCM text2image](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/latent_consistency_models/pipeline_latent_consistency_text2img.py)
- [LCM on Hugging Face Hub](https://huggingface.co/docs/diffusers/api/pipelines/latent_consistency_models#latent-consistency-models)
- [Official paper *Effective Quantization for Diffusion Models on CPUs*](https://arxiv.org/pdf/2311.16133.pdf)
- [WGET for windows](https://eternallybored.org/misc/wget/)
- [StableDiffusion1.5 on Hugging Face](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main)





