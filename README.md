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
