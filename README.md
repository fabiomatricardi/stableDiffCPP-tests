# PoorGPUguy use StableDiffusion on CPU
### AKA super powers of stable-diffusion-cpp-python
Porting to python of the original project [stable-diffusion-cpp](https://github.com/leejet/stable-diffusion.cpp) inspired by Gerganov's llama.cpp
this is the best I can do... starting from 0.<br>
But it is not bad right?<br>
<img src='https://github.com/fabiomatricardi/stableDiffCPP-tests/raw/main/repo-banner.png' width=800>

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


### Image-to_Image
this fantasticlibrary can also be used with image2image pipeline<br>
<img src='https://github.com/fabiomatricardi/stableDiffCPP-tests/raw/main/img-2-img.png' width=800>

---

There is not so much informtion. What you need is to be understood from the code
- [stable_diffusion.py](https://github.com/william-murray1204/stable-diffusion-cpp-python/blob/main/stable_diffusion_cpp/stable_diffusion.py)
- [stable_diffusion_cpp.py](https://github.com/william-murray1204/stable-diffusion-cpp-python/blob/main/stable_diffusion_cpp/stable_diffusion_cpp.py)
- And the entire [folder of inference examples](https://github.com/william-murray1204/stable-diffusion-cpp-python/tree/main/tests)  but they use Low Level API


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
- [OpenELM: An Efficient Language Model Family with Open-source Training and Inference Framework](https://arxiv.org/pdf/2404.14619v1)
- [WebllM Web Assembly](https://webllm.mlc.ai/#chat-demo)
- [General Tutorial to Generate IMAGES](https://www.geeksforgeeks.org/generate-images-from-text-in-python-stable-diffusion/)

Further inspirations from [CPUONLY SD](https://github.com/darkhemic/stable-diffusion-cpuonly)

> Automatic1111 features and explanations [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#lora)

---

### Where to find Loras

You have to register, for free, to [Civita.ai](https://civitai.com/)

<img src='https://github.com/fabiomatricardi/stableDiffCPP-tests/blob/main/civitaAI.png' width=450>

And then [search for Lora(s) compatible with SD1.5](https://civitai.com/search/models?baseModel=SD%201.5&modelType=LORA&sortBy=models_v8)

There is also a convenient [Guide to Generation](https://civitai.com/articles/160)









