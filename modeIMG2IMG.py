"""
STABLE DIFFUSION CPP PYTHON
https://github.com/william-murray1204/stable-diffusion-cpp-python/blob/main/README.md
origninal: https://github.com/leejet/stable-diffusion.cpp
can be used also with LCM/LCM-LoRA
https://github.com/leejet/stable-diffusion.cpp#lcmlcm-lora
- Download LCM-LoRA form https://huggingface.co/latent-consistency/lcm-lora-sdv1-5
- Specify LCM-LoRA by adding <lora:lcm-lora-sdv1-5:1> to prompt
- It's advisable to set --cfg-scale to 1.0 instead of the default 7.0. For --steps, a range of 2-8 steps is recommended. For --sampling-method, lcm/euler_a is recommended.

MORE ADVANCED TECHNIQUES 
https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#lora


python scripts/img2img.py --prompt "A fantasy landscape, trending on artstation" --init-img <path-to-img.jpg> --strength 0.8



UPSCALER
--------
https://civitai.com/models/141491/4x-nmkd-superscale


MANAGE IMAGE METADATA
https://stackoverflow.com/questions/58399070/how-do-i-save-custom-information-to-a-png-image-file-in-python

--sampling-method {euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, lcm}
class SampleMethod(IntEnum):
    EULER_A = 0
    EULER = 1
    HEUN = 2
    DPM2 = 3
    DPMPP2S_A = 4
    DPMPP2M = 5
    DPMPP2Mv2 = 6
    LCM = 7
    N_SAMPLE_METHODS = 8
"""

from stable_diffusion_cpp import StableDiffusion
import random
import string
import sys
import datetime
from PIL import Image
from PIL.PngImagePlugin import PngInfo


def callback(step: int, steps: int, time: float):
    print("Completed step: {} of {}".format(step, steps))

def genRANstring(n):
    """
    n = int number of char to randomize
    """
    N = n
    res = ''.join(random.choices(string.ascii_uppercase +
                                string.digits, k=N))
    return res

start = datetime.datetime.now()
print('1. Loading pipeline MODEL SD15 with LORAS')

stable_diffusion = StableDiffusion(
      model_path="model/v1-5-pruned-emaonly.safetensors",
      wtype="q5_0", # Weight type (options: default, f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0)
      lora_model_dir="loras/",
      # seed=1337, # Uncomment to set a specific seed
)
delta = datetime.datetime.now() - start
print(f'Completed in {delta}')

#prompt = "Self-portrait cartoon style, a beautiful cyborg with golden hair, 8k"
posprompt = "A fantasy landscape, trending on artstation, fantasy art"
#negprompt = "bad, ugly, deformed, out of frame, watermark"
negprompt = 'bad, ugly, deformed, out of frame, watermark'
#loras = '<lora:retrowave_0.12:0.8>, <lora:zeekars:0.6>'
#---
########################### IMG2IMG ############################################################################
##### https://github.com/william-murray1204/stable-diffusion-cpp-python/blob/main/tests/test_img2img.py
################################################################################################################
SampleStps = 5
SampleMethod = 6
CfgScale = 10
mystrenght = 1.0
print(f'2. starting IMAGE to IMAGE and {SampleStps} SAMPLE STEPS...')
print("\033[91;1m")  #red
print(f'POSITIVE PROMPT: {posprompt}\nNEGATIVE PROMPT: {negprompt}')
print("\033[1;30m")  #dark grey
input_image = "sketch-mountains-input.jpg"
start = datetime.datetime.now()
output = stable_diffusion.img_to_img(
    prompt=posprompt, 
    #negative_prompt = negprompt,    
    image=input_image, 
    #sample_method = SampleMethod, # 6 = 'DPMPP2Mv2',
    #cfg_scale = CfgScale,  #1 for lowe steps and shorter time, 8+ for higher steps    
    #progress_callback=callback,
    #strength = mystrenght
)
delta = datetime.datetime.now() - start
fname = f"image-{genRANstring(5)}.png"
print("\033[92;1m")
print(f'Completed in {delta}')
print(f'3. Save image to file with metadata in {fname}...')
metadata = PngInfo()
metadata.add_text("ImageName", fname)
metadata.add_text("Prompt", posprompt)
metadata.add_text("NegativePrompt", negprompt)
metadata.add_text("OriginalIMAGE", input_image)
metadata.add_text("PIPELINE", 'img_to_img')
metadata.add_text("SDMODEL", 'StableDiffusion 1.5 pruned-emaonly')
metadata.add_text("Strenght", str(mystrenght))
metadata.add_text("SampleMethod", f'{SampleMethod} - DPMPP2Mv2')
metadata.add_text("CfgScale", str(CfgScale))
metadata.add_text("SampleSteps", str(SampleStps))
metadata.add_text("GenTime", str(delta))
#SAVE IMAGE WITH METADATA
output[0].save(fname,pnginfo=metadata )
print('Image saved')
targetImage = Image.open(fname)
print(targetImage.text)
targetImage.show()
