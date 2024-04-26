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

USAGE WITH LORA
zyd232's Ink Style
https://civitai.com/models/73305/zyd232s-ink-style?modelVersionId=78018
Trigger Words: ZYDINK  MONOCHROME   INK SKETCH

Industrial Machines
Trigger Words: MSHN   MSHN ROBOT

Badass Cars
https://civitai.com/models/54798/badass-cars?modelVersionId=59176
Trigger Words: ZEEKARS

MANAGE IMAGE METADATA
https://stackoverflow.com/questions/58399070/how-do-i-save-custom-information-to-a-png-image-file-in-python

"""

from stable_diffusion_cpp import StableDiffusion
import random
import string
import sys
import datetime
from PIL import Image
from PIL.PngImagePlugin import PngInfo

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
posprompt = "Photography, futuristic car, red car, ((skidding on a bend)), drifting in a snowy road, turning snow road, drift smoke, powerful drift, sport car, wonderful, frozen lake background, realistic, photo, 4k, professional photography, epic, intense, 4k, beautiful"
negprompt = "bad, ugly, deformed, out of frame, watermark"
loras1 = '<lora:zyd232_InkStyle_v1_0:0.8>'
p1 = f"A beauftiful sunset in the mountains {loras1}, ther is a forest and a lake, reflecting the lights from the sunset, zydink, ink sketch"
#---
loras2 = '<lora:zeekars:0.8>'
p2 = f"a 3/4 front view of ((futuristic dark blue lamborghini zeekars)) (with glowing tires), orange glow tires, at the parking lot, {loras2}"
##### 5 STEPS ############################################################################
SampleStps = 5
print(f'2. starting image genration with LORAS {loras2} and {SampleStps} SAMPLE STEPS...')
print("\033[91;1m")  #red
print(f'POSITIVE PROMPT: {p2}\nNEGATIVE PROMPT: {negprompt}')
print("\033[1;30m")  #dark grey
start = datetime.datetime.now()
output = stable_diffusion.txt_to_img(        
    prompt =  p2,   #"a lovely cat", # Prompt
    negative_prompt = negprompt,    
    sample_steps=SampleStps
)
delta = datetime.datetime.now() - start
fname = f"image-{genRANstring(5)}.png"
print("\033[92;1m")
print(f'Completed in {delta}')
print(f'3. Save image to file with metadata in {fname}...')
metadata = PngInfo()
metadata.add_text("ImageName", fname)
metadata.add_text("PositivePrompt", p2)
metadata.add_text("NegativePrompt", negprompt)
metadata.add_text("SDMODEL", 'StableDiffusion 1.5 pruned-emaonly')
metadata.add_text("LoraSettings", loras2)
metadata.add_text("SampleSteps", str(SampleStps))
metadata.add_text("GenTime", str(delta))
#SAVE IMAGE WITH METADATA
output[0].save(fname,pnginfo=metadata )
print('Image saved')
targetImage = Image.open(fname)
print(targetImage.text)
targetImage.show()
##### 10 STEPS ############################################################################
SampleStps = 10
print(f'2. starting image genration with LORAS {loras2} and {SampleStps} SAMPLE STEPS...')
print("\033[91;1m")  #red
print(f'POSITIVE PROMPT: {p2}\nNEGATIVE PROMPT: {negprompt}')
print("\033[1;30m")  #dark grey
start = datetime.datetime.now()
output = stable_diffusion.txt_to_img(        
    prompt =  p2,   #"a lovely cat", # Prompt
    negative_prompt = negprompt,    
    sample_steps=SampleStps
)
delta = datetime.datetime.now() - start
fname = f"image-{genRANstring(5)}.png"
print("\033[92;1m")
print(f'Completed in {delta}')
print(f'3. Save image to file with metadata in {fname}...')
metadata = PngInfo()
metadata.add_text("ImageName", fname)
metadata.add_text("PositivePrompt", p2)
metadata.add_text("NegativePrompt", negprompt)
metadata.add_text("SDMODEL", 'StableDiffusion 1.5 pruned-emaonly')
metadata.add_text("LoraSettings", loras2)
metadata.add_text("SampleSteps", str(SampleStps))
metadata.add_text("GenTime", str(delta))
#SAVE IMAGE WITH METADATA
output[0].save(fname,pnginfo=metadata )
print('Image saved')
targetImage = Image.open(fname)
print(targetImage.text)
targetImage.show()
##### 20 STEPS ############################################################################
SampleStps = 20
print(f'2. starting image genration with LORAS {loras2} and {SampleStps} SAMPLE STEPS...')
print("\033[91;1m")  #red
print(f'POSITIVE PROMPT: {p2}\nNEGATIVE PROMPT: {negprompt}')
print("\033[1;30m")  #dark grey
start = datetime.datetime.now()
output = stable_diffusion.txt_to_img(        
    prompt =  p2,   #"a lovely cat", # Prompt
    negative_prompt = negprompt,    
    sample_steps=SampleStps
)
delta = datetime.datetime.now() - start
fname = f"image-{genRANstring(5)}.png"
print("\033[92;1m")
print(f'Completed in {delta}')
print(f'3. Save image to file with metadata in {fname}...')
metadata = PngInfo()
metadata.add_text("ImageName", fname)
metadata.add_text("PositivePrompt", p2)
metadata.add_text("NegativePrompt", negprompt)
metadata.add_text("SDMODEL", 'StableDiffusion 1.5 pruned-emaonly')
metadata.add_text("LoraSettings", loras2)
metadata.add_text("SampleSteps", str(SampleStps))
metadata.add_text("GenTime", str(delta))
#SAVE IMAGE WITH METADATA
output[0].save(fname,pnginfo=metadata )
print('Image saved')
targetImage = Image.open(fname)
print(targetImage.text)
targetImage.show()
##### 30 STEPS ############################################################################
SampleStps = 30
print(f'2. starting image genration with LORAS {loras2} and {SampleStps} SAMPLE STEPS...')
print("\033[91;1m")  #red
print(f'POSITIVE PROMPT: {p2}\nNEGATIVE PROMPT: {negprompt}')
print("\033[1;30m")  #dark grey
start = datetime.datetime.now()
output = stable_diffusion.txt_to_img(        
    prompt =  p2,   #"a lovely cat", # Prompt
    negative_prompt = negprompt,    
    sample_steps=SampleStps
)
delta = datetime.datetime.now() - start
fname = f"image-{genRANstring(5)}.png"
print("\033[92;1m")
print(f'Completed in {delta}')
print(f'3. Save image to file with metadata in {fname}...')
metadata = PngInfo()
metadata.add_text("ImageName", fname)
metadata.add_text("PositivePrompt", p2)
metadata.add_text("NegativePrompt", negprompt)
metadata.add_text("SDMODEL", 'StableDiffusion 1.5 pruned-emaonly')
metadata.add_text("LoraSettings", loras2)
metadata.add_text("SampleSteps", str(SampleStps))
metadata.add_text("GenTime", str(delta))
#SAVE IMAGE WITH METADATA
output[0].save(fname,pnginfo=metadata )
print('Image saved')
targetImage = Image.open(fname)
print(targetImage.text)
targetImage.show()
##### 40 STEPS ############################################################################
SampleStps = 40
print(f'2. starting image genration with LORAS {loras2} and {SampleStps} SAMPLE STEPS...')
print("\033[91;1m")  #red
print(f'POSITIVE PROMPT: {p2}\nNEGATIVE PROMPT: {negprompt}')
print("\033[1;30m")  #dark grey
start = datetime.datetime.now()
output = stable_diffusion.txt_to_img(        
    prompt =  p2,   #"a lovely cat", # Prompt
    negative_prompt = negprompt,    
    sample_steps=SampleStps
)
delta = datetime.datetime.now() - start
fname = f"image-{genRANstring(5)}.png"
print("\033[92;1m")
print(f'Completed in {delta}')
print(f'3. Save image to file with metadata in {fname}...')
metadata = PngInfo()
metadata.add_text("ImageName", fname)
metadata.add_text("PositivePrompt", p2)
metadata.add_text("NegativePrompt", negprompt)
metadata.add_text("SDMODEL", 'StableDiffusion 1.5 pruned-emaonly')
metadata.add_text("LoraSettings", loras2)
metadata.add_text("SampleSteps", str(SampleStps))
metadata.add_text("GenTime", str(delta))
#SAVE IMAGE WITH METADATA
output[0].save(fname,pnginfo=metadata )
print('Image saved')
targetImage = Image.open(fname)
print(targetImage.text)
targetImage.show()