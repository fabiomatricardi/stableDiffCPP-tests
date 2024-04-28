"""
INFERENCE TEXT2IMAGE WITH
v1-5-pruned-emaonly.safetensors
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

modelname = 'v1-5-pruned-emaonly.safetensors'
stable_diffusion = StableDiffusion(
      model_path=f"model/{modelname}",#model_path="" model/v1-5-pruned-emaonly.q5_0.gguf
      wtype="default", # Weight type (options: default, f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0)
      # seed=1337, # Uncomment to set a specific seed
)
delta = datetime.datetime.now() - start
print(f'Completed in {delta}')

posprompt = "futuristic sci fi spaceship, star atlas, into earth orbit, 8k render, glowing blue light details"
negprompt = 'ugly, deformed, out of frame, watermark, blurry, blurred, amateur, low resolution, warped, crooked, low detail'

##### 20 STEPS ######
SampleStps = 20
SampleMethod = 6
CfgScale = 7
stylestrengt =0.4
iWidth = 1024
iHeight = 576
ClipSkip = 2
print(f'2. starting image genration with {SampleStps} SAMPLE STEPS...')
print("\033[91;1m")  #red
print(f'POSITIVE PROMPT: {posprompt}\nNEGATIVE PROMPT: {negprompt}')
print("\033[1;30m")  #dark grey
start = datetime.datetime.now()
output = stable_diffusion.txt_to_img(        
    prompt =  posprompt,   #"a lovely cat", # Prompt
    negative_prompt = negprompt,
    sample_method = SampleMethod, # 6 = 'DPMPP2Mv2',
    cfg_scale = CfgScale,  #1 for lowe steps and shorter time, 8+ for higher steps
    sample_steps=SampleStps,
    style_strength = stylestrengt,
    width = iWidth,
    height = iHeight, 
    clip_skip = ClipSkip,       
)
delta = datetime.datetime.now() - start
#PREPARE THE METADATA
fname = f"image-{genRANstring(5)}.png"
print("\033[92;1m")
print(f'Completed in {delta}')
print(f'3. Save image to file with metadata in {fname}...')
metadata = PngInfo()
metadata.add_text("ImageName", fname)
metadata.add_text("PositivePrompt", posprompt)
metadata.add_text("NegativePrompt", negprompt)
metadata.add_text("SDMODEL", modelname)
metadata.add_text("SampleMethod", f'{SampleMethod} - DPMPP2Mv2')
metadata.add_text("ImageSize", f'WxH: {iWidth} x {iHeight}')
metadata.add_text("CfgScale", str(CfgScale))
metadata.add_text("SampleSteps", str(SampleStps))
metadata.add_text("style_strength", str(stylestrengt))
metadata.add_text("ClipSkip", str(ClipSkip))
metadata.add_text("GenTime", str(delta))
#SAVE IMAGE WITH METADATA
output[0].save(fname,pnginfo=metadata )
print('Image saved')
targetImage = Image.open(fname)
print(targetImage.text)
targetImage.show()
