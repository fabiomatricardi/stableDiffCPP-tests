from stable_diffusion_cpp import StableDiffusion
import random
import string
import sys
import datetime
from PIL import Image

def genRANstring(n):
    """
    n = int number of char to randomize
    """
    N = n
    res = ''.join(random.choices(string.ascii_uppercase +
                                string.digits, k=N))
    return res

start = datetime.datetime.now()
print('1. Loading pipeline MODEL SD v1-5-pruned-emaonly')

stable_diffusion = StableDiffusion(
      model_path="model/v1-5-pruned-emaonly.safetensors",
      wtype="q5_0", # Weight type (options: default, f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0)
      # seed=1337, # Uncomment to set a specific seed
)
delta = datetime.datetime.now() - start
print(f'Completed in {delta}')

#prompt = "Self-portrait cartoon style, a beautiful cyborg with golden hair, 8k"
posprompt = "Photography, futuristic car, red car, ((skidding on a bend)), drifting in a snowy road, turning snow road, drift smoke, powerful drift, sport car, wonderful, frozen lake background, realistic, photo, 4k, professional photography, epic, intense, 4k, beautiful"
negprompt = "bad, ugly, deformed, out of frame, watermark, text, written text, signature, solo, far target, bad target, far zoom"
print('2. starting image genration...')
print("\033[91;1m")  #red
print(f'POSITIVE PROMPT: {posprompt}\nNEGATIVE PROMPT: {negprompt}')
print("\033[1;30m")  #dark grey
start = datetime.datetime.now()
output = stable_diffusion.txt_to_img(        
    prompt =  posprompt,   #"a lovely cat", # Prompt
    negative_prompt = negprompt,    
    sample_steps=20
)
delta = datetime.datetime.now() - start
fname = f"image-{genRANstring(5)}.png"
print("\033[92;1m")
print(f'Completed in {delta}')
print(f'3. Same image to file {fname}...')
output[0].save(fname)
print('Image saved')
output[0].show()