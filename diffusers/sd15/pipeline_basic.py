from diffusers import StableDiffusionPipeline
import torch
import os
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
os.makedirs("results", exist_ok=True)
image.save("results/astronaut_rides_horse.png")
