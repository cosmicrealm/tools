import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

prompt = "An astronaut riding a green horse"

images = pipe(prompt=prompt).images[0]
images.save("astronaut.png")