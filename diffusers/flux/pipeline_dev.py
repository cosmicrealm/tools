import torch
from diffusers import FluxPipeline
import os
# huggingface-cli download --token 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 --resume-download black-forest-labs/FLUX.1-dev --local-dir FLUX.1-dev
pipe = FluxPipeline.from_pretrained("~/models/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

prompt = "a tiny astronaut hatching from an egg on the moon"
out = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    height=768,
    width=1360,
    num_inference_steps=50,
).images[0]
os.makedirs("results", exist_ok=True)
out.save("results/image_dev.png")

