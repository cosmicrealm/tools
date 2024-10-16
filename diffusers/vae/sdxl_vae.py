from diffusers.models import AutoencoderKL
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

if __name__ == "__main__":
    vae_sdxl = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae", torch_dtype=torch.float32).to("cuda")
    image = Image.open("../asserts/baby.png")
    
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    
    with torch.no_grad():
        image_tensor = transform(image.convert("RGB"))
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.cuda()
        latents = vae_sdxl.encode(image_tensor).latent_dist.sample()
        recovery_img = vae_sdxl.decode(latents).sample
        recovery_img = recovery_img.squeeze(0)
        recovery_img = recovery_img.permute(1, 2, 0)
        recovery_img = recovery_img.detach().cpu().numpy()
    recovery_img = (recovery_img + 1) / 2
    recovery_img = np.clip(recovery_img, 0, 1)
    recovery_img = (recovery_img * 255).astype(np.uint8)
    recovery_img = Image.fromarray(recovery_img)
    recovery_img.save("baby_recovered.png")