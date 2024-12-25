from diffusers.models.unets.unet_2d import UNet2DModel
import torch
input_x = torch.randn((2,3,512,512))
model = UNet2DModel(in_channels=3, out_channels=3, 
                    down_block_types = ("DownBlock2D", "AttnDownBlock2D", ),
                    up_block_types = ("AttnUpBlock2D", "AttnUpBlock2D"),
                    block_out_channels = (128, 128),
                    layers_per_block = 2,)
output = model(input_x)
print(output.shape)