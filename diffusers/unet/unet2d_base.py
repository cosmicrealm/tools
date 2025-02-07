from diffusers.models.unets.unet_2d import UNet2DModel, UNet2DOutput
import torch
from typing import Optional, Tuple, Union

if __name__ == '__main__':
    
    unet2d_kwargs = {
        "sample_size": None,
        "in_channels": 3,
        "out_channels": 3,
        "center_input_sample": False,
        "time_embedding_type": "fourier",
        "freq_shift": 0,
        "flip_sin_to_cos": True,
        "down_block_types": ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        "up_block_types": ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        "block_out_channels": (224, 448, 672, 896),
        "layers_per_block": 2,
        "mid_block_scale_factor": 1,
        "downsample_padding": 1,
        "downsample_type": "conv",
        "upsample_type": "conv",
        "dropout": 0.0,
        "act_fn": "silu",
        "attention_head_dim": 8,
        "norm_num_groups": 32,
        "attn_norm_num_groups": None,
        "norm_eps": 1e-5,
        "resnet_time_scale_shift": "default",
        "add_attention": True,
        "class_embed_type": None,
        "num_class_embeds": None,
        "num_train_timesteps": None,
    }
    
    x_input = torch.randn(1, 3, 64, 64)
    # 创建 UNet2DModel 实例
    unet2d_model = UNet2DModel(**unet2d_kwargs)
    
    # 传递输入张量并获取输出
    output = unet2d_model(x_input, timestep=torch.tensor([2]))
    
    print("Output shape:", output.sample.shape)