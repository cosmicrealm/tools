import diffusers    
import torch
from diffusers.models.unets.unet_1d import UNet1DModel, UNet1DOutput
from typing import Optional, Tuple

if __name__ == '__main__':
    print("diffusers version:", diffusers.__version__)
    
    batch_size = 2
    in_channels = 64
    out_channels = 128
    sample_size = 512
    
    network_unet1d = UNet1DModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,
        extra_in_channels=0,
        time_embedding_type="positional",
        flip_sin_to_cos=True,
        use_timestep_embedding=True,
        freq_shift=0.0,
        down_block_types=("DownResnetBlock1D", "DownResnetBlock1D"),
        up_block_types=("UpResnetBlock1D", "UpResnetBlock1D"),
        mid_block_type="MidResTemporalBlock1D",
        out_block_type= "OutConv1DBlock",
        block_out_channels=(32, 32,),
        act_fn="mish",
    )
    input_tensor = torch.randn(batch_size, in_channels, sample_size)
    time_step = torch.tensor([0] * batch_size)  # 确保 time_step 是一维张量
    output_tensor = network_unet1d(input_tensor, timestep=time_step)
    print("output_tensor.shape:", output_tensor.sample.shape)