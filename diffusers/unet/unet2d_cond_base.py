from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel, UNet2DConditionOutput
import torch
from typing import Optional, Tuple, Union

if __name__ == '__main__':
    
    unet2d_cond_kwargs = {
        "sample_size": None,
        "in_channels": 4,
        "out_channels": 4,
        "center_input_sample": False,
        "flip_sin_to_cos": True,
        "freq_shift": 0,
        "down_block_types": (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        "mid_block_type": "UNetMidBlock2DCrossAttn",
        "up_block_types": ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        "only_cross_attention": False,
        "block_out_channels": (32, 64, 128, 128),
        "layers_per_block": 2,
        "downsample_padding": 1,
        "mid_block_scale_factor": 1,
        "dropout": 0.0,
        "act_fn": "silu",
        "norm_num_groups": 16,
        "norm_eps": 1e-5,
        "cross_attention_dim": 64,
        "transformer_layers_per_block": 1,
        "reverse_transformer_layers_per_block": None,
        "encoder_hid_dim": None,
        "encoder_hid_dim_type": None,
        "attention_head_dim": 8,
        "num_attention_heads": None,
        "dual_cross_attention": False,
        "use_linear_projection": False,
        "class_embed_type": None,
        "addition_embed_type": None,
        "addition_time_embed_dim": None,
        "num_class_embeds": None,
        "upcast_attention": False,
        "resnet_time_scale_shift": "default",
        "resnet_skip_time_act": False,
        "resnet_out_scale_factor": 1.0,
        "time_embedding_type": "positional",
        "time_embedding_dim": None,
        "time_embedding_act_fn": None,
        "timestep_post_act": None,
        "time_cond_proj_dim": None,
        "conv_in_kernel": 3,
        "conv_out_kernel": 3,
        "projection_class_embeddings_input_dim": None,
        "attention_type": "default",
        "class_embeddings_concat": False,
        "mid_block_only_cross_attention": None,
        "cross_attention_norm": None,
        "addition_embed_type_num_heads": 64,
    }
    
    # 创建 UNet2DConditionModel 实例
    unet2d_cond_model = UNet2DConditionModel(**unet2d_cond_kwargs)
    
    # 传递输入张量并获取输出
    x_input = torch.randn(1, 4, 64, 64)
    timestep = torch.tensor([1])
    # 多个 encoder_hidden_states，形状为 (batch_size, num_encoder_hidden_states, hidden_state_dim)，一般为文本或者图像的编码
    encoder_hidden_states = torch.randn(1, 4, 64)
    output = unet2d_cond_model(x_input, timestep=timestep, encoder_hidden_states=encoder_hidden_states)
    
    print("Output shape:", output.sample.shape)