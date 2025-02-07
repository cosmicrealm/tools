import torch
from diffusers.models.unets.unet_2d_blocks import (
    AttnUpDecoderBlock2D,
    AttnSkipUpBlock2D,
    SkipUpBlock2D,
    ResnetUpsampleBlock2D,
    SimpleCrossAttnUpBlock2D,
    KUpBlock2D,
    KCrossAttnUpBlock2D,
    KAttentionBlock,
    AutoencoderTinyBlock,
    UNetMidBlock2D,
    UNetMidBlock2DCrossAttn,
    UNetMidBlock2DSimpleCrossAttn,
    AttnDownBlock2D,
    CrossAttnDownBlock2D,
    DownBlock2D,
    DownEncoderBlock2D,
    AttnDownEncoderBlock2D,
    AttnSkipDownBlock2D,
    SkipDownBlock2D,
    ResnetDownsampleBlock2D,
    SimpleCrossAttnDownBlock2D,
    KDownBlock2D,
    KCrossAttnDownBlock2D,
    AttnUpBlock2D,
    CrossAttnUpBlock2D,
    UpBlock2D,
    UpDecoderBlock2D,
    AttnUpDecoderBlock2D,
)

def test_autoencoder_tiny_block():
    block = AutoencoderTinyBlock(in_channels=3, out_channels=3, act_fn="relu")
    x = torch.randn(1, 3, 64, 64)
    y = block(x)
    print("AutoencoderTinyBlock output shape:", y.shape)

def test_unet_mid_block_2d():
    block = UNetMidBlock2D(in_channels=3, temb_channels=3, resnet_groups=3)
    x = torch.randn(1, 3, 64, 64)
    temb = torch.randn(1, 3)
    y = block(x, temb)
    print("UNetMidBlock2D output shape:", y.shape)

def test_unet_mid_block_2d_cross_attn():
    block = UNetMidBlock2DCrossAttn(in_channels=3, temb_channels=3, resnet_groups=3, cross_attention_dim=3)
    x = torch.randn(1, 3, 64, 64)
    temb = torch.randn(1, 3)
    y = block(x, temb)
    print("UNetMidBlock2DCrossAttn output shape:", y.shape)

def test_unet_mid_block_2d_simple_cross_attn():
    block = UNetMidBlock2DSimpleCrossAttn(in_channels=3, temb_channels=3, resnet_groups=3, cross_attention_dim=3)
    x = torch.randn(1, 3, 64, 64)
    temb = torch.randn(1, 3)
    y = block(x, temb)
    print("UNetMidBlock2DSimpleCrossAttn output shape:", y.shape)

def test_attn_down_block_2d():
    block = AttnDownBlock2D(in_channels=3, out_channels=3, temb_channels=3, resnet_groups=3)
    x = torch.randn(1, 3, 64, 64)
    temb = torch.randn(1, 3)
    y, _ = block(x, temb)
    print("AttnDownBlock2D output shape:", y.shape)

def test_cross_attn_down_block_2d():
    block = CrossAttnDownBlock2D(in_channels=3, out_channels=3, temb_channels=3, resnet_groups=3, cross_attention_dim=3)
    x = torch.randn(1, 3, 64, 64)
    temb = torch.randn(1, 3)
    y, _ = block(x, temb)
    print("CrossAttnDownBlock2D output shape:", y.shape)

def test_down_block_2d():
    block = DownBlock2D(in_channels=3, out_channels=3, temb_channels=3, resnet_groups=3)
    x = torch.randn(1, 3, 64, 64)
    temb = torch.randn(1, 3)
    y, _ = block(x, temb)
    print("DownBlock2D output shape:", y.shape)

def test_down_encoder_block_2d():
    block = DownEncoderBlock2D(in_channels=3, out_channels=3, resnet_groups=3)
    x = torch.randn(1, 3, 64, 64)
    y = block(x)
    print("DownEncoderBlock2D output shape:", y.shape)

def test_attn_down_encoder_block_2d():
    block = AttnDownEncoderBlock2D(in_channels=3, out_channels=3, resnet_groups=3, cross_attention_dim=3)
    x = torch.randn(1, 3, 64, 64)
    y = block(x)
    print("AttnDownEncoderBlock2D output shape:", y.shape)

def test_attn_skip_down_block_2d():
    block = AttnSkipDownBlock2D(in_channels=32, out_channels=32, temb_channels=32)
    x = torch.randn(1, 32, 64, 64)
    temb = torch.randn(1, 32)
    y, _, _ = block(x, temb)
    print("AttnSkipDownBlock2D output shape:", y.shape)

def test_skip_down_block_2d():
    block = SkipDownBlock2D(in_channels=3, out_channels=3, temb_channels=3, resnet_groups=3)
    x = torch.randn(1, 3, 64, 64)
    temb = torch.randn(1, 3)
    y, _, _ = block(x, temb)
    print("SkipDownBlock2D output shape:", y.shape)

def test_resnet_downsample_block_2d():
    block = ResnetDownsampleBlock2D(in_channels=3, out_channels=3, temb_channels=3, resnet_groups=3)
    x = torch.randn(1, 3, 64, 64)
    temb = torch.randn(1, 3)
    y, _ = block(x, temb)
    print("ResnetDownsampleBlock2D output shape:", y.shape)

def test_simple_cross_attn_down_block_2d():
    block = SimpleCrossAttnDownBlock2D(in_channels=3, out_channels=3, temb_channels=3, cross_attention_dim=3)
    x = torch.randn(1, 3, 64, 64)
    temb = torch.randn(1, 3)
    y, _ = block(x, temb)
    print("SimpleCrossAttnDownBlock2D output shape:", y.shape)

def test_k_down_block_2d():
    block = KDownBlock2D(in_channels=3, out_channels=3, temb_channels=3, resnet_groups=3)
    x = torch.randn(1, 3, 64, 64)
    temb = torch.randn(1, 3)
    y, _ = block(x, temb)
    print("KDownBlock2D output shape:", y.shape)

def test_k_cross_attn_down_block_2d():
    block = KCrossAttnDownBlock2D(in_channels=3, out_channels=3, temb_channels=3, cross_attention_dim=3, resnet_groups=3)
    x = torch.randn(1, 3, 64, 64)
    temb = torch.randn(1, 3)
    y, _ = block(x, temb)
    print("KCrossAttnDownBlock2D output shape:", y.shape)

def test_attn_up_block_2d():
    block = AttnUpBlock2D(in_channels=3, prev_output_channel=3, out_channels=3, temb_channels=3, resnet_groups=3, cross_attention_dim=3)
    x = torch.randn(1, 3, 64, 64)
    res_hidden_states_tuple = (torch.randn(1, 3, 64, 64),)
    y = block(x, res_hidden_states_tuple)
    print("AttnUpBlock2D output shape:", y.shape)

def test_cross_attn_up_block_2d():
    block = CrossAttnUpBlock2D(in_channels=3, out_channels=3, prev_output_channel=3, temb_channels=3, cross_attention_dim=3, resnet_groups=3)
    x = torch.randn(1, 3, 64, 64)
    res_hidden_states_tuple = (torch.randn(1, 3, 64, 64),)
    y = block(x, res_hidden_states_tuple)
    print("CrossAttnUpBlock2D output shape:", y.shape)

def test_up_block_2d():
    block = UpBlock2D(in_channels=3, prev_output_channel=3, out_channels=3, temb_channels=3, resnet_groups=3)
    x = torch.randn(1, 3, 64, 64)
    res_hidden_states_tuple = (torch.randn(1, 3, 64, 64),)
    y = block(x, res_hidden_states_tuple)
    print("UpBlock2D output shape:", y.shape)

def test_up_decoder_block_2d():
    block = UpDecoderBlock2D(in_channels=3, out_channels=3)
    x = torch.randn(1, 3, 64, 64)
    y = block(x)
    print("UpDecoderBlock2D output shape:", y.shape)

def test_attn_up_decoder_block_2d():
    block = AttnUpDecoderBlock2D(in_channels=3, out_channels=3, resnet_groups=3, cross_attention_dim=3)
    x = torch.randn(1, 3, 64, 64)
    y = block(x)
    print("AttnUpDecoderBlock2D output shape:", y.shape)

def test_attn_skip_up_block_2d():
    block = AttnSkipUpBlock2D(in_channels=3, prev_output_channel=3, out_channels=3, temb_channels=3, resnet_groups=3, cross_attention_dim=3)
    x = torch.randn(1, 3, 64, 64)
    res_hidden_states_tuple = (torch.randn(1, 3, 64, 64),)
    y, _ = block(x, res_hidden_states_tuple)
    print("AttnSkipUpBlock2D output shape:", y.shape)

def test_skip_up_block_2d():
    block = SkipUpBlock2D(in_channels=3, prev_output_channel=3, out_channels=3, temb_channels=3, resnet_groups=3)
    x = torch.randn(1, 3, 64, 64)
    res_hidden_states_tuple = (torch.randn(1, 3, 64, 64),)
    y, _ = block(x, res_hidden_states_tuple)
    print("SkipUpBlock2D output shape:", y.shape)

def test_resnet_upsample_block_2d():
    block = ResnetUpsampleBlock2D(in_channels=3, prev_output_channel=3, out_channels=3, temb_channels=3, resnet_groups=3)
    x = torch.randn(1, 3, 64, 64)
    res_hidden_states_tuple = (torch.randn(1, 3, 64, 64),)
    y = block(x, res_hidden_states_tuple)
    print("ResnetUpsampleBlock2D output shape:", y.shape)

def test_simple_cross_attn_up_block_2d():
    block = SimpleCrossAttnUpBlock2D(in_channels=3, out_channels=3, prev_output_channel=3, temb_channels=3, cross_attention_dim=3, resnet_groups=3)
    x = torch.randn(1, 3, 64, 64)
    res_hidden_states_tuple = (torch.randn(1, 3, 64, 64),)
    y = block(x, res_hidden_states_tuple)
    print("SimpleCrossAttnUpBlock2D output shape:", y.shape)

def test_k_up_block_2d():
    block = KUpBlock2D(in_channels=3, out_channels=3, temb_channels=3, resnet_groups=3, resolution_idx=0)
    x = torch.randn(1, 3, 64, 64)
    res_hidden_states_tuple = (torch.randn(1, 3, 64, 64),)
    y = block(x, res_hidden_states_tuple)
    print("KUpBlock2D output shape:", y.shape)

def test_k_cross_attn_up_block_2d():
    block = KCrossAttnUpBlock2D(in_channels=3, out_channels=3, temb_channels=3, resnet_groups=3, resolution_idx=0, cross_attention_dim=3)
    x = torch.randn(1, 3, 64, 64)
    res_hidden_states_tuple = (torch.randn(1, 3, 64, 64),)
    y = block(x, res_hidden_states_tuple)
    print("KCrossAttnUpBlock2D output shape:", y.shape)

def test_k_attention_block():
    block = KAttentionBlock(dim=3, num_attention_heads=1, attention_head_dim=3)
    x = torch.randn(1, 3, 64, 64)
    y = block(x)
    print("KAttentionBlock output shape:", y.shape)

if __name__ == '__main__':
    test_autoencoder_tiny_block()
    test_unet_mid_block_2d()
    test_unet_mid_block_2d_cross_attn()
    test_unet_mid_block_2d_simple_cross_attn()
    test_attn_down_block_2d()
    test_cross_attn_down_block_2d()
    test_down_block_2d()
    test_down_encoder_block_2d()
    # test_attn_down_encoder_block_2d()
    # test_attn_skip_down_block_2d()
    test_skip_down_block_2d()
    test_resnet_downsample_block_2d()
    test_simple_cross_attn_down_block_2d()
    test_k_down_block_2d()
    test_k_cross_attn_down_block_2d()
    test_attn_up_block_2d()
    test_cross_attn_up_block_2d()
    test_up_block_2d()
    test_up_decoder_block_2d()
    test_attn_up_decoder_block_2d()
    test_attn_skip_up_block_2d()
    test_skip_up_block_2d()
    test_resnet_upsample_block_2d()
    test_simple_cross_attn_up_block_2d()
    test_k_up_block_2d()
    test_k_cross_attn_up_block_2d()
    test_k_attention_block()