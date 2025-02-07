import torch
from diffusers.models.unets.unet_1d_blocks import (
    DownResnetBlock1D,
    UpResnetBlock1D,
    ValueFunctionMidBlock1D,
    MidResTemporalBlock1D,
    OutConv1DBlock,
    OutValueFunctionBlock,
    DownBlock1D,
    DownBlock1DNoSkip,
    AttnDownBlock1D,
    UpBlock1D,
    UpBlock1DNoSkip,
    AttnUpBlock1D,
    UNetMidBlock1D,
)

def test_down_resnet_block_1d():
    block = DownResnetBlock1D(in_channels=32, out_channels=32, num_layers=2)
    x = torch.randn(1, 32, 64)
    temb = torch.randn(1, 32)
    y, _ = block(x, temb)
    print("DownResnetBlock1D output shape:", y.shape)

def test_up_resnet_block_1d():
    block = UpResnetBlock1D(in_channels=16, out_channels=32, num_layers=2)
    x = torch.randn(1, 32, 64)
    temb = torch.randn(1, 32)
    y = block(x, temb=temb)
    print("UpResnetBlock1D output shape:", y.shape)

def test_value_function_mid_block_1d():
    block = ValueFunctionMidBlock1D(in_channels=32, out_channels=32, embed_dim=32)
    x = torch.randn(1, 32, 64)
    temb = torch.randn(1, 32)
    y = block(x, temb)
    print("ValueFunctionMidBlock1D output shape:", y.shape)

def test_mid_res_temporal_block_1d():
    block = MidResTemporalBlock1D(in_channels=16, out_channels=32, embed_dim=32, num_layers=2)
    x = torch.randn(1, 16, 64)
    temb = torch.randn(1, 32)
    y = block(x, temb)
    print("MidResTemporalBlock1D output shape:", y.shape)

def test_out_conv_1d_block():
    block = OutConv1DBlock(num_groups_out=8, out_channels=32, embed_dim=16, act_fn="relu")
    x = torch.randn(1, 16, 64)
    y = block(x)
    print("OutConv1DBlock output shape:", y.shape)


def test_down_block_1d():
    block = DownBlock1D(out_channels=32, in_channels=16)
    x = torch.randn(1, 16, 64)
    y, _ = block(x)
    print("DownBlock1D output shape:", y.shape)

def test_attn_down_block_1d():
    block = AttnDownBlock1D(out_channels=32, in_channels=16)
    x = torch.randn(1, 16, 64)
    y, _ = block(x)
    print("AttnDownBlock1D output shape:", y.shape)

def test_up_block_1d():
    block = UpBlock1D(in_channels=16, out_channels=32)
    x = torch.randn(1, 16, 64)
    res_hidden_states = torch.randn(1, 16, 64)
    y = block(x, (res_hidden_states,))
    print("UpBlock1D output shape:", y.shape)

def test_up_block_1d_no_skip():
    block = UpBlock1DNoSkip(in_channels=16, out_channels=32)
    x = torch.randn(1, 16, 64)
    res_hidden_states = torch.randn(1, 16, 64)
    y = block(x, (res_hidden_states,))
    print("UpBlock1DNoSkip output shape:", y.shape)

def test_attn_up_block_1d():
    block = AttnUpBlock1D(in_channels=16, out_channels=32)
    x = torch.randn(1, 16, 64)
    res_hidden_states = torch.randn(1, 16, 64)
    y = block(x, (res_hidden_states,))
    print("AttnUpBlock1D output shape:", y.shape)

def test_unet_mid_block_1d():
    block = UNetMidBlock1D(mid_channels=32, in_channels=16, out_channels=32)
    x = torch.randn(1, 16, 64)
    y = block(x)
    print("UNetMidBlock1D output shape:", y.shape)

if __name__ == '__main__':
    test_down_resnet_block_1d()
    test_up_resnet_block_1d()
    test_value_function_mid_block_1d()
    test_mid_res_temporal_block_1d()
    test_out_conv_1d_block()
    test_down_block_1d()
    test_attn_down_block_1d()
    test_up_block_1d()
    test_up_block_1d_no_skip()
    test_attn_up_block_1d()
    test_unet_mid_block_1d()