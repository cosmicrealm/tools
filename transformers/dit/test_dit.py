from diffusers.models.transformers import DiTTransformer2DModel
import torch

if __name__ == "__main__":
    dit_model = DiTTransformer2DModel(
        num_attention_heads = 16,
        attention_head_dim= 72,
        in_channels = 4,
        out_channels = None,
        num_layers = 28,
        dropout = 0.0,
        norm_num_groups = 32,
        attention_bias = True,
        sample_size = 32,
        patch_size = 2,
        activation_fn = "gelu-approximate",
        num_embeds_ada_norm = 1000,
        upcast_attention = False,
        norm_type= "ada_norm_zero",
        norm_elementwise_affine = False,
        norm_eps = 1e-5,
    )
    hidden_states = torch.randn(3, 4, 8, 8)
    timestep = torch.Tensor([0, 1, 2,])
    class_labels = torch.randint(0, 1000, (1,))
    output = dit_model(hidden_states, timestep, class_labels).sample
    print(output.shape)
    