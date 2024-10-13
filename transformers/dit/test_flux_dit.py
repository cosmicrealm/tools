from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
import torch

if __name__ == "__main__":
    flux_model = FluxTransformer2DModel(
        patch_size = 1,
        in_channels = 64,
        num_layers = 19,
        num_single_layers = 38,
        attention_head_dim = 128,
        num_attention_heads = 24,
        joint_attention_dim = 4096,
        pooled_projection_dim = 768,
        guidance_embeds = False,
        axes_dims_rope = [16, 56, 56],
    )
    hidden_states = torch.randn(3, 4, 8, 8)
    timestep = torch.Tensor([0, 1, 2,])
    class_labels = torch.randint(0, 1000, (1,))
    output = flux_model(hidden_states, timestep, class_labels).sample
    print(output.shape)