import torch
import torch.nn.functional as F
import time

class SelfAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        self.out = torch.nn.Linear(d_model, d_model)

    def forward(self, x):
        # Batch size, sequence length, model dimension
        B, N, D = x.shape
        
        # Linear transformations
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Reshape for multi-head attention
        Q = Q.view(B, N, self.n_heads, D // self.n_heads).transpose(1, 2)
        K = K.view(B, N, self.n_heads, D // self.n_heads).transpose(1, 2)
        V = V.view(B, N, self.n_heads, D // self.n_heads).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(B, N, D)
        
        # Final linear transformation
        out = self.out(context)
        return out

class FlashAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads, chunk_size):
        super(FlashAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.chunk_size = chunk_size
        
        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        self.out = torch.nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, D = x.shape
        C = self.chunk_size
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        Q = Q.view(B, N, self.n_heads, D // self.n_heads).transpose(1, 2)
        K = K.view(B, N, self.n_heads, D // self.n_heads).transpose(1, 2)
        V = V.view(B, N, self.n_heads, D // self.n_heads).transpose(1, 2)
        
        output_chunks = []
        for i in range(0, N, C):
            Q_chunk = Q[:, :, i:i+C]
            K_chunk = K[:, :, i:i+C]
            V_chunk = V[:, :, i:i+C]
            
            scores = torch.matmul(Q_chunk, K_chunk.transpose(-2, -1)) / (D ** 0.5)
            attention_weights = F.softmax(scores, dim=-1)
            context = torch.matmul(attention_weights, V_chunk)
            
            output_chunks.append(context)
        
        context = torch.cat(output_chunks, dim=2)
        context = context.transpose(1, 2).contiguous().view(B, N, D)
        
        out = self.out(context)
        return out

# Example usage, O(N^2) vs O(N)
'''
简易版本的 attention 和 flash attention 的对比
1.	计算复杂度：
	•	传统自注意力机制计算整个序列的注意力权重矩阵，复杂度为 O(N^2)。
	•	Flash Attention 将序列划分为较小的块，计算块内部的注意力，复杂度降低。
2.	内存占用：
    •	传统自注意力需要存储整个注意力权重矩阵，内存占用高。
    •	Flash Attention 通过分块处理减少了中间结果的存储，内存占用较低。
3.	计算效率：
    •	Flash Attention 通过并行计算块内的注意力，可以更好地利用硬件资源，计算效率更高。
'''
x = torch.rand(2, 10000, 64)  # Batch size 2, sequence length 10, model dimension 64
self_attention = SelfAttention(d_model=64, n_heads=8)
t_start = time.time()
output = self_attention(x)
t_end = time.time()
print(f"normal attention cost time:{(t_end-t_start):.4f}", output.shape, )  # Output shape: (2, 10, 64)

flash_attention = FlashAttention(d_model=64, n_heads=8, chunk_size=4)
t_start = time.time()
output = flash_attention(x)
t_end = time.time()
print(f"flash attention cost time:{(t_end-t_start):.4f}", output.shape, )  # Output shape: (2, 10, 64)