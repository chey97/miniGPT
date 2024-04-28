# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from selfAttentionModule import SelfAttention

# Define Transformer block module
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, head_count):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, head_count)  # Self-attention layer
        self.norm1 = nn.LayerNorm(embed_size)  # Layer normalization
        self.norm2 = nn.LayerNorm(embed_size)  # Layer normalization
        
        # Feed-forward neural network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size)
        )
    
    def forward(self, embeddings):
        attention = self.attention(embeddings)
        
        # Apply residual connections and layer normalization
        out = self.norm1(attention + embeddings)
        out = attention + self.feed_forward(out)
        out = self.norm2(out)
        return out