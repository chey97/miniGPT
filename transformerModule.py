# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import pprint
from selfAttentionModule import SelfAttention
from transformerBlock import TransformerBlock

# Define Transformer module
class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, head_count):
        super(Transformer, self).__init__()
        self.embed_size = embed_size  # Size of word embeddings
        self.vocab_size = vocab_size  # Size of vocabulary
        self.word_embedding = nn.Embedding(vocab_size, embed_size)  # Embedding layer
        
        # List of transformer blocks
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, head_count) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)  # Final linear layer to produce logits

    def forward(self, input_tokens, mask=None):
        batch_size, token_count = input_tokens.shape[:2]
        out = self.word_embedding(input_tokens)  # Obtain word embeddings
        
        # Compute position encodings and add to word embeddings
        positions = torch.arange(0, token_count).expand(batch_size, token_count).to(input_tokens.device)
        position_encoding = self.position_encoding(positions, self.embed_size)
        out += position_encoding.reshape(out.shape)
        
        # Pass through each transformer block
        for layer in self.layers:
            out = layer(out)
        
        # Produce logits for the final token in each sequence
        out = self.fc_out(out[:, -1, :].reshape(batch_size, self.embed_size)).reshape(batch_size, self.vocab_size)
        return torch.nn.functional.softmax(out, dim=1)  # Apply softmax to obtain probabilities

    def position_encoding(self, positions, embed_size):
        # Compute position encoding for each position and dimension
        angle_rads = self.get_angles(
            positions.unsqueeze(2).float(), 
            torch.arange(embed_size)[None, None, :].float().to(positions.device), 
            embed_size
        )
        sines = torch.sin(angle_rads[:, :, 0::2])  # Compute sine of angle for even dimensions
        cosines = torch.cos(angle_rads[:, :, 1::2])  # Compute cosine of angle for odd dimensions
        pos_encoding = torch.cat([sines, cosines], dim=-1)  # Concatenate sine and cosine values
        pos_encoding = pos_encoding[None, ...]
        return pos_encoding

    def get_angles(self, pos, i, embed_size):
        # Compute angle rate for each position and dimension
        angle_rates = 1 / torch.pow(10000, (2 * (i//2)) / embed_size)