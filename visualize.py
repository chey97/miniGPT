import sys
# Increase recursion limit
sys.setrecursionlimit(10**6) 

import torch
import torch.nn as nn
from torchviz import make_dot

# Define Self-Attention module
class SelfAttention(nn.Module):
    def __init__(self, embed_size, head_count):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size  # Size of word embeddings
        self.head_count = head_count  # Number of attention heads
        
        # Create linear layers for query, key and value projections for each head
        self.query_layers = nn.ModuleList([nn.Linear(embed_size, embed_size, bias=False) for _ in range(head_count)])
        self.key_layers = nn.ModuleList([nn.Linear(embed_size, embed_size, bias=False) for _ in range(head_count)])
        self.value_layers = nn.ModuleList([nn.Linear(embed_size, embed_size, bias=False) for _ in range(head_count)])
        
         # Final linear layer to combine head outputs
        self.fc_out = nn.Linear(head_count * embed_size, embed_size) 

    def forward(self, embeddings):
        batch_size, token_count = embeddings.shape[:2]
        qkvs = torch.zeros(self.head_count, 3, batch_size, token_count, self.embed_size).to(embeddings.device)
        
        # Loop over heads and compute query, key and value projections
        for i in range(self.head_count):
            qkvs[i, 0] = self.query_layers[i](embeddings)
            qkvs[i, 1] = self.key_layers[i](embeddings)
            qkvs[i, 2] = self.value_layers[i](embeddings)
        
        # Compute energy terms for each head, batch, and pair of tokens
        energy = torch.zeros(self.head_count, batch_size, token_count, token_count).to(embeddings.device)
        # Create a mask with false on and below the diagonal, and true above the diagonal
        mask = torch.triu(torch.ones((token_count, token_count)), diagonal=1).bool()
        
        for h in range(self.head_count):
            for b in range(batch_size):
                for i in range(token_count):
                    for j in range(token_count):
                        energy[h, b, i, j] = torch.dot(qkvs[h, 0, b, i], qkvs[h, 1, b, j])
                energy[h, b] = energy[h, b].masked_fill(mask, float('-inf')) # Apply mask
        
        # Compute attention scores
        attention = torch.nn.functional.softmax(energy, dim=3)
        
        # Compute weighted sum of values for each head and token
        out = torch.zeros(batch_size, token_count, self.head_count, self.embed_size).to(embeddings.device)
        for h in range(self.head_count):
            for b in range(batch_size):
                for i in range(token_count):
                    for j in range(token_count):
                        out[b, i, h] += (attention[h, b, i, j] * qkvs[h, 2, b, j])
        
        # Reshape and pass through final linear layer
        out = out.reshape(batch_size, token_count, self.head_count * self.embed_size)
        return self.fc_out(out)
    
    def masked_attention(self, energy):
        # Assume scores has shape (batch_size, max_token_count, embed_size, embed_size)
        max_token_count, embed_size, _ = energy.size()

        # Create a mask with zeros on and below the diagonal, and negative infinity above the diagonal
        mask = torch.triu(torch.ones((max_token_count, max_token_count)), diagonal=1) * float('-inf')
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add dimensions for batch and embedding size
        mask = mask.expand(batch_size, embed_size, -1, -1)  # Expand mask to match batch and embedding size

        # Apply the mask to the scores
        masked_scores = energy + mask.to(energy.device)

        return masked_scores.to(energy.device)
    
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
        return pos * angle_rates

# Create an instance of your Transformer model
vocab_size = 25 # Example vocabulary size
embed_size = 512  # Example embedding size
num_layers = 4  # Example number of layers
head_count = 3  # Example number of attention heads
model = Transformer(vocab_size, embed_size, num_layers, head_count)

# Generate a dummy input
dummy_input = torch.randint(0, vocab_size, (1, 10))  # Example input sequence length of 10

# Create a graph of your model
output = model(dummy_input)

# Visualize the graph
dot = make_dot(output, params=dict(model.named_parameters()))

# Save the visualization as a PNG image
dot.format = 'png'
dot.render('transformer_model', directory='./', cleanup=True)