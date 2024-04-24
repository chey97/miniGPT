# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import pprint

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
        self.fc_out = nn.Linear(head_count * embed_size, embed_size)  # Final linear layer to combine head outputs

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