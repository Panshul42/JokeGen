"""
Transformer based model for joke generation, we will implement transformers and multi-head attention from scratch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.09):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim # embed_dim = d_model = 512 (in the original paper)
        self.num_heads = num_heads # num_heads = 8 (in the original paper)
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim) # project input into Q, K and V matrices
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, E = x.shape

        qkv = self.qkv_proj(x)  # [B, T, 3*E]
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [B, T, E]

        # Split into heads, swap dimensions [B, T, E] -> [B, T, h, h_dim] -> [B, h, T, h_dim]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # q: [B, h, T, h_dim], k.transpose(-2, -1): [B, h, h_dim, T]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, h, T, T]

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, heads, T, T]
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)  # [B, heads, T, head_dim]

        # .view(B,T,E) throws an error if data is not contiguous
        out = attn_output.transpose(1, 2).contiguous().view(B, T, E)  # [B, T, E]
        return self.dropout(self.out_proj(out))

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.13):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_hidden_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        # we use pre-normalization as it is a popular practice
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x

# Decoder only transformer since this isn't a seq -> seq task

class Decoder(nn.Module):
    def __init__(self, vocab_size, max_length, embed_dim=512, num_heads=8, num_layers=6, ff_hidden_dim=1024):
        super().__init__()
        # Learnable token embeddings map from vocab_size to embed_dim
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # Learnable (vs fixed sinusoidal) positional embeddings
        self.pos_embed = nn.Embedding(max_length, embed_dim)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_hidden_dim)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.MLP = nn.Linear(embed_dim, embed_dim//2)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.GELU(),
            nn.Linear(embed_dim//2, 2)
        )

        self.max_length = max_length

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        positions = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
    
        # Step 1: Embed tokens + positions → shape [B, T, D]
        x = self.token_embed(input_ids) + self.pos_embed(positions)  # [B, T, D]
    
        # Step 2: Expand attention mask for compatibility
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]  # [B, 1, 1, T]
    
        # Step 3: Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask=attention_mask)  # [B, T, D]
    
        x = self.ln_f(x)  # [B, T, D]
    
        # Step 4: Mean Pooling over tokens → [B, D]
        if attention_mask is not None:
            mask = attention_mask.squeeze(1).squeeze(1)  # [B, T]
            mask = mask.unsqueeze(-1)  # [B, T, 1]
            x = x * mask
            sum_x = x.sum(dim=1)
            lengths = mask.sum(dim=1)  # [B, 1]
            pooled = sum_x / lengths.clamp(min=1e-9)  # [B, D]
        else:
            pooled = x.mean(dim=1)
    
        # Step 5: Final classification head
        logits = self.head(pooled)  # [B, 2]
        return logits


def load_model_and_tokenizer(tokenizer_name="gpt2", max_length=64):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token  # Make sure pad token is set if not already

    # Initialize model
    model = Decoder(
        vocab_size=tokenizer.vocab_size,
        max_length=max_length,
        embed_dim=1024,
        num_heads=16,
        num_layers=12,
        ff_hidden_dim=2048
    ).to(device)

    return model, tokenizer
        

