import math
import torch.nn.functional as F

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8):
        """
        Args:
            embed_dim: dimension of embedding vector output
            n_heads: number of self attention heads
        """
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads

        # (e.g. 512/8=64 key, query, and value will have dim (64))
        self.single_head_dim = int(self.embed_dim / self.n_heads)

        # Create key, query, and value encoders that will generate the embeddings
        self.key_encoder = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.query_encoder = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.value_encoder = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.out = nn.Linear(self.n_heads * self.single_head_dim, self.embed_dim)

    def forward(self, key, query, value, mask=None):
        """
        Each of the input vectors will have dimension (batch_size, sequence_length, embedding_dim)

        Args:
           key : key vector
           query : query vector
           value : value vector
           mask: mask for decoder
        Returns:
           output vector from multihead attention
        """
        batch_size = key.size(0)
        seq_length = key.size(1)

        # The query dimension can change in decoder during inference, so we can't take general seq_length
        seq_length_query = query.size(1)

        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)
        query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim)
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim)

        k = self.key_encoder(key)
        q = self.query_encoder(query)
        v = self.value_encoder(value)

        # Transposes q, k, v to (batch_size, n_heads, seq_len, single_head_dim)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # Transpose key for matrix multiplication
        k_transposed = k.transpose(-1, -2)  # (batch_size, n_heads, single_head_dim, seq_len)

        # Computes attention matrix (non-normalized)
        qk_out = torch.matmul(q, k_transposed)  # (b x nh x seq x sh) x (b x nh x sh x seq) = (b x nh x seq x seq)

        # For decoder: where mask positions are 0, fill non-normalized attention matrix with -1e20
        if mask is not None:
            qk_out = qk_out.masked_fill(mask == 0, float("-1e20"))

        # Reduces outliers, used to increase stability of model
        qk_out = qk_out / math.sqrt(self.single_head_dim)

        # Computes the normalized attention matrix
        attention_weights = F.softmax(qk_out, dim=-1)

        # Computes the attention-weighted values: (b x nh x seq x seq) x (b x nh x seq x sh) = (b x nh x seq x sh)
        attention_weighted_input = torch.matmul(attention_weights, v)

        # Concatenate outputs of each head, to generate final output
        concat = attention_weighted_input.transpose(1, 2).contiguous().view(batch_size, seq_length_query,
                                                                            self.single_head_dim * self.n_heads)
        # (b x nh x seq x sh) -> (b x seq x nh x sh) -> (b x seq x embed_dim)

        # Another trainable layer applied on the output of the attention layer
        output = self.out(concat)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8, dropout_ratio=0.2):
        super(TransformerBlock, self).__init__()
        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: factor which expands inner dimension of feed forward layer
           n_heads: number of attention heads
        """
        self.attention = MultiHeadAttention(embed_dim, n_heads)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.ReLU(),
            nn.Linear(expansion_factor * embed_dim, embed_dim)
        )

        self.dropout1 = nn.Dropout(dropout_ratio)
        self.dropout2 = nn.Dropout(dropout_ratio)

    def forward(self, key, query, value):
        """
        Args:
           key: key vector
           query: query vector
           value: value vector
           norm2_out: output of transformer block
        """
        attention_out = self.attention(key, query, value)  # (batch_size, sequence_length, embedding_dim)

        # Residual connection allows input data to be joined with weighted attention values
        attention_residual_out = attention_out + value
        norm1_out = self.dropout1(self.norm1(attention_residual_out))

        feed_fwd_out = self.feed_forward(norm1_out)
        feed_fwd_residual_out = feed_fwd_out + norm1_out
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out))

        return norm2_out
