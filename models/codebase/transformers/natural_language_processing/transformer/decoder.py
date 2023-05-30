from torch import nn
import torch.nn.functional as F

from models.codebase.transformer.encoder import PositionalEmbedding
from models.codebase.transformer.utilities import MultiHeadAttention, TransformerBlock


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8, dropout_ratio=0.2):
        super(DecoderBlock, self).__init__()
        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: factor which expands inner dimension of feed forward layer
           n_heads: number of attention heads
        """
        self.attention = MultiHeadAttention(embed_dim, n_heads=8)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_ratio)
        self.transformer_block = TransformerBlock(embed_dim, expansion_factor, n_heads)

    def forward(self, key, query, dec_pos_emb, mask):
        """
        Args:
           key: key vector
           query: query vector
           dec_pos_emb: decoder's positional embedding
           mask: mask for multi head attention
        Returns:
           out: output of transformer block
        """
        # (batch_size, sequence_length, embedding_dim)
        masked_attention_out = self.attention(dec_pos_emb, dec_pos_emb, dec_pos_emb, mask=mask)
        value = self.dropout(self.norm(masked_attention_out + dec_pos_emb))

        out = self.transformer_block(key, query, value)

        return out


class TransformerDecoder(nn.Module):
    def __init__(self, output_size, embed_dim, seq_len, num_layers=2, expansion_factor=4, n_heads=8, dropout_ratio=0.2):
        super(TransformerDecoder, self).__init__()
        """  
        Args:
           output_size: size of output
           embed_dim: dimension of embedding
           seq_len : length of input sequence
           num_layers: number of encoder layers
           expansion_factor: factor which expands inner dimension of feed forward layer
           n_heads: number of heads in multihead attention
        """
        self.word_embedding = nn.Embedding(output_size, embed_dim)
        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_dim, expansion_factor=expansion_factor, n_heads=n_heads, dropout_ratio=dropout_ratio)
                for _ in range(num_layers)
            ]

        )
        self.fc_out = nn.Linear(embed_dim, output_size)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x, enc_out, mask):
        """
        Args:
            x: decoder input (batch_size, sequence_length)
            enc_out : output from encoder layer
            mask: mask for decoder self attention
        Returns:
            out: output vector
        """

        embedding = self.word_embedding(x)  # (batch_size, sequence_length, embedding_dim)
        positional_embedding = self.positional_encoder(embedding)  # (batch_size, sequence_length, embedding_dim)
        positional_embedding = self.dropout(positional_embedding)

        for layer in self.layers:
            positional_embedding = layer(enc_out, enc_out, positional_embedding, mask)

        out = F.softmax(self.fc_out(positional_embedding))

        return out
