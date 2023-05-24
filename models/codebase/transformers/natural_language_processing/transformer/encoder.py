import torch.nn as nn
import torch
import math

from models.codebase.transformer.utilities import TransformerBlock


# Embedding: only used for NLP problems
class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        """
        Args:
            vocab_size: size of vocabulary
            embed_dim: the size of each embedding vector (e.g. 256)
        """
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        """
        Args:
            x: input vector (e.g. (batch_size, sequence_length))
        Returns:
            out: embedding vector (e.g. (batch_size, sequence_length, embed_dim))
        """
        out = self.embed(x)
        return out


class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len, embed_model_dim):
        """
        There are many variants for PositionalEmbedding, this is the original one.
        It takes in the embedding of your input and adds positional information.
        Args:
            seq_len: length of input sequence
            embed_model_dim: the size of each embedding vector (e.g. 256)
        """
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_model_dim

        pe = torch.zeros(seq_len, self.embed_dim)
        for pos in range(seq_len):
            for i in range(0, self.embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / self.embed_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # Typically used to store tensors that are not considered as model parameters
        # but still need to be saved and loaded along with the model state.

    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            x: output
        """

        # Make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)

        # Add constant to embedding
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:, :seq_len], requires_grad=False)
        return x


class TransformerEncoder(nn.Module):
    """
    Args:
        seq_len : length of input sequence
        embed_dim: the size of each embedding vector (e.g. 256)
        num_layers: number of encoder layers
        expansion_factor: factor which expands inner dimension of feed forward layer
        n_heads: number of heads in multihead attention
    Returns:
        out: output of the encoder
    """
    def __init__(self, seq_len, vocab_size, embed_dim, num_layers=2, expansion_factor=4, n_heads=8, dropout_ratio=0.2):
        super(TransformerEncoder, self).__init__()

        self.embedding_layer = Embedding(vocab_size, embed_dim)  # Only needed for NLP-problems

        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList([TransformerBlock(embed_dim, expansion_factor, n_heads, dropout_ratio) for i in range(num_layers)])

    def forward(self, x):
        embedding = self.embedding_layer(x)
        positional_embedding = self.positional_encoder(embedding)
        for layer in self.layers:
            positional_embedding = layer(positional_embedding, positional_embedding, positional_embedding)

        return positional_embedding  # (batch_size, sequence_length, embedding_dim)
