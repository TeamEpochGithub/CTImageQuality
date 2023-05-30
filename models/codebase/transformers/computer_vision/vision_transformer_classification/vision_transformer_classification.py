import torch.nn as nn
import torch
import math

from einops.layers.torch import Rearrange

from models.codebase.transformer.utilities import TransformerBlock
from models.codebase.vision_transformer_classification.utilities import pair


class PositionalEmbedding(nn.Module):
    def __init__(self, num_patches, embed_model_dim):
        """
        There are many variants for PositionalEmbedding, this is the original one.
        It takes in the embedding of your input and adds positional information.
        Args:
            num_patches: length of input sequence
            embed_model_dim: the size of each embedding vector (e.g. 256)
        """
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_model_dim

        pe = torch.zeros(num_patches, self.embed_dim)
        for pos in range(num_patches):
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


class VisionTransformerClassification(nn.Module):
    """
    Args:
        num_patches : length of input sequence
        embed_dim: the size of each embedding vector (e.g. 512)  # Heavily influences memory use
        num_layers: number of encoder layers
        expansion_factor: factor which expands inner dimension of feed forward layer
        n_heads: number of heads in multihead attention
    Returns:
        out: output of the encoder
    """
    def __init__(self, image_size, patch_size, channels, num_classes, embed_dim, num_layers=2, expansion_factor=4,
                 n_heads=8, dropout_ratio=0.2):
        super(VisionTransformerClassification, self).__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.positional_encoder = PositionalEmbedding(num_patches, embed_dim)

        self.layers = nn.ModuleList(
            [TransformerBlock(embed_dim, expansion_factor, n_heads, dropout_ratio) for i in range(num_layers)])

        self.linear_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)

        positional_embedding = self.positional_encoder(x)
        for layer in self.layers:
            positional_embedding = layer(positional_embedding, positional_embedding, positional_embedding)

        positional_embedding = positional_embedding.mean(dim=1)

        positional_embedding = self.linear_head(positional_embedding)

        return positional_embedding  # (batch_size, sequence_length, embedding_dim)
