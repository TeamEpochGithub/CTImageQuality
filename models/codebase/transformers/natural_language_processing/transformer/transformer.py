import torch
from torch import nn

from models.codebase.transformer.decoder import TransformerDecoder
from models.codebase.transformer.encoder import TransformerEncoder


class Transformer(nn.Module):
    def __init__(self, embed_dim, src_vocab_size, output_size, seq_length, num_layers=2, expansion_factor=4,
                 n_heads=8, dropout_ratio=0.2):
        super(Transformer, self).__init__()
        """  
        Args:
           embed_dim: dimension of embedding 
           src_vocab_size: vocabulary size of source
           target_vocab_size: vocabulary size of target
           seq_length : length of input sequence
           num_layers: number of encoder layers
           expansion_factor: factor which expands inner dimension of feed forward layer
           n_heads: number of heads in multihead attention

        """
        self.encoder = TransformerEncoder(seq_length, src_vocab_size, embed_dim, num_layers=num_layers,
                                          expansion_factor=expansion_factor, n_heads=n_heads,
                                          dropout_ratio=dropout_ratio)
        self.decoder = TransformerDecoder(output_size, embed_dim, seq_length, num_layers=num_layers,
                                          expansion_factor=expansion_factor, n_heads=n_heads,
                                          dropout_ratio=dropout_ratio)

    def make_target_mask(self, target_seq):
        """
        Args:
            target_seq: target sequence
        Returns:
            mask: target mask
        """
        batch_size, trg_len = target_seq.shape

        # Mask: the lower triangular part of matrix filled with ones
        mask = torch.tril(torch.ones((trg_len, trg_len))).expand(batch_size, 1, trg_len, trg_len)

        return mask

    def decode(self, source, target):
        """
        Used for inference
        Args:
            source: input to encoder
            target: input to decoder
        out:
            out_labels : returns final prediction of sequence
        """
        mask = self.make_target_mask(target)
        enc_out = self.encoder(source)
        seq_len = source.shape[1]
        out = target

        out_labels = []
        for i in range(seq_len):
            out = self.decoder(out, enc_out, mask)  # (batch_size, sequence_length, vocab_dim)

            # Takes the last token
            out = out[:, -1, :]

            out = out.argmax(axis=-1)
            out_labels.append(out.item())
            out = torch.unsqueeze(torch.tensor(out_labels), dim=0)

        return out_labels

    def forward(self, src, trg):
        """
        Args:
            src: input to encoder
            trg: input to decoder
        out:
            out: final vector which returns probabilities of each target word
        """
        target_mask = self.make_target_mask(trg)

        enc_out = self.encoder(src)
        outputs = self.decoder(trg, enc_out, target_mask)
        return outputs
