import torch
from torch import nn
from torch import Tensor
import numpy as np

from utils.transformer_blocks import Encoder, Decoder, get_extended_attention_mask, get_causal_extended_attention_mask


class Transformer(nn.Module):
    def __init__(
        self, 
        encoder_vocab_size, 
        decoder_vocab_size, 
        hidden_size, 
        n_head, 
        intermediate_size, 
        encoder_max_len, 
        decoder_max_len, 
        n_layers, 
        drop_prob=0.1
    ):

        super().__init__()

        self.encoder = Encoder(encoder_vocab_size, encoder_max_len, hidden_size, intermediate_size, n_head, n_layers, drop_prob)
        self.decoder = Decoder(decoder_vocab_size, decoder_max_len, hidden_size, intermediate_size, n_head, n_layers, drop_prob)

    def forward(self, src_input_ids, trg_input_ids, src_attention_mask=None, trg_attention_mask=None) -> torch.Tensor:
        
        """
        src_input_ids: (bs, src_len)
        trg_input_ids: (bs, trg_len)
        src_attention_mask: (bs, src_len) optional – 1 for real, 0 for pad
        trg_attention_mask: (bs, trg_len) optional – 1 for real, 0 for pad
        """
        
        bs, src_len = src_input_ids.shape
        _, trg_len = trg_input_ids.shape

        if src_attention_mask is None:
            src_attention_mask = (src_input_ids != 0).to(src_input_ids.device)
        if trg_attention_mask is None:
            trg_attention_mask = (trg_input_ids != 0).to(trg_input_ids.device)

        extended_src_mask = get_extended_attention_mask(src_attention_mask, dtype=torch.float).to(trg_input_ids.device)

        extended_trg_mask = get_causal_extended_attention_mask(trg_attention_mask, dtype=torch.float).to(trg_input_ids.device)

        encoder_output = self.encoder(src_input_ids, extended_src_mask)
        decoder_output = self.decoder(trg_input_ids, extended_trg_mask, encoder_output, extended_src_mask)

        return decoder_output  # (bs, trg_len, decoder_vocab_size)
