import torch
from torch import nn
from torch import Tensor
import numpy as np


class Embedding(nn.Module):

    def __init__(self, vocab_size, hidden_size, max_len, drop_prob=0.1):

        super().__init__()
        
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_len, hidden_size)

        self.register_buffer(
            "position_ids", torch.arange(max_len).expand((1, -1)), persistent=False
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, input_ids) -> Tensor:

        assert input_ids.ndim == 2
        tok_emb = self.word_embeddings(input_ids)

        position_ids = self.position_ids[:, :input_ids.shape[1]]
        pos_emb = self.position_embeddings(position_ids)

        emb = tok_emb + pos_emb
        return self.dropout(self.layer_norm(emb))


class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_size, n_head, drop_prob=0.1):

        super().__init__()

        assert hidden_size % n_head == 0

        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(drop_prob)
        self.scale = self.head_dim ** -0.5

    def forward(self, q, k, v, attention_mask=None) -> Tensor:
            
        """
        q, k, v: [bs, seq_len, hidden_size]
        attention_mask: [bs, 1, 1/seq_len, seq_len]
        """
        
        assert q.ndim == k.ndim == v.ndim == 3
        assert attention_mask.ndim == 4

        bs = q.size(0)

        queries = self.query(q)
        keys = self.key(k)
        values = self.value(v)

        queries = queries.view(bs, -1, self.n_head, self.head_dim).transpose(1, 2)
        keys = keys.view(bs, -1, self.n_head, self.head_dim).transpose(1, 2)
        values = values.view(bs, -1, self.n_head, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(queries, keys.transpose(-1, -2))/np.sqrt(self.head_dim)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        output = torch.matmul(attention_probs, values)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.hidden_size)
        output = self.out(output)

        return output


class FeedForward(nn.Module):

    def __init__(self, hidden_size, intermediate_size, drop_prob=0.1):

        super().__init__()

        self.u = nn.Linear(hidden_size, intermediate_size)
        self.relu = nn.ReLU()
        self.v = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(drop_prob)
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states) -> Tensor:
        
        """
        hidden_states: [bs, seq_len, hidden_size]
        """
        
        assert hidden_states.ndim == 3

        hidden_states_transfromed = self.relu(self.u(hidden_states))
        hidden_states_transfromed = self.v(hidden_states_transfromed)
        hidden_states_transfromed = self.dropout(hidden_states_transfromed)
        hidden_states_transfromed += hidden_states
        hidden_states_transfromed = self.layernorm(hidden_states_transfromed)

        return hidden_states_transfromed


class EncoderBlock(nn.Module):

    def __init__(self, hidden_size, intermediate_size, n_head, drop_prob=0.1):

        super().__init__()

        self.multiheadattention = MultiHeadAttention(hidden_size, n_head, drop_prob)
        self.dropout = nn.Dropout(drop_prob)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.feedforward = FeedForward(hidden_size, intermediate_size, drop_prob)

    def forward(self, hidden_states, attention_mask) -> Tensor:
        
        """
        hidden_states: [bs, seq_len, hidden_size]
        attention_mask: [bs, 1, 1, seq_len]
        """
        
        assert hidden_states.ndim == 3
        assert attention_mask.ndim == 4

        hidden_states_transformed = self.multiheadattention(hidden_states, hidden_states, hidden_states, attention_mask)
        hidden_states_transformed = self.dropout(hidden_states_transformed)
        hidden_states_transformed += hidden_states
        hidden_states_transformed = self.layernorm(hidden_states_transformed)
        hidden_states_transformed = self.feedforward(hidden_states_transformed)

        return hidden_states_transformed


class Encoder(nn.Module):

    def __init__(self, vocab_size, max_len, hidden_size, intermediate_size, n_head, n_layers, drop_prob=0.1):

        super().__init__()

        self.embeddings = Embedding(vocab_size, hidden_size, max_len, drop_prob)
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(hidden_size, intermediate_size, n_head, drop_prob) for _ in range(n_layers)]
        )

    def forward(self, input_ids, attention_mask=None) -> Tensor:
        
        """
        input_ids: [bs, seq_len]
        attention_mask: [bs, 1, 1, seq_len]
        """
        
        assert input_ids.ndim == 2
        assert attention_mask.ndim == 4

        x = self.embeddings(input_ids)

        for encoder in self.encoder_blocks:
            x = encoder(x, attention_mask)

        return x


class DecoderBlock(nn.Module):

    def __init__(self, hidden_size, intermediate_size, n_head, drop_prob=0.1):

        super().__init__()

        self.attention = MultiHeadAttention(hidden_size, n_head, drop_prob)
        self.layernorm_att = nn.LayerNorm(hidden_size)
        
        self.cross_attention = MultiHeadAttention(hidden_size, n_head, drop_prob)
        self.layernorm_cross_att = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(drop_prob)
        
        self.feedforward = FeedForward(hidden_size, intermediate_size, drop_prob)
        self.layernorm_ff = nn.LayerNorm(hidden_size)


    def forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask) -> Tensor:
        
        """
        hidden_states: [bs, trg_seq_len, hidden_size]
        attention_mask: [bs, 1, trg_seq_len, trg_seq_len]
        encoder_hidden_states: [bs, src_seq_len, hidden_size]
        encoder_attention_mask: [bs, 1, 1, src_seq_len]
        """
        
        assert hidden_states.ndim == encoder_hidden_states.ndim == 3
        assert attention_mask.ndim == encoder_attention_mask.ndim == 4

        input_hidden_states = hidden_states

        hidden_states = self.attention(hidden_states, hidden_states, hidden_states, attention_mask)
        hidden_states = self.dropout(hidden_states)
        hidden_states += input_hidden_states
        hidden_states = self.layernorm_att(hidden_states)

        if encoder_hidden_states is not None:

            input_hidden_states = hidden_states
            
            hidden_states = self.cross_attention(hidden_states, encoder_hidden_states, encoder_hidden_states, encoder_attention_mask)
            hidden_states = self.dropout(hidden_states)
            hidden_states += input_hidden_states
            hidden_states = self.layernorm_cross_att(hidden_states)

        input_hidden_states = hidden_states
        
        hidden_states = self.feedforward(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states += input_hidden_states
        hidden_states = self.layernorm_ff(hidden_states)

        return hidden_states


class Decoder(nn.Module):

    def __init__(self, vocab_size, max_len, hidden_size, intermediate_size, n_head, n_layers, drop_prob=0.1):

        super().__init__()

        self.embeddings = Embedding(vocab_size, hidden_size, max_len, drop_prob)
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(hidden_size, intermediate_size, n_head, drop_prob) for _ in range(n_layers)]
        )
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask) -> Tensor:
        
        """
        input_ids: [bs, seq_len]
        attention_mask: [bs, 1, trg_seq_len, trg_seq_len]
        encoder_hidden_states: [bs, src_seq_len, hidden_size]
        encoder_attention_mask: [bs, 1, 1, src_seq_len]
        """
        
        assert input_ids.ndim == 2
        assert encoder_hidden_states.ndim == 3
        assert attention_mask.ndim == encoder_attention_mask.ndim == 4

        x = self.embeddings(input_ids)

        for decoder in self.decoder_blocks:
            x = decoder(x, attention_mask, encoder_hidden_states, encoder_attention_mask)

        x = self.output(x)
        
        return x

def get_extended_attention_mask(attention_mask, dtype=torch.float):
    
    min_value = torch.finfo(dtype).min
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    extended_attention_mask = (1.0 - extended_attention_mask) * min_value
    
    return extended_attention_mask.to(dtype)


def get_causal_extended_attention_mask(attention_mask, dtype=torch.float):   
    
    device = attention_mask.device
    batch_size, seq_len = attention_mask.shape

    seq_ids = torch.arange(seq_len, device=device)
    causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_len, 1) <= seq_ids[None, :, None]    
    causal_mask = causal_mask.to(attention_mask.dtype)

    extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :].to(dtype=dtype)
    
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    
    return extended_attention_mask
    
