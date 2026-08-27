import torch
import torch.nn.functional as F
from torch import nn

from model.attention import MultiHeadAttention, PositionwiseFeedForward


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, drop_prob):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        return self.dropout(self.tok_emb(x))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask=None):
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        x = self.norm1(x + _x)
        x = self.dropout1(x)

        if enc is not None:
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            x = self.norm2(x + _x)
            x = self.dropout2(x)

        _x = x
        x = self.ffn(x)
        x = self.norm3(x + _x)
        x = self.dropout3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, n_head, ffn_hidden, dec_voc_size, drop_prob, n_layers):
        super().__init__()
        self.output_dim = dec_voc_size
        self.emb = TransformerEmbedding(vocab_size=dec_voc_size, d_model=d_model, drop_prob=drop_prob)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head, drop_prob=drop_prob)
            for _ in range(n_layers)
        ])
        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask=None):
        trg = self.emb(trg)
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)
        output = self.linear(trg)
        return output, trg

    def weight_forward(self, trg_input, enc_src, trg_mask, src_mask=None, use_embedding_mixup=False):
        if use_embedding_mixup:
            embeddeding = self.emb.tok_emb.weight
            embedded = []
            inputs = trg_input if isinstance(trg_input, list) else [trg_input]
            for item in inputs:
                if item.shape[-1] == self.output_dim:
                    softmax_input = F.softmax(item, dim=-1)
                    emb = torch.einsum('bv,ve->be', softmax_input, embeddeding).unsqueeze(1)
                    embedded.append(self.emb.dropout(emb))
                else:
                    embedded.append(self.emb(item.long()))
            trg = torch.cat(embedded, dim=1)
        else:
            trg = self.emb(trg_input.long())

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)
        output = self.linear(trg)
        return output, trg
