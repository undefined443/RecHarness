import random
import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, use_embedding_mixup, max_values_per_column, feat_dim, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.use_embedding_mixup = use_embedding_mixup
        self.device = device

        # User feature embeddings
        self.user_id_embedding = nn.Embedding(int(max_values_per_column[0]) + 1, feat_dim)
        self.user_fea_embedding = nn.ModuleList([
            nn.Embedding(int(max_values_per_column[i]) + 1, feat_dim)
            for i in range(1, 19)
        ])
        # Item feature embeddings
        self.item_id_embedding = nn.Embedding(int(max_values_per_column[19]) + 1, feat_dim)
        self.item_fea_embedding = nn.ModuleList([
            nn.Embedding(int(max_values_per_column[i]) + 1, feat_dim)
            for i in range(20, 24)
        ])
        self.item_duration_embedding = nn.Embedding(int(max_values_per_column[-1]) + 1, feat_dim)

    def make_trg_mask(self, trg, pad_idx=0):
        trg_pad_mask = (trg != pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len, device=self.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def get_encoder_result(self, X, qmsk, imsk):
        # User embedding: user_id (col 0) + 18 user features (cols 1-18)
        user_id_emb = self.user_id_embedding(X[:, 0].long())
        user_fea_embs = [
            self.user_fea_embedding[i](X[:, i + 1].long())
            for i in range(18)
        ]
        # Mean pool over unmasked user features
        user_fea_stack = torch.stack(user_fea_embs, dim=1)  # (B, 18, feat_dim)
        qmsk_f = qmsk[:, 1:].float().unsqueeze(-1)  # (B, 18, 1); qmsk[:, 0] is user_id
        user_emb = (user_fea_stack * qmsk_f).sum(dim=1) / (qmsk_f.sum(dim=1) + 1e-8)

        # Item embedding: item_id (col 19) + 4 item features (cols 20-23) + duration (last col)
        item_id_emb = self.item_id_embedding(X[:, 19].long())
        item_fea_embs = [
            self.item_fea_embedding[i](X[:, 20 + i].long())
            for i in range(4)
        ]
        item_fea_stack = torch.stack(item_fea_embs, dim=1)  # (B, 4, feat_dim)
        imsk_f = imsk[:, 1:].float().unsqueeze(-1)  # (B, 4, 1); imsk[:, 0] is item_id
        item_emb = (item_fea_stack * imsk_f).sum(dim=1) / (imsk_f.sum(dim=1) + 1e-8)
        item_dur_emb = self.item_duration_embedding(X[:, -1].long())

        combined = torch.cat([
            user_id_emb, user_emb, item_id_emb, item_emb, item_dur_emb
        ], dim=-1)  # (B, feat_dim * 5 = input_dim * 2) if feat_dim == hidden_dim

        enc_src = self.encoder(combined)  # (1, B, hidden_dim)
        enc_src = enc_src.permute(1, 0, 2)  # (B, 1, hidden_dim)
        src_mask = torch.ones(X.shape[0], 1, device=self.device).unsqueeze(1).unsqueeze(2)
        return enc_src, src_mask

    def forward(self, X, qmsk, imsk, trg, trg_mask_seq, teacher_force_ratio):
        enc_src, src_mask = self.get_encoder_result(X, qmsk, imsk)
        trg_mask = self.make_trg_mask(trg)

        output, _ = self.decoder(trg, enc_src, trg_mask, src_mask)

        input_teacher = []
        use_pred = [random.random() > teacher_force_ratio for _ in range(1, output.shape[1])]
        pred_tokens = output.argmax(dim=-1)

        for i in range(output.shape[1]):
            if i != 0 and use_pred[i - 1]:
                if self.use_embedding_mixup:
                    input_teacher.append(output[:, i - 1, :])
                else:
                    input_teacher.append(pred_tokens[:, i - 1].unsqueeze(1))
            else:
                input_teacher.append(trg[:, i].unsqueeze(1))

        if not self.use_embedding_mixup:
            input_teacher = torch.cat(input_teacher, dim=1)

        output, _ = self.decoder.weight_forward(
            input_teacher, enc_src, trg_mask, src_mask, self.use_embedding_mixup
        )
        return output
