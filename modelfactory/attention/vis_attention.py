import torch
import torch.nn as nn


# 视觉注意力机制
class VisAttention(nn.Module):
    def __init__(self, vis_enc_dim, sent_hidden_dim, att_dim):
        super(VisAttention, self).__init__()
        # 视觉特征注意力
        self.encoder_att = nn.Linear(vis_enc_dim, att_dim)
        self.decoder_att = nn.Linear(sent_hidden_dim, att_dim)
        self.tanh = nn.Tanh()
        self.full_att = nn.Linear(att_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, vis_enc_out, decoder_hidden_state):
        # 视觉特征注意力
        # vis_enc_out: image features
        # (batch_size, num_pixels, att_dim)
        vis_enc_att = self.encoder_att(vis_enc_out)
        # 隐藏层注意力
        # (batch_size, att_dim)
        dec_out_att = self.decoder_att(decoder_hidden_state)
        # 拼接（相加）
        # (batch_size, num_pixels, att_dim)
        join_att = self.tanh(vis_enc_att + dec_out_att.unsqueeze(dim=1))
        # (batch_size, num_pixels)
        join_att = self.full_att(join_att).squeeze(2)
        # (batch_size, num_pixels)
        att_scores = self.softmax(join_att)
        # attention_weighted_encoding
        att_output = torch.sum(att_scores.unsqueeze(2) * vis_enc_att, dim=1)
        return att_output, att_scores
