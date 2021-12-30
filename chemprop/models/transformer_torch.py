import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class TransformerModel(nn.Module):

    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048,
                 num_encoder_layers=6,num_decoder_layers=6,  dropout=0.1,
                 batch_first=True,device=None, protein_dim=321):
        super(TransformerModel, self).__init__()

        self.model_type = 'Transformer'
        self.device = device
        # self.embeding_layer = nn.Linear(in_features=234, out_features=d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.ft = nn.Linear(protein_dim, d_model)
        ###encoder###
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                          dropout=dropout, batch_first=batch_first, device=device)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        # ###decoder###
        # decoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
        #                                             dropout=dropout, batch_first=batch_first, device=device)
        # self.transformer_decoder = nn.TransformerEncoder(decoder_layers, num_decoder_layers)


        # self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
        #                                   num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
        #                                   dropout=dropout, batch_first=batch_first, device=device)

        # self.init_weights()

    def init_weights(self):
        # initrange = 0.1
        # self.embeding_layer.bias.data.zero_()
        # self.embeding_layer.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.transformer.parameters():
            if p.dim() > 1:
                # 这里初始化采用的是nn.init.xavier_uniform
                nn.init.xavier_uniform_(p)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        ##padding tensor
        src, src_key_padding_mask = self.seq_padding(src)
        # src = self.embeding_layer(src)
        # ##Positional encoder
        # src = self.ft(src)
        src = self.pos_encoder(src)

        tgt, tgt_key_padding_mask = self.seq_padding(tgt)
        tgt = self.pos_encoder(tgt)
        tgt_mask = self._subsequent_mask(tgt.size(1),src.size(1))
        ##forward

        # memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        # output = self.decoder(tgt, memory,
        #                       tgt_key_padding_mask=tgt_key_padding_mask,
        #                       memory_key_padding_mask=src_key_padding_mask)
        # output = self.transformer(src=src, tgt=tgt,tgt_mask = tgt_mask,
        #                           src_key_padding_mask=src_key_padding_mask,
        #                           tgt_key_padding_mask=tgt_key_padding_mask,
        #                           memory_key_padding_mask=src_key_padding_mask)

        output = self.transformer(src=src, src_key_padding_mask=src_key_padding_mask)
        output = torch.mean(output,dim=1)
        return output

    def _subsequent_mask(self,tgt_size, src_size):
        """
        deocer层self attention需要使用一个mask矩阵，
        :param size: 句子维度
        :return: 右上角(不含对角线)全为True，左下角全为False的mask矩阵
        """
        "Mask out subsequent positions."
        # 设定subsequent_mask矩阵的shape
        attn_shape = (tgt_size, src_size)
        # 生成一个右上角(不含主对角线)为全True，左下角(含主对角线)为全False的subsequent_mask矩阵
        attn_mask = torch.triu(torch.ones(attn_shape), diagonal=1) == 1

        return attn_mask.cuda(self.device)

    def seq_padding(self, seq, pad=0):
        seq_lens = [len(x) for x in seq]
        max_len = max(seq_lens)
        #     ##pad_sequence
        seq = nn.utils.rnn.pad_sequence(
            seq, batch_first=True, padding_value=pad
        ).cuda(self.device)
        #     #mask tensor ==-1
        batch_size = seq.size(0)
        seq_mask = torch.ones((batch_size, max_len))
        for i in range(batch_size):
            seq_mask[i, :seq_lens[i]] = 0
        seq_mask = seq_mask.bool().to(seq.device)
        return seq, seq_mask


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        # 之所以用log再exp,可能是考虑到数值过大溢出的问题
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


def norm_output(output,hid_dim):
    # output = [batch size, compound_len, hidden_size]
    """Use norm to determine which atom is significant. """
    norm = torch.norm(output, dim=2)
    # norm = [batch size,compound len]
    norm = F.softmax(norm, dim=1)
    sum = torch.zeros((output.shape[0], hid_dim)).to(output.device)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            v = output[i, j,]
            v = v * norm[i, j]
            sum[i,] += v
    # sum = [batch size,hid_dim]
    return sum





