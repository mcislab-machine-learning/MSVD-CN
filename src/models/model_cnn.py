import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageModelConfig:
    def __init__(self):
        self.image_feature_dim = 4096
        self.feat_dim = 512
        self.vocab_size = 5000
        self.num_layers = 2
        self.kernel_size = 5
        self.dropout = 0.1

        self.use_attention = False


def Conv1d(in_channels, out_channels, kernel_size, padding, dropout=0.0):
    m = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
    std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, 0.1)
    return m


def Linear(in_features, out_features, dropout=0.):
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


class AttentionLayer(nn.Module):
    def __init__(self, conv_channels, embed_dim):
        super(AttentionLayer, self).__init__()
        self.in_projection = Linear(conv_channels, embed_dim)
        self.out_projection = Linear(embed_dim, conv_channels)
        self.bmm = torch.bmm


class LanguageModelConv(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dropout = self.config.dropout
        self.dropout = dropout
        feat_dim = self.config.feat_dim
        self.input_word_embedding_0 = Embedding(self.config.vocab_size, feat_dim, padding_idx=0)
        self.input_word_embedding_1 = Linear(feat_dim, feat_dim, dropout=dropout)
        self.image_embedding = Linear(self.config.image_feature_dim, feat_dim, dropout=dropout)
        self.res_proj = Linear(feat_dim * 2, feat_dim, dropout=dropout)

        n_in = self.config.feat_dim * 2
        n_out = self.config.feat_dim
        self.convs = nn.ModuleList()
        self.attention = nn.ModuleList()
        kernel_size = self.config.kernel_size
        pad = kernel_size - 1
        self.pad = pad

        for i in range(self.config.num_layers):
            self.convs.append(Conv1d(n_in, 2*n_out, self.config.kernel_size, pad, dropout))
            if self.config.use_attention:
                self.attention.append(AttentionLayer(n_out, feat_dim))
            n_in = n_out

        self.output_word_embedding_0 = Linear(feat_dim, (feat_dim // 2))
        self.output_word_embedding_1 = Linear((feat_dim // 2), self.config.vocab_size, dropout=dropout)

    def forward(self, image_feature, image_feature_fc7, input_word):
        attn_buffer = None
        input_word_embedding = self.input_word_embedding_0(input_word)
        input_word_embedding = self.input_word_embedding_1(input_word_embedding)

        x = input_word_embedding.transpose(2, 1)
        batchsize, wordembdim, maxtokens = x.size()

        y = F.relu(self.image_embedding(image_feature_fc7))
        y = y.unsqueeze(2).expand(batchsize, self.config.feat_dim, maxtokens)
        x = torch.cat([x, y], 1)

        for i, conv in enumerate(self.convs):
            if i == 0:
                x = x.transpose(2, 1)
                residual = self.res_proj(x)
                residual = residual.transpose(2, 1)
                x = x.transpose(2, 1)
            else:
                residual = x

            x = F.dropout(x, p=self.dropout, training=self.training)

            x = conv(x)
            x = x[:, :, :-self.pad]

            x = F.glu(x, dim=1)

            if self.config.use_attention:
                attn = self.attention[i]
                x = x.transpose(2, 1)
                x, attn_buffer = attn(x, input_word_embedding, image_feature)
                x = x.transpose(2, 1)

            x = (x + residual) * math.sqrt(.5)

        x = x.transpose(2, 1)

        x = self.output_word_embedding_0(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.output_word_embedding_1(x)

        x = x.transpose(2, 1)

        return x, attn_buffer

    def inference_step(self, inputs, states):
        pass