import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional

import data_video
from data_video import MSVDDatasetMultiModal
from vocabulary import Vocabulary
from torch.utils.data.dataloader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vocab_path = data_video.msvd_bilingual_vocab_char_path
vocab = Vocabulary.load(vocab_path)


class Config:
    def __init__(self):
        self.vocab_size = 2000
        self.embed_dim = 512
        self.dropout_ratio = 0.5
        self.hidden_dim = 512
        self.vis_dim = 512

        self.attn_option = 'e'

        if self.attn_option == 'v':
            self.vis_num = 49
        elif self.attn_option == 'e':
            self.vis_num = 13
        elif self.attn_option == 'both':
            self.vis_num = 49 + 13

        self.eng_caption_max_len = 15


class LanguageModelLSTM1(nn.Module):
    def __init__(self, config, _device):
        super().__init__()

        global device
        device = _device

        self.config = config

        print('attention:', self.config.attn_option)

        vocab_size = config.vocab_size
        embed_dim = config.embed_dim
        dropout_ratio = config.dropout_ratio
        vis_dim = config.vis_dim
        hidden_dim = config.hidden_dim
        num_layers = 1
        vis_num = config.vis_num

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio < 1 else None
        self.lstm_cell = nn.LSTMCell(embed_dim + vis_dim, hidden_dim, num_layers)
        self.fc_dropout = nn.Dropout(dropout_ratio) if dropout_ratio < 1 else None
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        # attention
        self.att_vw = nn.Linear(vis_dim, vis_dim, bias=False)
        self.att_hw = nn.Linear(hidden_dim, vis_dim, bias=False)
        self.att_bias = nn.Parameter(torch.zeros(vis_num))
        self.att_w = nn.Linear(vis_dim, 1, bias=False)

        self.att_fc_v = nn.Linear(in_features=vis_dim, out_features=vis_dim)
        self.att_fc_e = nn.Linear(in_features=vis_dim, out_features=vis_dim)

        self.eng_conv1d = nn.Conv1d(512, 512, kernel_size=3)

        self.img_embed = nn.Linear(in_features=2048, out_features=512)

        self.to(device)

    def _attention_layer(self, features, hiddens):
        """
        :param features:  batch_size * 49 * 512
        :param hiddens:  batch_size * hidden_dim
        :return:
        """
        att_fea = self.att_vw(features)
        # N-L-D
        att_h = self.att_hw(hiddens).unsqueeze(1)
        # N-1-D
        att_full = nn.ReLU()(att_fea + att_h + self.att_bias.view(1, -1, 1))
        att_out = self.att_w(att_full).squeeze(2)
        alpha = nn.Softmax()(att_out)
        # N-L
        context = torch.sum(features * alpha.unsqueeze(2), 1)
        return context, alpha

    def get_attn_input(self, features, eng_embedding):
        """

        :param features: batch_size * 512 * 1 * 7 * 7
        :param eng_embedding: batch_size * 15 * 512
        :return: feas, features_attn
        """
        features = features.squeeze(2)  # batch_size * 512 * 7 * 7

        if self.config.attn_option == 'e':
            feas = eng_embedding.mean(1)
        else:
            feas = features.reshape(features.shape[0], features.shape[1], features.shape[2] * features.shape[3]).mean(2)

        if self.config.attn_option in ['v', 'both']:
            features_v = features
            features_v = features_v.reshape(features_v.shape[0], features_v.shape[1],
                                            features_v.shape[2] * features_v.shape[3])  # batch_size * 512 * 49
            features_v = features_v.transpose(1, 2)  # batch_size * 49 * 512
            features_v = self.att_fc_v(features_v)    # batch_size * 49 * 512
            features_v = torch.nn.functional.normalize(features_v, dim=2)

        if self.config.attn_option in ['e', 'both']:
            # eng_embedding = self.att_fc_e(eng_embedding)
            eng_embedding = eng_embedding.transpose(1, 2)       # batch_size * 512 * 15
            eng_embedding = self.eng_conv1d(eng_embedding)
            eng_embedding = torch.nn.functional.normalize(eng_embedding, dim=2)     # batch_size * 512 * 15
            eng_embedding = eng_embedding.transpose(1, 2)

        if self.config.attn_option == 'v':
            features_all = features_v
        elif self.config.attn_option == 'e':
            features_all = eng_embedding
        elif self.config.attn_option == 'both':
            features_all = torch.cat([features_v, eng_embedding], dim=1)  # used for attention
        # features_all = features_v

        return feas, features_all

    def forward(self, feature_0, features, eng_embedding, captions, lengths):
        """
        :param feature_0: batch_size * 2048
        :param features: batch_size * 512 * 1 * 7 * 7
        :param eng_embedding: batch_size * 15 * 512
        :param captions:
        :param lengths:
        :return:
        """
        # features = features.squeeze(2)  # batch_size * 512 * 7 * 7
        #
        # features_conv_1 = self.att_conv2d_v_1(features) # batch_size * 512 * 5 * 5
        #
        # features_v = features_conv_1
        # features_v = features_v.reshape(features_v.shape[0], features_v.shape[1], features_v.shape[2] * features_v.shape[3])    # batch_size * 512 * 25
        # features_v = features_v.transpose(1, 2)     # batch_size * 25 * 512
        #
        # features_all = torch.cat([features_v, eng_embedding], dim=1)  # used for attention
        # # features_all = eng_embedding        # eng caption only

        # feas, features_all = self.get_attn_input(features, eng_embedding)
        _, features_all = self.get_attn_input(features, eng_embedding)

        feas = self.img_embed.forward(feature_0)

        batch_size, time_step = captions.data.shape
        vocab_size = self.config.vocab_size

        dropout = self.dropout
        attention_layer = self._attention_layer
        lstm_cell = self.lstm_cell
        fc_dropout = self.fc_dropout
        fc_out = self.fc_out
        embed = self.embed

        word_embeddings = embed(captions)
        word_embeddings = dropout(word_embeddings) if dropout is not None else word_embeddings
        # feas = features.mean(3).mean(2)  # batch_size * 512     # step = 0
        h0, c0 = torch.zeros(batch_size, self.config.hidden_dim), torch.zeros(batch_size, self.config.hidden_dim)
        h0 = h0.to(device)
        c0 = c0.to(device)

        predicts = torch.zeros(batch_size, time_step, vocab_size).to(device)

        for step in range(time_step):
            batch_size = sum(i >= step for i in lengths)
            if step > 0:
                # p = h0[:batch_size, :]
                p = word_embeddings[:batch_size, step - 1, :]
                feas, alpha = attention_layer(features_all[:batch_size, :], p) # features:(32, 49, 512) h0:(32, 512)
            words = (word_embeddings[:batch_size, step, :]).squeeze(1)
            inputs = torch.cat([feas, words], 1)
            h0, c0 = lstm_cell(inputs, (h0[:batch_size, :], c0[:batch_size, :]))
            outputs = fc_out(fc_dropout(h0)) if fc_dropout is not None else fc_out(h0)
            predicts[:batch_size, step, :] = outputs

        return predicts


if __name__ == '__main__':
    conf = LanguageModelConfig()
    lm = LanguageModelLSTM(conf)

    input_res5b = torch.zeros(2, 512, 7, 7)
    # out = lm.conv2d_video(input_res5b)    # (N * 256 * 3 * 3)
    # print(out.shape)

    input_eng = torch.zeros(2, 512, 15)
    # out = lm.conv1d_emb(input_eng)      # (N * 256 * 13)
    # print(out.shape)

    out = lm.attn(input_res5b, input_eng, torch.tensor([1, 2]))
    print(out.shape)



