import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional

import data_video
from data_video import MSVDDatasetMultiModal
from vocabulary import Vocabulary
from torch.utils.data.dataloader import DataLoader


vocab_path = data_video.msvd_bilingual_vocab_char_path
vocab = Vocabulary.load(vocab_path)


# class LanguageModelConfig:
#     def __init__(self):
#         self.image_feature_dim = 2048
#         self.image_embedding_dim = 512
#         self.word_embedding_dim = 512
#         self.lstm_output_size = 512
#         self.vocab_size = 5000
#
#         self.eng_caption_max_len = 15
#         self.attn_out_dim = 512
#         self.attn_emb_dim = 512

#
# class LanguageModelLSTM(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#
#         # image embedding
#         self.image_embedding_layer = nn.Linear(in_features=self.config.image_feature_dim, out_features=self.config.image_embedding_dim,
#                                                bias=False)
#
#         # word embedding
#         self.input_word_embeddings = nn.Embedding(self.config.vocab_size, self.config.word_embedding_dim)
#
#         attn_out_dim = self.config.attn_out_dim
#         attn_emb_dim = self.config.attn_emb_dim
#         self.conv2d_video = nn.Conv2d(in_channels=512, out_channels=attn_out_dim, kernel_size=(3, 3), stride=1)
#         self.conv1d_emb = nn.Conv1d(in_channels=attn_emb_dim, out_channels=attn_out_dim, kernel_size=3)
#         self.attn_layer = nn.Linear(in_features=self.config.word_embedding_dim, out_features=self.config.attn_out_dim)
#         self.attn_output_emb = nn.Linear(in_features=attn_out_dim, out_features=self.config.word_embedding_dim)
#
#         self.lstm = nn.LSTM(self.config.word_embedding_dim, self.config.lstm_output_size)
#         self.output_word_layer = nn.Linear(in_features=self.config.lstm_output_size,
#                                            out_features=self.config.vocab_size, bias=False)
#
#     def get_image_embedding(self, image_feature):
#         return self.image_embedding_layer(image_feature)
#
#     def get_word_embedding(self, words):
#         return self.input_word_embeddings(words)
#
#     def get_context_matrix(self, res5b_feature, eng_embedding):
#         """
#
#         :param res5b_feature: N * 512 * 1 * 7 * 7
#         :param eng_embedding: N * 15 * 512
#         :return:
#         """
#         res5b_feature = res5b_feature.squeeze(dim=2)                # N * 512 * 7 * 7
#         m = res5b_feature.reshape(res5b_feature.shape[0], res5b_feature.shape[1], -1)
#         # res5b_conv = self.conv2d_video.forward(res5b_feature)     # N * 256 * 5 * 5
#         # res5b_conv = res5b_conv.reshape((res5b_conv.shape[0], res5b_conv.shape[1], -1))     # N * 256 * 25
#         # eng_embedding = eng_embedding.transpose(1, 2)           # eng_embedding = N * 512 * 15
#         # emb_conv = self.conv1d_emb.forward(eng_embedding)       # N * 256 * 13
#         # m = torch.cat([res5b_conv, emb_conv], dim=2)            # N * 256 * 38
#         # m = res5b_conv
#         return m
#
#     def attn(self, res5b_feature, eng_embedding, previous_word_id):
#         """
#
#         :param res5b_feature: N * 512 * 1 * 7 * 7
#         :param eng_embedding: N * 15 * 512
#         :param previous_word_id: N integers
#         :return: attention output n * 256
#         """
#         # m = self.get_context_matrix(res5b_feature, eng_embedding)       # N * 256 * 38
#         # previous_embedding = self.get_word_embedding(previous_word_id)  # N * 512
#         # s = self.attn_layer(previous_embedding).unsqueeze(1)            # N * 1 * 256
#         # score = torch.nn.functional.softmax(torch.bmm(s, m), dim=-1)    # N * 1 * 38
#         # score = score.transpose(1, 2)      # N * 38 * 1
#         # attn_output = torch.bmm(m, score).squeeze(2)                    # N * 256
#         # return attn_output
#         return self.input_word_embeddings(previous_word_id)
#
#     def forward(self, image_feature, res5b_feature, eng_embedding, input_words, lengths):     # for training only
#         """
#
#         :param image_feature: batch_size * feature_dim
#         :param res5b_feature: batch_size * 512 * 1 * 7 * 7
#         :param eng_embedding: batch_size * 15 * 512
#         :param input_words: batch_size * max_len
#         :param lengths:
#         :return:
#         """
#         # input_word_embedding = self.input_word_embeddings(input_words)
#         image_feature = self.get_image_embedding(image_feature)
#
#         batch_size = image_feature.shape[0]
#         max_length = input_words.shape[1]
#         # res5b_feature = res5b_feature.expand(max_length, batch_size, res5b_feature.shape[1], res5b_feature.shape[2],
#         #                                      res5b_feature.shape[3], res5b_feature.shape[4])
#         res5b_feature = res5b_feature.unsqueeze(0).repeat([max_length, 1, 1, 1, 1, 1])
#         res5b_feature = res5b_feature.reshape(max_length * batch_size, res5b_feature.shape[2], res5b_feature.shape[3],
#                                              res5b_feature.shape[4], res5b_feature.shape[5])
#         # eng_embedding = eng_embedding.expand(max_length, batch_size, eng_embedding.shape[1], eng_embedding.shape[2])
#         eng_embedding = eng_embedding.unsqueeze(0).repeat([max_length, 1, 1, 1])
#         eng_embedding = eng_embedding.reshape(batch_size * max_length, eng_embedding.shape[2], eng_embedding.shape[3])
#         prev_words = input_words.reshape([batch_size * max_length,])
#
#         attn_output = self.attn(res5b_feature, eng_embedding, prev_words)   # batch_size * max_length * 256
#         attn_output_embeddings = self.attn_output_emb(attn_output)                     # batch_size * max_length * 512
#         attn_output_embeddings = attn_output_embeddings.reshape(batch_size, max_length, -1)
#
#         # embeddings = torch.cat((image_feature.unsqueeze_(1), input_word_embedding), 1)
#         embeddings = torch.cat((image_feature.unsqueeze_(1), attn_output_embeddings), 1)
#         embeddings_packed = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True)
#
#         lstm_output_packed, (h_n, c_n) = self.lstm(embeddings_packed)
#         lstm_output = lstm_output_packed[0]
#
#         word_output = self.output_word_layer(lstm_output)
#         return word_output, h_n
#
#     # not used
#     # def inference_step(self, inputs, states):   # for inference only
#     #     hiddens, states = self.lstm(inputs, states)  # hiddens:     states: [1, 3, 512] [1, 3, 512]
#     #     outputs = self.output_word_layer(hiddens.squeeze(0))  # lstm outputs:
#     #     softmax = self.output_softmax(outputs)
#     #     return softmax
#

class Config:
    def __init__(self):
        self.vocab_size = 2000
        self.embed_dim = 512
        self.dropout_ratio = 0.5
        self.hidden_dim = 512
        self.vis_dim = 512
        # self.vis_num = 49         # v only
        self.vis_num = 49 + 15  # no conv
        # self.vis_num = 25 + 15

        self.eng_caption_max_len = 15


class LanguageModelLSTM1(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        self.device = device

        self.config = config
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

        self.att_conv2d_v_1 = nn.Conv2d(in_channels=vis_dim, out_channels=vis_dim, kernel_size=3)
        # self.att_conv2d_v_2 = nn.Conv2d(in_channels=vis_dim, out_channels=vis_dim, kernel_size=3)
        self.att_fc = nn.Linear(in_features=vis_dim, out_features=vis_dim)

        self.att_layer_norm = torch.nn.LayerNorm([49, 512])

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

        feas = features.reshape(features.shape[0], features.shape[1], features.shape[2] * features.shape[3]).mean(2)

        # features_conv_1 = self.att_conv2d_v_1(features)  # batch_size * 512 * 5 * 5
        # features_v = features_conv_1
        features_v = features

        features_v = features_v.reshape(features_v.shape[0], features_v.shape[1],
                                        features_v.shape[2] * features_v.shape[3])  # batch_size * 512 * 49
        features_v = features_v.transpose(1, 2)  # batch_size * 49 * 512

        features_v = self.att_fc(features_v)    # batch_size * 49 * 512
        features_v = torch.nn.functional.normalize(features_v, dim=2)
        # features_v = self.att_layer_norm(features_v)    # layer norm, batch_size * 49 * 512

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
        h0 = h0.to(self.device)
        c0 = c0.to(self.device)

        predicts = torch.zeros(batch_size, time_step, vocab_size).to(self.device)

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



