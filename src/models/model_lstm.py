import os

import numpy as np
import torch
import torch.nn as nn


class LanguageModelConfig:
    def __init__(self):
        self.image_feature_dim = 2048
        self.image_embedding_dim = 512
        self.word_embedding_dim = 512
        self.lstm_output_size = 512
        self.vocab_size = 5000


class LanguageModelLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # image embedding
        self.image_embedding_layer = nn.Linear(in_features=self.config.image_feature_dim, out_features=self.config.image_embedding_dim,
                                               bias=False)

        # word embedding
        self.input_word_embeddings = nn.Embedding(self.config.vocab_size, self.config.word_embedding_dim)
        self.lstm = nn.LSTM(self.config.word_embedding_dim, self.config.lstm_output_size)
        self.output_word_layer = nn.Linear(in_features=self.config.lstm_output_size,
                                           out_features=self.config.vocab_size, bias=False)

    def get_image_embedding(self, image_feature):
        return self.image_embedding_layer(image_feature)

    def get_word_embedding(self, words):
        return self.input_word_embeddings(words)

    def forward(self, image_feature, input_words, lengths):     # for training only
        input_word_embedding = self.input_word_embeddings(input_words)
        image_feature = self.get_image_embedding(image_feature)

        embeddings = torch.cat((image_feature.unsqueeze_(1), input_word_embedding), 1)
        embeddings_packed = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True)

        lstm_output_packed, (h_n, c_n) = self.lstm(embeddings_packed)
        lstm_output = lstm_output_packed[0]

        word_output = self.output_word_layer(lstm_output)
        return word_output, h_n

    def inference_step(self, inputs, states):   # for inference only
        hiddens, states = self.lstm(inputs, states)  # hiddens:     states: [1, 3, 512] [1, 3, 512]
        outputs = self.output_word_layer(hiddens.squeeze(0))  # lstm outputs:
        softmax = self.output_softmax(outputs)
        return softmax


