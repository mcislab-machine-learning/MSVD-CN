import os
import pickle
import argparse
from collections import Counter
import json
import csv

import nltk

from preprocess import data_clean

start_word = '<S>'
end_word = '</S>'
unknown_word = '<unk>'

lang_chs = '<CHS>'
lang_en = '<EN>'


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def get_index(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def get_word(self, index):
        index = int(index)
        if index not in self.idx2word:
            return unknown_word
        return self.idx2word[index]

    def __len__(self):
        return len(self.word2idx)

    def tokenize_id(self, sentence):
        """

        :param sentence: "hello world"
        :return: [<S>, hello, world, </S>]
        """
        tokens = nltk.tokenize.word_tokenize(sentence.lower())
        tokenized_caption = []
        tokenized_caption.append(self.get_index(start_word))
        tokenized_caption.extend([self.get_index(t) for t in tokens])
        tokenized_caption.append(self.get_index(end_word))
        return tokenized_caption

    @staticmethod
    def load(vocab_path):
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
            print('vocab size:', vocab.idx)
            return vocab


def gen_vocab(tokenized_sentence_list, min_threshold):
    vocab = Vocabulary()
    counter = Counter()
    for i, sentence in enumerate(tokenized_sentence_list):
        tokens = sentence.strip().split()
        counter.update(tokens)

    words = [word for word, cnt in counter.items() if cnt >= min_threshold]

    preset_symbols = [start_word, end_word, unknown_word, lang_chs, lang_en]
    for sym in preset_symbols:
        vocab.add_word(sym)

    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def gen_vocab_chinese(vocab_path, segment_method='char'):
    all_captions = [i[2] for i in data_clean.read_filtered_captions()]

    if segment_method == 'char':
        tokenized_sentence_list = [' '.join(i) for i in all_captions]
    elif segment_method == 'word':
        tokenized_sentence_list = data_clean.segment_sentences(all_captions)

    vocab = gen_vocab(tokenized_sentence_list=tokenized_sentence_list, min_threshold=0)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)


def gen_vocab_bilingual(vocab_path, chn_segment_method='char'):
    # Chinese
    captions_chn = [i[2] for i in data_clean.read_filtered_captions()]

    if chn_segment_method == 'char':
        captions_chn_tokenized = [' '.join(i) for i in captions_chn]
    elif chn_segment_method == 'word':
        captions_chn_tokenized = data_clean.segment_sentences(captions_chn)

    # English
    f_captions = open("..\\data\\video-descriptions.csv", "r", encoding='utf-8')
    f_video_mapping = open("..\\data\\youtube_mapping.txt", 'r', encoding='utf-8')
    reader = csv.reader(f_captions)
    all_captions = {}
    captions_eng = []

    video_name_map = dict()
    for line in f_video_mapping:
        strs = line.strip().split()
        video_name = strs[0][:11]
        vid = strs[1] + '.mp4'
        video_name_map[video_name] = vid

    for strs in reader:
        # strs = line.strip().split(',')
        video_name = strs[0]
        if video_name not in video_name_map:
            continue
        language = strs[6]
        caption = strs[-1].lower()  # use lower case
        if not (strs[4] == 'clean' and language.lower() == 'english'):
            continue
        vid = video_name_map[video_name]
        if vid not in all_captions:
            all_captions[vid] = []
        all_captions[vid].append(caption)
        captions_eng.append(caption)

    captions_eng_tokenized = captions_eng

    tokenized_sentence_list = captions_chn_tokenized + captions_eng_tokenized
    vocab = gen_vocab(tokenized_sentence_list=tokenized_sentence_list, min_threshold=0)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)


def main():
    train_caption_file = '/home/mcislab/zwt/caption_dataset/MSCOCO/annotations/captions_train2014.json'
    val_caption_file = '/home/mcislab/zwt/caption_dataset/MSCOCO/annotations/captions_val2014.json'
    captions_list = []
    f_train_annotation = open(train_caption_file, 'r')
    f_val_annotation = open(val_caption_file, 'r')

    d = json.load(f_train_annotation)
    captions_list.extend([a['caption'] for a in d['annotations']])
    d = json.load(f_val_annotation)
    captions_list.extend([a['caption'] for a in d['annotations']])

    vocab = gen_vocab(captions_list, 5)
    vocab_save_path = 'vocabulary.pkl'
    with open(vocab_save_path, 'wb') as f:
        pickle.dump(vocab, f)

    f_train_annotation.close()
    f_val_annotation.close()


if __name__ == '__main__':
    # gen_vocab_chinese(vocab_path=r"D:\Code\Projects\PyCharmProjects\caption_models_Chinese\data\vocabulary_Chinese_char.pkl")
    # gen_vocab_chinese(vocab_path=r"D:\Code\Projects\PyCharmProjects\caption_models_Chinese\data\vocabulary_Chinese_word.pkl", segment_method='word')
    gen_vocab_bilingual(vocab_path=r"D:\Code\Projects\PyCharmProjects\caption_models_Chinese\data\vocabulary_bilingual_char.pkl")
    gen_vocab_bilingual(vocab_path=r"D:\Code\Projects\PyCharmProjects\caption_models_Chinese\data\vocabulary_bilingual_word.pkl", chn_segment_method='word')
