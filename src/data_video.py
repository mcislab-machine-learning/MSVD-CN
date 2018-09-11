import sqlite3
from collections import namedtuple
from functools import lru_cache
import random

import numpy as np
import scipy.io
import torch
import torch.utils.data as data

from util.segment_sentences_chn import *
from vocabulary import *

CaptionItem = namedtuple('CaptionItem', 'video_id, caption, tokenized_caption')

res5c_feature_path = '/home/mcislab/houjingyi/cvpr19/msvd_res/res5c'
pool5_feature_path = '/home/mcislab/houjingyi/cvpr19/msvd_res/pool5'
# res5c_feature_path = '/media/mcislab3d/Seagate Backup Plus Drive/zwt/caption_models_Chinese/msvd_feature/res5c'
# pool5_feature_path = '/media/mcislab3d/Seagate Backup Plus Drive/zwt/caption_models_Chinese/msvd_feature/pool5'

c3d_feature_path = '/media/mcislab/sdb1/home/mcislab/nieyuxiang/extract_C3Dv1.1_features/MSVD_prefix'

msvd_chinese_vocab_word_path = os.path.join('..', 'data', 'vocabulary_Chinese_word.pkl')
msvd_chinese_vocab_char_path = os.path.join('..', 'data', 'vocabulary_Chinese_char.pkl')

msvd_bilingual_vocab_word_path = os.path.join('..', 'data', 'vocabulary_bilingual_word.pkl')
msvd_bilingual_vocab_char_path = os.path.join('..', 'data', 'vocabulary_bilingual_char.pkl')

chn_db_filename = os.path.join('..', 'data', 'captions.db')


def read_binary_blob(file_name):
    fid = open(file_name, 'rb')

    # s contains size of the blob e.g. num x chanel x length x height x width
    s = np.fromfile(fid, np.int32, 5)

    m = s[0] * s[1] * s[2] * s[3] * s[4]

    # data is the blob binary data in single precision (e.g float in C++)
    data = np.fromfile(fid, np.float32, m)
    data = data.reshape(s)

    fid.close()
    return data


@lru_cache(maxsize=4000)
def get_c3d_feature_of_video(video_name, layer_name='res5b'):
    """

    :param video_name: vid1.mp4
    :param layer_name:
    :return:
    """
    if video_name.endswith('.mp4'):
        video_name = video_name.replace('.mp4', '.avi')
    if re.match('^vid[0-9]+$', video_name) is not None:
        video_name += '.avi'

    feature_folder = os.path.join(c3d_feature_path, video_name)
    postfix = '.' + layer_name
    feature_files = filter(lambda x: x.endswith(postfix), os.listdir(feature_folder))
    feature_list = [read_binary_blob(os.path.join(feature_folder, feature_file)) for feature_file in feature_files]
    return np.mean(feature_list, axis=0)


@lru_cache(maxsize=2000)
def get_res5c_feature_of_video(video_name):
    feature_file = os.path.join(res5c_feature_path, video_name + '.mat')
    feature = np.array(scipy.io.loadmat(feature_file)['feat_res5c'], dtype=np.float32)
    feature = np.mean(feature, axis=0).reshape([2048, -1]).mean(axis=1)  # shape=(2048,)
    feature_tensor = torch.tensor(feature)
    return feature_tensor


def read_msvd_generated_captions_eng():
    """
    vid = 'vid1.mp4'
    all_captions[vid] = ['caption 1', 'caption 2']
    :return:
    """
    all_captions = {}
    vid = 1301
    with open('../data/youtube2text_scn_test.txt', 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().lower()
            all_captions['vid{}.mp4'.format(vid + i)] = [line]
    return all_captions


def read_msvd_captions_eng():
    # English
    f_captions = open("../data/MSR Video Description Corpus.csv", "r", encoding='utf-8')
    f_video_mapping = open("../data/youtube_mapping.txt", 'r', encoding='utf-8')
    reader = csv.reader(f_captions)
    all_captions = {}
    captions_eng = []

    video_name_map = dict()
    for line in f_video_mapping:
        strs = line.strip().split()
        video_name = strs[0]
        vid = strs[1] + '.mp4'
        assert video_name not in video_name_map
        video_name_map[video_name] = vid

    for i, strs in enumerate(reader):
        if i == 0:
            continue

        video_name, s, e = strs[0], strs[1], strs[2]
        video_name = '{}_{}_{}'.format(video_name, s, e)

        if video_name not in video_name_map:
            continue

        language = strs[6]
        caption = strs[-1].lower()  # use lower case

        vid = video_name_map[video_name]
        video_index = int(vid.split('.')[0][3:])
        if language.lower() != 'english':
            continue
        if strs[4] != 'clean':
            continue

        if vid not in all_captions:
            all_captions[vid] = []
        all_captions[vid].append(caption)
        captions_eng.append(caption)

    f_captions.close()

    f_captions = open("../data/MSR Video Description Corpus.csv", "r", encoding='utf-8')
    reader = csv.reader(f_captions)

    missing_list = []
    for i in range(1301, 1971):
        vid = 'vid{}.mp4'.format(i)
        if vid not in all_captions.keys():
            missing_list.append(vid)
    print('missing:', missing_list)

    for i, strs in enumerate(reader):
        if i == 0:
            continue
        video_name, s, e = strs[0], strs[1], strs[2]
        video_name = '{}_{}_{}'.format(video_name, s, e)
        if video_name not in video_name_map:
            continue
        language = strs[6]
        if language.lower() != 'english':
            continue

        caption = strs[-1].lower()  # use lower case
        vid = video_name_map[video_name]

        if vid not in missing_list:
            continue
        assert vid in missing_list
        if vid not in all_captions:
            all_captions[vid] = []
        all_captions[vid].append(caption)
        captions_eng.append(caption)

    return all_captions


def read_msvd_captions_eng_downsample():
    all_captions = read_msvd_captions_eng()
    all_captions_downsample = {}
    for vid, captions in all_captions.items():
        if len(captions) > 6:
            captions_downsample = random.sample(captions, k=6)
            all_captions_downsample[vid] = captions_downsample
        else:
            all_captions_downsample[vid] = captions
    return all_captions_downsample


class MSVDDatasetCHN(data.Dataset):
    def __init__(self, vocab, segment_method='char', caption_mode='token', split='train', feature='resnet'):
        self.db_file_name = chn_db_filename

        self.data = []
        self.caption_mode = caption_mode
        assert(split in ['all', 'train', 'val', 'test'])
        self.feature = feature

        all_caption_list = []
        video_name_list = []

        start_id = vocab.get_index(start_word)
        end_id = vocab.get_index(end_word)

        conn = sqlite3.connect(self.db_file_name)
        cursor = conn.cursor()
        cursor.execute('select id, video_id, sentence from captions_filter where deleted=0')
        result = cursor.fetchall()
        for item in result:
            id, vid, caption = item
            video_name_list.append(vid)
            all_caption_list.append(caption)
        cursor.close()
        conn.close()

        if segment_method == 'char':
            segmented_sentences = segment_sentences_char(all_caption_list)
        elif segment_method == 'word':
            segmented_sentences = segment_sentences(all_caption_list)

        assert len(video_name_list) == len(all_caption_list) and len(segmented_sentences) == len(all_caption_list)

        for i in range(len(segmented_sentences)):
            vid = video_name_list[i]
            video_index = int(vid.split('.')[0][3:])

            is_valid_index = True
            if split == 'train':
                is_valid_index = 1 <= video_index <= 1200
            elif split == 'val':
                is_valid_index = 1201 <= video_index <= 1300
            elif split == 'test':
                is_valid_index = 1301 <= video_index <= 1970
            if not is_valid_index:
                continue

            caption_raw = all_caption_list[i]
            tokens = segmented_sentences[i].strip().split()
            tokenized_caption = [start_id]
            tokenized_caption.extend([vocab.get_index(i) for i in tokens])
            tokenized_caption.append(end_id)
            self.data.append(CaptionItem(video_id=vid, caption=caption_raw, tokenized_caption=tokenized_caption))

    def __getitem__(self, index):
        caption_item = self.data[index]
        video_name = caption_item.video_id.split('.')[0]
        tokenized_caption = torch.tensor(caption_item.tokenized_caption)
        # feature_file = os.path.join(res5c_feature_path, video_name + '.mat')
        # feature = np.array(scipy.io.loadmat(feature_file)['feat_res5c'], dtype=np.float32)
        # feature = np.mean(feature, axis=0).reshape([2048, -1]).mean(axis=1)     # shape=(2048,)
        # feature_tensor = torch.tensor(feature)

        if self.feature == 'resnet':
            feature_tensor = get_res5c_feature_of_video(video_name)
        elif self.feature == 'c3d_pool5':
            feature_tensor = get_c3d_feature_of_video(video_name, layer_name='pool5')
            feature_tensor = feature_tensor.reshape([512,])
            feature_tensor = torch.tensor(feature_tensor)

        if self.caption_mode == 'token':
            return video_name, feature_tensor, tokenized_caption
        else:
            return video_name, feature_tensor, caption_item.caption

    def __len__(self):
        return len(self.data)


class MSVDDatasetBilingual(data.Dataset):
    def __init__(self, vocab, segment_method='char', caption_mode='token', split='train', feature='resnet'):
        self.db_file_name = chn_db_filename

        self.data = []
        self.caption_mode = caption_mode
        assert(split in ['all', 'train', 'val', 'test'])
        self.split = split
        self.feature = feature

        all_caption_list = []
        video_name_list = []

        start_id = vocab.get_index(start_word)
        end_id = vocab.get_index(end_word)

        lang_token_chs = vocab.get_index(lang_chs)
        lang_token_en = vocab.get_index(lang_en)

        conn = sqlite3.connect(self.db_file_name)
        cursor = conn.cursor()
        cursor.execute('select id, video_id, sentence from captions_filter where deleted=0')
        result = cursor.fetchall()
        for item in result:
            id, vid, caption = item
            video_name_list.append(vid)
            all_caption_list.append(caption)
        cursor.close()
        conn.close()

        if segment_method == 'char':
            segmented_sentences = segment_sentences_char(all_caption_list)
        elif segment_method == 'word':
            segmented_sentences = segment_sentences(all_caption_list)

        assert len(video_name_list) == len(all_caption_list) and len(segmented_sentences) == len(all_caption_list)

        count_chn = 0
        count_eng = 0

        for i in range(len(segmented_sentences)):
            vid = video_name_list[i]
            if not self.__is_valid_video_name(vid):
                continue

            caption_raw = all_caption_list[i]
            tokens = segmented_sentences[i].strip().split()
            tokenized_caption = [lang_token_chs]        # start with <CHS>
            tokenized_caption.extend([vocab.get_index(i) for i in tokens])
            tokenized_caption.append(end_id)
            count_chn += 1
            self.data.append(CaptionItem(video_id=vid, caption=caption_raw, tokenized_caption=tokenized_caption))

        # caption_eng = read_msvd_captions_eng()
        caption_eng = read_msvd_captions_eng_downsample()   # FIXME:
        for vid, captions in caption_eng.items():
            if not self.__is_valid_video_name(vid):
                continue

            for c in captions:
                caption_raw = c
                tokens = c.strip().split()
                tokenized_caption = [lang_token_en]     # start with <EN>
                tokenized_caption.extend([vocab.get_index(i) for i in tokens])
                tokenized_caption.append(end_id)
                count_eng += 1
                self.data.append(CaptionItem(video_id=vid, caption=caption_raw, tokenized_caption=tokenized_caption))

        print('English: {}, Chinese: {}'.format(count_eng, count_chn))

    def __is_valid_video_name(self, video_name):
        video_index = int(video_name.split('.')[0][3:])
        split = self.split
        is_valid_index = True
        if split == 'train':
            is_valid_index = 1 <= video_index <= 1200
        elif split == 'val':
            is_valid_index = 1201 <= video_index <= 1300
        elif split == 'test':
            is_valid_index = 1301 <= video_index <= 1970
        return is_valid_index

    def __getitem__(self, index):
        caption_item = self.data[index]
        video_name = caption_item.video_id.split('.')[0]
        tokenized_caption = torch.tensor(caption_item.tokenized_caption)
        # feature_file = os.path.join(res5c_feature_path, video_name + '.mat')
        # feature = np.array(scipy.io.loadmat(feature_file)['feat_res5c'], dtype=np.float32)
        # feature = np.mean(feature, axis=0).reshape([2048, -1]).mean(axis=1)     # shape=(2048,)
        # feature_tensor = torch.tensor(feature)

        if self.feature == 'resnet':
            feature_tensor = get_res5c_feature_of_video(video_name)
        elif self.feature == 'c3d_pool5':
            feature_tensor = get_c3d_feature_of_video(video_name, layer_name='pool5')
            feature_tensor = feature_tensor.reshape([512,])
            feature_tensor = torch.tensor(feature_tensor)

        if self.caption_mode == 'token':
            return video_name, feature_tensor, tokenized_caption
        elif self.caption_mode == 'text':
            return video_name, feature_tensor, caption_item.caption, tokenized_caption

    def __len__(self):
        return len(self.data)


class MSVDDatasetMultiModal(data.Dataset):
    def __init__(self, vocab, segment_method='char', caption_mode='token', split='train', feature='resnet', english_src='groundtruth'):
        self.db_file_name = chn_db_filename

        self.data = []
        self.caption_mode = caption_mode
        assert(split in ['all', 'train', 'val', 'test'])
        self.split = split
        self.feature = feature

        all_caption_list = []
        video_name_list = []

        start_id = vocab.get_index(start_word)
        end_id = vocab.get_index(end_word)

        conn = sqlite3.connect(self.db_file_name)
        cursor = conn.cursor()
        cursor.execute('select id, video_id, sentence from captions_filter where deleted=0')
        result = cursor.fetchall()
        for item in result:
            id, vid, caption = item
            video_name_list.append(vid)
            all_caption_list.append(caption)
        cursor.close()
        conn.close()

        if segment_method == 'char':
            segmented_sentences = segment_sentences_char(all_caption_list)
        elif segment_method == 'word':
            segmented_sentences = segment_sentences(all_caption_list)

        assert len(video_name_list) == len(all_caption_list) and len(segmented_sentences) == len(all_caption_list)

        for i in range(len(segmented_sentences)):
            vid = video_name_list[i]

            if not self.__is_valid_video_name(vid):
                continue

            caption_raw = all_caption_list[i]
            tokens = segmented_sentences[i].strip().split()
            tokenized_caption = [start_id]
            tokenized_caption.extend([vocab.get_index(i) for i in tokens])
            tokenized_caption.append(end_id)
            self.data.append(CaptionItem(video_id=vid, caption=caption_raw, tokenized_caption=tokenized_caption))

        lang_token_en = vocab.get_index(lang_en)

        if english_src == 'groundtruth':
            caption_eng = read_msvd_captions_eng()
        elif english_src == 'generated':
            assert split == 'test'
            caption_eng = read_msvd_generated_captions_eng()

        self.caption_eng = {}
        for vid, captions in caption_eng.items():
            if not self.__is_valid_video_name(vid):
                continue
            self.caption_eng[vid] = []
            for c in captions:
                caption_raw = c
                tokens = c.strip().split()
                tokenized_caption = [lang_token_en]  # start with <EN>
                tokenized_caption.extend([vocab.get_index(i) for i in tokens])
                tokenized_caption.append(end_id)
                self.caption_eng[vid].append(CaptionItem(video_id=vid, caption=caption_raw, tokenized_caption=tokenized_caption))

        self.data = list(filter(lambda x: x.video_id in self.caption_eng, self.data))

    def __getitem__(self, index):
        caption_item = self.data[index]
        video_name = caption_item.video_id.split('.')[0]
        tokenized_caption = torch.tensor(caption_item.tokenized_caption)
        # feature_file = os.path.join(res5c_feature_path, video_name + '.mat')
        # feature = np.array(scipy.io.loadmat(feature_file)['feat_res5c'], dtype=np.float32)
        # feature = np.mean(feature, axis=0).reshape([2048, -1]).mean(axis=1)     # shape=(2048,)
        # feature_tensor = torch.tensor(feature)

        if self.feature == 'resnet':
            feature_tensor = get_res5c_feature_of_video(video_name)
        elif self.feature == 'c3d_pool5':
            feature_tensor = get_c3d_feature_of_video(video_name, layer_name='pool5')
            feature_tensor = feature_tensor.reshape([512,])
            feature_tensor = torch.tensor(feature_tensor)

        res5b_feature = get_c3d_feature_of_video(video_name, 'res5b')
        res5b_feature = torch.tensor(res5b_feature).squeeze(0)

        eng_caption = random.sample(self.caption_eng[caption_item.video_id], k=1)[0].tokenized_caption

        if self.caption_mode == 'token':
            return video_name, feature_tensor, tokenized_caption, res5b_feature, eng_caption
        else:
            return video_name, feature_tensor, caption_item.caption, res5b_feature, eng_caption

    def __len__(self):
        return len(self.data)

    def __is_valid_video_name(self, video_name):
        video_index = int(video_name.split('.')[0][3:])
        split = self.split
        is_valid_index = True
        if split == 'train':
            is_valid_index = 1 <= video_index <= 1200
        elif split == 'val':
            is_valid_index = 1201 <= video_index <= 1300
        elif split == 'test':
            is_valid_index = 1301 <= video_index <= 1970
        return is_valid_index


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image_filename, image, caption).
            - image_filename: string
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        image_filenames: list
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[2]), reverse=True)
    image_filenames, features, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    features = torch.stack(features, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return image_filenames, features, targets, lengths


def collate_fn_1(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image_filename, image, caption).
            - image_filename: string
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        image_filenames: list
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[2]), reverse=True)
    image_filenames, features, captions, res5b_features, captions_eng = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    features = torch.stack(features, 0)
    res5b_features = torch.stack(res5b_features, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    captions_eng_cut = []
    for c in captions_eng:
        captions_eng_cut.append(c[1:min(16, len(c) - 1)])

    return image_filenames, features, targets, res5b_features, captions_eng_cut, lengths


