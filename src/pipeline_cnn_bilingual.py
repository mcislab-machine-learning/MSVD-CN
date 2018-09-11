import os

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import sys
import traceback

import numpy as np
import json

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import IPython

import data_video
from vocabulary import *
from models import cnn, model_cnn
from util.beam_search_util_cnn import BeamSearch
from util.coco_result_generator import COCOResultGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


segment_method = 'char'
vocab_path = data_video.msvd_bilingual_vocab_char_path
# segment_method = 'word'
# vocab_path = data_video.msvd_bilingual_vocab_word_path


def repeat_features(imgsfeats, imgsfc7, ncap_per_img):
    """Repeat image features ncap_per_img times"""

    batchsize, featdim, feat_h, feat_w = imgsfeats.size()
    batchsize_cap = batchsize * ncap_per_img
    imgsfeats = imgsfeats.unsqueeze(1).expand(batchsize, ncap_per_img, featdim, feat_h, feat_w)
    imgsfeats = imgsfeats.contiguous().view(batchsize_cap, featdim, feat_h, feat_w)

    batchsize, featdim = imgsfc7.size()
    batchsize_cap = batchsize * ncap_per_img
    imgsfc7 = imgsfc7.unsqueeze(1).expand(batchsize, ncap_per_img, featdim)
    imgsfc7 = imgsfc7.contiguous().view(batchsize_cap, featdim)

    return imgsfeats, imgsfc7


def train():
    saved_model_path = None

    use_multiple_caption = False
    captions_per_image = 5
    save_steps = 10000

    n_epoch = 500
    learning_rate = 5e-5
    scheduler_step_size = 30
    batch_size = 32

    vocab = Vocabulary.load(vocab_path)
    vocab_size = (vocab.idx // 100 + 1) * 100

    # if use_multiple_caption:
    #     dataset = data.COCODemoDataset2(split='train', vocab=vocab, mode='train', image_size=224,
    #                                     captions_per_image=captions_per_image)
    #     collate_fn = data.collate_fn_2
    # else:
    #     dataset = data.COCODemoDataset1(split='train', vocab=vocab, mode='train', image_size=224)
    #     collate_fn = data.collate_fn

    dataset = data_video.MSVDDatasetBilingual(vocab=vocab, segment_method=segment_method, split='train')
    collate_fn = data_video.collate_fn
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

    lm_config = model_cnn.LanguageModelConfig()
    lm_config.image_feature_dim = 2048
    lm_config.vocab_size = vocab_size
    lm_config.use_attention = False
    lm_cnn = model_cnn.LanguageModelConv(lm_config)
    lm_cnn.to(device)
    lm_cnn.train(True)

    epoch = 0
    global_step = 0

    optimizer = torch.optim.RMSprop(lm_cnn.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=.5)

    criterion = nn.CrossEntropyLoss()

    while epoch < n_epoch:
        scheduler.step()
        print('1 epoch = {} steps'.format(len(data_loader)))        # 831 steps
        for _,  (image_filename_list, image_feature, captions, lengths) in enumerate(data_loader):
            global_step += 1

            mask = torch.zeros(captions.shape)
            for i in range(len(captions)):
                mask[i][:len(captions[i])] = 1

            batch_size, max_caption_len = captions.shape

            # TODO: caption = ['<S>', 'hello', 'world'], no ending symbol ?
            captions = captions.to(device)  # (batch_size, caption_length)

            image_feature = image_feature.to(device)
            # image feature
            # image_feature, image_feature_fc7 = image_cnn.forward(images)  # (batch_size, feature_dim)
            # if use_multiple_caption:
            #     image_feature, image_feature_fc7 = repeat_features(image_feature, image_feature_fc7, captions_per_image)

            # word output
            word_output, attn = lm_cnn.forward(None, image_feature, input_word=captions)   # (batch_size, vocab_size, max_len)
            word_output = word_output[:, :, :-1]    # (batch_size, vocab_size, max_len - 1)

            captions = captions[:, 1:].contiguous()
            mask = mask[:, 1:].contiguous()

            word_output_t = word_output.permute(0, 2, 1).contiguous().view(batch_size * (max_caption_len - 1), -1)
            captions_t = captions.view(batch_size * (max_caption_len - 1), 1)
            maskids = torch.nonzero(mask.view(-1)).numpy().reshape(-1)

            caption_loss = criterion(word_output_t[maskids, :], captions_t[maskids, :].view(maskids.shape[0]))

            optimizer.zero_grad()
            caption_loss.backward()
            optimizer.step()

            print(epoch, global_step, 'loss:', caption_loss)

            if global_step % 10000 == 0:
                # save_model('../models_cnn_bilingual/model-{}-ep{}'.format(global_step, epoch), (lm_cnn, optimizer, epoch, global_step))
                test1(lm_cnn, global_step)

        epoch += 1


def save_model(save_path, items):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    lm_cnn, optimizer, epoch, global_step = items
    state_dict = {'lm_cnn': lm_cnn,
                  'optimizer': optimizer,
                  'epoch': epoch,
                  'global_step': global_step}
    torch.save(state_dict, save_path)
    print('model saved at {}'.format(save_path))


def load_model(save_path):
    state_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
    lm_cnn, optimizer, epoch, global_step = state_dict['lm_cnn'], \
                                            state_dict['optimizer'], state_dict['epoch'], \
                                            state_dict['global_step']
    print('loaded {}'.format(save_path))
    return lm_cnn, optimizer, epoch, global_step


def test1(lm_cnn, global_step):
    annotation_file_name = '../all_results/results_cnn_bilingual/msvd_annotation_bil_{}.json'.format(global_step)
    output_file_name = '../all_results/results_cnn_bilingual/msvd_result_bil_{}.json'.format(global_step)

    if not os.path.exists(os.path.dirname(annotation_file_name)):
        os.makedirs(os.path.dirname(annotation_file_name))

    beam_size = 3
    data_loader_batch_size = 4
    max_sentence_length = 15

    vocab = Vocabulary.load(vocab_path)
    dataset = data_video.MSVDDatasetCHN(vocab=vocab, segment_method=segment_method, caption_mode='text', split='test')
    # dataset = data.DemoDataset()
    data_loader = DataLoader(dataset=dataset, batch_size=data_loader_batch_size, shuffle=False, num_workers=0)

    # start_word_id = vocab.get_index(start_word)
    start_word_id = vocab.get_index(lang_chs)
    end_word_id = vocab.get_index(end_word)

    # result_set = set()
    # result_obj = []
    # annotation_obj = {'info': 'N/A', 'licenses': 'N/A', 'type': 'captions', 'images': [], 'annotations': []}
    # caption_id = 0

    result_generator = COCOResultGenerator()

    for index, (image_filename_list, image_feature, captions) in enumerate(data_loader):
        # print('images', len(images))
        # print('captions', len(captions))
        # captions[0] = list of 5 captions

        batch_size = image_feature.shape[0]       # actual batch size may be smaller than specified batch size !!!

        flag = True
        for _i in range(batch_size):
            result_generator.add_annotation(image_id=image_filename_list[_i], caption_raw=captions[_i])
            flag &= result_generator.has_output(image_filename_list[_i])
        if flag:
            continue

        beam_searcher = BeamSearch(beam_size, batch_size, max_sentence_length)

        # print('image_feature', image_feature.shape)
        # print('image_feature_fc7', image_feature_fc7.shape)

        image_feature = image_feature.to(device)
        image_feature_fc7 = image_feature
        # b, d, h, w = image_feature.shape
        # image_feature = image_feature.unsqueeze(1).expand(b, beam_size, d, h, w)
        # image_feature = image_feature.contiguous().view(b * beam_size, d, h, w)

        b, d = image_feature_fc7.shape
        image_feature_fc7 = image_feature_fc7.unsqueeze(1).expand(b, beam_size, d)
        image_feature_fc7 = image_feature_fc7.contiguous().view(b * beam_size, d)

        wordclass_feed = np.zeros((beam_size * batch_size, max_sentence_length), dtype='int64')
        wordclass_feed[:, 0] = start_word_id

        outcaps = np.empty((batch_size, 0)).tolist()

        for j in range(max_sentence_length - 1):
            wordclass = Variable(torch.from_numpy(wordclass_feed)).to(device)

            wordact, _ = lm_cnn.forward(None, image_feature_fc7, wordclass)
            wordact = wordact[:, :, :-1]
            wordact_j = wordact[..., j]

            beam_indices, wordclass_indices = beam_searcher.expand_beam(wordact_j)

            if len(beam_indices) == 0 or j == (max_sentence_length - 2):  # Beam search is over.
                generated_captions = beam_searcher.get_results()
                for k in range(batch_size):
                    g = generated_captions[:, k]
                    outcaps[k] = [vocab.get_word(int(x.cpu())) for x in g]
            else:
                wordclass_feed = wordclass_feed[beam_indices]
                image_feature_fc7 = image_feature_fc7.index_select(0,
                                                                   Variable(torch.LongTensor(beam_indices).to(device)))
                # image_feature = image_feature.index_select(0, Variable(torch.LongTensor(beam_indices).to(device)))
                for i, wordclass_idx in enumerate(wordclass_indices):
                    wordclass_feed[i, j + 1] = wordclass_idx

        for j in range(batch_size):
            num_words = len(outcaps[j])
            if end_word in outcaps[j]:
                num_words = outcaps[j].index(end_word)
            outcaps[j] = outcaps[j][:num_words]
            outcaps[j] = [i for i in outcaps[j] if i != end_word and i != start_word]
            outcap = ' '.join(outcaps[j][:num_words])

            print(image_filename_list[j], outcap)

            if not (result_generator.has_output(image_filename_list[j])):
                result_generator.add_output(image_filename_list[j], ''.join(outcap.split()))
        # print('--------')

    annotation_obj, result_obj = result_generator.get_annotation_and_output()

    f_output = open(output_file_name, 'w')
    json.dump(result_obj, f_output, indent=4)
    f_output.close()
    f_ann = open(annotation_file_name, 'w')
    json.dump(annotation_obj, f_ann, indent=4)
    f_ann.close()


def test():
    saved_model_path = '../models_cnn/model-400000'

    beam_size = 3
    data_loader_batch_size = 4
    max_sentence_length = 15

    vocab = Vocabulary.load(vocab_path)
    dataset = data_video.MSVDDatasetCHN(vocab=vocab, segment_method=segment_method, caption_mode='text', split='test')
    # dataset = data.DemoDataset()
    data_loader = DataLoader(dataset=dataset, batch_size=data_loader_batch_size, shuffle=False, num_workers=0)

    lm_cnn, optimizer, epoch, global_step = load_model(saved_model_path)
    lm_cnn.to(device)
    lm_cnn.train(False)

    # start_word_id = vocab.get_index(start_word)
    start_word_id = vocab.get_index(lang_chs)
    end_word_id = vocab.get_index(end_word)

    result_set = set()
    result_obj = []
    annotation_obj = {'info': 'N/A', 'licenses': 'N/A', 'type': 'captions', 'images': [], 'annotations': []}
    caption_id = 0

    for index, (image_filename_list, image_feature, captions) in enumerate(data_loader):
        # print('images', len(images))
        # print('captions', len(captions))
        # captions[0] = list of 5 captions

        batch_size = image_feature.shape[0]       # actual batch size may be smaller than specified batch size !!!
        beam_searcher = BeamSearch(beam_size, batch_size, max_sentence_length)

        # print('image_feature', image_feature.shape)
        # print('image_feature_fc7', image_feature_fc7.shape)

        image_feature = image_feature.to(device)
        image_feature_fc7 = image_feature
        # b, d, h, w = image_feature.shape
        # image_feature = image_feature.unsqueeze(1).expand(b, beam_size, d, h, w)
        # image_feature = image_feature.contiguous().view(b * beam_size, d, h, w)

        b, d = image_feature_fc7.shape
        image_feature_fc7 = image_feature_fc7.unsqueeze(1).expand(b, beam_size, d)
        image_feature_fc7 = image_feature_fc7.contiguous().view(b * beam_size, d)

        wordclass_feed = np.zeros((beam_size * batch_size, max_sentence_length), dtype='int64')
        wordclass_feed[:, 0] = start_word_id

        outcaps = np.empty((batch_size, 0)).tolist()

        for j in range(max_sentence_length - 1):
            wordclass = Variable(torch.from_numpy(wordclass_feed)).to(device)

            wordact, _ = lm_cnn.forward(None, image_feature_fc7, wordclass)
            wordact = wordact[:, :, :-1]
            wordact_j = wordact[..., j]

            beam_indices, wordclass_indices = beam_searcher.expand_beam(wordact_j)

            if len(beam_indices) == 0 or j == (max_sentence_length - 2):  # Beam search is over.
                generated_captions = beam_searcher.get_results()
                for k in range(batch_size):
                    g = generated_captions[:, k]
                    outcaps[k] = [vocab.get_word(int(x.cpu())) for x in g]
            else:
                wordclass_feed = wordclass_feed[beam_indices]
                image_feature_fc7 = image_feature_fc7.index_select(0,
                                                                   Variable(torch.LongTensor(beam_indices).to(device)))
                # image_feature = image_feature.index_select(0, Variable(torch.LongTensor(beam_indices).to(device)))
                for i, wordclass_idx in enumerate(wordclass_indices):
                    wordclass_feed[i, j + 1] = wordclass_idx

        for j in range(batch_size):
            num_words = len(outcaps[j])
            if end_word in outcaps[j]:
                num_words = outcaps[j].index(end_word)
            outcaps[j] = outcaps[j][:num_words]
            outcaps[j] = [i for i in outcaps[j] if i != end_word and i != start_word]
            outcap = ' '.join(outcaps[j][:num_words])

            if image_filename_list[j] not in result_set:
                result = {'image_id': image_filename_list[j], 'caption': ''.join(outcap.split()), 'image_filename': image_filename_list[j]}
                print(result)
                result_obj.append(result)
                result_set.add(image_filename_list[j])

            annotation_obj['images'].append({'id': image_filename_list[j]})
            caption = captions[j]
            annotation_obj['annotations'].append({'image_id': image_filename_list[j], 'caption': caption, 'id': caption_id})
            caption_id += 1
        # print('--------')

    f_output = open('..//val_output_coco.json', 'w')
    json.dump(result_obj, f_output, indent=4)
    f_output.close()
    f_ann = open('..//val_annotation_coco.json', 'w')
    json.dump(annotation_obj, f_ann, indent=4)
    f_ann.close()


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'test':
        test()






