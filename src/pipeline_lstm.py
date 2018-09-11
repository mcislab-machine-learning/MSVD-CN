import os
import sys
import traceback
import math

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import IPython
import numpy as np

import data_video
from vocabulary import *
from models import model_lstm
from util.beam_search_util_lstm import *
from util.coco_result_generator import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# segment_method = 'char'
# vocab_path = data_video.msvd_chinese_vocab_char_path
segment_method = 'word'
vocab_path = data_video.msvd_chinese_vocab_word_path

feature_type = 'c3d_pool5'


def train():
    save_steps = 10000

    n_epoch = 1000
    learning_rate = 1e-4
    scheduler_step_size = 15
    batch_size = 32

    vocab = Vocabulary.load(vocab_path)
    vocab_size = (vocab.idx // 100 + 1) * 100

    dataset = data_video.MSVDDatasetCHN(vocab=vocab, segment_method=segment_method, split='train', feature=feature_type)
    collate_fn = data_video.collate_fn
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

    lm_config = model_lstm.LanguageModelConfig()
    lm_config.vocab_size = vocab_size
    lm_config.image_feature_dim = 512
    lm_lstm = model_lstm.LanguageModelLSTM(lm_config)
    lm_lstm.to(device)
    lm_lstm.train(True)

    epoch = 0
    global_step = 0

    optimizer = torch.optim.Adam(lm_lstm.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=.1)

    criterion = nn.CrossEntropyLoss()

    while epoch < n_epoch:
        scheduler.step()
        print('1 epoch = {} steps'.format(len(data_loader)))
        for _,  (image_filename_list, features, captions, lengths) in enumerate(data_loader):
            global_step += 1

            features = features.to(device)
            captions = captions.to(device)

            word_prob_output, last_hidden_state = lm_lstm.forward(features, captions, lengths, )
            # print(word_prob_output.shape)   # (batch, seq_len, vocab_size)
            target = torch.nn.utils.rnn.pack_padded_sequence(captions, lengths=lengths, batch_first=True)[0]

            loss = criterion(word_prob_output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('epoch {}, global step: {}, loss: {}'.format(epoch, global_step, loss))

            if global_step % 10 == 0:
                print(data_video.get_res5c_feature_of_video.cache_info())

            if global_step % save_steps == 0 and global_step > 0:
                test1(lm_lstm, global_step)
                # save_model('../models/model-lstm-{}'.format(global_step), (lm_lstm, optimizer, epoch, global_step))
        epoch += 1


def save_model(save_path, items):
    dirname = os.path.dirname(save_path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    lm, optimizer, epoch, global_step = items
    state_dict = {'lm_lstm': lm,
                  'optimizer': optimizer,
                  'epoch': epoch,
                  'global_step': global_step}
    torch.save(state_dict, save_path)
    print('model saved at {}'.format(save_path))


def load_model(save_path):
    if device.type == 'cuda':
        state_dict = torch.load(save_path)
    else:
        state_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
    lm_lstm, optimizer, epoch, global_step = state_dict['lm_lstm'], state_dict['optimizer'], \
                                             state_dict['epoch'], state_dict['global_step']
    print('loaded {}'.format(save_path))
    return lm_lstm, optimizer, epoch, global_step


def test1(lm_lstm, global_step):
    lm_lstm.train(False)

    annotation_file_name = '../all_results/results_lstm_chn_word_c3dpool5/msvd_annotation{}.json'.format(global_step)
    output_file_name = '../all_results/results_lstm_chn_word_c3dpool5/msvd_result{}.json'.format(global_step)

    if not os.path.exists(os.path.dirname(annotation_file_name)):
        os.makedirs(os.path.dirname(annotation_file_name))

    length_normalization_factor = 0.0
    beam_size = 3
    max_sentence_length = 15

    vocab = Vocabulary.load(vocab_path)

    dataset = data_video.MSVDDatasetCHN(vocab=vocab, segment_method=segment_method, caption_mode='text', split='test', feature=feature_type)
    collate_fn = data_video.collate_fn
    # data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=4)
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)

    dataset_size = len(dataset)

    start_word_id = vocab.get_index(start_word)
    end_word_id = vocab.get_index(end_word)

    result_generator = COCOResultGenerator()

    for index, (image_filename_list, features, captions) in enumerate(data_loader):
        # print(index)
        # continue
        # write annotation
        image_id = image_filename_list[0]
        caption = captions[0]

        result_generator.add_annotation(image_id, caption)
        if result_generator.has_output(image_id):
            continue

        if len(result_generator.test_image_set) >= 1970:
            break

        # extract image feature
        features = features.to(device)
        image_embedding = lm_lstm.get_image_embedding(features)
        image_embedding = image_embedding.repeat([beam_size, 1])
        inputs = image_embedding.unsqueeze_(0)
        states = None

        initial_beam = Caption(
            sentence=[start_word_id],
            state=states,
            logprob=0.0,
            score=0.0,
            metadata=[""]
        )
        partial_captions = TopN(beam_size)
        partial_captions.push(initial_beam)
        complete_captions = TopN(beam_size)

        output_softmax = nn.Softmax(dim=-1)

        for j in range(max_sentence_length):
            try:
                partial_captions_list = partial_captions.extract()
                partial_captions.reset()
                if j > 0:
                    ii = torch.tensor([c.sentence[-1] for c in partial_captions_list])
                    ii = ii.to(device)
                    inputs = lm_lstm.get_word_embedding(ii)
                    inputs = inputs.unsqueeze_(0)
                    states = [None, None]
                    states[0] = torch.cat([c.state[0] for c in partial_captions_list], dim=1)  # (1, 3, 512)
                    states[1] = torch.cat([c.state[1] for c in partial_captions_list], dim=1)  # (1, 3, 512)

                hiddens, states = lm_lstm.lstm(inputs, states)
                outputs = lm_lstm.output_word_layer(hiddens.squeeze(0))  # lstm outputs:
                softmax = output_softmax(outputs)

                for (i, partial_caption) in enumerate(partial_captions_list):
                    word_probabilities = softmax[i].detach().cpu().numpy()  # cuda tensors -> cpu for sorting
                    # state = (states[0][0][i].detach().cpu().numpy(), states[1][0][i].detach().cpu().numpy())
                    state = (states[0][:, i:i + 1], states[1][:, i:i + 1])
                    words_and_probs = list(enumerate(word_probabilities))
                    words_and_probs.sort(key=lambda x: -x[1])
                    words_and_probs = words_and_probs[0:beam_size]

                    # print([(self.vocab.get_word(w), p) for w, p in words_and_probs])

                    for w, p in words_and_probs:
                        if p < 1e-12:
                            continue  # Avoid log(0).
                        sentence = partial_caption.sentence + [w]
                        logprob = partial_caption.logprob + math.log(p)
                        score = logprob

                        metadata_list = None
                        if w == end_word_id:
                            if length_normalization_factor > 0:
                                score /= len(sentence) ** length_normalization_factor
                            beam = Caption(sentence, state, logprob, score, metadata_list)
                            complete_captions.push(beam)
                        else:
                            beam = Caption(sentence, state, logprob, score, metadata_list)
                            partial_captions.push(beam)
                    if partial_captions.size() == 0:
                        break
            except Exception as e:
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info)
                IPython.embed()

        if not complete_captions.size():
            complete_captions = partial_captions
        captions = complete_captions.extract(sort=True)

        print(len(result_generator.test_image_set))
        print('{}, {}/{} {}'.format(
            image_id, index, dataset_size, result_generator.has_output(image_id)))

        for i, caption in enumerate(captions):
            sentence = [vocab.get_word(w) for w in caption.sentence]
            # print(sentence)
            sentence = [w for w in sentence if (w != start_word and w != end_word)]     # ignore start and end tokens
            sentence = "".join(sentence)

            print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
            if i == 0:
                print(sentence)
                result_generator.add_output(image_id, sentence)

    annotation_obj, result_obj = result_generator.get_annotation_and_output()
    # print(annotation_obj)
    with open(annotation_file_name, 'w') as f:
        json.dump(annotation_obj, f)
    with open(output_file_name, 'w') as f:
        json.dump(result_obj, f)

    lm_lstm.train(True)


def test():
    saved_model_path = '../models_cnn/model-136427-ep600'

    beam_size = 3
    data_loader_batch_size = 4
    max_sentence_length = 15

    vocab = Vocabulary.load(vocab_path)
    dataset = data_video.MSVDDatasetCHN(vocab=vocab, segment_method=segment_method, caption_mode='text', split='test', feature=feature_type)
    # dataset = data.DemoDataset()
    data_loader = DataLoader(dataset=dataset, batch_size=data_loader_batch_size, shuffle=False, num_workers=0)

    lm_cnn, optimizer, epoch, global_step = load_model(saved_model_path)
    lm_cnn.to(device)
    lm_cnn.train(False)

    start_word_id = vocab.get_index(start_word)
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