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

import data_video
from vocabulary import *
from models import cnn, model_lstm
from util.beam_search_util_lstm import *
from util.coco_result_generator import *

# os.environ['CUDA_VISIBLE_DEVICES'] = ''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

segment_method = 'char'
vocab_path = data_video.msvd_bilingual_vocab_char_path
# segment_method = 'word'
# vocab_path = data_video.msvd_bilingual_vocab_word_path


def train():
    save_steps = 10000

    n_epoch = 1000
    learning_rate = 1e-4
    scheduler_step_size = 15
    batch_size = 32

    vocab = Vocabulary.load(vocab_path)
    vocab_size = (vocab.idx // 100 + 1) * 100

    dataset = data_video.MSVDDatasetBilingual(vocab=vocab, segment_method=segment_method, split='train')
    collate_fn = data_video.collate_fn
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

    lm_config = model_lstm.LanguageModelConfig()
    lm_config.vocab_size = vocab_size
    # lm_config.image_feature_dim += 1
    lm_lstm = model_lstm.LanguageModelLSTM(lm_config)
    lm_lstm.to(device)
    lm_lstm.train(True)

    epoch = 0
    global_step = 0

    optimizer = torch.optim.Adam(lm_lstm.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=.5)

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

            if global_step % 20 == 0:
                print(data_video.get_res5c_feature_of_video.cache_info())

            if global_step == 5000 or (global_step % save_steps == 0 and global_step > 0):
                test1(lm_lstm, global_step)
                # save_model('../models/model-lstm-bilingual-{}'.format(global_step), (lm_lstm, optimizer, epoch, global_step))

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


def test():
    saved_model_path = '../models/model-lstm-bilingual-10000'

    annotation_file_name = '../all_results/results_lstm_bilingual/msvd_annotation_chn.json'
    output_file_name = '../all_results/results_lstm_bilingual/msvd_result_chn.json'

    length_normalization_factor = 0.0
    beam_size = 3
    max_sentence_length = 15

    vocab = Vocabulary.load(vocab_path)

    # use chinese
    start_token_id = vocab.get_index(lang_chs)

    dataset = data_video.MSVDDatasetBilingual(vocab=vocab, segment_method=segment_method, caption_mode='text', split='test')
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)

    dataset_size = len(dataset)

    lm_lstm, optimizer, epoch, global_step = load_model(saved_model_path)
    lm_lstm.to(device)
    lm_lstm.train(False)

    start_word_id = vocab.get_index(start_word)
    end_word_id = vocab.get_index(end_word)

    result_generator = COCOResultGenerator()

    for index, (image_filename_list, features, captions, tokenized_captions) in enumerate(data_loader):
        # print(index)
        # continue
        # write annotation
        image_id = image_filename_list[0]
        caption = captions[0]

        annotation_start_token = tokenized_captions[0][0]
        if not annotation_start_token == start_token_id:
            continue

        result_generator.add_annotation(image_id, caption)
        if result_generator.has_output(image_id):
            continue

        if len(result_generator.test_image_set) >= 1970:
            break

        # extract image feature
        features = features.to(device)
        image_embedding = lm_lstm.get_image_embedding(features)
        image_embedding = image_embedding.repeat([beam_size, 1])
        # j = 0
        inputs = image_embedding.unsqueeze_(0)
        states = None

        initial_beam = Caption(
            sentence=[start_word_id, start_token_id],
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
                    ii.to(device)

                    # print(ii, type(ii), ii.shape)

                    inputs = lm_lstm.get_word_embedding(ii)
                    inputs = inputs.unsqueeze_(0)
                    states = [None, None]
                    states[0] = torch.cat([c.state[0] for c in partial_captions_list], dim=1)  # (1, 3, 512)
                    states[1] = torch.cat([c.state[1] for c in partial_captions_list], dim=1)  # (1, 3, 512)

                hiddens, states = lm_lstm.lstm(inputs, states)  # hiddens:     states: [1, 3, 512] [1, 3, 512] FIXME: here?
                outputs = lm_lstm.output_word_layer(hiddens.squeeze(0))  # lstm outputs:
                softmax = output_softmax(outputs)

                for (i, partial_caption) in enumerate(partial_captions_list):
                    word_probabilities = softmax[i].detach().cpu().numpy()  # cuda tensors -> cpu for sorting
                    # state = (states[0][0][i].detach().cpu().numpy(), states[1][0][i].detach().cpu().numpy())
                    state = (states[0][:, i:i + 1], states[1][:, i:i + 1])
                    words_and_probs = list(enumerate(word_probabilities))
                    words_and_probs.sort(key=lambda x: -x[1])
                    words_and_probs = words_and_probs[0:beam_size]

                    # print(j, [(vocab.get_word(w), p) for w, p in words_and_probs])

                    if j == 0:      # force generate chinese
                        words_and_probs_chs = []
                        for w, p in words_and_probs:
                            words_and_probs_chs.append((start_token_id, p))
                        words_and_probs = words_and_probs_chs

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

            sentence = [w for w in sentence if (w != start_word and w != end_word and w != lang_chs and w != lang_en)]     # ignore start and end tokens
            sentence = [w for w in sentence if (w not in [start_word, end_word, lang_chs, lang_en])]     # ignore start and end tokens
            sentence = " ".join(sentence)

            print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
            if i == 0:
                print(sentence)
                # chinese
                output_sentence = ''.join(sentence.split())
                if len(output_sentence) == 0:
                    output_sentence = '一'
                result_generator.add_output(image_id, output_sentence)

    annotation_obj, result_obj = result_generator.get_annotation_and_output()
    with open(annotation_file_name, 'w') as f:
        json.dump(annotation_obj, f, indent=4)
    with open(output_file_name, 'w') as f:
        json.dump(result_obj, f, indent=4)

    print('annotation file:', annotation_file_name)
    print('result file:', output_file_name)
    # os.system('{} {} {}'.format(r'/media/mcislab3d/Seagate Backup Plus Drive/zwt/coco-caption-master/eval.py', annotation_file_name, output_file_name))


def test1(lm_lstm, global_step):
    annotation_file_name = '../all_results/results_lstm_bilingual/msvd_annotation_bil_{}.json'.format(global_step)
    output_file_name = '../all_results/results_lstm_bilingual/msvd_result_bil_{}.json'.format(global_step)

    if not os.path.exists(os.path.dirname(annotation_file_name)):
        os.makedirs(os.path.dirname(annotation_file_name))

    length_normalization_factor = 0.0
    beam_size = 3
    max_sentence_length = 15

    vocab = Vocabulary.load(vocab_path)

    # use chinese
    start_token_id = vocab.get_index(lang_chs)

    dataset = data_video.MSVDDatasetBilingual(vocab=vocab, segment_method=segment_method, caption_mode='text', split='test')
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)

    dataset_size = len(dataset)

    # lm_lstm, optimizer, epoch, global_step = load_model(saved_model_path)
    # lm_lstm.to(device)
    # lm_lstm.train(False)
    lm_lstm.train(False)

    start_word_id = vocab.get_index(start_word)
    end_word_id = vocab.get_index(end_word)

    result_generator = COCOResultGenerator()

    for index, (image_filename_list, features, captions, tokenized_captions) in enumerate(data_loader):
        # print(index)
        # continue
        # write annotation
        image_id = image_filename_list[0]
        caption = captions[0]

        annotation_start_token = tokenized_captions[0][0]
        if not annotation_start_token == start_token_id:
            continue

        result_generator.add_annotation(image_id, caption)
        if result_generator.has_output(image_id):
            continue

        if len(result_generator.test_image_set) >= 1970:
            break

        # extract image feature
        features = features.to(device)
        image_embedding = lm_lstm.get_image_embedding(features)
        image_embedding = image_embedding.repeat([beam_size, 1])
        # j = 0
        inputs = image_embedding.unsqueeze_(0)
        states = None

        initial_beam = Caption(
            sentence=[start_word_id, start_token_id],
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

                if len(partial_captions_list) == 0:
                    break

                if j > 0:
                    ii = torch.tensor([c.sentence[-1] for c in partial_captions_list])
                    ii = ii.to(device)

                    # print(ii, type(ii), ii.shape)

                    inputs = lm_lstm.get_word_embedding(ii)
                    inputs = inputs.unsqueeze_(0)
                    states = [None, None]
                    states[0] = torch.cat([c.state[0] for c in partial_captions_list], dim=1)  # (1, 3, 512)
                    states[1] = torch.cat([c.state[1] for c in partial_captions_list], dim=1)  # (1, 3, 512)

                hiddens, states = lm_lstm.lstm(inputs, states)  # hiddens:     states: [1, 3, 512] [1, 3, 512] FIXME: here?
                outputs = lm_lstm.output_word_layer(hiddens.squeeze(0))  # lstm outputs:
                softmax = output_softmax(outputs)

                for (i, partial_caption) in enumerate(partial_captions_list):
                    word_probabilities = softmax[i].detach().cpu().numpy()  # cuda tensors -> cpu for sorting
                    # state = (states[0][0][i].detach().cpu().numpy(), states[1][0][i].detach().cpu().numpy())
                    state = (states[0][:, i:i + 1], states[1][:, i:i + 1])
                    words_and_probs = list(enumerate(word_probabilities))
                    words_and_probs.sort(key=lambda x: -x[1])
                    words_and_probs = words_and_probs[0:beam_size]

                    # print(j, [(vocab.get_word(w), p) for w, p in words_and_probs])

                    if j == 0:      # force generate chinese
                        words_and_probs_chs = []
                        for w, p in words_and_probs:
                            words_and_probs_chs.append((start_token_id, p))
                        words_and_probs = words_and_probs_chs

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

            sentence = [w for w in sentence if (w != start_word and w != end_word and w != lang_chs and w != lang_en)]     # ignore start and end tokens
            sentence = [w for w in sentence if (w not in [start_word, end_word, lang_chs, lang_en])]     # ignore start and end tokens
            sentence = " ".join(sentence)

            print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
            if i == 0:
                print(sentence)
                # chinese
                output_sentence = ''.join(sentence.split())
                if len(output_sentence) == 0:
                    output_sentence = '一'
                result_generator.add_output(image_id, output_sentence)

    annotation_obj, result_obj = result_generator.get_annotation_and_output()
    with open(annotation_file_name, 'w') as f:
        json.dump(annotation_obj, f, indent=4)
    with open(output_file_name, 'w') as f:
        json.dump(result_obj, f, indent=4)

    print('annotation file:', annotation_file_name)
    print('result file:', output_file_name)

    lm_lstm.train(True)


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'test':
        test()