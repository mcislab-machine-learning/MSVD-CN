import os
import sys
import traceback
import math
import cProfile

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import IPython
import numpy as np
from tensorboardX import SummaryWriter

import data_video
from vocabulary import *
from models import model_lstm_multimodal_1 as model_lstm_multimodal
from util.beam_search_util_lstm import *
from util.coco_result_generator import *
from util.timer import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

segment_method = 'char'
vocab_path = data_video.msvd_bilingual_vocab_char_path

feature_type = 'resnet'

os.environ["JAVA_HOME"] = r"/usr/local/lib/jre1.8.0_161"

run_name = sys.argv[1]
assert(len(run_name) > 0)
result_folder = '../all_results/{}'.format(run_name)
assert (not os.path.exists(result_folder)), 'result folder {} exists!'.format(result_folder)
print('run:', run_name)


def train():
    save_steps = 10000

    n_epoch = 500
    learning_rate = 1e-4
    scheduler_step_size = 15
    batch_size = 32

    writer = SummaryWriter(log_dir=os.path.join('runs', run_name))

    vocab = Vocabulary.load(vocab_path)
    vocab_size = (vocab.idx // 100 + 1) * 100

    dataset = data_video.MSVDDatasetMultiModal(vocab=vocab, segment_method=segment_method, split='train', feature=feature_type)
    collate_fn = data_video.collate_fn_1
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

    lm_config = model_lstm_multimodal.Config()
    lm_config.vocab_size = vocab_size
    lm_lstm = model_lstm_multimodal.LanguageModelLSTM1(lm_config, device)
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
        for _,  (image_filename_list, features, captions, res5b_features, caption_eng, lengths) in enumerate(data_loader):
            timer = Timer()
            timer.start()

            global_step += 1

            features_resnet = features.to(device)  # batch_size * 2048
            captions = captions.to(device)
            res5b_features = res5b_features.to(device)

            eng_embedding = torch.zeros(len(caption_eng), lm_config.eng_caption_max_len, lm_config.vis_dim)
            for _i in range(len(caption_eng)):
                l = len(caption_eng[_i])
                tokens = torch.tensor(caption_eng[_i]).to(device)
                eng_embedding[_i][:l] = lm_lstm.embed(tokens)
            eng_embedding = eng_embedding.to(device)    # batch_size * 15 * 512

            timer.step('to gpu')

            # word_prob_output, last_hidden_state = lm_lstm.forward(features, res5b_feature=res5b_features,
            #                                                       eng_embedding=eng_embedding,
            #                                                       input_words=captions, lengths=lengths)
            # # print(word_prob_output.shape)   # (batch, seq_len, vocab_size)
            # target = torch.nn.utils.rnn.pack_padded_sequence(captions, lengths=lengths, batch_first=True)[0]
            #
            # loss = criterion(word_prob_output, target)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            optimizer.zero_grad()

            predicts = lm_lstm.forward(features_resnet, res5b_features, eng_embedding, captions, lengths)
            predicts = torch.nn.utils.rnn.pack_padded_sequence(predicts, [l - 1 for l in lengths], batch_first=True)[0]
            targets = torch.nn.utils.rnn.pack_padded_sequence(captions[:, 1:], [l - 1 for l in lengths], batch_first=True)[0]

            timer.step('forward')

            loss = criterion(predicts, targets)     # loss.device is 'cuda'
            loss.backward()
            optimizer.step()

            timer.step('optimize')

            print('epoch {}, global step: {}, loss: {:.8f}, lr: [{}]'.format(epoch, global_step, loss, ' '.join(
                '{}'.format(param_group['lr']) for param_group in optimizer.param_groups)))

            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("loss", loss, global_step=global_step)
            writer.add_scalar("lr", lr, global_step=global_step)

            if global_step % 10 == 0:
                print(data_video.get_c3d_feature_of_video.cache_info())
                timer.print()

            if (global_step % save_steps == 0 and global_step > 0):
                test1(lm_lstm, global_step, test_index=0, split='val')
                test1(lm_lstm, global_step, test_index=1, split='val')
                test1(lm_lstm, global_step, test_index=0, split='test')
                save_model(os.path.join(result_folder, 'models', 'model-{}'.format(global_step)), (lm_lstm, optimizer, epoch, global_step))
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


def test1(lm_lstm, global_step, test_index, split='test'):
    lm_lstm.train(False)

    assert split in ['test', 'val']

    annotation_file_name = os.path.join(result_folder, 'msvd_annotation_{}_{}_{}.json'.format(split, global_step, test_index))
    output_file_name = os.path.join(result_folder, 'msvd_result_{}_{}_{}.json'.format(split, global_step, test_index))
    eval_file_name = os.path.join(result_folder, 'eval_{}_{}_{}.txt'.format(split, global_step, test_index))

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    length_normalization_factor = 0.0
    beam_size = 3
    max_sentence_length = 15

    vocab = Vocabulary.load(vocab_path)

    dataset = data_video.MSVDDatasetMultiModal(vocab=vocab, segment_method=segment_method, caption_mode='text', split=split, feature=feature_type)
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)

    dataset_size = len(dataset)

    start_word_id = vocab.get_index(start_word)
    end_word_id = vocab.get_index(end_word)

    en_token_id = vocab.get_index(lang_en)
    chs_token_id = vocab.get_index(lang_chs)
    unk_token_id = vocab.get_index(unknown_word)

    result_generator = COCOResultGenerator()

    for index, (image_filename_list, features, captions, res5b_feature, caption_eng) in enumerate(data_loader):
        try:
            image_id = image_filename_list[0]
            caption = captions[0]

            result_generator.add_annotation(image_id, caption)
            if result_generator.has_output(image_id):
                continue

            initial_beam = Caption(
                sentence=[start_word_id],
                state=None,
                logprob=0.0,
                score=0.0,
                metadata=[""]
            )
            partial_captions = TopN(beam_size)
            partial_captions.push(initial_beam)
            complete_captions = TopN(beam_size)

            output_softmax = nn.Softmax(dim=-1)

            h0, c0 = torch.zeros(beam_size, lm_lstm.config.hidden_dim), torch.zeros(beam_size, lm_lstm.config.hidden_dim)
            h0 = h0.to(device)
            c0 = c0.to(device)

            caption_eng = [i for i in caption_eng if i not in [start_word_id, end_word_id, en_token_id, chs_token_id, unk_token_id]]
            caption_eng = caption_eng[:min(len(caption_eng), lm_lstm.config.eng_caption_max_len)]
            eng_embedding = torch.zeros(lm_lstm.config.eng_caption_max_len, lm_lstm.config.vis_dim)
            eng_embedding[:len(caption_eng), :] = lm_lstm.embed(torch.tensor(caption_eng).to(device))
            eng_embedding = eng_embedding.unsqueeze(0).to(device)   # 512 * 15

            # # res5b_feature: 1 * 512 * 1 * 7 * 7
            # # res5b_feature = res5b_feature.reshape(512, 49).to(device)
            # res5b_feature = res5b_feature.squeeze(2)    # 1 * 512 * 7 * 7
            # feas = res5b_feature.mean(3).mean(2).squeeze(0)     # 512
            # feas = feas.expand(beam_size, feas.shape[0])    # beam_size * 512
            #
            # features_v = lm_lstm.att_conv2d_v_1(res5b_feature)  # 1 * 512 * 5 * 5
            # features_all = torch.cat([features_v, eng_embedding], dim=1)  # used for attention
            res5b_feature = res5b_feature.to(device)

            # feas, features_all = lm_lstm.get_attn_input(res5b_feature, eng_embedding)
            _, features_all = lm_lstm.get_attn_input(res5b_feature, eng_embedding)

            features = features.to(device)
            feas = lm_lstm.img_embed.forward(features)

            # feas = feas.expand([beam_size, feas.shape[1]])
            feas = feas.repeat([beam_size, 1])
            words = lm_lstm.embed(torch.tensor([start_word_id] * beam_size).to(device))

            for j in range(max_sentence_length):
                partial_captions_list = partial_captions.extract()
                partial_captions.reset()

                if len(partial_captions_list) == 0:
                    break

                if j > 0:
                    ii = torch.tensor([c.sentence[-1] for c in partial_captions_list]).to(device)
                    words = lm_lstm.embed(ii)

                    beam_size = len(ii)
                    res5b_feature_expand = features_all.expand(beam_size, features_all.shape[1], features_all.shape[2])

                    h0 = torch.cat([c.state[0].unsqueeze(0) for c in partial_captions_list], dim=0)
                    c0 = torch.cat([c.state[1].unsqueeze(0) for c in partial_captions_list], dim=0)

                    feas, alpha = lm_lstm._attention_layer(res5b_feature_expand, h0)

                inputs = torch.cat([feas, words], 1)
                h0, c0 = lm_lstm.lstm_cell(inputs, (h0, c0))
                outputs = lm_lstm.fc_out(h0)

                # hiddens, states = lm_lstm.lstm(inputs, states)  # hiddens:     states: [1, 3, 512] [1, 3, 512] FIXME: here?
                # outputs = lm_lstm.output_word_layer(hiddens.squeeze(0))  # lstm outputs:
                softmax = output_softmax(outputs)

                for (i, partial_caption) in enumerate(partial_captions_list):
                    word_probabilities = softmax[i].detach().cpu().numpy()  # cuda tensors -> cpu for sorting
                    # state = (states[0][0][i].detach().cpu().numpy(), states[1][0][i].detach().cpu().numpy())
                    # state = (states[0][:, i:i + 1], states[1][:, i:i + 1])
                    state = (h0[i, :], c0[i, :])

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
        except Exception:
            IPython.embed()

    annotation_obj, result_obj = result_generator.get_annotation_and_output()
    # print(annotation_obj)
    with open(annotation_file_name, 'w') as f:
        json.dump(annotation_obj, f)
    with open(output_file_name, 'w') as f:
        json.dump(result_obj, f)

    print('annotation images:', len(annotation_obj['images']))
    print('output images:', len(result_obj))
    print('saved to {}'.format(output_file_name))

    eval_cmd = '{} {} {} {} {}'.format(sys.executable,
                                       r"/media/mcislab/sdb1/home/mcislab/zwt/coco-caption-master/eval.py",
                                       annotation_file_name,
                                       output_file_name,
                                       eval_file_name)

    os.system(eval_cmd)

    lm_lstm.train(True)


def test(model_path, result_folder, split='test'):
    lm_lstm, optimizer, epoch, global_step = load_model(model_path)

    lm_lstm.train(False)

    assert split in ['test', 'val']

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    annotation_file_name = os.path.join(result_folder, 'msvd_annotation_{}_{}.json'.format(split, global_step))
    output_file_name = os.path.join(result_folder, 'msvd_result_{}_{}.json'.format(split, global_step))
    eval_file_name = os.path.join(result_folder, 'eval_{}_{}.txt'.format(split, global_step))

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    length_normalization_factor = 0.0
    beam_size = 3
    max_sentence_length = 15

    vocab = Vocabulary.load(vocab_path)

    dataset = data_video.MSVDDatasetMultiModal(vocab=vocab, segment_method=segment_method, caption_mode='text',
                                               split=split, feature=feature_type, english_src='groundtruth')
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)

    dataset_size = len(dataset)

    start_word_id = vocab.get_index(start_word)
    end_word_id = vocab.get_index(end_word)

    en_token_id = vocab.get_index(lang_en)
    chs_token_id = vocab.get_index(lang_chs)
    unk_token_id = vocab.get_index(unknown_word)

    result_generator = COCOResultGenerator()

    for index, (image_filename_list, features, captions, res5b_feature, caption_eng) in enumerate(data_loader):
        image_id = image_filename_list[0]
        caption = captions[0]

        result_generator.add_annotation(image_id, caption)
        if result_generator.has_output(image_id):
            continue

        initial_beam = Caption(
            sentence=[start_word_id],
            state=None,
            logprob=0.0,
            score=0.0,
            metadata=[""]
        )
        partial_captions = TopN(beam_size)
        partial_captions.push(initial_beam)
        complete_captions = TopN(beam_size)

        output_softmax = nn.Softmax(dim=-1)

        h0, c0 = torch.zeros(beam_size, lm_lstm.config.hidden_dim), torch.zeros(beam_size, lm_lstm.config.hidden_dim)
        h0 = h0.to(device)
        c0 = c0.to(device)

        ee = caption_eng

        caption_eng = [i for i in caption_eng if
                       i not in [start_word_id, end_word_id, en_token_id, chs_token_id, unk_token_id]]
        caption_eng = caption_eng[:min(len(caption_eng), lm_lstm.config.eng_caption_max_len)]
        eng_embedding = torch.zeros(lm_lstm.config.eng_caption_max_len, lm_lstm.config.vis_dim)
        eng_embedding[:len(caption_eng), :] = lm_lstm.embed(torch.tensor(caption_eng).to(device))
        eng_embedding = eng_embedding.unsqueeze(0).to(device)  # 512 * 15

        # # res5b_feature: 1 * 512 * 1 * 7 * 7
        # # res5b_feature = res5b_feature.reshape(512, 49).to(device)
        # res5b_feature = res5b_feature.squeeze(2)    # 1 * 512 * 7 * 7
        # feas = res5b_feature.mean(3).mean(2).squeeze(0)     # 512
        # feas = feas.expand(beam_size, feas.shape[0])    # beam_size * 512
        #
        # features_v = lm_lstm.att_conv2d_v_1(res5b_feature)  # 1 * 512 * 5 * 5
        # features_all = torch.cat([features_v, eng_embedding], dim=1)  # used for attention
        res5b_feature = res5b_feature.to(device)

        # feas, features_all = lm_lstm.get_attn_input(res5b_feature, eng_embedding)
        _, features_all = lm_lstm.get_attn_input(res5b_feature, eng_embedding)

        features = features.to(device)
        feas = lm_lstm.img_embed.forward(features)

        # feas = feas.expand([beam_size, feas.shape[1]])
        feas = feas.repeat([beam_size, 1])
        words = lm_lstm.embed(torch.tensor([start_word_id] * beam_size).to(device))

        for j in range(max_sentence_length):
            partial_captions_list = partial_captions.extract()
            partial_captions.reset()

            if len(partial_captions_list) == 0:
                break

            if j > 0:
                ii = torch.tensor([c.sentence[-1] for c in partial_captions_list]).to(device)
                words = lm_lstm.embed(ii)

                beam_size = len(ii)
                res5b_feature_expand = features_all.expand(beam_size, features_all.shape[1], features_all.shape[2])

                h0 = torch.cat([c.state[0].unsqueeze(0) for c in partial_captions_list], dim=0)
                c0 = torch.cat([c.state[1].unsqueeze(0) for c in partial_captions_list], dim=0)

                feas, alpha = lm_lstm._attention_layer(res5b_feature_expand, h0)

            inputs = torch.cat([feas, words], 1)
            h0, c0 = lm_lstm.lstm_cell(inputs, (h0, c0))
            outputs = lm_lstm.fc_out(h0)

            # hiddens, states = lm_lstm.lstm(inputs, states)  # hiddens:     states: [1, 3, 512] [1, 3, 512] FIXME: here?
            # outputs = lm_lstm.output_word_layer(hiddens.squeeze(0))  # lstm outputs:
            softmax = output_softmax(outputs)

            for (i, partial_caption) in enumerate(partial_captions_list):
                word_probabilities = softmax[i].detach().cpu().numpy()  # cuda tensors -> cpu for sorting
                # state = (states[0][0][i].detach().cpu().numpy(), states[1][0][i].detach().cpu().numpy())
                # state = (states[0][:, i:i + 1], states[1][:, i:i + 1])
                state = (h0[i, :], c0[i, :])

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

    print('annotation images:', len(annotation_obj['images']))
    print('output images:', len(result_obj))
    print('saved to {}'.format(output_file_name))

    eval_cmd = '{} {} {} {} {}'.format(sys.executable,
                                       r"/media/mcislab/sdb1/home/mcislab/zwt/coco-caption-master/eval.py",
                                       annotation_file_name,
                                       output_file_name,
                                       eval_file_name)

    os.system(eval_cmd)


if __name__ == '__main__':
    # train()
    # cProfile.run('train()', filename='run_profile')
    test(model_path=r'/media/mcislab/sdb1/home/mcislab/zwt/caption_models_Chinese/all_results/results_lstm_multi_modal_both_e_conv_2/models/model-50000',
         result_folder=os.path.join('../all_results/', run_name))