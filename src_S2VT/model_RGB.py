"""
requires tensorflow 0.12.1
"""

#-*- coding: utf-8 -*-
import json
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
import pandas as pd
import numpy as np
import scipy.io
import sys
import time
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
import codecs

from lru import lru_cache_function

os.environ["JAVA_HOME"] = r"/usr/local/lib/jre1.8.0_161"

class Video_Caption_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, n_video_lstm_step, n_caption_lstm_step, bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.n_video_lstm_step=n_video_lstm_step
        self.n_caption_lstm_step=n_caption_lstm_step

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')
        #self.bemb = tf.Variable(tf.zeros([dim_hidden]), name='bemb')

        self.lstm1 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)
        self.lstm2 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)

        self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

    def build_model(self):
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step, self.dim_image])
        video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step])

        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_caption_lstm_step+1])
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_caption_lstm_step+1])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b ) # (batch_size*n_lstm_steps, dim_hidden)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_steps, self.dim_hidden])

        state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
        padding = tf.zeros([self.batch_size, self.dim_hidden])

        probs = []
        loss = 0.0

        ##############################  Encoding Stage ##################################
        for i in range(0, self.n_video_lstm_step):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(image_emb[:,i,:], state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat(1, [padding, output1]), state2)

        ############################# Decoding Stage ######################################
        for i in range(0, self.n_caption_lstm_step): ## Phase 2 => only generate captions
            #if i == 0:
            #    current_embed = tf.zeros([self.batch_size, self.dim_hidden])
            #else:
            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i])

            tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(padding, state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat(1, [current_embed, output1]), state2)

            labels = tf.expand_dims(caption[:, i+1], 1)
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
            concated = tf.concat(1, [indices, labels])
            onehot_labels = tf.sparse_to_dense(concated, tf.pack([self.batch_size, self.n_words]), 1.0, 0.0)

            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_words, onehot_labels)
            cross_entropy = cross_entropy * caption_mask[:,i]
            probs.append(logit_words)

            current_loss = tf.reduce_sum(cross_entropy)/self.batch_size
            loss = loss + current_loss

        return loss, video, video_mask, caption, caption_mask, probs


    def build_generator(self):
        video = tf.placeholder(tf.float32, [1, self.n_video_lstm_step, self.dim_image])
        video_mask = tf.placeholder(tf.float32, [1, self.n_video_lstm_step])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [1, self.n_video_lstm_step, self.dim_hidden])

        state1 = tf.zeros([1, self.lstm1.state_size])
        state2 = tf.zeros([1, self.lstm2.state_size])
        padding = tf.zeros([1, self.dim_hidden])

        generated_words = []

        probs = []
        embeds = []

        for i in range(0, self.n_video_lstm_step):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(image_emb[:, i, :], state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat(1, [padding, output1]), state2)

        for i in range(0, self.n_caption_lstm_step):
            tf.get_variable_scope().reuse_variables()

            if i == 0:
                with tf.device('/cpu:0'):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([1], dtype=tf.int64))     # 1 = <bos>

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(padding, state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat(1, [current_embed, output1]), state2)

            logit_words = tf.nn.xw_plus_b( output2, self.embed_word_W, self.embed_word_b)
            max_prob_index = tf.argmax(logit_words, 1)[0]
            generated_words.append(max_prob_index)
            probs.append(logit_words)

            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                current_embed = tf.expand_dims(current_embed, 0)

            embeds.append(current_embed)

        return video, video_mask, generated_words, probs, embeds


#=====================================================================================
# Global Parameters
#=====================================================================================
feature_type = 'c3d'
segment_method = 'char'

if feature_type == 'resnet':
    video_train_feat_path = '/media/mcislab/sdb1/home/mcislab/wuhuayi/res5c'
    video_test_feat_path = '/media/mcislab/sdb1/home/mcislab/wuhuayi/res5c'
elif feature_type == 'c3d':
    video_train_feat_path = '/media/mcislab/sdb1/home/mcislab/nieyuxiang/extract_C3Dv1.1_features/MSVD_prefix'
    video_test_feat_path = '/media/mcislab/sdb1/home/mcislab/nieyuxiang/extract_C3Dv1.1_features/MSVD_prefix'

if segment_method == 'char':
    video_train_data_path = './data/video_corpus_chn_segment_char.csv'
    video_test_data_path = './data/video_corpus_chn_segment_char.csv'
elif segment_method == 'word':
    video_train_data_path = './data/video_corpus_chn_segment_word.csv'
    video_test_data_path = './data/video_corpus_chn_segment_word.csv'

model_path = './models-{}-{}'.format(feature_type, segment_method)
data_path = './.data_{}_{}'.format(feature_type, segment_method)

video_mapping_file = 'data/youtube_mapping.txt'

#=======================================================================================
# Train Parameters
#=======================================================================================
# c3d
if feature_type == 'c3d':
    dim_image = 512
elif feature_type == 'resnet':
    dim_image = 2048

dim_hidden= 1000

if feature_type == 'c3d':
    n_video_lstm_step = 3
elif feature_type == 'resnet':
    n_video_lstm_step = 10

n_caption_lstm_step = 20

# TODO: ??
n_frame_step = 3
# n_frame_step = 10

n_epochs = 981
batch_size = 50
learning_rate = 0.0001


def get_video_train_data(video_data_path, video_feat_path):
    f_video_mapping = codecs.open(video_mapping_file, 'r', encoding='utf-8')
    video_name_map = dict()
    for line in f_video_mapping:
        strs = line.strip().split()
        video_name = strs[0]
        vid = strs[1]
        video_index = int(vid[3:])
        if 1 <= video_index <= 1200:
            video_name_map[video_name] = vid
    f_video_mapping.close()

    video_data = pd.read_csv(video_data_path, sep=',')
    video_data = video_data[video_data['Language'] == 'English']
    video_data['video_file_name'] = video_data.apply(lambda row: row['VideoID']+'_'+str(int(row['Start']))+'_'+str(int(row['End'])), axis=1)
    video_data = video_data[video_data['video_file_name'].isin(video_name_map.keys())]
    video_data['video_path'] = video_data.apply(lambda row: video_name_map[row['video_file_name']], axis=1)
    video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(video_feat_path, x))
    # video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists(x))]
    video_data = video_data[video_data['Description'].map(lambda x: isinstance(x, str))]
    
    unique_filenames = sorted(video_data['video_path'].unique())
    train_data = video_data[video_data['video_path'].map(lambda x: x in unique_filenames)]
    print 'loaded train data'
    return train_data


def get_video_test_data(video_data_path, video_feat_path):
    f_video_mapping = codecs.open(video_mapping_file, 'r', encoding='utf-8')
    video_name_map = dict()
    for line in f_video_mapping:
        strs = line.strip().split()
        video_name = strs[0]
        vid = strs[1]
        video_index = int(vid[3:])
        if 1301 <= video_index <= 1970:
            video_name_map[video_name] = vid
    f_video_mapping.close()

    video_data = pd.read_csv(video_data_path, sep=',')
    video_data = video_data[video_data['Language'] == 'English']
    video_data['video_file_name'] = video_data.apply(
        lambda row: row['VideoID'] + '_' + str(int(row['Start'])) + '_' + str(int(row['End'])), axis=1)
    video_data = video_data[video_data['video_file_name'].isin(video_name_map.keys())]
    video_data['video_path'] = video_data.apply(lambda row: video_name_map[row['video_file_name']], axis=1)
    video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(video_feat_path, x))
    # video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists( x ))]
    video_data = video_data[video_data['Description'].map(lambda x: isinstance(x, str))]

    unique_filenames = sorted(video_data['video_path'].unique())
    test_data = video_data[video_data['video_path'].map(lambda x: x in unique_filenames)]
    print 'loaded test data'
    return test_data


def preProBuildWordVocab(sentence_iterator, word_count_threshold=5):
    # borrowed this function from NeuralTalk
    print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold)
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
           word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print 'filtered words from %d to %d' % (len(word_counts), len(vocab))

    ixtoword = {}
    ixtoword[0] = '<pad>'
    ixtoword[1] = '<bos>'
    ixtoword[2] = '<eos>'
    ixtoword[3] = '<unk>'

    wordtoix = {}
    wordtoix['<pad>'] = 0
    wordtoix['<bos>'] = 1
    wordtoix['<eos>'] = 2
    wordtoix['<unk>'] = 3

    for idx, w in enumerate(vocab):
        wordtoix[w] = idx+4
        ixtoword[idx+4] = w

    word_counts['<pad>'] = nsents
    word_counts['<bos>'] = nsents
    word_counts['<eos>'] = nsents
    word_counts['<unk>'] = nsents

    bias_init_vector = np.array([1.0 * word_counts[ ixtoword[i] ] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range

    return wordtoix, ixtoword, bias_init_vector


@lru_cache_function(max_size=2000, expiration=120*60)   # expire in 2 hrs
def load_res5c_feature(vid):
    vid += '.mat'

    feature_file_name = vid

    feature = np.array(scipy.io.loadmat(feature_file_name)['feat_res5c'], dtype=np.float32)
    feature = np.reshape(feature, [feature.shape[0], feature.shape[1], feature.shape[2] * feature.shape[3]]).mean(
        axis=-1)
    return feature


@lru_cache_function(max_size=2000, expiration=120*60)
def get_c3d_feature_of_video(video_feature_path, layer_name='pool5'):

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

    def sample(l, k=3):
        n = len(l)
        seq = [int(float(n) / k * i) for i in range(k)]
        return [l[i] for i in seq]

    video_feature_path += '.avi'

    feature_folder = video_feature_path
    postfix = '.' + layer_name
    feature_files = filter(lambda x: x.endswith(postfix), os.listdir(feature_folder))
    feature_files = sample(feature_files)

    feature_list = [read_binary_blob(os.path.join(feature_folder, feature_file)) for feature_file in feature_files]

    feature = np.array(feature_list)
    # print('feature', feature.shape)
    feature = feature.squeeze(1).squeeze(4).squeeze(3).squeeze(2)
    return feature


if feature_type == 'c3d':
    load_video_feature = get_c3d_feature_of_video
elif feature_type == 'resnet':
    load_video_feature = load_res5c_feature


def train():
    train_data = get_video_train_data(video_train_data_path, video_train_feat_path)
    train_captions = train_data['Description'].values
    test_data = get_video_test_data(video_test_data_path, video_test_feat_path)
    test_captions = test_data['Description'].values

    captions_list = list(train_captions) + list(test_captions)
    captions = np.asarray(captions_list, dtype=np.object)

    captions = map(lambda x: x.replace('.', ''), captions)
    captions = map(lambda x: x.replace(',', ''), captions)
    captions = map(lambda x: x.replace('"', ''), captions)
    captions = map(lambda x: x.replace('\n', ''), captions)
    captions = map(lambda x: x.replace('?', ''), captions)
    captions = map(lambda x: x.replace('!', ''), captions)
    captions = map(lambda x: x.replace('\\', ''), captions)
    captions = map(lambda x: x.replace('/', ''), captions)

    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions, word_count_threshold=0)

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    np.save(os.path.join(data_path, "wordtoix"), wordtoix)
    np.save(os.path.join(data_path, 'ixtoword'), ixtoword)
    np.save(os.path.join(data_path, "bias_init_vector"), bias_init_vector)

    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(wordtoix),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            n_video_lstm_step=n_video_lstm_step,
            n_caption_lstm_step=n_caption_lstm_step,
            bias_init_vector=bias_init_vector)

    tf_loss, tf_video, tf_video_mask, tf_caption, tf_caption_mask, tf_probs = model.build_model()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    
    # my tensorflow version is 0.12.1, I write the saver with version 1.0
    saver = tf.train.Saver(max_to_keep=100, write_version=1)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
    tf.global_variables_initializer().run()

    #new_saver = tf.train.Saver()
    #new_saver = tf.train.import_meta_graph('./rgb_models/model-1000.meta')
    #new_saver.restore(sess, tf.train.latest_checkpoint('./models/'))

    loss_fd = open('loss.txt', 'w')
    loss_to_draw = []

    for epoch in range(0, n_epochs):
        loss_to_draw_epoch = []

        index = list(train_data.index)
        np.random.shuffle(index)
        train_data = train_data.ix[index]

        # current_train_data = train_data.groupby('video_path').apply(lambda x: x.irow(np.random.choice(len(x))))
        current_train_data = train_data.groupby('video_path').apply(lambda x: x.iloc[np.random.choice(len(x))])
        current_train_data = current_train_data.reset_index(drop=True)

        for start, end in zip(
                range(0, len(current_train_data), batch_size),
                range(batch_size, len(current_train_data), batch_size)):

            start_time = time.time()

            current_batch = current_train_data[start:end]
            current_videos = current_batch['video_path'].values

            current_feats = np.zeros((batch_size, n_video_lstm_step, dim_image))

            read_data_start = time.time()

            current_feats_vals = map(lambda vid: load_video_feature(vid), current_videos)

            read_data_time = time.time()

            current_video_masks = np.zeros((batch_size, n_video_lstm_step))

            for ind,feat in enumerate(current_feats_vals):
                current_feats[ind][:len(current_feats_vals[ind])] = feat
                current_video_masks[ind][:len(current_feats_vals[ind])] = 1

            current_captions = current_batch['Description'].values
            current_captions = map(lambda x: '<bos> ' + x, current_captions)
            current_captions = map(lambda x: x.replace('.', ''), current_captions)
            current_captions = map(lambda x: x.replace(',', ''), current_captions)
            current_captions = map(lambda x: x.replace('"', ''), current_captions)
            current_captions = map(lambda x: x.replace('\n', ''), current_captions)
            current_captions = map(lambda x: x.replace('?', ''), current_captions)
            current_captions = map(lambda x: x.replace('!', ''), current_captions)
            current_captions = map(lambda x: x.replace('\\', ''), current_captions)
            current_captions = map(lambda x: x.replace('/', ''), current_captions)

            for idx, each_cap in enumerate(current_captions):
                word = each_cap.lower().split(' ')
                if len(word) < n_caption_lstm_step:
                    current_captions[idx] = current_captions[idx] + ' <eos>'
                else:
                    new_word = ''
                    for i in range(n_caption_lstm_step-1):
                        new_word = new_word + word[i] + ' '
                    current_captions[idx] = new_word + '<eos>'

            current_caption_ind = []
            for cap in current_captions:
                current_word_ind = []
                for word in cap.lower().split(' '):     # FIXME: split here
                    if word in wordtoix:
                        current_word_ind.append(wordtoix[word])
                    else:
                        current_word_ind.append(wordtoix['<unk>'])
                current_caption_ind.append(current_word_ind)

            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=n_caption_lstm_step)
            current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix), 1] ) ] ).astype(int)
            current_caption_masks = np.zeros( (current_caption_matrix.shape[0], current_caption_matrix.shape[1]) )
            nonzeros = np.array( map(lambda x: (x != 0).sum() + 1, current_caption_matrix ) )

            for ind, row in enumerate(current_caption_masks):
                row[:nonzeros[ind]] = 1

            probs_val = sess.run(tf_probs, feed_dict={
                tf_video:current_feats,
                tf_caption: current_caption_matrix
                })

            _, loss_val = sess.run(
                    [train_op, tf_loss],
                    feed_dict={
                        tf_video: current_feats,
                        tf_video_mask : current_video_masks,
                        tf_caption: current_caption_matrix,
                        tf_caption_mask: current_caption_masks
                        })
            loss_to_draw_epoch.append(loss_val)

            print 'idx: ', start, " Epoch: ", epoch, " loss: ", loss_val, ' Elapsed time: ', str((time.time() - start_time))
            print 'read data used', read_data_time - read_data_start
            loss_fd.write('epoch ' + str(epoch) + ' loss ' + str(loss_val) + '\n')

        # draw loss curve every epoch
        # loss_to_draw.append(np.mean(loss_to_draw_epoch))
        # plt_save_dir = "./loss_imgs"
        # plt_save_img_name = str(epoch) + '.png'
        # plt.plot(range(len(loss_to_draw)), loss_to_draw, color='g')
        # plt.grid(True)
        # plt.savefig(os.path.join(plt_save_dir, plt_save_img_name))

        if np.mod(epoch, 20) == 0:
            print "Epoch ", epoch, " is done. Saving the model ..."
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)

    loss_fd.close()


def test(model_path, result_file_path):
    test_data = get_video_test_data(video_test_data_path, video_test_feat_path)
    test_videos = test_data['video_path'].unique()

    ixtoword = pd.Series(np.load(os.path.join(data_path, 'ixtoword.npy')).tolist())

    bias_init_vector = np.load(os.path.join(data_path, 'bias_init_vector.npy'))

    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            n_video_lstm_step=n_video_lstm_step,
            n_caption_lstm_step=n_caption_lstm_step,
            bias_init_vector=bias_init_vector)

    video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    test_output_txt_fd = open(result_file_path, 'w')

    result_obj = []

    for idx, video_feat_path in enumerate(test_videos):
        print(idx, video_feat_path)
        feature_file_name = os.path.basename(video_feat_path)
        vid = feature_file_name
        # vid = feature_file_name.replace('.mat', '.mp4')

        video_feat = load_video_feature(video_feat_path)[None,...]

        if video_feat.shape[1] == n_frame_step:
            video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
        else:
            continue

        generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
        generated_words = ixtoword[generated_word_index]

        punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
        generated_words = generated_words[:punctuation]

        generated_sentence = ' '.join(generated_words)
        generated_sentence = generated_sentence.replace('<bos> ', '')
        generated_sentence = generated_sentence.replace(' <eos>', '')
        # print generated_sentence,'\n'
        # test_output_txt_fd.write(video_feat_path + '\n')
        # test_output_txt_fd.write(generated_sentence + '\n\n')

        # vid = vid.replace('.mp4', '')
        result = {'image_id': vid, 'caption': ''.join(generated_sentence.split()),
                  'image_filename': vid}
        result_obj.append(result)
        print result

    json.dump(result_obj, test_output_txt_fd, indent=4)
    test_output_txt_fd.close()

    sess.close()


def test_all(all_model_path):
    for epoch in range(20, n_epochs, 20):
        cmd = ''
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            cmd += 'CUDA_VISIBLE_DEVICES={} '.format(os.environ['CUDA_VISIBLE_DEVICES'])
        cmd += '{} {} test {}'.format(sys.executable, os.path.realpath(__file__), epoch)
        os.system(cmd)


def test_one(all_model_path, epoch):
    ep_model_path = os.path.join(all_model_path, 'model-{}'.format(epoch))
    result_folder = os.path.join(all_model_path, 'results')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    result_file_path = os.path.join(result_folder, 'S2VT_results_{}.json'.format(epoch))
    result_txt_path = os.path.join(result_folder, 'results_{}.txt'.format(epoch))
    test(ep_model_path, result_file_path)

    eval_cmd = '{} {} {} {} {}'.format(r"/home/mcislab/library/miniconda3/bin/python3",
                                       r"/media/mcislab/sdb1/home/mcislab/zwt/coco-caption-master/eval.py",
                                       r"/media/mcislab/sdb1/home/mcislab/zwt/S2VT-master/annotation.json",
                                       result_file_path,
                                       result_txt_path)
    os.system(eval_cmd)


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'test':
        epoch = sys.argv[2]
        test_one(model_path, epoch)
    elif sys.argv[1] == 'testall':
        test_all(model_path)
    # test(os.path.join(model_path, 'model-500'), 'S2VT_results_c3d_char_500.json')

