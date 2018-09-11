

from vocabulary import *
import pipeline_lstm
import pipeline_cnn

# pipeline_lstm.train()
# pipeline_lstm.test()
# pipeline_cnn.test()

import data_video
from vocabulary import Vocabulary

vocab_path = data_video.msvd_bilingual_vocab_char_path
vocab = Vocabulary.load(vocab_path)
dataset = data_video.MSVDDatasetBilingual(vocab=vocab, segment_method='char', caption_mode='text', split='train')

dataset.data.sort(key=lambda x: x.video_id)

with open('all_captions.txt', 'w') as f:
    for d in dataset.data:
        f.write('{:>12} {}\n'.format(d.video_id, d.caption))