"""
generate data for video_corpus_chn_segment (S2VT)
"""

import os
import sys
import csv
import sqlite3
import re
from nltk.tokenize.stanford_segmenter import StanfordSegmenter

stanford_corenlp_path = r'D:\Desktop\stanford corenlp'


def segment_sentences(sentence_list):
    segmenter = StanfordSegmenter(
        java_class=r"edu.stanford.nlp.ie.crf.CRFClassifier",
        path_to_jar=os.path.join(stanford_corenlp_path, 'stanford-segmenter-2018-02-27', 'stanford-segmenter-3.9.1.jar'),
        path_to_slf4j=os.path.join(stanford_corenlp_path, 'slf4j-api-1.7.25.jar'),
        path_to_sihan_corpora_dict=os.path.join(stanford_corenlp_path, 'stanford-segmenter-2018-02-27', 'data'),
        path_to_model=os.path.join(stanford_corenlp_path, 'stanford-segmenter-2018-02-27', 'data', 'pku.gz'),
        path_to_dict=os.path.join(stanford_corenlp_path, 'stanford-segmenter-2018-02-27', 'data', 'dict-chris6.ser.gz'),
        sihan_post_processing='true'
    )
    result = segmenter.segment_sents(sentence_list)
    result = result.strip()
    segmented_list = re.split(os.linesep, result)
    if len(segmented_list[-1]) == 0:
        segmented_list = segmented_list[:-1]
    if len(segmented_list) != len(sentence_list):
        for i in range(len(segmented_list)):
            ss = ''.join(segmented_list[i].split())
            if ss != sentence_list[i]:
                print(i, '|', segmented_list[i], '|', sentence_list[i])
                # break
        print(len(segmented_list), len(sentence_list))
    assert len(segmented_list) == len(sentence_list)
    return segmented_list

def segment_sentences_char(sentence_list):
    return [' '.join(i) for i in sentence_list]

# f_csv = open('../data/video-descriptions.csv', 'r')
# csv_reader = csv.reader(f_csv)

map_dict = {}
f_mapping = open('../data/youtube_mapping.txt', 'r')
for line in f_mapping:
    strs = line.strip().split()
    map_dict[strs[1] + '.mp4'] = strs[0]

all_chn_captions = []
conn = sqlite3.connect('../data/captions.db')
cursor = conn.cursor()
cursor.execute('select id, video_id, sentence from captions_filter where deleted=0')
result = cursor.fetchall()
for item in result:
    id, vid, caption = item
    all_chn_captions.append([vid, caption])
cursor.close()
conn.close()

sentences = [i[1] for i in all_chn_captions]
# sentences = segment_sentences(sentences)
sentences = segment_sentences_char(sentences)
assert(len(sentences) == len(all_chn_captions))

for i in range(len(all_chn_captions)):
    all_chn_captions[i][1] = sentences[i]

f_csv = open('../data/video_corpus_chn_segment_char.csv', 'w', encoding='utf-8')
f_csv.write('VideoID,Start,End,WorkerID,Source,AnnotationTime,Language,Description,id' + '\n')
for vid, caption in all_chn_captions:
    ytb_id = map_dict[vid]
    split_index = [i for i, c in enumerate(ytb_id) if c == '_'][-2:]
    name, s, e = ytb_id[:split_index[0]], ytb_id[split_index[0] + 1 : split_index[1]], ytb_id[split_index[1] + 1 :]
    line = [name, s, e, '1', 'clean', '16', 'English', caption, vid]
    f_csv.write(','.join(line) + '\n')
f_csv.close()