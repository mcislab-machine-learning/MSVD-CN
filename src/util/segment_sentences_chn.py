from nltk.tokenize.stanford_segmenter import StanfordSegmenter
import re
import os

stanford_corenlp_path = r'/media/mcislab3d/Seagate Backup Plus Drive/zwt/stanford corenlp'


def segment_sentences_char(sentence_list):
    return [' '.join(i) for i in sentence_list]


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