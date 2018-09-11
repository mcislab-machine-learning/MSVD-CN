import sqlite3
from nltk.tokenize.stanford_segmenter import StanfordSegmenter


db_file_name = '/home/mcislab/zwt/download_youtube/captions.db'


def read_all_captions():
    """
    read all captions from database
    :return: a list, each element is (video_id, caption, user_name, user_ip, time)
    user_name, user_ip, time can be ignored
    """
    conn = sqlite3.connect(db_file_name)
    cursor = conn.cursor()
    cursor.execute('select * from captions')
    result = cursor.fetchall()
    conn.close()
    return result


def segment():
    """
    split a Chinese sentence into words
    :return:
    """
    segmenter = StanfordSegmenter(
        java_class=r"edu.stanford.nlp.ie.crf.CRFClassifier",
        path_to_jar=r"D:\Desktop\stanford corenlp\stanford-segmenter-2018-02-27\stanford-segmenter-3.9.1.jar",
        path_to_slf4j=r"D:\Desktop\stanford corenlp\slf4j-api-1.7.25.jar",
        path_to_sihan_corpora_dict=r"D:\Desktop\stanford corenlp\stanford-segmenter-2018-02-27\data",
        path_to_model=r"D:\Desktop\stanford corenlp\stanford-segmenter-2018-02-27\data\pku.gz",
        path_to_dict=r"D:\Desktop\stanford corenlp\stanford-segmenter-2018-02-27\data\dict-chris6.ser.gz",
        sihan_post_processing='true'
    )   # path to jar files should be changed

    # result = segmenter.segment(s)
    result = segmenter.segment_sents(["一个人在切西红柿", "这个把手该换了", "别把手放在我的肩膀上", "他正在量和服尺寸"])

    print(result)


segment()