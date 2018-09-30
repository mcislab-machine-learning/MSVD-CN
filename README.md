# MSVD-CN
Chinese corpus for MSVD-CN dataset and code for baseline methods

- Video of MSVD dataset can be downloaded from [youtubeclips.zip](http://upplysingaoflun.ecn.purdue.edu/~yu239/datasets/youtubeclips.zip), with video name mappings
- Orginial MSVD corpus: (https://www.microsoft.com/en-us/download/details.aspx?id=52422)
  - The Chinese captions in orginial MSVD corpus is not used
- Chinese corpus: (msvd_cn_captions_release_utf8.csv)
  - The file is encoded in UTF-8, and each row is seperated with a comma. 
  - Each row contains: sentence id, video id, sentence
  - The video id in each row corresponds to the video name in youtube_mapping.txt
  - If using Windows with Simplified Chinese language, Don't open this file directly with Microsoft Excel as Excel opens csv as GBK encoding without checking the real encoding.
