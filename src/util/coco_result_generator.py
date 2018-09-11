import os
import sys
import json


class COCOResultGenerator:
    def __init__(self):
        self.result_obj = []
        self.annotation_obj = {'info': 'N/A', 'licenses': 'N/A', 'type': 'captions', 'images': [], 'annotations': []}
        self.caption_id = 0
        self.annotation_image_set = set()
        self.test_image_set = set()

    def add_annotation(self, image_id, caption_raw):
        if image_id not in self.annotation_image_set:
            self.annotation_obj['images'].append({'id': image_id})
            self.annotation_image_set.add(image_id)
        self.annotation_obj['annotations'].append({'image_id': image_id, 'caption': caption_raw, 'id': self.caption_id})
        self.caption_id += 1

    def add_output(self, image_id, caption_output):
        assert(image_id in self.annotation_image_set and image_id not in self.test_image_set)
        self.result_obj.append({"image_id": image_id, "caption": caption_output, "image_filename": image_id})
        self.test_image_set.add(image_id)

    def has_output(self, image_id):
        return image_id in self.test_image_set

    def get_annotation_and_output(self):
        return self.annotation_obj, self.result_obj

