# coding=utf-8
# Copyleft 2019 project LXRT.

import json

import numpy as np
from torch.utils.data import Dataset

from param import args
from utils import load_obj_tsv

import sys
import csv

TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

csv.field_size_limit(10000000)

class NLVR2Dataset:

    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets to data
        self.data = []
        captions_path = r"C:\Users\asmit\Desktop\Guided Research\lxmert\data\mscoco_imgfeat\captions_val2014.json"

        data_annotations = {}
        data_keys = []
        with open(captions_path) as json_file:
            data = json.load(json_file)
            for p in data['annotations']:
                temp_id = p['image_id']
                if temp_id in data_keys:
                    temp_caption = data_annotations[temp_id]
                    temp_caption.append(p['caption'])
                    data_annotations[temp_id] = temp_caption
                else:
                    data_keys.append(temp_id)
                    data_annotations[temp_id] = [p['caption']]

        self.data = data_annotations

        print("Length of data annotations : ",len(data_annotations))
        print("Data annotations : ", data_annotations[203564])

    def __len__(self):
        return len(self.data)


"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""


class NLVR2TorchDataset(Dataset):
    def __init__(self, dataset: NLVR2Dataset):
        super().__init__()
        self.raw_dataset = dataset
        topk = 10

        # Loading detection features to img_data
        test_image_features_path = r"C:\Users\asmit\Desktop\Guided Research\lxmert\data\mscoco_imgfeat\val2014_obj36.tsv"
        img_data = []
        img_data.extend(load_obj_tsv(test_image_features_path, topk=topk))

        # Creating a dictionary with img_id as the key
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        count = len(self.imgid2img)
        print("Before precossing keys : ", count)
        # Change the image_ids so that they match with the ids of annotations
        for key in list(self.imgid2img.keys()):
            if count > 0:
                temp_feat = self.imgid2img[key]
                temp_id = key[13:]
                temp_id = str(int(temp_id))
                self.imgid2img[temp_id] = temp_feat
                del self.imgid2img[key]
                count -= 1
            else:
                break
        print("After precossing keys : ", len(self.imgid2img))

        # Filter out the dataset
        # data is a list with annotations(only list of sentences) and image_features in two separate lists
        self.data = {}
        count = 0
        for each_datum in self.raw_dataset.data:
            str_key = str(each_datum)
            if str_key in self.imgid2img:
                temp_img_info = self.imgid2img[str_key]
                for each_sent in self.raw_dataset.data[each_datum]:
                    temp_img_info['uid'] = str_key
                    temp_img_info['sent'] = each_sent
                    self.data[count] = temp_img_info
                    count += 1
        print("Use %d data in torch dataset" % (len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        sample_id = datum['uid']
        sent = datum['sent']

        # Get image info
        boxes2 = []
        feats2 = []

        img_info = self.imgid2img[sample_id]
        boxes = img_info['boxes'].copy()
        feats = img_info['features'].copy()
        assert len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes[..., (0, 2)] /= img_w
        boxes[..., (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        boxes2.append(boxes)
        feats2.append(feats)
        feats = np.stack(feats2)
        boxes = np.stack(boxes2)

        # Create target
        if 'label' in datum:
            label = datum['label']
            return sample_id, feats, boxes, sent, label
        else:
            return sample_id, feats, boxes, sent


class NLVR2Evaluator:
    def __init__(self, dataset: NLVR2Dataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans == label:
                score += 1
        return score / len(quesid2ans)

    '''

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump result to a CSV file, which is compatible with NLVR2 evaluation system.
        NLVR2 CSV file requirement:
            Each line contains: identifier, answer

        :param quesid2ans: nlvr2 uid to ans (either "True" or "False")
        :param path: The desired path of saved file.
        :return:
        """
        path = r"C:\Users\asmit\Desktop\Guided Research\lxmert\snap"
        with open(path, 'w') as f:
            for uid, ans in quesid2ans.items():
                idt = self.dataset.id2datum[uid]["identifier"]
                ans = 'True' if ans == 1 else 'False'
                f.write("%s,%s\n" % (idt, ans))
'''
