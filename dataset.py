from collections import defaultdict

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image


class datasetBBox(data.Dataset):
    def __init__(self, dir, img_dir):
        self.dir = dir
        self.img_dir = img_dir
        self.filenames = sorted(os.listdir(os.path.join(self.dir, self.img_dir)))

        self.bbox = pd.read_csv(os.path.join(self.dir, 'Anno/bounding_boxes.txt'),
                           delim_whitespace=True, header=None).astype(int)

        self.transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5))])


    def __getitem__(self, index):
        img_path = os.path.join(self.dir, self.img_dir, self.filenames[index])
        bbox = self.bbox.iloc[index][1:].tolist()

        img = Image.open(img_path).convert('RGB')
        width, height = img.size

        # BBOX?
        # r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        # center_x = int((2 * bbox[0] + bbox[2]) / 2)
        # center_y = int((2 * bbox[1] + bbox[3]) / 2)
        # y1 = np.maximum(0, center_y - r)
        # y2 = np.minimum(height, center_y + r)
        # x1 = np.maximum(0, center_x - r)
        # x2 = np.minimum(width, center_x + r)
        # img = img.crop([x1, y1, x2, y2])

        img = self.transform(img)

        return img


    def __len__(self):
        return len(self.filenames)
