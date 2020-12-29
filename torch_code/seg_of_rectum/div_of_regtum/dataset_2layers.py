import os
import imageio
import numpy as np
from torch.utils.data import Dataset
import torch

class Dataset2Layers(Dataset):
    def __init__(self, img_path, ori_seg_path, label_path, train_phrase):
        super(Dataset2Layers, self).__init__()
        self.img_path = img_path
        self.ori_seg_path = ori_seg_path
        self.label_path = label_path
        self.train_phrase = train_phrase
        self.names = self.get_names()
    
    def get_names(self):
        path = os.path.join(self.img_path, self.train_phrase)
        arr = [x for x in os.listdir(path) if x.endswith('.png')]
        arr.sort()
        return arr

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        file_name = self.names[idx]

        img_path = os.path.join(self.img_path, self.train_phrase, file_name)
        img = imageio.imread(img_path)

        ori_seg_img_path = os.path.join(self.ori_seg_path, self.train_phrase, file_name)
        if os.path.exists(ori_seg_img_path):
            ori_seg_img = imageio.imread(ori_seg_img_path)
        else:
            ori_seg_img = np.zeros(img.shape)
        features = [img, ori_seg_img]
        features = np.array(features).astype('float32')
        features = torch.from_numpy(features / features.max())

        label_path = os.path.join(self.label_path, self.train_phrase, file_name)
        label = imageio.imread(label_path)
        labels = [label]
        labels = np.array(labels).astype('float32')
        labels = torch.from_numpy(labels / 255)

        return {
            'file_name': file_name,
            'features': features,
            'labels': labels
        }
