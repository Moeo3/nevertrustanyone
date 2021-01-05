import os
import imageio
import numpy as np
from torch.utils.data import Dataset
import torch
from itertools import groupby

class Dataset3Layers(Dataset):
    def __init__(self, img_path, label_path, train_phrase):
        super(Dataset3Layers, self).__init__()
        self.img_path = img_path
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

    def __getitem__(self, idx, exp_=1e-6):
        file_name = self.names[idx]
        file_name_split = [''.join(list(g)) for k, g in groupby(file_name, key=lambda x: x.isdigit())]
        layer = int(file_name_split[1])

        img_path = os.path.join(self.img_path, self.train_phrase, file_name)
        img = imageio.imread(img_path)
        last_img = self.get_layer(file_name_split[0], layer - 1, img)
        next_img = self.get_layer(file_name_split[0], layer + 1, img)

        img = self.normalize(img)
        last_img = self.normalize(last_img)
        next_img = self.normalize(next_img)

        features = [last_img, img, next_img]
        features = np.array(features).astype('float32')
        features = torch.from_numpy(features)

        label_path = os.path.join(self.label_path, self.train_phrase, file_name)
        label = imageio.imread(label_path)
        label = self.normalize(label)
        labels = [label]
        labels = np.array(labels).astype('float32')
        labels = torch.from_numpy(labels)

        return {
            'file_name': file_name,
            'features': features,
            'labels': labels
        }

    def get_layer(self, patient_name, layer_num, self_img):
        layer_num = str(layer_num).zfill(2)
        img_path = os.path.join(self.img_path, self.train_phrase, f'{patient_name}{layer_num}.png')
        if os.path.exists(img_path):
            img = imageio.imread(img_path)
        else:
            img = self_img
        return img

    def normalize(self, img):
        if img.max() != 0:
            return img / img.max()
        return img
