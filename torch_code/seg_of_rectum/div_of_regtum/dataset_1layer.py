import os
import imageio
import numpy as np
from torch.utils.data import Dataset
import torch

class Dataset1Layer(Dataset):
    def __init__(self, img_path, label_path, train_phrase):
        super(Dataset1Layer, self).__init__()
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

    def __getitem__(self, idx):
        file_name = self.names[idx]

        img_path = os.path.join(self.img_path, self.train_phrase, file_name)
        img = imageio.imread(img_path)
        img = self.normalize(img)
        features = [img]
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

    def normalize(self, img):
        if img.max() != 0:
            return img / img.max()
        return img
