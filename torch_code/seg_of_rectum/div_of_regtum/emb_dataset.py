from torch.utils.data import Dataset
import imageio
import numpy as np
import os, torch

class EmbDataset(Dataset):
    def __init__(self, img_path, mask_path, model_res_path, model_set, train_phrase):
        super(EmbDataset, self).__init__()
        self.img_path = img_path
        self.mask_path = mask_path
        self.model_res_path = model_res_path
        self.model_set = model_set
        self.train_phrase = train_phrase
        self.ids = self.get_ids()
        pass

    def get_ids(self):
        path = os.path.join(self.img_path, self.train_phrase)
        arr = [x for x in os.listdir(path) if x.endswith('.png')]
        arr.sort()
        return arr

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        name = self.ids[index]
        feature_path = os.path.join(self.img_path, self.train_phrase, name)
        feature = imageio.imread(feature_path)
        feature = self.normalize(feature)
        features = [feature]

        for model in self.model_set:
            feature_path = os.path.join(self.model_res_path, self.train_phrase, model, name)
            feature = imageio.imread(feature_path)
            feature = self.normalize(feature)
            features.append(feature)
            pass

        features = torch.from_numpy(np.array(features).astype('float32'))

        label_path = os.path.join(self.mask_path, self.train_phrase, name)
        label = imageio.imread(label_path)
        label = self.normalize(label)
        labels = [label]
        
        labels = torch.from_numpy(np.array(labels).astype('float32'))

        return {
            'file_name': name,
            'features': features,
            'labels': labels
        }
    
    def normalize(self, img):
        if img.max() != 0:
            return img / img.max()
        return img
