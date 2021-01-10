import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import SimpleITK as sitk
import random
import torchvision.transforms.functional as tf
import cv2
import csv
import imageio


class ModelDataSet(Dataset):
    def __init__(self, img_path='', label_path='', train_phase='train', rand_transform=False):
        super(ModelDataSet, self).__init__()
        self.train_phase = train_phase
        # self.config = get_config('data_config')
        self.img_path = img_path
        # self.label_path = label_path
        self.ids = self.get_ids()
        self.rand_transform = rand_transform
        self.csv_name, self.csv_score = self.get_label(label_path)

    def __len__(self):
        # return 10
        return len(self.ids)

    def get_label(self, label_path):
        # global label, score
        csv_name = []
        csv_score = []
        f = csv.reader(open(os.path.join(label_path, self.train_phase + '.csv'), 'r'))
        for i in f:
            csv_name.append(i[0])
            csv_score.append(i[1])
        return csv_name, csv_score

    def __getitem__(self, index):
        id = self.ids[index]
        sample = dict()
        sample['id'] = id
        sample['data'], sample['label'] = self.get_sample_data(id)
        sample = self.data_transform(sample)
        return sample

    def get_sample_data(self, id):
        image_1 = imageio.imread(os.path.join(self.img_path, self.train_phase, id))
        if not image_1.shape[0] == 48:
            image_1 = np.r_[image_1, np.zeros([48 - image_1.shape[0], image_1.shape[1]])]
        if not image_1.shape[1] == 48:
            image_1 = np.c_[image_1, np.zeros([image_1.shape[0], 48 - image_1.shape[1]])]
        label = torch.from_numpy(np.array([float(self.csv_score[self.csv_name.index(id)])]))
        return image_1, label.type(torch.float32)

    def get_ids(self):
        arr = []
        path = os.path.join(self.img_path, self.train_phase)
        # path = os.path.join(self.img_path)
        for i in os.listdir(path):
            if os.path.splitext(i)[1] == ".png":
                arr.append(i)
        arr.sort()
        return arr

    def data_transform(self, sample):
        data_transform = transforms.Compose([
            transforms.Lambda(lambd=lambda x: torch.from_numpy((x.astype(np.float32)) / x.max() * 1.0 - 0.0)),
            transforms.Lambda(lambd=lambda x: x[np.newaxis, ...]),
        ])
        # label_transform = transforms.Compose([
        #     transforms.Lambda(lambd=lambda x: torch.from_numpy((x.astype(np.float32)) / 1.0 * 1.0 - 0.0)),
        #     transforms.Lambda(lambd=lambda x: x[np.newaxis, ...]),
        # ])

        # random augmentation
        if self.rand_transform:
            # if random.random() > 0.5:
            #     sample['data'] = cv2.flip(sample['data'], 0)
            #     sample['label'] = cv2.flip(sample['label'], 0)
            # if random.random() > 0.5:
            #     sample['data'] = cv2.flip(sample['data'], 1)
            #     sample['label'] = cv2.flip(sample['label'], 1)
            random.seed()
            rand = random.random()
            if rand > 0.25 and rand < 0.5:
                sample['data'] = cv2.rotate(sample['data'], cv2.ROTATE_90_CLOCKWISE)
                # sample['label'] = cv2.rotate(sample['label'], cv2.ROTATE_90_CLOCKWISE)
            if rand > 0.5 and rand < 0.75:
                sample['data'] = cv2.rotate(sample['data'], cv2.ROTATE_180)
                # sample['label'] = cv2.rotate(sample['label'], cv2.ROTATE_180)
            if rand > 0.75:
                sample['data'] = cv2.rotate(sample['data'], cv2.ROTATE_90_COUNTERCLOCKWISE)
                # sample['label'] = cv2.rotate(sample['label'], cv2.ROTATE_90_COUNTERCLOCKWISE)

        # normalization
        sample['data'] = data_transform(sample['data'])
        # sample['label'] = label_transform(sample['label'])
        return sample
