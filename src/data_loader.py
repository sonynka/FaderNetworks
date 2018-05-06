# coding=utf-8

import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import pandas as pd


class FashionDataset(Dataset):
    def __init__(self, image_path, metadata_path, transform, mode):
        self.image_path = image_path
        self.transform = transform
        self.mode = mode
        self.lines = pd.read_csv(metadata_path, sep='\t', encoding='utf-8')
        self.num_data = self.lines.shape[0]
        self.attr2idx = {}
        self.idx2attr = {}

        print ('Start preprocessing dataset..!')
        random.seed(1234)
        self.preprocess()
        print ('Finished preprocessing dataset..!')

        if self.mode == 'train':
            self.num_data = len(self.train_filenames)
        elif self.mode == 'test':
            self.num_data = len(self.test_filenames)

    def preprocess(self):

        self.selected_attrs = [u'ärmellänge_langarm', u'ärmellänge_viertelarm', u'ärmellänge_ärmellos',
                               u'muster_all-over-muster', u'muster_geblümt/floral', u'muster_gestreift']

        self.train_filenames = []
        self.train_labels = []
        self.test_filenames = []
        self.test_labels = []

        lines = self.lines.copy()
        # lines = lines[lines['category_kleider'] == 1]
        lines = lines[['img_path'] + self.selected_attrs]
        train_set = lines.sample(frac=0.8)
        test_set = lines.drop(train_set.index)

        self.train_filenames = train_set['img_path'].tolist()
        self.train_labels = train_set.iloc[:, 1:].as_matrix().tolist()

        self.test_filenames = test_set['img_path'].tolist()
        self.test_labels = test_set.iloc[:, 1:].as_matrix().tolist()


    def __getitem__(self, index):
        if self.mode == 'train':
            file_path = u''.join((self.image_path, self.train_filenames[index])).encode('utf-8').strip()
            image = Image.open(file_path)
            label = self.train_labels[index]
        elif self.mode in ['test']:
            file_path = u''.join((self.image_path, self.test_filenames[index])).encode('utf-8').strip()
            image = Image.open(file_path)
            label = self.test_labels[index]

        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        return self.num_data



def get_loader(image_path, metadata_path, crop_size, image_size, batch_size, mode='train'):
    """Build and return data loader."""

    if mode == 'train':
        transform = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Resize(image_size, interpolation=Image.ANTIALIAS),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Scale(image_size, interpolation=Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = FashionDataset(image_path, metadata_path, transform, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader