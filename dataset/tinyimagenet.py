# adapted from https://github.com/ehuynh1106/TinyImageNet-Transformers/blob/main/dataset.py
from torch import FloatTensor, div
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from pathlib import Path
import torchvision.transforms as t

import pickle, torch
import numpy as np

class TinyImagenetWrapper:
    def __init__(self, cfg):
        dataset = {}

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        self.img_size = cfg.task.data.img_size
        transform_train = self.create_train_transform(
                                    rand_aug_n_ops=cfg.task.data.rand_aug_n_ops,
                                    rand_aug_mag=cfg.task.data.rand_aug_mag)

        transform_val = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
        ])


        transform_normalize = transforms.Compose([
            transforms.Normalize(
                    mean=self.mean,
                    std=self.std
                )
            ])

        tinyimagenet_path = Path(cfg.path.data) / 'tiny-imagenet-200'
        with open(tinyimagenet_path / 'train_dataset.pkl', 'rb') as f:
            train_data, train_labels = pickle.load(f)
        dataset['train'] = ImageNetDataset(train_data, train_labels.type(torch.LongTensor),
                                           transform_train, transform_normalize, self.mean, self.std)

        with open(tinyimagenet_path / 'val_dataset.pkl', 'rb') as f:
            val_data, val_labels = pickle.load(f)
        dataset['val'] = ImageNetDataset(val_data, val_labels.type(torch.LongTensor),
                                         transform_val, transform_normalize, self.mean, self.std)

        with open(tinyimagenet_path / 'test_dataset.pkl', 'rb') as f:
            test_data, test_labels = pickle.load(f)
        dataset['test'] = ImageNetDataset(test_data, test_labels.type(torch.LongTensor),
                                          transform_val, transform_normalize, self.mean, self.std)

        self.dataset = dataset
        self.class_names = list(range(200))

    def create_train_transform(self, **kwargs):
        # cf fashionmnist, add resize, remove normalize (because it'll be separately applied below, in the dataset) and ToTensor.
        return t.Compose([
            t.Resize(self.img_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
            t.RandAugment(num_ops=kwargs['rand_aug_n_ops'], magnitude=kwargs['rand_aug_mag']),
            t.RandomHorizontalFlip(),
        ])

class ImageNetDataset(Dataset):
    """Dataset class for ImageNet"""
    def __init__(self, dataset, labels, transform=None, normalize=None, mean=None, std=None):
        super(ImageNetDataset, self).__init__()
        assert(len(dataset) == len(labels))
        self.images = dataset
        self.labels = labels
        self.transform = transform
        self.normalize = normalize
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        data = self.images[idx]
        if self.transform:
            data = self.transform(data)

        data = div(data.type(FloatTensor), 255)
        if self.normalize:
            data = self.normalize(data)

        return data, self.labels[idx]

if __name__ == '__main__':
    tinyimagenet_path = Path('data/tiny-imagenet-200')
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    img_size = 128
    with open(tinyimagenet_path / 'test_dataset.pkl', 'rb') as f:
        val_data, val_labels = pickle.load(f)
    transform_val = transforms.Compose([
        transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
    ])

    transform_normalize = transforms.Compose([
        transforms.Normalize(
            mean=mean,
            std=std
        )
    ])
    d = ImageNetDataset(val_data, val_labels.type(torch.LongTensor),
                                     transform_val, transform_normalize, mean, std)