from pathlib import Path

import numpy as np
from torchvision.io import read_image

import cv2 as cv
import os, pickle, torch
import pandas as pd
import random

NUM_CLASSES = 200
IMGS_PER_CLASS = 500

def get_imagenet_ids_to_class_indices_mapping(tinyimagenet_dir):
    # read wnids.txt
    with open(tinyimagenet_dir / 'wnids.txt', 'r') as f:
        wnids = f.read().splitlines()
    mapping = {}
    for i, wnid in enumerate(wnids):
        mapping[wnid] = i
    return mapping


def get_train_data(tinyimagenet_dir):
    train_data = torch.Tensor().type(torch.ByteTensor)
    mapping = get_imagenet_ids_to_class_indices_mapping(tinyimagenet_dir)
    labels = []
    i = 1
    train_dir = tinyimagenet_dir / 'train'
    for subdir in train_dir.iterdir():
        if subdir.is_dir():
            images_dir = subdir / 'images'
            if not (images_dir.exists() and images_dir.is_dir()):
                continue
            one_class = torch.Tensor().type(torch.ByteTensor)
            for img_path in images_dir.glob('*'):
                img = read_image(str(img_path))
                if img.shape[0] == 1:
                    img = torch.tensor(cv.cvtColor(img.permute(1, 2, 0).numpy(), cv.COLOR_GRAY2RGB)).permute(2, 0, 1)
                one_class = torch.cat((one_class, img), 0)
                labels.append(mapping[subdir.name])

            one_class = one_class.reshape(-1, 3, 64, 64)
            print(i, '/', NUM_CLASSES)
            i += 1
            train_data = torch.cat((train_data, one_class), 0)

    return train_data, torch.Tensor(labels)


def get_val_data(tinyimagenet_dir, one_hot=False):
    val_data = torch.Tensor().type(torch.ByteTensor)
    mapping = get_imagenet_ids_to_class_indices_mapping(tinyimagenet_dir)
    labels = []
    val_annotations_path = tinyimagenet_dir / 'val' / 'val_annotations.txt'
    val_annotations = pd.read_csv(val_annotations_path, sep='\t',
                                  names=['filename', 'label_str', 'x_min', 'y_min', 'x_max', 'y_max'])
    val_images_dir = tinyimagenet_dir / 'val' / 'images'
    num_imgs = len(list(val_images_dir.iterdir()))

    i = 1
    for img_path in val_images_dir.iterdir():
        img = read_image(str(img_path))
        if img.shape[0] == 1:
            img = torch.tensor(cv.cvtColor(img.permute(1, 2, 0).numpy(), cv.COLOR_GRAY2RGB)).permute(2, 0, 1)
        val_data = torch.cat((val_data, img), 0)
        class_name = val_annotations.loc[val_annotations['filename'] == img_path.name]['label_str'].item()
        labels.append(mapping[class_name])
        print(i, '/', num_imgs)
        i += 1

    return val_data.reshape(-1, 3, 64, 64), torch.Tensor(labels)

if __name__ == '__main__':
    random.seed(0)
    path = Path('data/tiny-imagenet-200')
    data, labels = get_train_data(path)
    print('train+val', data.shape, labels.shape)

    val_samples = 10000
    total_samples = len(data)
    train_samples = total_samples - val_samples

    indices = list(range(total_samples))
    random.shuffle(indices)

    train_indices = indices[:train_samples]
    val_indices = indices[train_samples:]

    train_data = data[train_indices]
    train_labels = labels[train_indices]
    val_data = data[val_indices]
    val_labels = labels[val_indices]

    print('train', train_data.shape, train_labels.shape)
    with open(path / 'train_dataset.pkl', 'wb') as f:
        pickle.dump((train_data, train_labels), f)

    print('val', val_data.shape, val_labels.shape)
    with open(path / 'val_dataset.pkl', 'wb') as f:
        pickle.dump((val_data, val_labels), f)

    test_data, test_labels = get_val_data(path)
    print('test', test_data.shape, test_labels.shape)
    with open(path / 'test_dataset.pkl', 'wb') as f:
        pickle.dump((test_data, test_labels), f)