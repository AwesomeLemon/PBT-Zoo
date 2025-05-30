import copy
from pathlib import Path
import torch
import torchvision.datasets as datasets
from torch.utils.data import random_split
import torchvision.transforms as t

class FashionMnistWrapper:
    def __init__(self, cfg):
        dataset = {}

        self.mean = (0.5,)
        self.std = (0.5,)

        transforms = {'train': self.create_train_transform(
                                    rand_aug_n_ops=cfg.task.data.rand_aug_n_ops,
                                    rand_aug_mag=cfg.task.data.rand_aug_mag),
                      'val': t.Compose([
                                    t.ToTensor(),
                                    t.Normalize(self.mean, self.std),
                                ])
        }

        d = datasets.FashionMNIST(Path(cfg.path.data) / 'FashionMnist' / 'train', train=True,
                                  transform=transforms['train'], download=True)

        total_samples = len(d)
        generator = torch.Generator().manual_seed(42)
        dataset['train'], dataset['val'] = random_split(d, [total_samples - cfg.task.data.val_size,
                                                            cfg.task.data.val_size], generator=generator)

        # print(f"{dataset['train'].dataset.transform=}")
        dataset['val'].dataset = copy.deepcopy(dataset['val'].dataset)
        dataset['val'].dataset.transform = transforms['val']
        # print(f"{dataset['val'].dataset.transform=}")

        dataset['test'] = datasets.FashionMNIST(Path(cfg.path.data) / 'FashionMnist' / 'test', train=False,
                                                transform=transforms['val'], download=True)

        self.dataset = dataset
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def create_train_transform(self, **kwargs):
        return t.Compose([
            t.RandAugment(num_ops=kwargs['rand_aug_n_ops'], magnitude=kwargs['rand_aug_mag']),
            t.ToTensor(),
            t.Normalize(self.mean, self.std),
            t.RandomHorizontalFlip(),
        ])