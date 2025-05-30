import copy
from pathlib import Path
import torch
import torchvision.datasets as datasets
from torch.utils.data import random_split
import torchvision.transforms as t

class Cifar10Wrapper:
    def __init__(self, cfg):
        dataset = {}

        self.mean = (0.5,)
        self.std = (0.5,)

        self.img_size = cfg.task.data.get('img_size', 32)
        transforms = {'train': self.create_train_transform(
                                    rand_aug_n_ops=cfg.task.data.rand_aug_n_ops,
                                    rand_aug_mag=cfg.task.data.rand_aug_mag),
                      'val': t.Compose(
                          ([t.Resize(self.img_size)] if self.img_size != 32 else []) +
                          [t.ToTensor(),
                           t.Normalize(self.mean, self.std),
                          ])
        }

        d = datasets.CIFAR10(Path(cfg.path.data) / 'cifar10', train=True,
                                  transform=transforms['train'], download=True)

        total_samples = len(d)
        generator = torch.Generator().manual_seed(42)
        dataset['train'], dataset['val'] = random_split(d, [total_samples - cfg.task.data.val_size,
                                                            cfg.task.data.val_size], generator=generator)
        dataset['val'].dataset = copy.deepcopy(dataset['val'].dataset)
        dataset['val'].dataset.transform = transforms['val']

        dataset['test'] = datasets.CIFAR10(Path(cfg.path.data) / 'cifar10', train=False,
                                                transform=transforms['val'], download=True)

        self.dataset = dataset
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck']

    def create_train_transform(self, **kwargs):
        resize = [t.RandomResizedCrop(self.img_size)] if self.img_size != 32 else []
        return t.Compose(resize + [
            t.RandAugment(num_ops=kwargs['rand_aug_n_ops'], magnitude=kwargs['rand_aug_mag']),
            t.ToTensor(),
            t.Normalize(self.mean, self.std),
            t.RandomHorizontalFlip(),
        ])