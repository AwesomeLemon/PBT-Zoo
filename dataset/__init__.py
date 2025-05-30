from dataset.cifar10 import Cifar10Wrapper
from dataset.cifar100 import Cifar100Wrapper
from dataset.fashion_mnist import FashionMnistWrapper
from dataset.tinyimagenet import TinyImagenetWrapper

def get_dataset(cfg):
    if cfg.task.data.name == 'FashionMnist':
        d = FashionMnistWrapper(cfg)
        return d
    elif cfg.task.data.name == 'Cifar10':
        d = Cifar10Wrapper(cfg)
        return d
    elif cfg.task.data.name == 'Cifar100':
        d = Cifar100Wrapper(cfg)
        return d
    elif cfg.task.data.name == 'TinyImagenet':
        d = TinyImagenetWrapper(cfg)
        return d
    else:
        raise ValueError(cfg.task.data.name)