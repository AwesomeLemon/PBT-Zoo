import logging
import random
import sys

import math
import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
import seaborn as sns

def set_random_seeds(seed):
    torch.backends.cudnn.benchmark = True
    seed = int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f'random seed set to {seed}')


class LogWriter:
    def __init__(self, log_fun):
        self.log_fun = log_fun
        self.buf = []
        self.is_tqdm_msg_fun = lambda msg: '%|' in msg
        # annoyingly, ray doesn't allow to disable colors in output, and they make logs unreadable, so:
        self.replace_garbage = lambda msg: msg.replace('[2m[36m', '').replace('[0m', '').replace('[32m', '').replace('[36m', '')

    def write(self, msg):
        is_tqdm = self.is_tqdm_msg_fun(msg)
        has_newline = msg.endswith('\n')
        if has_newline or is_tqdm:
            self.buf.append(msg)  # .rstrip('\n'))
            self.log_fun(self.replace_garbage(''.join(self.buf)))
            self.buf = []
        else:
            self.buf.append(msg)

    def flush(self):
        pass

    def close(self):
        self.log_fun.close()

    @staticmethod
    def isatty(): # ray logging expects this function to exist
        return True


def setup_logging(log_path):
    from importlib import reload
    reload(logging)
    logging.StreamHandler.terminator = ''  # don't add new line, I'll do it myself; this line affects both handlers
    stream_handler = logging.StreamHandler(sys.__stdout__)
    file_handler = logging.FileHandler(log_path, mode='a')
    # don't want a bazillion tqdm lines in the log:
    # file_handler.filter = lambda record: '%|' not in record.msg or '100%|' in record.msg
    file_handler.filter = lambda record: '[A' not in record.msg and ('%|' not in record.msg or '100%|' in record.msg)
    handlers = [
        file_handler,
        stream_handler]
    logging.basicConfig(level=logging.INFO,
                        # format='%(asctime)s %(message)s',
                        # https://docs.python.org/3/library/logging.html#logrecord-attributes
                        # https://docs.python.org/3/library/logging.html#logging.Formatter
                        # format='%(process)d %(message)s',
                        format='%(message)s',
                        handlers=handlers,
                        datefmt='%H:%M')
    sys.stdout = LogWriter(logging.info)
    sys.stderr = LogWriter(logging.error)

def set_plot_style():
    plt.style.use('seaborn-v0_8')
    # plt.rcParams.update({'font.size': 90}) # seaborn style ignores font size
    plt.rcParams['axes.grid'] = True
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    # sns.set(font_scale=1.2)
    from cycler import cycler
    palette = sns.color_palette("colorblind", 10)
    plt.rcParams['axes.prop_cycle'] = cycler(color=palette)

def set_plot_style_paper():
    plt.style.use('seaborn-v0_8')
    # plt.rcParams.update({'font.size': 90}) # seaborn style ignores font size
    plt.rcParams['axes.grid'] = True
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    sns.set(font_scale=1.3)
    from cycler import cycler
    palette = sns.color_palette("colorblind", 10)
    plt.rcParams['axes.prop_cycle'] = cycler(color=palette)

def solution_history_to_str(solution_history):
    def _format_single(y):
        if type(y) == str:
            return y
        if type(y) == int:
            return f'{y}'
        return f'{y:.3f}'
    rounded = [[_format_single(y) for y in x ] for x in solution_history]
    out_str = '-'.join(['[' + '|'.join(x) + ']' for x in rounded])
    return out_str


def adjust_optimizer_settings(optimizer, optimizer_hps):
    for param_group in optimizer.param_groups:
        for hp_name, hp_val in optimizer_hps.items():
            param_group[hp_name] = hp_val

    return optimizer

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def save_yaml(data, path):
    with open(path, 'w') as f:
        yaml.safe_dump(data, f)

def convert_from_logarithmic(key, value):
    assert key.startswith('log')
    exp = int(key[3:key.index('_')])
    return exp ** value

def convert_to_logarithmic(key, value):
    assert key.startswith('log')
    exp = int(key[3:key.index('_')])
    return math.log(value, exp)

def convert_config_from_logarithmic(config_dict):
    config_out = {}
    for key, val in config_dict.items():
        if key.startswith('log'):
            config_out[key[key.index('_')+1:]] = convert_from_logarithmic(key, val)
        else:
            config_out[key] = val
    return config_out