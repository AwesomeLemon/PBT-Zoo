import shutil

import torch

from task.toy_deceptive import ToyDeceptiveTask

def save_explored_ckpt_to_path(task, explored_hps, ckpt_chosen_path, ckpt_path):
    '''
    This function is needed to correctly compute penalty in ToyDeceptiveTask
    '''
    if task is ToyDeceptiveTask:
        loaded = torch.load(ckpt_chosen_path)
        loaded['lr_history'][-1] = explored_hps[0]
        torch.save(loaded, ckpt_path)
    else:
        shutil.copy(ckpt_chosen_path, ckpt_path)
