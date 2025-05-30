import random

import numpy as np
import torch

class ToyPlainTask:
    """
    See Section 5.2 of "To Be Greedy, or Not to Be - That Is the Question for Population Based Training Variants".
    The optimal behavior is to reduce the hyperparameter to 0 as fast as possible.
    """
    def __init__(self, **__):
        self.obj = lambda theta: 1.2 - torch.sum(theta ** 2)
        self.obj_surrogate = lambda theta, h: 1.2 - np.sum((2 - h) * theta ** 2) # don't need it because gradient can be computed analytically, but have it here as a reminder

    def __call__(self, seed, solution, t, t_step, cpkt_loaded, tb_dir, only_evaluate):
        theta = torch.Tensor(cpkt_loaded['model_state_dict'])
        if only_evaluate:
            res = self._eval(theta)
            return {'fitness': res, 'val': res, 'test': res}

        solution = torch.Tensor(solution)
        curve = []
        for t_cur in range(t, t + t_step):
            grad_obj_surrogate = -2.0 * (2 - solution) * theta
            theta += 0.001 * grad_obj_surrogate
            curve.append((t_cur, self._eval(theta)))

        dict_to_save = {'model_state_dict': theta}
        return {'fitness': curve[-1][1], 'dict_to_save': dict_to_save, 'curve': curve} # dummy curve


    def _eval(self, theta):
        return self.obj(theta).item()

    def prepare_initial_ckpt(self, solution):
        return  {'model_state_dict': torch.tensor([0.9] * len(solution))} # arbitrary dimensionality.
        # Initial theta always 0.9 (note that this is not the hyperparameter).

    def init_sample(self, sampled_dict):
        sampled_dict['lr'] = float(random.uniform(0.9, 1.1))
        return sampled_dict