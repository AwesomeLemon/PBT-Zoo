import random
import numpy as np
import torch

class ToyDeceptiveTask:
    """
    See Section 5.2 of "To Be Greedy, or Not to Be - That Is the Question for Population Based Training Variants".
    The optimal behavior for final performance is to reduce the hyperparameter to 0 linearly.
    The greedily optimal behavior is to reduce the hyperparameter to 0 as fast as possible.
    """
    def __init__(self, t_max, **__):
        self.t_max = t_max

    def __call__(self, seed, solution, t, t_step, cpkt_loaded, tb_dir, only_evaluate):
        theta = torch.Tensor(cpkt_loaded['model_state_dict'])
        lr_history = cpkt_loaded['lr_history']
        solution = solution[0] # this is a 1-D problem
        lr_history.append(solution)
        if only_evaluate:
            res = self._eval(theta, lr_history, t, t_step)[0][-1][1]
            return {'fitness': res, 'val': res, 'test': res}

        curve, theta = self._eval(theta, lr_history, t, t_step)

        dict_to_save = {'model_state_dict': theta, 'lr_history': lr_history}
        return {'fitness': curve[-1][1], 'dict_to_save': dict_to_save, 'curve': curve}

    def _eval(self, theta, lr_history, t, t_step):
        diffs = []
        for i, v in enumerate(lr_history):
            target = (self.t_max - i * t_step) / self.t_max
            diffs.append(abs(v - target))
        sum_diff = np.sum(diffs).item()

        curve = []
        for t_cur in range(t, t + t_step):
            penalty_mul = 0.2

            penalty = penalty_mul * sum_diff

            # gradient step
            obj = (1.2 - torch.sum(theta ** 2)).item()
            grad_obj_surrogate = -2.0 * max(2 - lr_history[-1] - penalty, 0) * theta
            theta += 0.001 * grad_obj_surrogate

            res = obj

            curve.append((t_cur, res))
        if t_step > 0:
            print(f'lr={lr_history[-1]:.4f} {obj=:.4f} {penalty=:.4f}')
        else: # eval at init
            obj = (1.2 - torch.sum(theta ** 2)).item()
            curve = [(0, obj)]
            print(f'lr={lr_history[-1]:.4f}')

        return curve, theta

    def prepare_initial_ckpt(self, solution):
        assert len(solution) == 1
        return  {'model_state_dict': torch.tensor([0.9] * len(solution)),
                 'lr_history': []}

    def init_sample(self, sampled_dict):
        sampled_dict['lr'] = float(random.uniform(0.9, 1.1))
        return sampled_dict