import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ray
import torch

from algo.base import BaseAlgo
from algo.pbt_utils import save_explored_ckpt_to_path
from utils.util_fns import solution_history_to_str, convert_from_logarithmic, convert_to_logarithmic
import operator


class PBT(BaseAlgo):
    """
    Population Based Training
    https://arxiv.org/abs/1711.09846
    """
    def __init__(self, cfg, search_space, task, **__):
        super().__init__(cfg, search_space, task)
        self.quant_top = cfg.algo.quant_top
        self.quant_bottom = cfg.algo.quant_bottom
        self.perturb_factor = cfg.algo.get('perturb_factor', None)  # the child algo PB2 doesn't have this.

        self.min_steps_before_eval = cfg.algo.get('min_steps_before_eval', 0)

    def tick(self):
        # 1. start evaluations
        st = time.time()
        results = self._schedule_all_populations()
        self.train_and_eval_times = pd.concat([self.train_and_eval_times,
                                               pd.DataFrame({'t': [self.t_cur], 'time': [time.time() - st]})],
                                              ignore_index=True)
        print(f'_schedule_all_populations time: {time.time() - st:.2f} s')
        # 2. save checkpoints
        ckpts_to_delete = self._save_checkpoints(results)
        # 3. log history
        fitnesses = []
        for i, res_dict in enumerate(results):
            fitnesses.append(res_dict['fitness'])

        for i, (fitness, p) in enumerate(zip(fitnesses, self.pop)):
            self.fitness_history[i].append(fitness)
            self.solution_history[i].append(copy.deepcopy(p))
            print(f'{i}: {solution_history_to_str(self.solution_history[i])}: {fitness:.4f}')

        # 4. update population
        if self.min_steps_before_eval <= self.t_cur < self.t_max - self.t_step:
            st = time.time()
            self._exploit_and_explore(fitnesses)
            self.update_times = pd.concat([self.update_times,
                                           pd.DataFrame({'t': [self.t_cur], 'time': [time.time() - st]})],
                                          ignore_index=True)

        # 5. delete old checkpoints
        for p in ckpts_to_delete:
            p.unlink()

        # save
        self.update_times.to_csv(self.exp_dir / 'update_times.csv', index=False)
        self.train_and_eval_times.to_csv(self.exp_dir / 'train_and_eval_times.csv', index=False)

    def _schedule_all_populations(self):
        futures = []
        for i, p in enumerate(self.pop):
            ckpt_path = Path(self.cpkt_dir) / f'pop_{i}_t{self.t_cur}.pt'
            if self.t_cur == 0:
                self.prepare_initial_ckpt(ckpt_path, p)
            ckpt_loaded = torch.load(ckpt_path)

            seed = self.seed_base * 100 + int(self.t_cur / self.t_max * 1e6) + i
            tb_dir = Path(self.exp_dir) / 'tb' / f'pop_{i}'
            tb_dir.mkdir(parents=True, exist_ok=True)
            f = self._task_fn_ray.options(**self.ray_options).remote(self.task, seed, p, self.t_cur, self.t_step,
                                                                     ckpt_loaded, tb_dir, None)
            futures.append(f)

        results = ray.get(futures)
        return results

    def _save_checkpoints(self, results):
        ckpts_to_delete = []  # first collect, then delete after everything is saved

        for i, res_dict in enumerate(results):
            ckpt_path = Path(self.cpkt_dir) / f'pop_{i}_t{self.t_cur + self.t_step}.pt'
            torch.save(res_dict['dict_to_save'], ckpt_path)
            if self.delete_old_ckpts:
                ckpts_to_delete.append(Path(self.cpkt_dir) / f'pop_{i}_t{self.t_cur}.pt')
        return ckpts_to_delete

    def _exploit_and_explore(self, fitnesses):
        idx_top = np.argsort(fitnesses)[-int(self.quant_top * self.pop_size):]
        idx_bottom = np.argsort(fitnesses)[:int(self.quant_bottom * self.pop_size)]
        bounds_cont = self.search_space.get_bounds_cont()
        for i in range(len(self.pop)):
            if i not in idx_bottom:
                continue
            # replace bottom
            chosen_idx = np.random.choice(idx_top)
            self.pop[i] = copy.deepcopy(self.pop[chosen_idx])
            # replace its history
            self.fitness_history[i] = copy.deepcopy(self.fitness_history[chosen_idx])
            self.solution_history[i] = copy.deepcopy(self.solution_history[chosen_idx])
            # explore
            for i_var in range(self.search_space.n_vars):
                hp_name = self.search_space.get_hp_name(i_var)
                if hp_name in bounds_cont.keys():
                    op = operator.mul if np.random.rand() < 0.5 else operator.truediv

                    if hp_name.startswith('log'):
                        # if var is log, convert to normal space, perturb, and convert back
                        self.pop[i][i_var] = convert_from_logarithmic(hp_name, self.pop[i][i_var])
                    elif bounds_cont[hp_name][0] * self.perturb_factor > bounds_cont[hp_name][1]:
                        # if the bounds are so tight that perturbation factor only flips between min/max, normalize, perturb, denormalize
                        self.pop[i][i_var] = (self.pop[i][i_var] - bounds_cont[hp_name][0]) / (
                                bounds_cont[hp_name][1] - bounds_cont[hp_name][0])

                    self.pop[i][i_var] = op(self.pop[i][i_var], self.perturb_factor)

                    if hp_name.startswith('log'):
                        self.pop[i][i_var] = convert_to_logarithmic(hp_name, self.pop[i][i_var])
                    elif bounds_cont[hp_name][0] * self.perturb_factor > bounds_cont[hp_name][1]:
                        self.pop[i][i_var] = self.pop[i][i_var] * (bounds_cont[hp_name][1] - bounds_cont[hp_name][0]) + \
                                             bounds_cont[hp_name][0]

                    self.pop[i][i_var] = float(np.clip(self.pop[i][i_var], *bounds_cont[hp_name]))
                else:
                    resample_prob = 1.0  # this seems high but this is the standard mutation, as also used in PBT and PB2-Rand (in the PB2-Mix paper)
                    if np.random.rand() < resample_prob:
                        self.pop[i][i_var] = self.search_space.sample()[i_var]
                    else:
                        self.pop[i][i_var] = self.pop[chosen_idx][i_var]

            # replace checkpoint
            ckpt_path = Path(self.cpkt_dir) / f'pop_{i}_t{self.t_cur + self.t_step}.pt'
            ckpt_path.unlink()
            ckpt_chosen_path = Path(self.cpkt_dir) / f'pop_{chosen_idx}_t{self.t_cur + self.t_step}.pt'
            save_explored_ckpt_to_path(self.task, self.pop[i], ckpt_chosen_path, ckpt_path)
            print(
                f'Replaced {i}  with {chosen_idx}, unperturbed values {self.pop[chosen_idx]}, perturbed values {self.pop[i]}')
