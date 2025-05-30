import copy
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ray
import torch
import yaml
from matplotlib import pyplot as plt

from algo.plot_utils import plot_pop_history
from utils.util_fns import save_yaml, set_random_seeds, set_plot_style


class BaseAlgo:
    '''
    Base class implementing common functionality for all algorithms.
    '''
    def __init__(self, cfg, search_space, task, **__):
        self.cfg = cfg
        self.task = task
        self.search_space = search_space

        if not hasattr(self, 'pop_size'):  # FirePBT initializes pop_size in its __init__ differently
            self.pop_size = cfg.algo.pop_size
        self.exp_dir = Path(cfg.path.dir_exp)
        self.exp_name = Path(cfg.general.exp_name)

        self.cpkt_dir = Path(cfg.path.dir_ckpt)
        self.cpkt_dir.mkdir(parents=True, exist_ok=True)

        self._init_pop()

        self.t_cur = 0
        self.update_times = pd.DataFrame(columns=['t', 'time'])
        self.tick_times = pd.DataFrame(columns=['t', 'time'])
        self.train_and_eval_times = pd.DataFrame(columns=['t', 'time'])
        if self.cfg.general.continue_auto:
            self.t_cur = yaml.safe_load(open(self.exp_dir / 'last_finished_tick.yaml'))
            self.update_times = pd.read_csv(self.exp_dir / 'update_times.csv')
            self.tick_times = pd.read_csv(self.exp_dir / 'tick_times.csv')
            self.train_and_eval_times = pd.read_csv(self.exp_dir / 'train_and_eval_times.csv')
        self.t_step = cfg.algo.t_step
        self.t_max = cfg.algo.t_max

        self.ray_options = {'num_cpus': cfg.general.num_cpus, 'num_gpus': cfg.general.num_gpus}

        self.delete_old_ckpts = cfg.algo.get('delete_old_ckpts', False)
        self.delete_all_ckpts_at_the_end = cfg.algo.get('delete_all_ckpts_at_the_end', False)

        self.seed_base = cfg.general.seed_base

    def _init_pop(self):
        if not self.cfg.general.continue_auto:
            self.pop = []
            for i in range(self.pop_size):
                self.pop.append(self.search_space.sample(self.task, True))
            self.fitness_history = [[] for _ in range(self.pop_size)]
            self.solution_history = [[] for _ in range(self.pop_size)]
            self.population_history = {1: {}}
        else:
            self.pop = yaml.safe_load(open(self.exp_dir / 'population.yaml'))
            self.fitness_history = yaml.safe_load(open(self.exp_dir / 'history_fitness.yaml'))
            self.solution_history = yaml.safe_load(open(self.exp_dir / 'history_solution.yaml'))
            self.population_history = yaml.safe_load(open(self.exp_dir / 'history_population.yaml'))

    def run(self):
        while self.t_cur < self.t_max:
            print('---' * 10)
            print(f'Start tick {self.t_cur}')
            st = time.time()

            self.extend_population_history()

            self.tick()

            # saving state after tick so that continuing an interrupted experiment would work properly.
            self.save_state()

            tick_time = time.time() - st
            print(f'End tick {self.t_cur} | time: {tick_time :.2f} s')
            self.tick_times = pd.concat([self.tick_times,
                                         pd.DataFrame({'t': [self.t_cur], 'time': [tick_time]})],
                                        ignore_index=True)
            self.t_cur += self.t_step
            self.save_fitnesses_at_tick()

            save_yaml(self.t_cur, self.exp_dir / 'last_finished_tick.yaml')
            self.tick_times.to_csv(self.exp_dir / 'tick_times.csv', index=False)

        try:
            self.save_best()
        except Exception as e:
            print(f'Error saving best: {e}')
        plot_pop_history(self.exp_dir, self.exp_name, self.cfg.general.seed_offset)

        if self.delete_all_ckpts_at_the_end:
            for p in Path(self.cpkt_dir).glob('*.pt'):
                p.unlink()

    def save_state(self):
        save_yaml(self.population_history, self.exp_dir / 'history_population.yaml')
        save_yaml(self.pop, self.exp_dir / 'population.yaml')
        save_yaml(self.fitness_history, self.exp_dir / 'history_fitness.yaml')
        save_yaml(self.solution_history, self.exp_dir / 'history_solution.yaml')

    def extend_population_history(self):
        if type(self.pop) is list:  # a single population
            self.population_history[1][self.t_cur] = copy.deepcopy(self.pop)
        else:  # multiple populations
            for pop_id, subpop in self.pop.items():
                self.population_history[pop_id][self.t_cur] = copy.deepcopy(subpop)

    def save_fitnesses_at_tick(self):
        if type(self.fitness_history) is list:
            fitnesses = {1: [h[-1] for h in self.fitness_history]}
        else:
            fitnesses = {pop_id: [h[-1] for h in history] if len(history[0]) > 0 else [] for pop_id, history in
                         self.fitness_history.items()}
        fitnesses_at_tick_dir = self.exp_dir / 'fitnesses_at_tick'
        fitnesses_at_tick_dir.mkdir(parents=True, exist_ok=True)
        save_yaml(fitnesses, fitnesses_at_tick_dir / f'{self.t_cur:09d}.yaml')

    def tick(self):
        raise NotImplementedError

    @staticmethod
    @ray.remote(num_cpus=1, num_gpus=0.25)
    def _task_fn_ray(task, seed, *args):
        set_random_seeds(seed)
        return task(seed, *args)

    def prepare_initial_ckpt(self, ckpt_path, p):
        # need to create a checkpoint so that the evaluator population of FirePBT could start from that
        dict_to_save = self.task.prepare_initial_ckpt(p)
        torch.save(dict_to_save, ckpt_path)

    def save_best(self):
        # save best to yaml
        best_idx = int(np.argmax([self.fitness_history[i][-1] for i in range(self.pop_size)]))
        best = self.pop[best_idx]
        best_fitness = self.fitness_history[best_idx][-1]
        print(f'Best solution: {self.solution_history[best_idx]}')

        f = (self._task_fn_ray.options(**self.ray_options)
             .remote(self.task, self.seed_base, best, 0, 0,
                     torch.load(Path(self.cpkt_dir) / f'pop_{best_idx}_t{self.t_cur}.pt'),
                     None, ['test']))
        res = ray.get(f)
        with open(self.exp_dir / 'best_info.yaml', 'w') as f:
            yaml.safe_dump({'solution': best, 'fitness': best_fitness,
                            'fitness_history': self.fitness_history[best_idx],
                            'solution_history': self.solution_history[best_idx],
                            'solution_id': best_idx,
                            'test': res['test']
                            }, f)
        print(f'Val: {best_fitness:.4f}, Test: {res["test"]:.4f}')

        shutil.copy(Path(self.cpkt_dir) / f'pop_{best_idx}_t{self.t_cur}.pt',
                    Path(self.exp_dir) / 'best_model.pt')

        if 'policy_gif' in res:
            with open(Path(self.exp_dir) / 'policy.webp', 'wb') as f:
                f.write(res['policy_gif'])

        set_plot_style()

        # plot all fitnesses, with the best one highlighted
        plt.figure(figsize=(8, 5))
        for i in range(self.pop_size):
            linewidth = 1
            if i == best_idx:
                linewidth = 6
            plt.plot(range(0, self.t_max, self.t_step), self.fitness_history[i],
                     label=f'{i}' if i == best_idx else None,
                     linewidth=linewidth)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.xlabel('t')
        plt.ylabel('fitness')
        plt.title(f'{self.exp_dir.name}: fitness')
        plt.tight_layout()
        plt.savefig(self.exp_dir / 'fitness_history.png')
        plt.show()
        plt.close()
