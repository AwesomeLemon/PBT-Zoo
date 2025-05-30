import operator
import pickle

import copy
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import ray
import torch
import yaml
from matplotlib import pyplot as plt

from algo.base import BaseAlgo
from algo.firepbt_utils import smooth, best_score_diff, binom_test, find_overlap
from algo.pbt_utils import save_explored_ckpt_to_path
from utils.util_fns import (solution_history_to_str, save_yaml,
                            convert_from_logarithmic, convert_to_logarithmic, set_plot_style)
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


class FirePBT(BaseAlgo):
    """
    Faster Improvement Rate Population Based Training
    https://arxiv.org/abs/2109.13800
    """
    def __init__(self, cfg, search_space, task, **__):
        self.pop_size = {1: cfg.algo.pop1_size, 2: cfg.algo.pop2_size}
        self.pop_size_eval = cfg.algo.popEval_size

        super().__init__(cfg, search_space, task)  # will call init_pop among other things

        self.quant_top = cfg.algo.quant_top
        self.quant_bottom = cfg.algo.quant_bottom
        self.perturb_factor = cfg.algo.perturb_factor
        self.max_eval_steps = cfg.algo.max_eval_steps
        self.min_steps_before_eval = cfg.algo.min_steps_before_eval  # should have value for each sub-population
        self.p_stat = 0.01

    def _init_pop(self):
        if not self.cfg.general.continue_auto:
            self.pop = {}
            self.solution_history = {}
            self.fitness_history = {}
            self.curves = {}  # related to fitness_history but for simplicity keep separate for now
            self.time_since_eval = {}
            self.just_perturbed = {}  # when evaluators choose the best performer from prev iteration,
            #                           they may choose the perturbed version - this needs to be avoided.
            for i in self.pop_size.keys():
                self.pop[i] = []
                self.solution_history[i] = []
                self.fitness_history[i] = []
                self.curves[i] = {}
                self.just_perturbed[i] = []
                for j in range(self.pop_size[i]):
                    self.pop[i].append(self.search_space.sample(self.task, True))
                    self.solution_history[i].append([])
                    self.fitness_history[i].append([])
                    self.curves[i][j] = []
                    if i != 1:  # evaluators are not responsbile for P1, for other populations init with 0
                        self.time_since_eval[(i, j)] = 0
            self.population_history = {1: {}, 2: {}}

            # evaluator curves
            self.curves['eval'] = {}
            for i in range(self.pop_size_eval):
                self.curves['eval'][i] = []

            # evaluator state
            self.ev_states = {}
            for i in range(self.pop_size_eval):
                self.ev_states[i] = self._init_ev_state()

            # evaluator pop history - needed to set pop history if evaluator succeeds
            self.solution_history['eval'] = []
            for i in range(self.pop_size_eval):
                self.solution_history['eval'].append([])

            self.evaluating = set()
            # note that queue is needed for normal populations except the first one

        else:
            self.pop = yaml.safe_load(open(self.exp_dir / 'population.yaml'))
            self.fitness_history = yaml.safe_load(open(self.exp_dir / 'history_fitness.yaml'))
            self.solution_history = yaml.safe_load(open(self.exp_dir / 'history_solution.yaml'))
            self.population_history = yaml.safe_load(open(self.exp_dir / 'history_population.yaml'))
            self.curves = yaml.safe_load(open(self.exp_dir / 'curves.yaml'))
            self.time_since_eval = pickle.load(open(self.exp_dir / 'time_since_eval.pkl', 'rb'))
            self.ev_states = yaml.safe_load(open(self.exp_dir / 'ev_states.yaml'))
            self.evaluating = pickle.load(open(self.exp_dir / 'evaluating.pkl', 'rb'))
            self.just_perturbed = yaml.safe_load(open(self.exp_dir / 'just_perturbed.yaml'))

        self.curves_smooth = {}  # cache to save time on smoothing: maps curve -> smooth curve

    @staticmethod
    def _init_ev_state():
        return {'parent_pop': None, 'parent_id': None,
                'child_pop': None, 'child_id': None,
                'solution_being_evaluated': None,
                't_start': None,
                'status': 'idle'}  # idle, evaluating, stopped, success

    def tick(self):
        lost = {pop_idx: [False] * pop_size for pop_idx, pop_size in self.pop_size.items()}
        self.curves_smooth = {}
        # 1. start evaluations
        st = time.time()
        results = self._schedule_all_populations()
        self.train_and_eval_times = pd.concat([self.train_and_eval_times,
                                               pd.DataFrame({'t': [self.t_cur], 'time': [time.time() - st]})],
                                              ignore_index=True)
        print(f'_schedule_all_populations time: {time.time() - st:.2f} s')

        # 2. save checkpoints
        ckpts_to_delete = self._save_checkpoints(results)
        # 3. update pop 1 - note that it can be also updated afterwards based on the results
        # of the evaluator population, though that only updates the weights, not hparams
        fitnesses_1 = []
        for i, res_dict in enumerate(results[1]):
            fitnesses_1.append(res_dict['fitness'])
        # # history
        fit_hist_cur = self.fitness_history[1]
        sol_hist_cur = self.solution_history[1]
        pop_cur = self.pop[1]
        pop_cur_size = self.pop_size[1]
        print(f'Updating population 1')
        for i, (fitness, p) in enumerate(zip(fitnesses_1, pop_cur)):
            fit_hist_cur[i].append(float(fitness))
            print(f'1_{i}: {solution_history_to_str(sol_hist_cur[i])}: {fitness:.4f}')

        # # update pop
        if (self.t_cur < self.t_max - self.t_step) \
                and (self.t_cur >= self.min_steps_before_eval[1]):
            st = time.time()
            # # get indices of top and bottom quantiles in the results
            idx_top = np.argsort(fitnesses_1)[-int(self.quant_top * pop_cur_size):]
            idx_bottom = np.argsort(fitnesses_1)[:int(self.quant_bottom * pop_cur_size)]

            self._exploit_and_explore(1, idx_top, idx_bottom, lost)
        # 4. update evaluator population & the rest of normal populations | if t_cur > min_steps_before_eval
        # Steps:
        #  - evaluators return curves [already handled above]
        #  - compute fitnesses for pop2
        #  - update intermediate populations (pop 2, 3..)
        #  - check stopping & success criteria

        if (self.t_cur < self.t_max - self.t_step) \
                and (self.t_cur >= self.min_steps_before_eval[2]):
            #  - compute fitnesses for pop2
            ids_pop2, fitnesses_and_ids_2 = self._compute_fitness_subpopulation()

            #  - update intermediate populations (pop 2, 3..)
            #  -- history
            fit_hist_cur = self.fitness_history[2]
            sol_hist_cur = self.solution_history[2]
            pop_cur = self.pop[2]
            pop_cur_size = self.pop_size[2]

            print(f'Updating population 2')
            for p2_id in range(pop_cur_size):
                if p2_id in ids_pop2:
                    fitness = [fi[0] for fi in fitnesses_and_ids_2 if fi[1] == p2_id][0]
                    fit_hist_cur[p2_id].append(float(fitness))
                    print(f'2_{p2_id}: {solution_history_to_str(sol_hist_cur[p2_id])}: {fitness:.4f}')
                else:
                    fit_hist_cur[p2_id].append(-1)
                    print(f'2_{p2_id}: {solution_history_to_str(sol_hist_cur[p2_id])}: NO FITNESS')

            #  -- update pop
            if self.t_cur < self.t_max - self.t_step:
                fai2_sorted = list(sorted(fitnesses_and_ids_2, key=lambda x: x[0]))
                idx_top = [x[1] for x in fai2_sorted[-int(self.quant_top * pop_cur_size):]]
                idx_bottom = [x[1] for x in fai2_sorted[:int(self.quant_bottom * pop_cur_size)]]
                self._exploit_and_explore(2, idx_top, idx_bottom, lost)

            #  - check stopping & success criteria
            self._check_evaluator_stop_and_success(lost)
        else:
            if self.t_cur == self.t_max - self.t_step:
                # add fake fitness - so the shape would be the same.
                for i in range(self.pop_size[2]):
                    self.fitness_history[2][i].append(-1)

        if (self.t_cur < self.t_max - self.t_step) \
                and (self.t_cur >= self.min_steps_before_eval[1]):
            self.update_times = pd.concat([self.update_times,
                                           pd.DataFrame({'t': [self.t_cur], 'time': [time.time() - st]})],
                                          ignore_index=True)

        for p in ckpts_to_delete:
            p.unlink()

        # save stuff not saved by basealgo
        save_yaml(self.curves, self.exp_dir / 'curves.yaml')
        pickle.dump(self.time_since_eval, open(self.exp_dir / 'time_since_eval.pkl', 'wb'))
        save_yaml(self.ev_states, self.exp_dir / 'ev_states.yaml')
        pickle.dump(self.evaluating, open(self.exp_dir / 'evaluating.pkl', 'wb'))
        save_yaml(self.just_perturbed, self.exp_dir / 'just_perturbed.yaml')
        self.update_times.to_csv(self.exp_dir / 'update_times.csv', index=False)
        self.train_and_eval_times.to_csv(self.exp_dir / 'train_and_eval_times.csv', index=False)

    def _schedule_all_populations(self):
        futures = {1: [], 2: [], 'eval': []}
        # # normal populations
        for i_pop, pop in self.pop.items():
            for i, p in enumerate(pop):
                ckpt_path = Path(self.cpkt_dir) / f'pop{i_pop}_{i}_t{self.t_cur}.pt'
                if self.t_cur == 0:
                    self.prepare_initial_ckpt(ckpt_path, p)
                ckpt_loaded = torch.load(ckpt_path)
                # print(f'Start task for individual {i_pop}_{i}')
                seed = self.seed_base * 100 + int(self.t_cur / self.t_max * 1e6) + i_pop * 100 + i
                tb_dir = Path(self.exp_dir) / 'tb' / f'pop{i_pop}_{i}'
                tb_dir.mkdir(parents=True, exist_ok=True)
                f = self._task_fn_ray.options(**self.ray_options).remote(self.task, seed, p, self.t_cur, self.t_step,
                                                                         ckpt_loaded, tb_dir, None)
                futures[i_pop].append(f)

                self.solution_history[i_pop][i].append(copy.deepcopy(p))

        # # evaluator population
        if (self.t_cur >= self.min_steps_before_eval[2]):
            time_since_eval_sorted = list(sorted(self.time_since_eval.items(), key=lambda x: -x[1])) # I probably should've just used deque.
            time_since_eval_sorted = [x for x in time_since_eval_sorted if x[0] not in self.evaluating]
            if self.t_cur == 0:
                # choose randomly but the same for everyone (for comparability between evaluator curves)
                child_id_random_per_pop = {pop_id: np.random.choice(self.pop_size[pop_id])
                                           for pop_id in self.pop_size.keys()}

            for i in range(self.pop_size_eval):
                if self.ev_states[i]['status'] == 'idle':
                    if len(time_since_eval_sorted) == 0:
                        print(f'All models are already being evaluated, evaluator {i} staying idle')
                    else:
                        pop_id, solution_id = time_since_eval_sorted.pop(0)[0]
                        print(f'Queue: popped ({pop_id}, {solution_id}) {time_since_eval_sorted}')

                        self.ev_states[i]['parent_pop'] = parent_pop = pop_id
                        self.ev_states[i]['parent_id'] = parent_id = solution_id
                        # to get child index, find the best value in the population with previous index:
                        self.ev_states[i]['child_pop'] = child_pop = pop_id - 1

                        if self.t_cur == 0:
                            self.ev_states[i]['child_id'] = child_id = child_id_random_per_pop[child_pop]
                        else:
                            self.ev_states[i]['child_id'] = child_id = np.argmax(
                                [h[-1] if j not in self.just_perturbed[child_pop] else np.NINF
                                 for j, h in enumerate(self.fitness_history[child_pop])]
                            ).item()

                        self.ev_states[i]['t_start'] = self.t_cur

                        # copy ckpt
                        # target ckpt must exist
                        ckpt_t = self.t_cur
                        # first, need to remove the previous Eval ckpt, if it exists
                        ckpt_path = Path(self.cpkt_dir) / f'popEval_{i}_t{ckpt_t}.pt'
                        if ckpt_path.exists():
                            ckpt_path.unlink()

                        shutil.copy(Path(self.cpkt_dir) / f'pop{parent_pop}_{parent_id}_t{ckpt_t}.pt',
                                    ckpt_path)

                        # copy hyperparameters from child
                        self.ev_states[i]['solution_being_evaluated'] = copy.deepcopy(self.pop[child_pop][child_id])

                        ckpt_loaded = torch.load(ckpt_path)
                        seed = self.seed_base * 100 + int(self.t_cur / self.t_max * 1e6) + 9 * 100 + i
                        tb_dir = Path(self.exp_dir) / 'tb' / f'popEval_{i}'
                        tb_dir.mkdir(parents=True, exist_ok=True)
                        f = self._task_fn_ray.options(**self.ray_options).remote(self.task, seed, self.ev_states[i][
                            'solution_being_evaluated'],
                                                                                 self.t_cur, self.t_step, ckpt_loaded,
                                                                                 tb_dir, None)
                        futures['eval'].append(f)

                        self.ev_states[i]['status'] = 'evaluating'
                        self.curves['eval'][i] = []
                        self.evaluating.add((parent_pop, parent_id))
                        self.time_since_eval[(parent_pop, parent_id)] = 0
                        self.solution_history['eval'][i] = copy.deepcopy(
                            self.solution_history[parent_pop][parent_id][:-1])
                        # the last value in solution_history[eval] should come from the child
                        self.solution_history['eval'][i].append(self.ev_states[i]['solution_being_evaluated'])
                else:
                    print(f'Evaluator {i}: continue')
                    assert self.ev_states[i]['status'] == 'evaluating'
                    ckpt_path = Path(self.cpkt_dir) / f'popEval_{i}_t{self.t_cur}.pt'
                    ckpt_loaded = torch.load(ckpt_path)

                    seed = self.seed_base * 100 + int(self.t_cur / self.t_max * 1e6) + 9 * 100 + i
                    tb_dir = Path(self.exp_dir) / 'tb' / f'popEval_{i}'
                    tb_dir.mkdir(parents=True, exist_ok=True)
                    f = self._task_fn_ray.options(**self.ray_options).remote(self.task, seed,
                                                                             self.ev_states[i]['solution_being_evaluated'],
                                                                             self.t_cur, self.t_step, ckpt_loaded,
                                                                             tb_dir, None)
                    futures['eval'].append(f)

                    # if continue with the same value, still need to add it to solution_history
                    self.solution_history['eval'][i].append(
                        copy.deepcopy(self.ev_states[i]['solution_being_evaluated']))

        results = {pop_name: ray.get(fs) for pop_name, fs in futures.items()}
        return results

    def _save_checkpoints(self, results):
        ckpts_to_delete = []  # first collect, then delete after everything is saved

        for pop_name, pop_results in results.items():
            if pop_name != 'eval':
                for i, res_dict in enumerate(pop_results):
                    ckpt_path = Path(self.cpkt_dir) / f'pop{pop_name}_{i}_t{self.t_cur + self.t_step}.pt'
                    torch.save(res_dict['dict_to_save'], ckpt_path)
                    self.curves[pop_name][i].extend(res_dict['curve'])
                    # print(f'Population {pop_name}: saved checkpoint: pop{pop_name}_{i}_t{self.t_cur}.pt')
                    if (pop_name != 1) and ((pop_name, i) not in self.evaluating):
                        self.time_since_eval[(pop_name, i)] += self.t_step

                    if self.delete_old_ckpts:
                        ckpts_to_delete.append(Path(self.cpkt_dir) / f'pop{pop_name}_{i}_t{self.t_cur}.pt')
            else:
                # Evaluator population is a special case:
                # Some evaluators may be idle, and then the numbering breaks down
                # E.g., if evaluator 1 is idle, the saved checkpoints for evaluators 0, 2 should still
                # have names 0.pt, 2.pt, and not 0.pt, 1.pt

                i = 0
                while len(pop_results) > 0:
                    res_dict = pop_results.pop(0)
                    while self.ev_states[i]['status'] != 'evaluating':
                        i += 1
                    ckpt_path = Path(self.cpkt_dir) / f'popEval_{i}_t{self.t_cur + self.t_step}.pt'
                    torch.save(res_dict['dict_to_save'], ckpt_path)
                    self.curves[pop_name][i].extend(res_dict['curve'])
                    # print(f'Evaluator {i}: saved checkpoint: popEval_{i}_t{self.t_cur}.pt')
                    if self.delete_old_ckpts:
                        ckpts_to_delete.append(Path(self.cpkt_dir) / f'popEval_{i}_t{self.t_cur}.pt')

                    i += 1

        return ckpts_to_delete

    def _get_smooth_curve(self, curve):
        c_vals_list = [x[1] for x in curve]
        c_tuple = tuple(c_vals_list)
        if c_tuple not in self.curves_smooth:
            self.curves_smooth[c_tuple] = smooth(c_vals_list, self.exp_dir)
        return self.curves_smooth[c_tuple]

    def _check_evaluator_stop_and_success(self, lost):
        # Steps:
        #  - check stopping criteria
        #  - check success criteria [not sure if this is independent of the step above, assume yes]
        #  - check stopping criteria again, now inspecting if one of the networks lost

        #  - check stopping criteria
        for ev_i, ev_s in self.ev_states.items():
            if ev_s['status'] == 'idle':
                continue
            print(f'Evaluator {ev_i}: Checking stopping criteria 1 and 2')
            T = self.t_cur + self.t_step - ev_s['t_start']
            curve_child = self.curves[ev_s['child_pop']][ev_s['child_id']]
            curve_ev = self.curves['eval'][ev_i]

            if len(curve_child) == 0:
                assert lost[ev_s['child_pop']][ev_s['child_id']]
                continue

            curve_child_smooth = self._get_smooth_curve(curve_child)
            curve_ev_smooth = self._get_smooth_curve(curve_ev)

            overlap_start, l = find_overlap(curve_ev_smooth, curve_child_smooth)
            if_no_overlap = overlap_start is None

            # criterion 1: The evaluator’s curve does not overlap with ρchild’s curve and T is greater than a hyperparameter max_eval_steps
            #  -- Note: the above implies that max_eval_steps is not a hard limit
            if T >= self.max_eval_steps and if_no_overlap:
                ev_s['status'] = 'stopped'
                print(f'Evaluator {ev_i}: Stopping criterion 1')

            # criterion 2: The evaluator’s curve does overlap with ρchild’s curve but binom_test(η, ρchild) is greater
            if ev_s['status'] != 'stopped':  # no need to check if already stopped
                if not if_no_overlap:
                    binom_test_res = binom_test(curve_ev, curve_child, curve_ev_smooth, curve_child_smooth)
                    print(f"Criterion 2: {T=} {binom_test_res=} {self.p_stat + max(0, 1 - T / self.max_eval_steps)=}")
                    ev_s['binom_test_res'] = float(binom_test_res)
                    if binom_test_res > self.p_stat + max(0, 1 - T / self.max_eval_steps):
                        ev_s['status'] = 'stopped'
                        print(f'Evaluator {ev_i}: Stopping criterion 2')

        #  - check success criteria [not sure if this is independent of the step above, assume yes]
        children_replaced_with_evaluators = set()  # if two evaluators succeed simultaneously,
        #                   and the first one is better than the second one, the second one should be stopped
        for ev_i, ev_s in self.ev_states.items():
            if ev_s['status'] == 'idle':
                continue
            print(f'Evaluator {ev_i}: Checking success criterion & stopping criterion 4')
            curve_child = self.curves[ev_s['child_pop']][ev_s['child_id']]
            curve_ev = self.curves['eval'][ev_i]

            if len(curve_child) == 0:
                assert lost[ev_s['child_pop']][ev_s['child_id']]
                print(f"Child was lost: {ev_s['child_pop']}_{ev_s['child_id']}")
                continue

            curve_child_smooth = self._get_smooth_curve(curve_child)
            curve_ev_smooth = self._get_smooth_curve(curve_ev)

            bsd = best_score_diff(curve_ev, curve_child, curve_ev_smooth, curve_child_smooth, self.exp_dir)
            if bsd > 0:
                binom_test_res = ev_s.get('binom_test_res', binom_test(curve_ev, curve_child,
                                                                       curve_ev_smooth, curve_child_smooth))
                if binom_test_res < self.p_stat:
                    ev_s['status'] = 'success'
                    children_replaced_with_evaluators.add((ev_s['child_pop'], ev_s['child_id']))
                    # need to copy the weights
                    ckpt_path = Path(
                        self.cpkt_dir) / f'pop{ev_s["child_pop"]}_{ev_s["child_id"]}_t{self.t_cur + self.t_step}.pt'
                    ckpt_path.unlink()
                    ckpt_chosen_path = Path(self.cpkt_dir) / f'popEval_{ev_i}_t{self.t_cur + self.t_step}.pt'
                    shutil.copy(ckpt_chosen_path, ckpt_path)

                    self.solution_history[ev_s["child_pop"]][ev_s["child_id"]] = copy.deepcopy(
                        self.solution_history['eval'][ev_i])

                    # also need to copy the curve: otherwise multiple evaluators that succeed simultaneously override each other without comparison
                    self.curves[ev_s['child_pop']][ev_s['child_id']] = copy.deepcopy(self.curves['eval'][ev_i])
                    print(f'Evaluator {ev_i}: Success criterion: '
                          f'Copied weights from [the evaluator copy of] {ev_s["parent_pop"]}_{ev_s["parent_id"]}'
                          f' to {ev_s["child_pop"]}_{ev_s["child_id"]}')
            if ev_s['status'] != 'success' and \
                    (ev_s['child_pop'], ev_s['child_id']) in children_replaced_with_evaluators:
                ev_s['status'] = 'stopped'
                print(f'Evaluator {ev_i}: Stopping criterion 4 (my): Child already replaced with a better evaluator')

            viz = False
            if viz:
                plt.plot([x[0] for x in curve_child], [x[1] for x in curve_child], 'x', color='black')
                plt.plot([x[0] for x in curve_child], [x for x in curve_child_smooth], color='black')
                plt.plot([x[0] for x in curve_ev], [x[1] for x in curve_ev], 'x', color='green')
                plt.plot([x[0] for x in curve_ev], [x for x in curve_ev_smooth], color='green')
                plt.title(f'Evaluator {ev_i} | {ev_s["status"]} | bsd={bsd:.6f} | {bsd > 0}')
                ev_pictures_dir = self.exp_dir / 'evaluator_viz'
                ev_pictures_dir.mkdir(exist_ok=True)
                plt.savefig(ev_pictures_dir / f't{self.t_cur:05d}_eval_{ev_i}.png')
                plt.close()

        #  - check stopping criteria again, now inspecting if one of the networks lost
        #  -- if the status is already 'stopped' or 'success', also set to 'idle'
        for ev_i, ev_s in self.ev_states.items():
            if ev_s['status'] == 'idle':
                continue
            print(f'Evaluator {ev_i}: Checking stopping criterion 3')
            if (ev_s['status'] in ['stopped', 'success']) \
                    or (lost[ev_s['child_pop']][ev_s['child_id']]) \
                    or (lost[ev_s['parent_pop']][ev_s['parent_id']]):
                if lost[ev_s['child_pop']][ev_s['child_id']]:
                    print(f'Evaluator {ev_i}: Stopping criterion 3: Child lost: {ev_s["child_pop"]}_{ev_s["child_id"]}')

                if lost[ev_s['parent_pop']][ev_s['parent_id']]:
                    print(
                        f'Evaluator {ev_i}: Stopping criterion 3: Parent lost: {ev_s["parent_pop"]}_{ev_s["parent_id"]}')

                self.evaluating.remove((ev_s['parent_pop'], ev_s['parent_id']))
                self.ev_states[ev_i] = self._init_ev_state()

    def _exploit_and_explore(self, pop_id, idx_top, idx_bottom, lost):
        fit_hist_cur = self.fitness_history[pop_id]
        sol_hist_cur = self.solution_history[pop_id]
        pop_cur = self.pop[pop_id]
        bounds_cont = self.search_space.get_bounds_cont()
        self.just_perturbed[pop_id] = []

        for i in range(len(pop_cur)):
            if i not in idx_bottom:
                continue
            # replace bottom
            lost[pop_id][i] = True
            self.just_perturbed[pop_id].append(i)
            chosen_idx = np.random.choice(idx_top)
            pop_cur[i] = copy.deepcopy(pop_cur[chosen_idx])
            # replace history
            fit_hist_cur[i] = copy.deepcopy(fit_hist_cur[chosen_idx])
            sol_hist_cur[i] = copy.deepcopy(sol_hist_cur[chosen_idx])
            self.curves[pop_id][i] = []  # "The training curve of ρchild starts from when it
            #                       last copied the weights of another member via an exploit-and-explore step"
            # perturb
            for i_var in range(self.search_space.n_vars):
                hp_name = self.search_space.get_hp_name(i_var)
                if hp_name in bounds_cont.keys():
                    op = operator.mul if np.random.rand() < 0.5 else operator.truediv

                    if hp_name.startswith('log'):
                        # if var is log, convert to normal space, perturb, and convert back
                        pop_cur[i][i_var] = convert_from_logarithmic(hp_name, pop_cur[i][i_var])
                    elif bounds_cont[hp_name][0] * self.perturb_factor > bounds_cont[hp_name][1]:
                        # if the bounds are so tight that perturbation factor only flips between min/max, normalize, perturb, denormalize
                        pop_cur[i][i_var] = (pop_cur[i][i_var] - bounds_cont[hp_name][0]) / (
                                bounds_cont[hp_name][1] - bounds_cont[hp_name][0])

                    pop_cur[i][i_var] = op(pop_cur[i][i_var], self.perturb_factor)

                    if hp_name.startswith('log'):
                        pop_cur[i][i_var] = convert_to_logarithmic(hp_name, pop_cur[i][i_var])
                    elif bounds_cont[hp_name][0] * self.perturb_factor > bounds_cont[hp_name][1]:
                        pop_cur[i][i_var] = pop_cur[i][i_var] * (
                                bounds_cont[hp_name][1] - bounds_cont[hp_name][0]) + bounds_cont[hp_name][0]

                    pop_cur[i][i_var] = float(np.clip(pop_cur[i][i_var], *bounds_cont[hp_name]))
                else:
                    resample_prob = 1.0  # this seems high but this is the standard mutation, as also used in PBT and PB2-Rand (in the PB2-Mix paper)
                    if np.random.rand() < resample_prob:
                        pop_cur[i][i_var] = self.search_space.sample()[i_var]
                    else:
                        pop_cur[i][i_var] = pop_cur[chosen_idx][i_var]

            if pop_id > 1:
                self.time_since_eval[(pop_id, i)] = -1  # lowest priority for evaluation immediately after exploration

            # replace checkpoint
            ckpt_path = Path(self.cpkt_dir) / f'pop{pop_id}_{i}_t{self.t_cur + self.t_step}.pt'
            ckpt_path.unlink()
            ckpt_chosen_path = Path(self.cpkt_dir) / f'pop{pop_id}_{chosen_idx}_t{self.t_cur + self.t_step}.pt'
            save_explored_ckpt_to_path(self.task, pop_cur[i], ckpt_chosen_path, ckpt_path)
            print(f'Replaced {pop_id}_{i} with {pop_id}_{chosen_idx}, '
                  f'unperturbed values {[f"{x:.2f}" for x in pop_cur[chosen_idx]]}, '
                  f'perturbed values {[f"{x:.2f}" for x in pop_cur[i]]}')

    def _compute_fitness_subpopulation(self):
        #  -- get curves of individuals who have evaluators assigned
        ids_pop2 = []
        curves_corresponding_eval = []
        for i, p in enumerate(self.pop[2]):
            if (2, i) in self.evaluating:
                ids_pop2.append(i)
                # get curve & id from evaluator:
                found = False
                for j, ev_state in self.ev_states.items():
                    if (ev_state['parent_pop'] == 2) and (ev_state['parent_id'] == i):
                        curves_corresponding_eval.append(self.curves['eval'][j])
                        found = True
                        break
                if not found:
                    raise ValueError('There should be a corresponding evaluator state')

        best_score_diffs_2 = defaultdict(lambda: defaultdict(lambda: 0))
        for i, i_id in enumerate(ids_pop2):
            for j, j_id in enumerate(ids_pop2):
                if i == j:
                    continue
                c_i_smooth = self._get_smooth_curve(curves_corresponding_eval[i])
                c_j_smooth = self._get_smooth_curve(curves_corresponding_eval[j])
                best_score_diffs_2[i_id][j_id] = best_score_diff(curves_corresponding_eval[i],
                                                                 curves_corresponding_eval[j],
                                                                 c_i_smooth, c_j_smooth, self.exp_dir)

                viz = False
                if viz:
                    plt.plot([x[0] for x in curves_corresponding_eval[i]], [x[1] for x in curves_corresponding_eval[i]],
                             'x', color='orange', label=str(i_id))
                    plt.plot([x[0] for x in curves_corresponding_eval[i]], [x for x in c_i_smooth], color='orange')
                    plt.plot([x[0] for x in curves_corresponding_eval[j]], [x[1] for x in curves_corresponding_eval[j]],
                             'x', color='blue', label=str(j_id))
                    plt.plot([x[0] for x in curves_corresponding_eval[j]], [x for x in c_j_smooth], color='blue')
                    plt.title(
                        f'{i_id} and {j_id} | bsd={best_score_diffs_2[i_id][j_id]:.6f} | {best_score_diffs_2[i_id][j_id] > 0}')
                    plt.legend()
                    pop2fit_pictures_dir = self.exp_dir / 'pop2fit_viz'
                    pop2fit_pictures_dir.mkdir(exist_ok=True)
                    plt.savefig(pop2fit_pictures_dir / f't{self.t_cur:05d}_{i_id}_{j_id}.png')
                    plt.close()

        fitnesses_and_ids_2 = []
        for i_id in ids_pop2:
            cur = 0
            for j_id in ids_pop2:
                if i_id == j_id:
                    continue
                cur += best_score_diffs_2[i_id][j_id]
            fitnesses_and_ids_2.append((cur, i_id))

        return ids_pop2, fitnesses_and_ids_2

    def save_best(self):
        # populations other than pop1 don't have proper fitness
        # => save best info & weights in pop1
        pop_id = 1
        best_idx = int(np.argmax([self.fitness_history[pop_id][i][-1] for i in range(self.pop_size[pop_id])]))
        best = self.pop[pop_id][best_idx]
        best_fitness = self.fitness_history[pop_id][best_idx][-1]
        print(f'Best solution: {self.solution_history[pop_id][best_idx]}')

        f = (self._task_fn_ray.options(**self.ray_options)
             .remote(self.task, self.seed_base, best, 0, 0,
                     torch.load(Path(self.cpkt_dir) / f'pop{pop_id}_{best_idx}_t{self.t_cur}.pt'),
                     None, ['test']))
        res = ray.get(f)
        with open(Path(self.exp_dir) / 'best_info.yaml', 'w') as f:
            yaml.safe_dump({'solution': best, 'fitness': best_fitness,
                            'fitness_history': self.fitness_history[pop_id][best_idx],
                            'solution_history': self.solution_history[pop_id][best_idx],
                            'solution_id': best_idx,
                            'test': res['test']
                            }, f)
        print(f'Val: {best_fitness:.4f}, Test: {res["test"]:.4f}')

        shutil.copy(Path(self.cpkt_dir) / f'pop{pop_id}_{best_idx}_t{self.t_cur}.pt',
                    Path(self.exp_dir) / 'best_model.pt')

        if 'policy_gif' in res:
            with open(Path(self.exp_dir) / 'policy.webp', 'wb') as f:
                f.write(res['policy_gif'])

        set_plot_style()

        # plot all fitnesses, with the best one highlighted
        plt.figure(figsize=(8, 5))
        for i in range(self.pop_size[pop_id]):
            linewidth = 1
            if i == best_idx:
                linewidth = 6
            plt.plot(range(0, self.t_max, self.t_step), self.fitness_history[pop_id][i], label=f'{i}',
                     linewidth=linewidth)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.xlabel('t')
        plt.ylabel('fitness')
        plt.title(f'{self.exp_dir.name}: fitness')
        plt.tight_layout()
        plt.savefig(Path(self.exp_dir) / 'fitness_history.png')
        plt.show()
        plt.close()
