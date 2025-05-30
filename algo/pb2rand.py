import copy
import time
from copy import deepcopy
from pathlib import Path

import GPy
import numpy as np
import pandas as pd
import ray
import torch
import yaml

from algo.pb2_utils import normalize, standardize, TV_SquaredExp, optimize_acq, UCB, viz_gp
from algo.pbt import PBT
from algo.pbt_utils import save_explored_ckpt_to_path
from utils.util_fns import solution_history_to_str, save_yaml


class PB2Rand(PBT):
    """
    Population-Based Bandits
    https://arxiv.org/abs/2002.02518

    Based on https://github.com/jparkerholder/procgen_autorl/blob/main/pbt.py
    and https://github.com/ray-project/ray/blob/master/python/ray/tune/schedulers/pb2.py
    """

    def __init__(self, cfg, search_space, task, **__):
        super().__init__(cfg, search_space, task)
        self.data = pd.DataFrame()
        self.trial_ids = {}  # current trial ids
        self.trial_id_counter = 0

        if self.cfg.general.continue_auto:
            self.data = pd.read_csv(Path(self.exp_dir) / 'data.csv',
                                    float_precision='round_trip')  # the code relies on df.groupby, which needs the exact float values.
            self.trial_ids, self.trial_id_counter = yaml.safe_load(open(self.exp_dir / 'active_trial_ids.yaml'))

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

            # update self.data
            lst = [[self.trial_ids[i], self.t_cur + self.t_step] + copy.deepcopy(p) + [fitness]]
            cols = ["Trial", "Time"] + self.search_space.get_hp_names() + ["Reward"]
            entry = pd.DataFrame(lst, columns=cols)
            self.data = pd.concat([self.data, entry]).reset_index(drop=True)
            self.data.Trial = self.data.Trial.astype("str")

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
        self.data.to_csv(self.exp_dir / 'data.csv', index=False)
        save_yaml((self.trial_ids, self.trial_id_counter), self.exp_dir / 'active_trial_ids.yaml')
        self.update_times.to_csv(self.exp_dir / 'update_times.csv', index=False)
        self.train_and_eval_times.to_csv(self.exp_dir / 'train_and_eval_times.csv', index=False)

    def _schedule_all_populations(self):
        futures = []
        futures_fitness_at_init = []  # PB2 builds GP based on difference from previous result => need results at 0
        for i, p in enumerate(self.pop):
            ckpt_path = Path(self.cpkt_dir) / f'pop_{i}_t{self.t_cur}.pt'
            if self.t_cur == 0:
                self.prepare_initial_ckpt(ckpt_path, p)
                f = self._task_fn_ray.options(**self.ray_options).remote(self.task, 0, p, 0, 0, torch.load(ckpt_path),
                                                                         None, ['val'])
                futures_fitness_at_init.append(f)

            ckpt_loaded = torch.load(ckpt_path)

            seed = self.seed_base * 100 + int(self.t_cur / self.t_max * 1e6) + i
            tb_dir = Path(self.exp_dir) / 'tb' / f'pop_{i}'
            tb_dir.mkdir(parents=True, exist_ok=True)
            f = self._task_fn_ray.options(**self.ray_options).remote(self.task, seed, p, self.t_cur, self.t_step,
                                                                     ckpt_loaded, tb_dir, None)
            futures.append(f)

        if self.t_cur == 0:
            for i, p, in enumerate(self.pop):
                fitness_at_init = ray.get(futures_fitness_at_init[i])['fitness']
                lst = [[str(self.trial_id_counter), 0] + copy.deepcopy(p) + [fitness_at_init]]
                cols = ["Trial", "Time"] + self.search_space.get_hp_names() + ["Reward"]
                entry = pd.DataFrame(lst, columns=cols)
                self.data = pd.concat([self.data, entry]).reset_index(drop=True)
                self.data.Trial = self.data.Trial.astype("str")
                self.trial_ids[i] = str(self.trial_id_counter)
                self.trial_id_counter += 1

        results = ray.get(futures)
        return results

    def _exploit_and_explore(self, fitnesses):
        idx_top = np.argsort(fitnesses)[-int(self.quant_top * self.pop_size):]
        idx_bottom = np.argsort(fitnesses)[:int(self.quant_bottom * self.pop_size)]
        # first add survivors to 'current', only then start adding new points
        current = []

        for i in range(len(self.pop)):
            if i not in idx_bottom:
                hp_values_cur = np.array(self.pop[i])
                hp_values_cur = hp_values_cur.reshape(1, -1)
                current.append(hp_values_cur)
                continue

        for i in range(len(self.pop)):
            if i not in idx_bottom:
                continue
            # replace bottom
            chosen_idx = np.random.choice(idx_top)
            self.pop[i] = copy.deepcopy(self.pop[chosen_idx])
            # replace history
            self.fitness_history[i] = copy.deepcopy(self.fitness_history[chosen_idx])
            self.solution_history[i] = copy.deepcopy(self.solution_history[chosen_idx])

            # <explore/> #####################################
            new_trial_id = str(self.trial_id_counter)
            current_stacked = np.concatenate(current, axis=0) if len(current) > 0 else None
            new_hp_values = self.explore(self.trial_ids[chosen_idx], new_trial_id, current_stacked)
            self.pop[i] = new_hp_values
            current.append(np.array(new_hp_values).reshape(1, -1))
            self.trial_ids[i] = new_trial_id
            self.trial_id_counter += 1
            ####################################### </explore>

            # replace checkpoint
            ckpt_path = Path(self.cpkt_dir) / f'pop_{i}_t{self.t_cur + self.t_step}.pt'
            ckpt_path.unlink()
            ckpt_chosen_path = Path(self.cpkt_dir) / f'pop_{chosen_idx}_t{self.t_cur + self.t_step}.pt'
            save_explored_ckpt_to_path(self.task, self.pop[i], ckpt_chosen_path, ckpt_path)
            print(f'Replaced {i}  with {chosen_idx}, '
                  f'unperturbed values {self.pop[chosen_idx]}, perturbed values {self.pop[i]}')

    def explore(self, base_trial_id, new_trial_id, current):
        df = self.data.sort_values(by="Time").reset_index(drop=True)
        bounds_cont = self.search_space.get_bounds_cont()
        hp_names = self.search_space.get_hp_names()

        # Group by trial ID and hyperparams.
        # Compute change in timesteps and reward.
        df["y"] = df.groupby(["Trial"] + hp_names)["Reward"].diff()
        df["t_change"] = df.groupby(["Trial"] + hp_names)["Time"].diff()

        # Delete entries without positive change in t. Me: there should be none (because sync)
        df = df[df["t_change"] > 0].reset_index(drop=True)
        df["R_before"] = df.Reward - df.y

        # Normalize the reward change by the update size.
        # For example if trials took diff lengths of time.
        df["y"] = df.y / df.t_change
        df = df[~df.y.isna()].reset_index(drop=True)
        df = df.sort_values(by="Time").reset_index(drop=True)

        # Only use the last 1k datapoints, so the GP is not too slow.
        df = df.iloc[-1000:, :].reset_index(drop=True)

        # We need this to know the T and Reward for the weights.
        dfnewpoint = df[df["Trial"] == str(base_trial_id)] # should never be empty in our sync case
        if dfnewpoint.empty:
            print('there is a bug somewhere, print the variables (the code will fail in the "newpoint=" line)')
            print(f'{df=}')
            print(f'{base_trial_id=} {new_trial_id=} {self.t_cur=}')

        # Now specify the dataset for the GP.
        y = np.array(df.y.values)
        # Meta data we keep -> episodes and reward.
        t_r = df[["Time", "R_before"]]

        cont_cat_ord_types = self.search_space.get_cont_cat_ord_types()
        cont_names = [hp for hp, type_ in zip(hp_names, cont_cat_ord_types) if type_ == 'cont']
        cat_names = [hp for hp, type_ in zip(hp_names, cont_cat_ord_types) if type_ == 'cat']
        ord_names = [hp for hp, type_ in zip(hp_names, cont_cat_ord_types) if type_ == 'ord']
        if len(ord_names) > 0:
            raise ValueError('Ordinal hyperparameters are not supported in PB2-Mix')

        hparams_cont = df[cont_names]
        if current is not None:
            cont_indices = [hp_names.index(cont_name) for cont_name in cont_names]
            current = current[:, cont_indices].astype(float)  # if there were strings in current,
            #                                                   float also became strings => cast back to float.

        cat_values_explored = {}  # randomly sample categorical variables first
        for cat_name in cat_names:
            cat_values_explored[cat_name] = self.search_space.sample_dict()[cat_name]

        X = pd.concat([t_r, hparams_cont], axis=1).values
        newpoint = df[df["Trial"] == str(base_trial_id)].iloc[-1, :][["Time", "R_before"]].values.astype(X.dtype)
        num_f = len(t_r.columns)
        new = _select_config(X, y, current, newpoint, bounds_cont, num_f=num_f)

        cont_values_explored_list = [float(new_) for new_ in new]  # these are guaranteed to be continuous parameters
        cont_values_explored = {cont_name: cont_values_explored_list[i] for i, cont_name in enumerate(cont_names)}

        values = []
        for hp_name in hp_names:
            if hp_name in cont_names:
                values.append(cont_values_explored[hp_name])
            else:
                values.append(cat_values_explored[hp_name])

        new_T = df[df["Trial"] == str(base_trial_id)].iloc[-1, :]["Time"]
        new_Reward = df[df["Trial"] == str(base_trial_id)].iloc[-1, :].Reward

        lst = [[new_trial_id] + [new_T] + values + [new_Reward]]
        cols = ["Trial", "Time"] + hp_names + ["Reward"]
        new_entry = pd.DataFrame(lst, columns=cols)

        # Create an entry for the new config, with the reward from the
        # copied agent.
        self.data = pd.concat([self.data, new_entry]).reset_index(drop=True)

        return values


def _select_config(
        Xraw: np.array,
        yraw: np.array,
        current: np.array,
        newpoint: np.array,
        bounds: dict,
        num_f: int,
) -> np.ndarray:
    """Selects the next hyperparameter config to try.

    This function takes the formatted data, fits the GP model and optimizes the
    UCB acquisition function to select the next point.

    Args:
        Xraw: The un-normalized array of hyperparams, Time and
            Reward
        yraw: The un-normalized vector of reward changes.
        current: The hyperparams of trials currently running. This is
            important so we do not select the same config twice. If there is
            data here then we fit a second GP including it
            (with fake y labels). The GP variance doesn't depend on the y
            labels so it is ok.
        newpoint: The Reward and Time for the new point.
            We cannot change these as they are based on the *new weights*.
        bounds: Bounds for the hyperparameters. Used to normalize.
        num_f: The number of fixed params. Almost always 2 (reward+time)

    Return:
        xt: A vector of new hyperparameters.
    """
    st = time.time()
    # Follow the PB2-Mix implementation in not selecting length.
    # length = select_length(Xraw, yraw, bounds, num_f)
    # print(f'Length selection time: {time.time() - st:.2f} s')
    # Xraw = Xraw[-length:, :]
    # yraw = yraw[-length:]

    base_vals = np.array(list(bounds.values())).T.astype(np.float32)
    # Me: if min == max, get nan. Therefore, +-eps, same as in PB2-Mix implementation
    max_is_min = base_vals[0] == base_vals[1]
    base_vals[0][max_is_min] -= 1e-8
    base_vals[1][max_is_min] += 1e-8

    oldpoints = Xraw[:, :num_f]
    old_lims = np.concatenate(
        (np.max(oldpoints, axis=0), np.min(oldpoints, axis=0))
    ).reshape(2, oldpoints.shape[1])
    # Me: if min == max, get nan. Therefore, +-eps, same as in PB2-Mix implementation.
    # Note that here max is 0, min is 1. This inconsistency is crazy but
    # I keep it to avoid introducing unnecessary changes wrt official implementations.
    max_is_min = old_lims[0] == old_lims[1]
    old_lims[0][max_is_min] += 1e-8
    old_lims[1][max_is_min] -= 1e-8

    limits = np.concatenate((old_lims, base_vals), axis=1)

    X = normalize(Xraw, limits)
    y = standardize(yraw).reshape(yraw.size, 1)

    fixed = normalize(newpoint, old_lims)

    # Me: add noise to duplicates to prevent singular matrices in GP
    X_without_fixed = X[:, num_f:]
    _, indices = np.unique(X_without_fixed, axis=0, return_index=True)
    duplicates = np.setdiff1d(np.arange(X_without_fixed.shape[0]), indices)
    if duplicates.size > 0:
        print(f'found duplicates! {len(duplicates)=} {X.shape=}')
        X[duplicates, num_f:] += 1e-4 * np.abs(np.random.randn(len(duplicates), X.shape[1] - (num_f)))

    kernel = TV_SquaredExp(
        input_dim=X.shape[1], variance=1.0, lengthscale=1.0, epsilon=0.1
    )

    m = GPy.models.GPRegression(X, y, kernel)
    # the try-except clauses that were used here in the original codebase make
    # no sense because X is not square-shaped, and in any case the LinAlgError is caused by a different matrix
    # Instead, I add noise to duplicates (above and below)

    st_m = time.time()
    m.optimize(messages=True, ipython_notebook=False)
    print(f'Optimize GP time [m]: {time.time() - st_m:.2f} s')

    m.kern.lengthscale.fix(m.kern.lengthscale.clip(1e-5, 1))

    if current is None:
        m1 = deepcopy(m)
    else:
        # add the current trials to the dataset
        padding = np.tile(fixed, (current.shape[0], 1))
        current = normalize(current, base_vals)
        current = np.hstack((padding, current))

        Xnew = np.vstack((X, current))
        ypad = np.zeros(current.shape[0])
        ypad = ypad.reshape(-1, 1)
        ynew = np.vstack((y, ypad))

        # Me: add noise to duplicates to prevent singular matrices in GP
        Xnew_without_fixed = Xnew[:, num_f:]
        _, indices = np.unique(Xnew_without_fixed, axis=0, return_index=True)
        duplicates = np.setdiff1d(np.arange(Xnew_without_fixed.shape[0]), indices)
        if duplicates.size > 0:
            print(f'found duplicates! {len(duplicates)=} {Xnew.shape=}')
            Xnew[duplicates, num_f:] += 1e-4 * np.abs(np.random.randn(len(duplicates), Xnew.shape[1] - (num_f)))

        kernel = TV_SquaredExp(
            input_dim=X.shape[1], variance=1.0, lengthscale=1.0, epsilon=0.1
        )
        m1 = GPy.models.GPRegression(Xnew, ynew, kernel)
        st_m1 = time.time()
        m1.optimize(messages=True, ipython_notebook=False)
        print(f'Optimize GP time [m1]: {time.time() - st_m1:.2f} s')

    st_acq = time.time()
    xt = optimize_acq(UCB, m, m1, fixed, num_f)
    print(f'Optimize acq time: {time.time() - st_acq:.2f} s')

    if_viz = False  # doesn't work for >1 variable; useful for debugging.
    if if_viz:
        viz_gp(X, Xraw, base_vals, current, fixed, limits, m, m1, num_f, xt, y)

    # convert back...
    xt = xt * (np.max(base_vals, axis=0) - np.min(base_vals, axis=0)) + np.min(
        base_vals, axis=0
    )

    xt = xt.astype(np.float32)
    print(f'Total _select_config time: {time.time() - st:.2f} s')
    return xt
