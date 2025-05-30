import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ray
import yaml

from algo.pbt import PBT
from algo.pbt_utils import save_explored_ckpt_to_path
from utils.util_fns import solution_history_to_str, save_yaml

from algo.bgpbt_utils import normalize, copula_standardize, train_gp, MAX_CHOLESKY_SIZE, MIN_CUDA, _Casmo
import gpytorch
import torch
import logging
import pickle


class BGPBT(PBT):
    """
    Bayesian Generational Population-Based Training
    https://arxiv.org/abs/2207.09405

    This is a simplified version of the original code: https://github.com/xingchenwan/bgpbt
    I also referred to https://github.com/facebookresearch/how-to-autorl/blob/main/hydra_plugins/hydra_pbt_sweeper/hydra_bgt.py

    Note: the objective is maximized overall but in the bayesian optimization part it is minimized
          (for simplicity; otherwise, would have to change, e.g., lcb=>ucb)
    """

    def __init__(self, cfg, search_space, task, **__):
        super().__init__(cfg, search_space, task)
        self.data = pd.DataFrame()

        self.trial_ids = {}  # current trial ids
        self.trial_id_counter = 0

        # BG-PBT variables:
        self.config_space = search_space.cs
        self.n_init = cfg.algo.n_init
        self.verbose = True
        self.casmo = _Casmo(self.config_space,
                            n_init=self.n_init,
                            max_evals=1,  # Me: default value; also, influences nothing.
                            batch_size=None,  # this will be updated later. batch_size=None signifies initialisation
                            verbose=self.verbose,
                            ard=False,  # Me: default value, the only one used.
                            acq='lcb',  # Me: default value, the only one used.
                            use_standard_gp=False,  # Me: default value, the only one used.
                            time_varying=False)  # Me: default value, but it's overriden below
        self.patience = 15
        self.n_fail = 0
        self.n_distills = 0

        if self.cfg.general.continue_auto:
            self.data = pd.read_csv(Path(self.exp_dir) / 'data.csv',
                                    float_precision='round_trip')  # the code relies on df.groupby, which needs the exact float values.
            self.trial_ids, self.trial_id_counter = yaml.safe_load(open(self.exp_dir / 'active_trial_ids.yaml'))
            self.n_distills, self.n_fail = yaml.safe_load(open(self.exp_dir / 'bgpbt_state.yaml'))
            self.casmo = pickle.load(open(self.exp_dir / 'casmo.pkl', 'rb'))

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
            lst = [[self.trial_ids[i], self.t_cur + self.t_step] +
                   copy.deepcopy(p) +
                   [fitness, "bo", self.n_distills]]
            cols = (["Trial", "Time"] +
                    self.search_space.get_hp_names() +
                    ["Reward", "config_source", "n_distills"])
            entry = pd.DataFrame(lst, columns=cols)
            self.data = pd.concat([self.data, entry]).reset_index(drop=True)
            self.data.Trial = self.data.Trial.astype("str")

        # 4. update population
        if self.min_steps_before_eval <= self.t_cur < self.t_max - self.t_step:
            st = time.time()
            # 4.1 adjust trust region length before exploiting and exploring
            self.adjust_tr_length(True)  # False in BGPBT-Arch

            # 4.2 exploit & explore
            self._exploit_and_explore(fitnesses)

            # 4.3 potentially restart GP
            best_fitness = self.data[self.data['n_distills'] == self.n_distills].Reward.max()
            if self.data[(self.data.Time == (self.t_cur + + self.t_step)) &
                         (self.data['n_distills'] == self.n_distills)].Reward.max() == best_fitness:
                self.n_fail = 0
            else:
                self.n_fail += 1
            if self.n_fail >= self.patience:
                self.n_fail = 0
                print('n_fail reached patience. Restarting GP')
                self._restart()

            print(f'n_fail: {self.n_fail}')
            self.update_times = pd.concat([self.update_times,
                                           pd.DataFrame({'t': [self.t_cur], 'time': [time.time() - st]})],
                                          ignore_index=True)

        # 5. delete old checkpoints
        for p in ckpts_to_delete:
            p.unlink()

        # save
        self.data.to_csv(self.exp_dir / 'data.csv', index=False)
        save_yaml((self.trial_ids, self.trial_id_counter), self.exp_dir / 'active_trial_ids.yaml')
        save_yaml((self.n_distills, self.n_fail), self.exp_dir / 'bgpbt_state.yaml')
        pickle.dump(self.casmo, open(self.exp_dir / 'casmo.pkl', 'wb'))
        self.update_times.to_csv(self.exp_dir / 'update_times.csv', index=False)
        self.train_and_eval_times.to_csv(self.exp_dir / 'train_and_eval_times.csv', index=False)

    def _schedule_all_populations(self):
        futures = []
        futures_fitness_at_init = [] # PB2 builds GP based on difference from previous result => need results at 0
        if self.t_cur == 0:
            for i, p in enumerate(self.pop):
                ckpt_path = Path(self.cpkt_dir) / f'pop_{i}_t{self.t_cur}.pt'
                self.prepare_initial_ckpt(ckpt_path, p)
                f = self._task_fn_ray.options(**self.ray_options).remote(self.task, 0, p, 0, 0,
                                                                         torch.load(ckpt_path),
                                                                         None, ['val'])
                futures_fitness_at_init.append(f)

            for i, p, in enumerate(self.pop):
                fitness_at_init = ray.get(futures_fitness_at_init[i])['fitness']
                lst = [[str(self.trial_id_counter), 0] +
                       copy.deepcopy(p) +
                       [fitness_at_init, "random", self.n_distills]] # I don't implement extra BO sampling at init (for now?)
                cols = (["Trial", "Time"] +
                        self.search_space.get_hp_names() +
                        ["Reward", "config_source", "n_distills"])
                entry = pd.DataFrame(lst, columns=cols)
                self.data = pd.concat([self.data, entry]).reset_index(drop=True)
                self.data.Trial = self.data.Trial.astype("str")
                self.trial_ids[i] = str(self.trial_id_counter)
                self.trial_id_counter += 1

        for i, p in enumerate(self.pop):
            ckpt_path = Path(self.cpkt_dir) / f'pop_{i}_t{self.t_cur}.pt'

            ckpt_loaded = torch.load(ckpt_path)

            seed = self.seed_base * 100 + int(self.t_cur / self.t_max * 1e6) + i
            tb_dir = Path(self.exp_dir) / 'tb' / f'pop_{i}'
            tb_dir.mkdir(parents=True, exist_ok=True)
            f = self._task_fn_ray.options(**self.ray_options).remote(self.task, seed, p, self.t_cur, self.t_step,
                                                                     ckpt_loaded, tb_dir, None)
            futures.append(f)

        results = ray.get(futures)
        return results

    def _exploit_and_explore(self, fitnesses):
        idx_top = np.argsort(fitnesses)[-int(self.quant_top * self.pop_size):]
        idx_bottom = np.argsort(fitnesses)[:int(self.quant_bottom * self.pop_size)]
        # first populate current, only then start adding stuff
        current = []

        for i in range(len(self.pop)):
            if i not in idx_bottom:
                # add to current
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
        bounds_cont = self.search_space.get_bounds_cont(treat_int_as_cont=True)
        bounds_noncont = self.search_space.get_bounds_noncont(treat_int_as_cont=True)
        bounds = {}
        for hp_name in self.search_space.get_hp_names():  # to preserve order
            if hp_name in bounds_cont:
                bounds[hp_name] = bounds_cont[hp_name]
            else:
                # don't normalize categorical hps
                bounds[hp_name] = (0, 1)

        # <diff wrt PB2/> #####################################
        # if a reset happened, we will have only the just-evaluated data, and therefore no diff, so we sample randomly
        df = df[df['n_distills'] == self.n_distills]
        df_check_too_few_data = df.copy()
        df_check_too_few_data["t_change"] = df_check_too_few_data.groupby(["Trial"] + list(bounds.keys()))["Time"].diff()
        df_check_too_few_data = df_check_too_few_data[df_check_too_few_data["t_change"] > 0].reset_index(drop=True)
        if df_check_too_few_data.shape[0] == 0:
            values = self.search_space.sample()
            df = self.data
            new_T = df[df["Trial"] == str(base_trial_id)].iloc[-1, :]["Time"]
            new_Reward = df[df["Trial"] == str(base_trial_id)].iloc[-1, :].Reward

            lst = [[new_trial_id, new_T] + values + [new_Reward, "random", self.n_distills]]
            cols = ["Trial", "Time"] + list(bounds) + ["Reward", "config_source", "n_distills"]
            new_entry = pd.DataFrame(lst, columns=cols)

            self.data = pd.concat([self.data, new_entry]).reset_index(drop=True)
            return values

        df = df.copy()
        df["Reward"] = -df["Reward"]  # minimization
        ####################################### </diff wrt PB2>

        # At this point, df contains only the good n_distill
        # Group by trial ID and hyperparams.
        # Compute change in timesteps and reward.
        df["y"] = df.groupby(["Trial"] + list(bounds.keys()))["Reward"].diff()
        df["t_change"] = df.groupby(["Trial"] + list(bounds.keys()))["Time"].diff()

        # Delete entries without positive change in t. Me: there should be none (because sync)
        df = df[df["t_change"] > 0].reset_index(drop=True)
        df["R_before"] = df.Reward - df.y

        # Normalize the reward change by the update size.
        # For example if trials took diff lengths of time.
        df["y"] = df.y / df.t_change
        df = df[~df.y.isna()].reset_index(drop=True)
        df = df.sort_values(by="Time").reset_index(drop=True)
        # <diff wrt PB2/> 100 last points, not 1000 </diff wrt PB2> ###########
        df = df.iloc[-100:, :].reset_index(drop=True)

        # We need this to know the T and Reward for the weights.
        dfnewpoint = df[(df["Trial"] == str(base_trial_id)) & (df["Time"] == df["Time"].max())]
        if dfnewpoint.empty:
            print('There is a bug somewhere, print the variables (the code will fail in the "newpoint=" line)')
            print(f'{df=}')
            print(f'{base_trial_id=} {new_trial_id=} {self.t_cur=}')
            # save df as tmp csv for debugging
            df.to_csv(self.exp_dir / 'data_DEBUG.csv', index=False)

        # Now specify the dataset for the GP.
        y = np.array(df.y.values)

        # Meta data we keep -> episodes and reward.
        r_t = df[["R_before", "Time"]]
        hparams = df[bounds.keys()]
        # for categorical hyperparameters, need to go from value to its index
        # (and same for current)
        # (no normalization should be done)
        if current is not None:
            current = pd.DataFrame(current, columns=list(bounds.keys()))
        for hp_noncont in bounds_noncont.keys():
            hparams.loc[:, hp_noncont] = hparams[hp_noncont].apply(
                lambda x: self.search_space.get_idx_by_value(hp_noncont, x)
            )
            dfnewpoint.loc[:, hp_noncont] = dfnewpoint[hp_noncont].apply(
                lambda x: self.search_space.get_idx_by_value(hp_noncont, x)
            )
            if current is not None:
                current.loc[:, hp_noncont] = current[hp_noncont].apply(
                    lambda x: self.search_space.get_idx_by_value(hp_noncont, x)
                )
        if current is not None:
            # current contained categorical values (likely strings) => cont values also became strings
            # => need to convert them back to float
            for hp_cont in bounds_cont.keys():
                current[hp_cont] = current[hp_cont].astype(float)
            current = current.values
        X = pd.concat([hparams, r_t], axis=1).values  # unlike in PB2, concat to the end
        X_best = dfnewpoint.iloc[-1, :][list(bounds.keys()) + ["R_before", "Time"]].values
        r_t_current = np.tile(dfnewpoint[["R_before", "Time"]].values, (current.shape[0], 1))

        current = np.hstack([current, r_t_current]).astype(float)
        new = _select_config(self.casmo, X, y, current, X_best, bounds, self.verbose)
        values = [fn_(new_) for fn_, new_ in
                  zip(self.search_space.get_fns_to_convert_from_encoding(treat_int_as_cont=True), new)]

        df["Reward"] = -df["Reward"] # we minimized in BO, but actually we maximize
        new_T = df[df["Trial"] == str(base_trial_id)].iloc[-1, :]["Time"]
        new_Reward = df[df["Trial"] == str(base_trial_id)].iloc[-1, :].Reward

        lst = [[new_trial_id] + [new_T] + values + [new_Reward, "bo",
                                                    self.n_distills]]
        cols = ["Trial", "Time"] + list(bounds) + ["Reward", "config_source", "n_distills"]
        new_entry = pd.DataFrame(lst, columns=cols)

        # Create an entry for the new config, with the reward from the
        # copied agent.
        self.data = pd.concat([self.data, new_entry]).reset_index(drop=True)

        return values

    def adjust_tr_length(self, restart=False):
        """Adjust trust region size -- the criterion is that whether any config sampled by BO outperforms the other config
        sampled otherwise (e.g. randomly, or carried from previous timesteps). If true, then it will be a success or
        failure otherwise."""
        agents = self.data[self.data.Time == (self.t_cur + self.t_step)]
        # get the negative reward
        best_reward = np.max(agents.Reward.values)
        # get the agents selected by Bayesian optimization
        bo_agents = agents[agents.config_source == 'bo']
        if bo_agents.shape[0] == 0:
            return

        # if the best reward is caused by a config suggested by BayesOpt
        if np.max(bo_agents.Reward.values) == best_reward:
            self.casmo.succcount += 1
            self.casmo.failcount = 0
        else:
            self.casmo.failcount += 1
            self.casmo.succcount = 0

        if self.casmo.succcount == self.casmo.succtol:  # Expand trust region
            self.casmo.length = min(
                [self.casmo.tr_multiplier * self.casmo.length, self.casmo.length_max])
            self.casmo.length_cat = min(
                self.casmo.length_cat * self.casmo.tr_multiplier, self.casmo.length_max_cat)
            self.casmo.succcount = 0
            logging.info(f'Expanding TR length to {self.casmo.length}')
        elif self.casmo.failcount == self.casmo.failtol:  # Shrink trust region
            self.casmo.failcount = 0
            self.casmo.length_cat = max(
                self.casmo.length_cat / self.casmo.tr_multiplier, self.casmo.length_min_cat)
            self.casmo.length = max(
                self.casmo.length / self.casmo.tr_multiplier, self.casmo.length_min)
            logging.info(f'Shrinking TR length to {self.casmo.length}')

        if restart and (self.casmo.length <= self.casmo.length_min
                        or self.casmo.length_max_cat <= self.casmo.length_min_cat):
            self._restart()

    def _restart(self):
        print('Restarting!')
        self.n_distills += 1  # this will cause the GP to reset in the next iteration
        self.casmo.length = self.casmo.length_init
        self.casmo.length_cat = self.casmo.length_init_cat
        self.casmo.failcount = self.casmo.succcount = 0


def _select_config(
        casmo: _Casmo,
        Xraw: np.array,
        yraw: np.array,
        current: np.array,
        X_best_raw: np.array,
        bounds: dict,
        verbose: bool
) -> np.ndarray:
    st = time.time()

    base_vals = np.array(list(bounds.values())).T.astype(np.float32)
    # Me: if min == max, get nan. Therefore, +-eps, same as in PB2-Mix implementation
    max_is_min = base_vals[0] == base_vals[1]
    base_vals[0][max_is_min] -= 1e-8
    base_vals[1][max_is_min] += 1e-8
    num_f = 2  # reward, time
    oldpoints = Xraw[:, -num_f:]
    old_lims = np.concatenate(
        (np.max(oldpoints, axis=0), np.min(oldpoints, axis=0))
    ).reshape(2, oldpoints.shape[1])
    # Me: if min == max, get nan. Therefore, +-eps, same as in PB2-Mix implementation.
    # Note that here max is index 0, min is 1. This inconsistency is crazy but
    # I keep it to avoid introducing unnecessary changes wrt official implementations.
    max_is_min = old_lims[0] == old_lims[1]
    old_lims[0][max_is_min] += 1e-8
    old_lims[1][max_is_min] -= 1e-8
    limits = np.concatenate((base_vals, old_lims), axis=1)

    X = normalize(Xraw, limits).astype(float)
    X_current = normalize(current, limits).astype(float)
    x_center = normalize(X_best_raw, limits).astype(float)

    X, t = X[:, :-1], X[:, -1]
    X_current, t_current = X_current[:, :-1], X_current[:, -1]
    x_center = x_center[:-1]

    # add fake "batch" dimension to x_center
    x_center = x_center.reshape(1, -1)

    y = yraw  # no standardizing yet

    hypers = {}
    use_time_varying_gp = np.unique(t).shape[0] > 1

    if X_current is not None:  # should be always True in the sync setup
        # 2. Train a GP conditioned on the *real* data which would give us the fantasised y output for the pending fixed_points
        y = copula_standardize(copy.deepcopy(y).ravel())
        if len(X) < MIN_CUDA:
            device, dtype = torch.device("cpu"), torch.float32
        else:
            device, dtype = casmo.device, casmo.dtype

        with gpytorch.settings.max_cholesky_size(MAX_CHOLESKY_SIZE):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            # here we replace the nan values with zero, but record the nan locations via the X_torch_nan_mask
            y_torch = torch.tensor(y).to(device=device, dtype=dtype)
            # add some noise to improve numerical stability
            y_torch += torch.randn(y_torch.size()) * 1e-5
            t_torch = torch.tensor(t).to(device=device, dtype=dtype)

            gp = train_gp(
                configspace=casmo.cs,
                train_x=X_torch,
                train_y=y_torch,
                use_ard=False,
                num_steps=200,
                time_varying=True if use_time_varying_gp else False,
                train_t=t_torch,
                verbose=verbose
            )
            hypers = gp.state_dict()

        # 3. Get the posterior prediction at the fantasised points
        gp.eval()
        if use_time_varying_gp:
            t_x_current = torch.hstack(
                (torch.tensor(t_current, dtype=dtype).reshape(-1, 1), torch.tensor(X_current, dtype=dtype)))
        else:
            t_x_current = torch.tensor(X_current, dtype=dtype)
        pred_ = gp(t_x_current).mean
        y_fantasised = pred_.detach().numpy()
        y = np.concatenate((y, y_fantasised))
        X = np.concatenate((X, X_current), axis=0)
        t = np.concatenate((t, t_current))
        del X_torch, y_torch, t_torch, gp

    y = copula_standardize(copy.deepcopy(y).ravel())
    next_config = casmo._create_and_select_candidates(X, y, length_cat=casmo.length_cat,
                                                      length_cont=casmo.length,
                                                      hypers=hypers, batch_size=1,
                                                      t=t if use_time_varying_gp else None,
                                                      time_varying=use_time_varying_gp,
                                                      x_center=x_center,
                                                      frozen_dims=None,
                                                      frozen_vals=None,
                                                      n_training_steps=1).flatten()

    next_config = next_config[:-1]  # remove reward

    # convert back...
    next_config = next_config * (np.max(base_vals, axis=0) - np.min(base_vals, axis=0)) + \
                  np.min(base_vals, axis=0)

    next_config = next_config.astype(np.float32)
    print(f'Total _select_config time: {time.time() - st:.2f} s')
    return next_config
