import time
from copy import deepcopy

import GPy
import numpy as np
import pandas as pd

from algo.pb2_utils import normalize, standardize, optimize_acq, UCB, TV_MixtureViaSumAndProduct
from algo.pb2mix_exp3 import exp3_get_cat
from algo.pb2rand import PB2Rand


class PB2Mix(PB2Rand):
    """
    Population-Based-Bandits-Mix
    https://arxiv.org/abs/2106.15883

    Based on https://github.com/jparkerholder/procgen_autorl/blob/main/pbt.py
    """

    def __init__(self, cfg, search_space, task, **__):
        super().__init__(cfg, search_space, task)

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

        # <diff wrt PB2-Rand/> #####################################
        if len(cat_names) == 0:
            raise ValueError('Categorical hyperparameters are required in PB2-Mix')

        hparams = df[hp_names]
        # to normalize categorical hyperparameters properly, need to go from value to its index
        # (and same for current)
        if current is not None:
            current = pd.DataFrame(current, columns=hp_names)
        for hp_noncont in cat_names:
            hparams.loc[:, hp_noncont] = hparams[hp_noncont].apply(
                lambda x: self.search_space.get_idx_by_value(hp_noncont, x)
            )
            if current is not None:
                current.loc[:, hp_noncont] = current[hp_noncont].apply(
                    lambda x: self.search_space.get_idx_by_value(hp_noncont, x)
                )
        if current is not None:
            # current contained categorical values (likely strings) => cont values also became strings
            # => need to convert them back to float
            for hp_cont in cont_names:
                current[hp_cont] = current[hp_cont].astype(float)
            current = current.values

        cat_values_explored = {}  # in CoCaBo, the categorical values are sampled first
        df_cat = df.copy()
        df_cat["y_exp3"] = normalize(df_cat['y'], df_cat['y'])
        df_cat[hp_names] = hparams[hp_names].copy()
        for cat_name in cat_names:
            # get all values for the current categorical valuable from 'current':
            pending_actions = current[:, hp_names.index(cat_name)] if current is not None else []
            num_rounds = self.t_max // self.t_step
            cat = exp3_get_cat(cat_name, list(range(len(self.search_space.get_choices_by_name(cat_name)))),
                               # categorical => has "choices", but I need their indices
                               df_cat, num_rounds, pending_actions, self.t_step)
            cat_values_explored[cat_name] = cat
        ####################################### </diff wrt PB2-Rand>

        X = pd.concat([t_r, hparams], axis=1).values

        # <diff wrt PB2-Rand/> #####################################
        newpoint = df_cat[df_cat["Trial"] == str(base_trial_id)].iloc[-1, :][
            ["Time", "R_before"] + cat_names].values.astype(X.dtype)

        cat_indices = [hp_names.index(cat_name) for cat_name in cat_names]
        cont_indices = [hp_names.index(cont_name) for cont_name in cont_names]
        if current is not None:
            current_cat = current[:, cat_indices]
            current_cont = current[:, cont_indices]
        else:
            current_cat, current_cont = None, None
        cat_indices_in_X = [t_r.shape[1] + c for c in cat_indices]
        cont_indices_in_X = [t_r.shape[1] + c for c in cont_indices]

        new = _select_config(X, y, current_cat, current_cont, newpoint, bounds_cont, len(t_r.columns),
                             cat_indices_in_X, cont_indices_in_X)

        cont_values_explored = {cont_name: new[i] for i, cont_name in enumerate(cont_names)}

        values = []
        for hp_name, fn_convert in zip(hp_names, self.search_space.get_fns_to_convert_from_encoding()):
            if hp_name in cont_names:
                values.append(fn_convert(cont_values_explored[hp_name]))
            else:
                values.append(fn_convert(cat_values_explored[hp_name]))

        ####################################### </diff wrt PB2-Rand>

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
        current_cat: np.array,
        current_cont: np.array,
        newpoint: np.array,
        bounds_cont: dict,
        num_f: int,
        cat_indices_in_X,
        cont_indices_in_X
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
        bounds_cont: Bounds for the continuous hyperparameters. Used to normalize.
        num_f: The number of fixed params. Almost always 2 (reward+time)

    Return:
        xt: A vector of new hyperparameters.
    """
    st = time.time()
    # no length selection here: https://github.com/jparkerholder/procgen_autorl/blob/main/pbt.py#L259

    X_cat = Xraw[:, cat_indices_in_X]
    X_cont = Xraw[:, cont_indices_in_X]
    oldpoints = Xraw[:, :num_f]
    Xraw_cont = np.concatenate((oldpoints, X_cont), axis=1)

    fixed_cat = newpoint[num_f:]  # cat variables
    newpoint = newpoint[:num_f]  # time, reward

    base_vals = np.array(list(bounds_cont.values())).T.astype(np.float32)
    # Me: if min == max, get nan. Therefore, +-eps, same as in the official PB2-Mix implementation
    max_is_min = base_vals[0] == base_vals[1]
    base_vals[0][max_is_min] -= 1e-8
    base_vals[1][max_is_min] += 1e-8

    fixed_points = np.concatenate((oldpoints, newpoint.reshape(1, -1)), axis=0)
    old_lims = np.concatenate(
        (np.max(fixed_points, axis=0), np.min(fixed_points, axis=0))
    ).reshape(2, oldpoints.shape[1])
    # Me: if min == max, get nan. Therefore, +-eps, same as in the official PB2-Mix implementation.
    # Note that here max is 0, min is 1. This inconsistency is crazy but
    # I keep it to avoid introducing unnecessary changes wrt official implementations.
    max_is_min = old_lims[0] == old_lims[1]
    old_lims[0][max_is_min] += 1e-8
    old_lims[1][max_is_min] -= 1e-8
    limits = np.concatenate((old_lims, base_vals), axis=1)

    X = normalize(Xraw_cont, limits)
    y = standardize(yraw).reshape(yraw.size, 1)

    fixed = normalize(newpoint, old_lims)

    X = np.concatenate((X[:, :num_f], X_cat, X[:, num_f:]), axis=1)

    # Me: add noise to duplicates to prevent singular matrices in GP
    X_without_fixed = X[:, num_f:]
    _, indices = np.unique(X_without_fixed, axis=0, return_index=True)
    duplicates = np.setdiff1d(np.arange(X_without_fixed.shape[0]), indices)
    if duplicates.size > 0:
        print(f'found duplicates! {len(duplicates)=} {X.shape=}')
        X[duplicates, num_f + len(cat_indices_in_X):] += 1e-4 * np.abs(
            np.random.randn(len(duplicates), X.shape[1] - (num_f + len(cat_indices_in_X))))

    kernel = TV_MixtureViaSumAndProduct(X.shape[1],
                                        variance_1=1.,
                                        variance_2=1.,
                                        variance_mix=1.,
                                        lengthscale=1.,
                                        epsilon_1=0.,
                                        epsilon_2=0.,
                                        mix=0.5,
                                        cat_dims=cat_indices_in_X)

    m = GPy.models.GPRegression(X, y, kernel)
    # the try-except clauses that were used here in the original codebase make
    # no sense because X is not square-shaped, and in any case the LinAlgError is caused by a different matrix
    # Instead, I add noise to duplicates (above and below)

    m.kern.lengthscale.constrain_positive()
    st_m = time.time()
    m.optimize()
    print(f'Optimize GP time [m]: {time.time() - st_m:.2f} s')

    m.kern.lengthscale.fix(m.kern.lengthscale.clip(1e-5, 1))

    if current_cont is None:  # doesn't matter which "current": they are both either None or not None.
        m1 = deepcopy(m)
    else:
        # add the current trials to the dataset
        # create padding in a proper numpy way:
        padding = np.tile(fixed, (current_cont.shape[0], 1))
        current_cont = normalize(current_cont, base_vals)

        current = np.concatenate((padding, current_cat, current_cont), axis=1)

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
            Xnew[duplicates, num_f + len(cat_indices_in_X):] += 1e-4 * np.abs(
                np.random.randn(len(duplicates), Xnew.shape[1] - (num_f + len(cat_indices_in_X))))

        kernel = TV_MixtureViaSumAndProduct(Xnew.shape[1],
                                            variance_1=1.,
                                            variance_2=1.,
                                            variance_mix=1.,
                                            lengthscale=1.,
                                            epsilon_1=0.,
                                            epsilon_2=0.,
                                            mix=0.5,
                                            cat_dims=cat_indices_in_X)

        m1 = GPy.models.GPRegression(Xnew, ynew, kernel)

        m1.kern.lengthscale.constrain_positive()

        st_m1 = time.time()
        m1.optimize()
        print(f'Optimize GP time [m1]: {time.time() - st_m1:.2f} s')
        m1.kern.lengthscale.fix(m1.kern.lengthscale.clip(1e-5, 1))

    fixed = np.concatenate((fixed.reshape(1, -1), fixed_cat.reshape(1, -1)), axis=1)
    st_acq = time.time()
    xt = optimize_acq(UCB, m, m1, fixed, num_f + len(cat_indices_in_X))
    print(f'Optimize acq time: {time.time() - st_acq:.2f} s')

    # convert back...
    if xt.shape[0] != base_vals.shape[1]:
        print(f'xt.shape[0] != base_vals.shape[1] {xt.shape=} {base_vals.shape=} {xt=} {base_vals=}')
    xt = xt * (np.max(base_vals, axis=0) - np.min(base_vals, axis=0)) + np.min(
        base_vals, axis=0
    )

    xt = xt.astype(np.float32)
    print(f'Total _select_config time: {time.time() - st:.2f} s')
    return xt
