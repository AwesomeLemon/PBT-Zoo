import copy
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, ticker
import yaml
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import os

PBT_ZOO_PATH = os.getenv("PBT_ZOO_PATH")
sys.path.insert(0, PBT_ZOO_PATH)

from utils.util_fns import convert_from_logarithmic

from plot.critdd_my import Diagrams

LOGS_DIR = f'{PBT_ZOO_PATH}/logs' if PBT_ZOO_PATH != "" else "logs"
palette = sns.color_palette("colorblind", 10)
algo_to_color = { # so that the algos have the same color in all plots
    'pbt': palette[0],
    'pb2rand': palette[1],
    'pb2mix': palette[3],
    'bgpbt': palette[4],
    'firepbt': palette[2],
}
algo_to_pretty_name = {
    'pbt': 'PBT',
    'pb2rand': 'PB2',
    'pb2mix': 'PB2-Mix',
    'bgpbt': 'BG-PBT',
    'firepbt': 'FIRE-PBT',
}

hp_to_pretty_name = {
    'lr': 'Learning rate'
}

def _get_fitness_at_ticks(seed_path):
    path = seed_path / 'fitnesses_at_tick'
    files = list(sorted([x for x in path.iterdir() if x.is_file()]))
    pop_to_fitnesses = {}
    ticks = []
    for file in files:
        loaded = yaml.safe_load(open(file, 'r'))
        for pop_id, fs in loaded.items():
            if pop_id not in pop_to_fitnesses:
                pop_to_fitnesses[pop_id] = []
            pop_to_fitnesses[pop_id].append(fs)

        tick = int(file.stem)
        ticks.append(tick)

    return pop_to_fitnesses, ticks

def _get_max_fitness_at_ticks(seed_path):
    # do it for subpop1 because that's always the proper fitnesses, even in FIRE
    pop_to_fitnesses, ticks = _get_fitness_at_ticks(seed_path)
    max_fitnesses = [max(fitnesses) for fitnesses in pop_to_fitnesses[1]]
    return max_fitnesses, ticks

def plot_fitness_at_ticks_avg_many_seeds_many_exps(exp_names, fitness_transform=lambda x: x, seeds=(),
                                                   yscale='log', ylim={'bottom': 85},
                                                   xlim={}, xlabel='', ylabel='', aggregate='mean', uncertainty='std', inset_params=None, figsize=(4, 4),
                                                   if_leftalign_xlabel=False, pad_inches=0.0, legend_on_the_side=False, legend_kwargs={},
                                                   xaxis_number_of_zeros=None,
                                                   yticks_pad=None, yticks_labelsize=None, force_n_yticks=None,
                                                   ):
    plt.figure(figsize=figsize)
    min_seeds = None

    ax = plt.gca()
    if inset_params:
        ax_inset = inset_axes(ax, width=f"{inset_params['width']}%", height=f"{inset_params['height']}%", loc='lower right')
        ax_inset.set_xlim(inset_params['xlim'])
        ax_inset.set_ylim(inset_params['ylim'])
        ax_inset.set_xticklabels([])
        ax_inset.set_yticklabels([])
        for spine in ax_inset.spines.values():
            spine.set_edgecolor('0.6')
            spine.set_linewidth(1)

    for exp_name in exp_names:
        exp_path = Path(LOGS_DIR) / exp_name

        seeds_cur = list(sorted([int(x.name) for x in exp_path.iterdir() if x.is_dir()]))
        seeds_cur = [x for x in seeds_cur if x in seeds]

        fitnesses_all_seeds = []
        for seed in seeds_cur:
            seed_path = exp_path / str(seed)
            fitnesses, ticks = _get_max_fitness_at_ticks(seed_path)
            fitnesses_all_seeds.append([fitness_transform(x) for x in fitnesses])

        fitnesses_all_seeds = np.array(fitnesses_all_seeds)
        if aggregate == 'mean':
            mean_fitnesses = np.mean(fitnesses_all_seeds, axis=0)
        elif aggregate == 'iqm': # inter-quartile mean
            q25 = np.quantile(fitnesses_all_seeds, 0.25, axis=0, method='nearest')
            q75 = np.quantile(fitnesses_all_seeds, 0.75, axis=0, method='nearest')
            mask = (fitnesses_all_seeds >= q25) & (fitnesses_all_seeds <= q75)
            interquartile_fitnesses = np.ma.array(fitnesses_all_seeds, mask=~mask)
            mean_fitnesses = interquartile_fitnesses.mean(axis=0).data
            lbs = q25
            ubs = q75

        if uncertainty == 'std':
            std_fitnesses = np.std(fitnesses_all_seeds, axis=0)
            lbs = mean_fitnesses - std_fitnesses
            ubs = mean_fitnesses + std_fitnesses
        elif uncertainty == 'iqr':
            pass  # already computed above

        algo_name = exp_name.split('_')[0]
        color_ = algo_to_color[algo_name]
        ax.plot(ticks,
                 mean_fitnesses,
                 color=color_, label=algo_to_pretty_name[algo_name])
        ax.fill_between(ticks,
                         lbs,
                         ubs,
                         color=color_, alpha=0.2)
        min_seeds = len(seeds_cur) if min_seeds is None else min(min_seeds, len(seeds_cur))

        if inset_params:
            ax_inset.plot(ticks, mean_fitnesses, color=color_)
            ax_inset.fill_between(ticks,
                            lbs,
                            ubs,
                            color=color_, alpha=0.2)

        print(f'{len(seeds_cur)=}')

    if inset_params:
        mark_inset(ax, ax_inset, loc1=inset_params.get('loc1', 1), loc2=inset_params.get('loc2', 3), fc="none", ec="0.6")

    ax.set_ylim(**ylim)
    ax.set_xlim(**xlim)

    label = ax.set_xlabel(xlabel)
    if if_leftalign_xlabel:
        label.set_horizontalalignment('left')  # Align text to the left
        label.set_position((0, 0))  # Position at the start of the x-axis
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)

    if (yticks_pad is not None) or (yticks_labelsize is not None):
        plt.tick_params(axis='y', which='both', pad=yticks_pad,
                        labelsize=yticks_labelsize)
    if force_n_yticks is not None:
        plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(force_n_yticks))

    if legend_on_the_side:
        lgnd_kwargs = dict(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
        # plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    else:
        lgnd_kwargs = dict(bbox_to_anchor=(0.5, 1), loc='lower center', borderaxespad=0., ncol=2)
    lgnd_kwargs.update(legend_kwargs)
    ax.legend(**lgnd_kwargs)

    if xaxis_number_of_zeros is not None:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-xaxis_number_of_zeros, xaxis_number_of_zeros))  # This will trigger scientific notation if out of this range.

        ax.xaxis.set_major_formatter(formatter)

        # Optionally force redraw of the plot to see the effects on the fly
        ax.figure.canvas.draw()

    plt.tight_layout()
    for exp_name in exp_names:
        print(f'{exp_name=}')
        plt.savefig(Path(LOGS_DIR) / exp_name / f'fitness_at_ticks_avg.pdf', bbox_inches='tight', pad_inches=pad_inches)
    plt.show()

def _plot_single_solution_history_mean_quartiles(solution_history_many_seeds, var_names, t_step, c, linewidth=6,
                                                 mid_stat_name='mean', uncertainty='std', overwrite_hp_to_pretty_name={}):
    hp_to_pretty_name_local = copy.deepcopy(hp_to_pretty_name)
    hp_to_pretty_name_local.update(overwrite_hp_to_pretty_name)
    num_vars = len(solution_history_many_seeds[0][0])

    # Plot each variable
    for var_index in range(num_vars):
        # Select subplot for current variable
        if num_vars > 1:
            plt.subplot(num_vars, 1, var_index + 1)

        mids, lbs, ubs = [], [], []

        for t in range(solution_history_many_seeds.shape[1]):

            var_across_seeds = solution_history_many_seeds[:, t, var_index]

            if var_names[var_index].startswith('log'):
                var_across_seeds = convert_from_logarithmic(var_names[var_index], var_across_seeds)
                # print('disabled log scale on y')
                if t == 0:
                    plt.yscale('log')

            if mid_stat_name == 'mean':
                mid = np.mean(var_across_seeds)
            elif mid_stat_name == 'median':
                mid = np.median(var_across_seeds)
            mids.append(mid)

            if uncertainty == 'std':
                lb = mid - np.std(var_across_seeds)
                ub = mid + np.std(var_across_seeds)
            elif uncertainty == 'quartiles':
                lb = np.quantile(var_across_seeds, 0.25)
                ub = np.quantile(var_across_seeds, 0.75)

            lbs.append(lb)
            ubs.append(ub)
            # set y axis name to var name:
            if t == 0:
                name_to_show = var_names[var_index]
                if name_to_show.startswith('log'):
                    name_to_show = name_to_show[name_to_show.index('_')+1:]
                name_to_show = hp_to_pretty_name_local.get(name_to_show, name_to_show)
                plt.ylabel(name_to_show)

        ts = [x * t_step for x in list(range(solution_history_many_seeds.shape[1]))]
        plt.plot(ts, mids,
                 color=c, linewidth=linewidth, marker="none")
        # plot with std:
        plt.fill_between(ts,
                         np.array(lbs),
                         np.array(ubs),
                         color=c, alpha=0.2)

def plot_best_solution_avg_many_seeds_many_exps(exp_names, fitness_transform=lambda x: x, seeds=(),
                                                uncertainty='std', mid_stat_name='mean', xlabel='', drop_last=False,
                                                overwrite_hp_to_pretty_name={}):
    plt.figure(figsize=(4, 4))
    used_colors = []
    final_fitness_means = []

    for i, exp_name in enumerate(exp_names):
        exp_path = Path(LOGS_DIR) / exp_name
        seeds_cur = list(sorted([int(x.name) for x in exp_path.iterdir() if x.is_dir()]))
        seeds_cur = [x for x in seeds_cur if x in seeds]

        best_solution_history_all = []

        fitnesses = []
        for seed in seeds_cur:
            seed_path = exp_path / str(seed)
            best_solution_history, best_fitness, _, t_step, _ = _get_best_solution_info(seed_path)
            if drop_last:
                best_solution_history = best_solution_history[:-1]
            best_solution_history_all.append(best_solution_history)

            config = yaml.safe_load(open(seed_path / 'config.yaml', 'r'))
            var_names = list(sorted([h['name'] for h in config['search_space']['hyperparameters']]))
            fitnesses.append(fitness_transform(best_fitness))

        algo_name = exp_name.split('_')[0]
        color_ = algo_to_color[algo_name]

        best_solution_history_all = np.array(best_solution_history_all)
        _plot_single_solution_history_mean_quartiles(best_solution_history_all, var_names, t_step, color_, linewidth=1,
                                                     mid_stat_name=mid_stat_name, uncertainty=uncertainty,
                                                     overwrite_hp_to_pretty_name=overwrite_hp_to_pretty_name)

        used_colors.append(color_)
        final_fitness_means.append(np.mean(fitnesses))

    ax = plt.gca()

    # def custom_thousands(x, pos):
    #     """Custom formatter to use a space as a thousand separator."""
    #     return f'{int(x):,}'.replace(',', '\u2009')
    # ax.xaxis.set_major_formatter(FuncFormatter(custom_thousands))

    handles = [matplotlib.lines.Line2D([], [], color=color, label=algo_to_pretty_name[exp_name.split('_')[0]]) for color, exp_name, fitness in
               zip(used_colors, exp_names, final_fitness_means)]
    # plt.legend(handles=handles, bbox_to_anchor=(1.01, 1),
    #                loc='upper left', borderaxespad=0., handlelength=0.7)
    plt.xlabel(xlabel)
    plt.legend(handles=handles, bbox_to_anchor=(0.5, 1), loc='lower center', borderaxespad=0., ncol=2)
    plt.tight_layout()

    for exp_name in exp_names:
        exp_path = Path(LOGS_DIR) / exp_name
        plt.savefig(exp_path / f'best_solution_{mid_stat_name}.pdf', bbox_inches='tight', pad_inches=0.0)
    plt.show()

def _get_best_solution_info(seed_path):
    path = seed_path / 'history_population.yaml'
    with open(path, 'r') as f:
        pop_history = yaml.safe_load(f)

    path = seed_path / 'best_info.yaml'
    with open(path, 'r') as f:
        best_info = yaml.safe_load(f)
        best_solution_history = best_info['solution_history']
        best_fitness = best_info['fitness']
        best_fitness_history = best_info['fitness_history']
        best_test = best_info['test']

    # compute t_step
    t_fst, t_snd = list(pop_history[1].keys())[:2]
    t_step = t_snd - t_fst
    return best_solution_history, best_fitness, best_fitness_history, t_step, best_test


def _get_best_fitness_history(exp_path, fitness_transform, seeds):
    seeds_cur = list(sorted([int(x.name) for x in exp_path.iterdir() if x.is_dir()]))
    seeds_cur = [x for x in seeds_cur if x in seeds]

    fitnesses_all = []
    for seed in seeds_cur:
        seed_path = exp_path / str(seed)
        _, best_fitness, best_fitness_history, t_step, _ = _get_best_solution_info(seed_path)

        fitnesses_all.append([fitness_transform(f) for f in best_fitness_history])

    fitnesses_all = np.array(fitnesses_all)
    return fitnesses_all, t_step, seeds_cur

def _get_test_of_best(exp_path, fitness_transform, seeds):
    seeds_cur = list(sorted([int(x.name) for x in exp_path.iterdir() if x.is_dir()]))
    seeds_cur = [x for x in seeds_cur if x not in seeds]

    test_all_seeds = []
    for seed in seeds_cur:
        seed_path = exp_path / str(seed)
        _, _, _, _, best_test = _get_best_solution_info(seed_path)

        test_all_seeds.append(fitness_transform(best_test))

    test_all_seeds = np.array(test_all_seeds)
    return test_all_seeds, seeds_cur

def plot_best_solution_fitness_avg_over_settings(exp_names_per_algo, algo_names, setting_names,
                                                 fitness_transform=lambda x: x, seeds=(), split='val',
                                                 xlabel='', ylabel='', aggregate='mean', uncertainty='std',
                                                 yticks_pad=None, yticks_labelsize=None, bar_width=0.2,
                                                 force_n_yticks=None,
                                                 legend_kwargs={}):
    plt.figure(figsize=(4, 4))
    min_seeds = None

    x_ticks_vals = np.arange(len(setting_names)) * 3

    records = []

    for i_algo, (algo_name, exp_names) in enumerate(zip(algo_names, exp_names_per_algo)):
        algo_to_agg_fitnesses = defaultdict(list)
        algo_to_uncertainty_fitnesses_lerror = defaultdict(list)
        algo_to_uncertainty_fitnesses_uerror = defaultdict(list)
        skipped = np.array([False] * len(setting_names))

        for i_setting, (setting_name, exp_name) in enumerate(zip(setting_names, exp_names)):
            if exp_name == 'skip':
                skipped[i_setting] = True
                continue
            exp_path = Path(LOGS_DIR) / exp_name
            if split == 'val':
                fitnesses_all, _, seeds = _get_best_fitness_history(exp_path, fitness_transform, seeds)
                fitnesses_all = fitnesses_all[:, -1]
            else:
                assert split == 'test'
                fitnesses_all, seeds = _get_test_of_best(exp_path, fitness_transform, seeds)
            if aggregate == 'mean':
                mean = np.mean(fitnesses_all)
            elif aggregate == 'iqm':
                q25 = np.quantile(fitnesses_all, 0.25, method='nearest')
                q75 = np.quantile(fitnesses_all, 0.75, method='nearest')
                mask = (q25 <= fitnesses_all) & (fitnesses_all <= q75)
                interquartile_fitnesses = np.ma.array(fitnesses_all, mask=~mask)
                mean = interquartile_fitnesses.mean()
                lerror = mean - q25
                uerror = q75 - mean

            if uncertainty == 'std':
                std = np.std(fitnesses_all)
                lerror = std
                uerror = std
            elif uncertainty == 'iqr':
                pass # already computed above

            algo_to_agg_fitnesses[algo_name].append(mean)
            algo_to_uncertainty_fitnesses_lerror[algo_name].append(lerror)
            algo_to_uncertainty_fitnesses_uerror[algo_name].append(uerror)
            print(f'{algo_name} | {setting_name} | {len(seeds)=} | {mean:.2f} ({lerror:.2f}, {uerror:.2f})')
            min_seeds = len(seeds) if min_seeds is None else min(min_seeds, len(seeds))

            for s, f in zip(seeds, fitnesses_all):
                records.append({'algo': algo_name, 'setting': setting_name, 'seed': s, 'fitness': f, 'task': exp_name.split('_')[1]})

        color_ = algo_to_color[algo_name]

        x_ticks_vals_cur = x_ticks_vals + i_algo * bar_width
        x_ticks_vals_cur = x_ticks_vals_cur[~skipped]
        plt.errorbar(x_ticks_vals_cur, algo_to_agg_fitnesses[algo_name],
                     yerr=[algo_to_uncertainty_fitnesses_lerror[algo_name],
                           algo_to_uncertainty_fitnesses_uerror[algo_name]],
                     fmt='o', color=color_, label=algo_to_pretty_name[algo_name],
                     capsize=2, elinewidth=2, capthick=2)

        plt.plot(x_ticks_vals_cur, algo_to_agg_fitnesses[algo_name], color=color_, alpha=0.4, linewidth=2)


    plt.xticks(x_ticks_vals + bar_width * (len(algo_names) - 1) / 2,
               setting_names)
    if (yticks_pad is not None) or (yticks_labelsize is not None):
        plt.tick_params(axis='y', which='both', pad=yticks_pad,
                        labelsize=yticks_labelsize)
    if force_n_yticks is not None:
        plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(force_n_yticks))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    lgnd_kwargs = dict(bbox_to_anchor=(0.5, 1), loc='lower center', borderaxespad=0., ncol=2,
               columnspacing=None, handletextpad=None, fontsize=None,
               borderpad=None)
    lgnd_kwargs.update(legend_kwargs)
    plt.legend(**lgnd_kwargs)
    plt.tight_layout()
    df = pd.DataFrame(records)
    for exp_name in exp_names:
        plt.savefig(Path(LOGS_DIR) / exp_name / f'vary_settings_{split}.pdf', bbox_inches='tight', pad_inches=0.0)
        df.to_csv(Path(LOGS_DIR) / exp_name / f'vary_settings_{split}.csv', index=False)
    plt.show()


def plot_iqm_diff_heatmap(exp_names_per_algo_1, exp_names_per_algo_2,
                          algo_names, setting_names,
                          seeds=(),
                          fitness_transform=lambda x: x,
                          split='val',
                          aggregate='iqm',  # or 'mean'
                          xlabel='Settings',
                          ylabel='Algorithm',
                          title='Difference in IQM (Group1 - Group2)',
                          cmap='PRGn', fmt='.2f',
                          vmin=None, vmax=None):

    num_algos = len(algo_names)
    num_settings = len(setting_names)

    # This array will hold the difference for each (algo, setting).
    diff_matrix = np.zeros((num_algos, num_settings), dtype=float)

    # Helper function to compute aggregator
    def compute_aggregator(values, agg):
        """
        Compute either mean or interquartile mean (IQM).
        """
        values = np.array(values)
        if agg == 'mean':
            return np.mean(values)
        elif agg == 'iqm':
            # Interquartile Mean:
            q25 = np.quantile(values, 0.25, method='nearest')
            q75 = np.quantile(values, 0.75, method='nearest')
            mask = (q25 <= values) & (values <= q75)
            return np.mean(values[mask])
        else:
            raise ValueError(f"Unknown aggregate '{agg}'")

    # Loop over algos and settings
    for i_algo, (algo_name, exp_list_1, exp_list_2) in enumerate(zip(algo_names, exp_names_per_algo_1, exp_names_per_algo_2)):
        print(algo_name)
        for i_setting, (setting_name, exp_name_1, exp_name_2) in enumerate(zip(setting_names, exp_list_1, exp_list_2)):

            # Skip if either experiment is marked as 'skip'
            if exp_name_1 == 'skip' or exp_name_2 == 'skip':
                diff_matrix[i_algo, i_setting] = np.nan
                continue

            # ======= First group =======
            exp_path_1 = Path(LOGS_DIR) / exp_name_1
            if split == 'val':
                fitnesses_all_1, _, seeds_1 = _get_best_fitness_history(exp_path_1, fitness_transform, seeds)
                fitnesses_all_1 = fitnesses_all_1[:, -1]  # final performance
            else:
                # must be 'test'
                fitnesses_all_1, seeds_1 = _get_test_of_best(exp_path_1, fitness_transform, seeds)

            val_1 = compute_aggregator(fitnesses_all_1, aggregate)

            # ======= Second group =======
            exp_path_2 = Path(LOGS_DIR) / exp_name_2
            if split == 'val':
                fitnesses_all_2, _, seeds_2 = _get_best_fitness_history(exp_path_2, fitness_transform, seeds)
                fitnesses_all_2 = fitnesses_all_2[:, -1]
            else:
                fitnesses_all_2, seeds_2 = _get_test_of_best(exp_path_2, fitness_transform, seeds)

            val_2 = compute_aggregator(fitnesses_all_2, aggregate)

            # Difference
            diff_matrix[i_algo, i_setting] = val_1 - val_2

    # ====== Now plot a heatmap ======
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        diff_matrix,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        center=0,
        xticklabels=setting_names,
        yticklabels=[algo_to_pretty_name[n] for n in algo_names],
        cbar_kws={'label': 'Difference'},
        linewidths=0,  # remove cell borders
        vmin=vmin, vmax=vmax,
    )

    # ax.set_yticklabels([algo_to_pretty_name[n] for n in algo_names], rotation=0)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Turn off the major figure grid lines
    plt.grid(False)

    plt.tight_layout()

    for exp_name in exp_names_per_algo_1[-1]:
        plt.savefig(Path(LOGS_DIR) / exp_name / f'diff_heatmap_{split}.pdf', bbox_inches='tight', pad_inches=0.0)

    plt.show()

    return diff_matrix

def plot_time_avg_over_settings(exp_names_per_algo, algo_names, setting_names,
                                seeds=(),
                                xlabel='', ylabel='', aggregate='mean', uncertainty='std',
                                cumulative=False, time_type='update_times', time_div_by=1, time_div_by_per_setting=None,
                                figsize=(4, 4), legend_on_the_side=False):
    plt.figure(figsize=figsize)
    min_seeds = None

    _mul = 3 if len(setting_names) == 4 else 2
    x_ticks_vals = np.arange(len(setting_names)) * 3
    bar_width = 0.2

    for i_algo, (algo_name, exp_names) in enumerate(zip(algo_names, exp_names_per_algo)):
        algo_to_agg_times = defaultdict(list)
        algo_to_uncertainty_times_lerror = defaultdict(list)
        algo_to_uncertainty_times_uerror = defaultdict(list)
        skipped = np.array([False] * len(setting_names))

        for i_setting, (setting_name, exp_name) in enumerate(zip(setting_names, exp_names)):
            if exp_name == 'skip':
                skipped[i_setting] = True
                continue
            exp_path = Path(LOGS_DIR) / exp_name

            seeds_cur = list(sorted([int(x.name) for x in exp_path.iterdir() if x.is_dir()]))
            seeds_cur = [x for x in seeds_cur if x in seeds]

            times_all_seeds = []
            for seed in seeds_cur:
                seed_path = exp_path / str(seed)

                df_times = pd.read_csv(seed_path / f'{time_type}.csv')
                times = df_times['time'].to_numpy() / time_div_by
                if time_div_by_per_setting is not None:
                    times /= time_div_by_per_setting[i_setting]
                if cumulative:
                    times = np.cumsum(times)
                times_all_seeds.append(times)

            times_all_seeds = np.array(times_all_seeds)

            if aggregate == 'mean':
                if not cumulative:
                    mean = np.mean(times_all_seeds)
                else:
                    mean = np.mean(times_all_seeds, axis=0)[-1]
            elif aggregate == 'iqm': # probably doesn't make sense for time
                q25 = np.quantile(times_all_seeds, 0.25, method='nearest', axis=0)
                q75 = np.quantile(times_all_seeds, 0.75, method='nearest', axis=0)
                mask = (q25 <= times_all_seeds) & (times_all_seeds <= q75)
                interquartile_fitnesses = np.ma.array(times_all_seeds, mask=~mask)
                if not cumulative:
                    raise ValueError('does not make sense: how would quartiles be combined?')
                else:
                    mean = interquartile_fitnesses.mean(axis=0).data[-1]
                    q25 = q25[-1]
                    q75 = q75[-1]
                lerror = mean - q25
                uerror = q75 - mean

            if uncertainty == 'std':
                if not cumulative:
                    std = np.std(times_all_seeds)
                else:
                    std = np.std(times_all_seeds, axis=0)[-1]
                lerror = std
                uerror = std
            elif uncertainty == 'iqr':
                pass # already computed above

            algo_to_agg_times[algo_name].append(mean)
            algo_to_uncertainty_times_lerror[algo_name].append(lerror)
            algo_to_uncertainty_times_uerror[algo_name].append(uerror)

            print(f'{len(seeds_cur)=}')
            min_seeds = len(seeds_cur) if min_seeds is None else min(min_seeds, len(seeds_cur))

        color_ = algo_to_color[algo_name]

        x_ticks_vals_cur = x_ticks_vals + i_algo * bar_width
        x_ticks_vals_cur = x_ticks_vals_cur[~skipped]
        plt.errorbar(x_ticks_vals_cur, algo_to_agg_times[algo_name],
                     yerr=[algo_to_uncertainty_times_lerror[algo_name],
                           algo_to_uncertainty_times_uerror[algo_name]],
                     fmt='o', color=color_, label=algo_to_pretty_name[algo_name],
                     capsize=2, elinewidth=2, capthick=2)

        plt.plot(x_ticks_vals_cur, algo_to_agg_times[algo_name], color=color_, alpha=0.4, linewidth=2)


    plt.xticks(x_ticks_vals + bar_width * (len(algo_names) - 1) / 2,
               setting_names)
    # plt.ylim(**ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend_on_the_side:
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    else:
        plt.legend(bbox_to_anchor=(0.5, 1), loc='lower center', borderaxespad=0., ncol=2)#(len(algo_names) + 1) // 2)
    plt.tight_layout()
    for exp_name in exp_names:
        plt.savefig(Path(LOGS_DIR) / exp_name / f'vary_settings_time.pdf', bbox_inches='tight', pad_inches=0.0)
    plt.show()

def crit_diff_diagram(exp_names, xlabel='', ylabel=''):
    df_path = Path(LOGS_DIR) / exp_names[0] / 'vary_settings_val.csv'
    df = pd.read_csv(df_path)
    for exp_name in exp_names[1:]:
        df_path = Path(LOGS_DIR) / exp_name / 'vary_settings_val.csv'
        df = pd.concat([df, pd.read_csv(df_path)], ignore_index=True)

    # combine task and seed columns:
    df['task_and_seed'] = df['task'] + '_' + df['seed'].astype(str)
    df = df.drop(columns=['task', 'seed'])

    # replace algo with pretty name
    df['algo'] = df['algo'].apply(lambda x: algo_to_pretty_name[x])

    # construct a sequence of CD diagrams
    treatment_names = df["algo"].unique()
    diagram_names = df["setting"].unique()
    Xs = []  # collect an (n,k)-shaped matrix for each diagram
    treatment_names_per_diagram = []
    for n in diagram_names:
        # ensure order even if a method is missing:
        treatment_names_per_diagram_cur = [x for x in treatment_names if x in df[df.setting == n]["algo"].unique()]
        diagram_df = df[df.setting == n].pivot(
            index="task_and_seed",
            columns="algo",
            values="fitness"
        )[treatment_names_per_diagram_cur]  # ensure a fixed order of treatments
        Xs.append(diagram_df.to_numpy())
        treatment_names_per_diagram.append(treatment_names_per_diagram_cur)
    # treatment_names = [algo_to_pretty_name[x] for x in treatment_names]
    two_dimensional_diagram = Diagrams(
        Xs,
        diagram_names=[str(n) for n in diagram_names],
        treatment_names_all=treatment_names,
        treatment_names_per_diagram=treatment_names_per_diagram,
        maximize_outcome=True
    )

    out_path = Path(LOGS_DIR) / exp_names[0] / 'crit_diff.pdf'
    # customize the style of the plot and export to PDF
    hex_colors = palette.as_hex()
    algo_to_color = {  # so that the algos have the same color in all plots
        'PBT': hex_colors[0][1:],
        'PB2': hex_colors[1][1:],
        'PB2-Mix': hex_colors[3][1:],
        'BG-PBT': hex_colors[4][1:],
        'FIRE-PBT': hex_colors[2][1:],
    }
    algo_to_tikz_color = {t: "\\definecolor{color" + str(i + 1) + "}{HTML}{" + algo_to_color[t] + "}" for i, t in enumerate(treatment_names)}
    algo_to_marker = {
        'PBT': 'mark=*',
        'PB2': 'mark=diamond*',
        'PB2-Mix': 'mark=triangle,thick',
        'BG-PBT': 'mark=square,thick',
        'FIRE-PBT': 'mark=pentagon,thick',
    }
    algo_to_tikz_marker = {t: "{color" + str(i + 1) + "," + algo_to_marker[t] + ",mark size=3pt}" for i, t in enumerate(treatment_names)}
    two_dimensional_diagram.to_file(
        out_path,
        preamble="\n".join([algo_to_tikz_color[t] for t in treatment_names]),
        axis_options={  # style the plot
            "cycle list": ",".join([algo_to_tikz_marker[t] for t in treatment_names]),
            "width": "0.8*\\axisdefaultwidth",
            "height": "0.75*\\axisdefaultheight",
            # "title": "critdd"
            "xlabel": xlabel,
            "ylabel": ylabel,
            "trim axis left": "",
            "trim axis right": "",
        },
    )

def clean_runtimes():
    p = Path(LOGS_DIR) / 'runtimes.csv'
    if p.exists():
        p.unlink()

def store_runtimes(exp_names, seeds_include=()):
    # df columns: exp_name, n_seeds, runtime_total
    n_gpus = 3

    data = []
    for exp_name in exp_names:
        exp_path = Path(LOGS_DIR) / exp_name

        seeds = list(sorted([int(x.name) for x in exp_path.iterdir() if x.is_dir()]))
        seeds = [x for x in seeds if x in seeds_include]

        runtime_sum = 0
        for seed in seeds:
            seed_path = exp_path / str(seed)
            runtime = yaml.safe_load(open(seed_path / 'runtime.yaml', 'r'))
            runtime_sum += runtime

        data.append([exp_name, len(seeds), runtime_sum * n_gpus])

    df = pd.DataFrame(data, columns=['exp_name', 'n_seeds', 'runtime_total'])
    if (Path(LOGS_DIR) / 'runtimes.csv').exists():
        df_old = pd.read_csv(Path(LOGS_DIR) / 'runtimes.csv')
        df = pd.concat([df_old, df], ignore_index=True)
    df.to_csv(Path(LOGS_DIR) / 'runtimes.csv', index=False)

def sum_runtimes():
    df = pd.read_csv(Path(LOGS_DIR) / 'runtimes.csv')
    total = df['runtime_total'].sum() # gpu-seconds
    total = total / 3600 / 24 # gpu-days
    print('Total GPU-days: ', total)
