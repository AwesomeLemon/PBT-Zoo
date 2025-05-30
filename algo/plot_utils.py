import itertools

import yaml
from matplotlib import pyplot as plt
from utils.util_fns import convert_from_logarithmic

def plot_pop_history(exp_path, exp_name, seed):
    path = exp_path / 'history_population.yaml'
    with open(path, 'r') as f:
        pop_history = yaml.safe_load(f)

    path = exp_path / 'best_info.yaml'
    with open(path, 'r') as f:
        best_info = yaml.safe_load(f)
        best_solution_history = best_info['solution_history']

    # compute t_step
    t_fst, t_snd = list(pop_history[1].keys())[:2]
    t_step = t_snd - t_fst
    it_cycle = itertools.cycle(iter(plt.rcParams['axes.prop_cycle']))

    var_names = list(sorted(
        [h['name'] for h in yaml.safe_load(open(exp_path / 'config.yaml', 'r'))['search_space']['hyperparameters']]))
    _plot_single_solution_history(best_solution_history, var_names, t_step, next(it_cycle)['color'])

    for pop_id in sorted(pop_history.keys()):
        _plot_single_pop_history(pop_history[pop_id], t_step, next(it_cycle)['color'])

    plt.yscale('log')
    plt.tight_layout()
    plt.title(f'{exp_name}: Population history + the best')
    plt.savefig(exp_path / f'history_population.png', bbox_inches='tight')
    plt.show()


def _plot_single_solution_history(solution_history, var_names, t_step, c, linewidth=6):
    num_vars = len(solution_history[0])

    # Plot each variable
    for var_index in range(num_vars):
        # Select subplot for current variable
        if num_vars > 1:
            plt.subplot(num_vars, 1, var_index + 1)

        prev = None
        for t, s in enumerate(solution_history):
            if var_names[var_index].startswith('log'):
                s[var_index] = convert_from_logarithmic(var_names[var_index], s[var_index])
                plt.yscale('log')
            plt.plot([t * t_step, (t + 1) * t_step], [s[var_index], s[var_index]],
                     color=c, linewidth=linewidth, marker="none")
            # set y axis name to var name:
            name_to_show = var_names[var_index]
            if name_to_show.startswith('log'):
                name_to_show = name_to_show[name_to_show.index('_') + 1:]
            plt.ylabel(name_to_show)
            if prev is not None:
                # vertical line
                if prev != s[var_index]:
                    plt.plot([t * t_step, t * t_step], [prev, s[var_index]],
                             color=c, linewidth=linewidth, marker="none")
            prev = s[var_index]


def _plot_single_pop_history(one_pop_history, t_step, c):
    for t, pop in one_pop_history.items():
        for i, s in enumerate(pop):
            # horizontal line for each solution from t to t+t_step
            plt.plot([t, t + t_step], [s[0], s[0]], color=c)
