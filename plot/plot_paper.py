from plot.plot_paper_fns import *
from utils.util_fns import set_plot_style_paper

def figure_2():
    ### PlainToy
    plot_fitness_at_ticks_avg_many_seeds_many_exps(['pbt_toyplain_00', 'pb2rand_toyplain_00',
                                                    'bgpbt_toyplain_00', 'firepbt_toyplain_00'],
                                    yscale='linear', seeds=list(range(5)),
                                    ylim={'bottom': 0.45}, xlabel='SGD steps', ylabel='Objective↑',
                                    aggregate='iqm', uncertainty='iqr',
                                    inset_params={'xlim': (80, 180), 'ylim': (0.7, 0.95),
                                                  'width': 65, 'height': 65})

    plot_best_solution_avg_many_seeds_many_exps(
        ['pbt_toyplain_00', 'pb2rand_toyplain_00', 'bgpbt_toyplain_00', 'firepbt_toyplain_00'],
        uncertainty='std', mid_stat_name='mean', seeds=list(range(5)),
        xlabel='SGD steps', drop_last=True,
        overwrite_hp_to_pretty_name={'lr': 'h'}
    )

    ### TimeLinkedToy
    plot_fitness_at_ticks_avg_many_seeds_many_exps(
        ['pbt_toydeceptive_00', 'pb2rand_toydeceptive_00', 'bgpbt_toydeceptive_00', 'firepbt_toydeceptive_00'],
        yscale='linear', seeds=list(range(5)),
        ylim={'bottom': 0.8}, xlabel='SGD steps', ylabel='Objective↑',
        aggregate='iqm', uncertainty='iqr',
        inset_params=None)

    plot_best_solution_avg_many_seeds_many_exps(
        ['pbt_toydeceptive_00', 'pb2rand_toydeceptive_00', 'bgpbt_toydeceptive_00', 'firepbt_toydeceptive_00'],
        uncertainty='std', mid_stat_name='mean', seeds=list(range(5)),
        xlabel='SGD steps', drop_last=True,
        overwrite_hp_to_pretty_name={'lr': 'h'}
    )

def figure_3():
    ### FashionMnist
    plot_fitness_at_ticks_avg_many_seeds_many_exps(['pbt_fmnist_16', 'pb2rand_fmnist_16', 'bgpbt_fmnist_16', 'firepbt_fmnist_16'],
                                   fitness_transform=lambda x: 100*x, yscale='linear', seeds=list(range(5)),
                                                   ylim={'bottom': 92}, xlabel='SGD steps', ylabel='Accuracy↑',
                                                   aggregate='iqm', uncertainty='iqr',
                                                   inset_params={'xlim': (4000, 13000), 'ylim': (93.5, 95),
                                                                 'width': 55, 'height': 55}
                                                   )

    plot_best_solution_avg_many_seeds_many_exps(
        ['pbt_fmnist_16', 'pb2rand_fmnist_16', 'bgpbt_fmnist_16', 'firepbt_fmnist_16'],
        fitness_transform=lambda x: 100*x, uncertainty='quartiles', mid_stat_name='median', seeds=list(range(5)),
        xlabel='SGD steps'
    )

    ### Cifar10
    plot_fitness_at_ticks_avg_many_seeds_many_exps(['pbt_c10_16', 'pb2rand_c10_16', 'bgpbt_c10_16', 'firepbt_c10_16'],
                                   fitness_transform=lambda x: 100*x, yscale='linear', seeds=list(range(5)),
                                                   ylim={'bottom': 85, 'top': 95.5}, xlabel='SGD steps', ylabel='Accuracy↑',
                                                   aggregate='iqm', uncertainty='iqr',
                                                   # inset_params={'xlim': (4200, 20000), 'ylim': (86, 92.5),
                                                   #               'width': 55, 'height': 35}
                                                   )

    plot_best_solution_avg_many_seeds_many_exps(
        ['pbt_c10_16', 'pb2rand_c10_16', 'bgpbt_c10_16', 'firepbt_c10_16'],
        fitness_transform=lambda x: 100*x, uncertainty='quartiles', mid_stat_name='median', seeds=list(range(5)),
        xlabel='SGD steps'
    )

def figure_4a():
    ### Hopper
    plot_fitness_at_ticks_avg_many_seeds_many_exps(['pbt_hopper_10', 'pb2rand_hopper_10', 'pb2mix_hopper_10',
                                                    'bgpbt_hopper_10', 'firepbt_hopper_10'],
                                   yscale='linear', seeds=list(range(7)),
                                   ylim={'bottom': 500, 'top': 2500}, xlabel='Environment steps', ylabel='Score↑',
                                   aggregate='iqm', uncertainty='iqr',
                                   if_leftalign_xlabel=True,
                                   pad_inches=0.0,
                                   legend_kwargs=dict(columnspacing=0.7,
                                                      fontsize=0.9 * plt.rcParams['legend.fontsize'],
                                                      borderpad=0.2, handlelength=0.7,
                                                      handletextpad=0.2, ncol=3),
                                   )

    ### Humanoid

    plot_fitness_at_ticks_avg_many_seeds_many_exps(['pbt_humanoid_13', 'pb2rand_humanoid_13',
                                                    'pb2mix_humanoid_13', 'bgpbt_humanoid_13', 'firepbt_humanoid_13'],
                                      yscale='linear', seeds=list(range(7)),
                                      ylim={'bottom': 2000, 'top': 9500}, xlabel='Environment steps', ylabel='Score↑',
                                      aggregate='iqm', uncertainty='iqr',
                                      if_leftalign_xlabel=True,
                                      pad_inches=0.0,
                                      legend_kwargs=dict(columnspacing=0.7,
                                                      fontsize=0.9 * plt.rcParams['legend.fontsize'],
                                                      borderpad=0.2, handlelength=0.7,
                                                      handletextpad=0.2, ncol=3),
                                   )

def figure_4b():
    ### Hopper
    plot_fitness_at_ticks_avg_many_seeds_many_exps(['pbt_hopper_13', 'pb2rand_hopper_13', 'pb2mix_hopper_13',
                                                    'bgpbt_hopper_13', 'firepbt_hopper_13'],
                                   yscale='linear', seeds=list(range(7)),
                                   ylim={'bottom': 500, 'top': 2100}, xlabel='Environment steps', ylabel='Score↑',
                                   aggregate='iqm', uncertainty='iqr',
                                   if_leftalign_xlabel=True,
                                   pad_inches=0.0,
                                   legend_kwargs=dict(columnspacing=0.7,
                                                      fontsize=0.9 * plt.rcParams['legend.fontsize'],
                                                      borderpad=0.2, handlelength=0.7,
                                                      handletextpad=0.2, ncol=3),
                                   )

    ### Humanoid

    plot_fitness_at_ticks_avg_many_seeds_many_exps(['pbt_humanoid_12', 'pb2rand_humanoid_12',
                                                    'pb2mix_humanoid_12', 'bgpbt_humanoid_12', 'firepbt_humanoid_12'],
                                      yscale='linear', seeds=list(range(7)),
                                      ylim={'bottom': 2000, 'top': 9500}, xlabel='Environment steps', ylabel='Score↑',
                                      aggregate='iqm', uncertainty='iqr',
                                      if_leftalign_xlabel=True,
                                      pad_inches=0.0,
                                      legend_kwargs=dict(columnspacing=0.7,
                                                          fontsize=0.9 * plt.rcParams['legend.fontsize'],
                                                          borderpad=0.2, handlelength=0.7,
                                                          handletextpad=0.2, ncol=3),
                                      )

def figure_5(split):
    #### vary step size

    plot_best_solution_fitness_avg_over_settings([
        ['pbt_fmnist_18', 'pbt_fmnist_17', 'pbt_fmnist_16', 'pbt_fmnist_15'],
        ['pb2rand_fmnist_18', 'pb2rand_fmnist_17', 'pb2rand_fmnist_16', 'pb2rand_fmnist_15'],
        ['bgpbt_fmnist_18', 'bgpbt_fmnist_17', 'bgpbt_fmnist_16', 'bgpbt_fmnist_15'],
        ['firepbt_fmnist_18', 'firepbt_fmnist_17', 'firepbt_fmnist_16', 'firepbt_fmnist_15']],
        ['pbt', 'pb2rand', 'bgpbt', 'firepbt'],
        ['3', '10', '30', '100'],
        seeds=list(range(5)), split=split, fitness_transform=lambda x: 100 * x,
        xlabel='Number of steps', ylabel='Accuracy↑',
        aggregate='iqm', uncertainty='iqr'
    )

    plot_best_solution_fitness_avg_over_settings([
        ['pbt_c10_18', 'pbt_c10_17', 'pbt_c10_16', 'pbt_c10_15'],
        ['pb2rand_c10_18', 'pb2rand_c10_17', 'pb2rand_c10_16', 'pb2rand_c10_15'],
        ['bgpbt_c10_18', 'bgpbt_c10_17', 'bgpbt_c10_16', 'bgpbt_c10_15'],
        ['firepbt_c10_18', 'firepbt_c10_17', 'firepbt_c10_16', 'firepbt_c10_15']],
        ['pbt', 'pb2rand', 'bgpbt', 'firepbt'],
        ['3', '10', '30', '100'],
        seeds=list(range(5)), split=split, fitness_transform=lambda x: 100 * x,
        xlabel='Number of steps', ylabel='Accuracy↑',
        aggregate='iqm', uncertainty='iqr'
    )
    crit_diff_diagram(['firepbt_fmnist_18', 'firepbt_c10_18'], xlabel='Rank', ylabel='Steps')

    plot_best_solution_fitness_avg_over_settings([
        ['pbt_hopper_14', 'pbt_hopper_12', 'pbt_hopper_10', 'pbt_hopper_13'],
        ['pb2rand_hopper_14', 'pb2rand_hopper_12', 'pb2rand_hopper_10', 'pb2rand_hopper_13'],
        ['pb2mix_hopper_14','pb2mix_hopper_12', 'pb2mix_hopper_10', 'pb2mix_hopper_13'],
        ['bgpbt_hopper_14','bgpbt_hopper_12', 'bgpbt_hopper_10', 'bgpbt_hopper_13'],
        ['firepbt_hopper_14','firepbt_hopper_12', 'firepbt_hopper_10', 'firepbt_hopper_13']],
        ['pbt', 'pb2rand', 'pb2mix', 'bgpbt', 'firepbt'],
        ['3', '10', '30', '100'],
        seeds=list(range(7)), split=split,
        xlabel='Number of steps', ylabel='Score↑',
        aggregate='iqm', uncertainty='iqr',
        yticks_pad=0, yticks_labelsize=0.7 * plt.rcParams['ytick.labelsize'],
        bar_width=0.2, force_n_yticks=3,
        legend_kwargs=dict(bbox_to_anchor=(1, 1), loc='lower right',
                           ncol=3, columnspacing=0.3, handletextpad=-0.25,
                           fontsize=0.9*plt.rcParams['legend.fontsize'],
                           borderpad=0.2)
    )

    plot_best_solution_fitness_avg_over_settings([
        ['pbt_humanoid_17', 'pbt_humanoid_14', 'pbt_humanoid_13', 'pbt_humanoid_12'],
        ['pb2rand_humanoid_17', 'pb2rand_humanoid_14', 'pb2rand_humanoid_13', 'pb2rand_humanoid_12'],
        ['pb2mix_humanoid_17','pb2mix_humanoid_14', 'pb2mix_humanoid_13', 'pb2mix_humanoid_12'],
        ['bgpbt_humanoid_17','bgpbt_humanoid_14', 'bgpbt_humanoid_13', 'bgpbt_humanoid_12'],
        ['firepbt_humanoid_17','firepbt_humanoid_14', 'firepbt_humanoid_13', 'firepbt_humanoid_12']],
        ['pbt', 'pb2rand', 'pb2mix', 'bgpbt', 'firepbt'],
        ['3', '10', '30', '100'],
        seeds=list(range(7)), split=split,
        xlabel='Number of steps', ylabel='Score↑',
        aggregate='iqm', uncertainty='iqr',
        yticks_pad=0, yticks_labelsize=0.7 * plt.rcParams['ytick.labelsize'],
        bar_width=0.2, force_n_yticks=3,
        legend_kwargs=dict(bbox_to_anchor=(1, 1), loc='lower right',
                           ncol=3, columnspacing=0.2, handletextpad=-0.3,
                           fontsize=0.9 * plt.rcParams['legend.fontsize'],
                           borderpad=0.2)
    )
    crit_diff_diagram(['firepbt_hopper_12', 'firepbt_humanoid_14'], xlabel='Rank', ylabel='Steps')

def figure_6(split):
    #### vary search space size

    plot_best_solution_fitness_avg_over_settings([
        ['pbt_fmnist_15', 'pbt_fmnist_19', 'pbt_fmnist_20'],
        ['pb2rand_fmnist_15', 'pb2rand_fmnist_19', 'pb2rand_fmnist_20'],
        ['skip', 'pb2mix_fmnist_19', 'pb2mix_fmnist_20'],
        ['bgpbt_fmnist_15', 'bgpbt_fmnist_19', 'bgpbt_fmnist_20'],
        ['firepbt_fmnist_15', 'firepbt_fmnist_19', 'firepbt_fmnist_20'],
    ],
        ['pbt', 'pb2rand', 'pb2mix', 'bgpbt', 'firepbt'],
        ['small', 'medium ', 'large'],
        seeds=list(range(5)), split=split, fitness_transform=lambda x: 100 * x,
        xlabel='Search space', ylabel='Accuracy↑',
        aggregate='iqm', uncertainty='iqr',
        yticks_pad=0.1, yticks_labelsize=0.9 * plt.rcParams['ytick.labelsize'], force_n_yticks=3,
        legend_kwargs=dict(bbox_to_anchor=(1, 1), loc='lower right',
                           ncol=3, columnspacing=0.2, handletextpad=-0.3,
                           fontsize=0.9 * plt.rcParams['legend.fontsize'],
                           borderpad=0.2)
    )

    plot_best_solution_fitness_avg_over_settings([
        ['pbt_c10_15', 'pbt_c10_19', 'pbt_c10_20'],
        ['pb2rand_c10_15', 'pb2rand_c10_19', 'pb2rand_c10_20'],
        ['skip', 'pb2mix_c10_19', 'pb2mix_c10_20'],
        ['bgpbt_c10_15', 'bgpbt_c10_19', 'bgpbt_c10_20'],
        ['firepbt_c10_15', 'firepbt_c10_19', 'firepbt_c10_20'],
        ],
        ['pbt', 'pb2rand', 'pb2mix', 'bgpbt', 'firepbt'],
        ['small', 'medium ', 'large'],
        seeds=list(range(5)), split=split, fitness_transform=lambda x: 100 * x,
        xlabel='Search space', ylabel='Accuracy↑',
        aggregate='iqm', uncertainty='iqr',
        yticks_pad=0.1, yticks_labelsize=0.9 * plt.rcParams['ytick.labelsize'], force_n_yticks=3,
        legend_kwargs=dict(bbox_to_anchor=(1, 1), loc='lower right',
                           ncol=3, columnspacing=0.2, handletextpad=-0.3,
                           fontsize=0.9 * plt.rcParams['legend.fontsize'],
                           borderpad=0.2)
    )
    crit_diff_diagram(['firepbt_fmnist_20', 'firepbt_c10_20'], xlabel='Rank', ylabel='Search space')

    plot_best_solution_fitness_avg_over_settings([
        ['pbt_hopper_11', 'pbt_hopper_17', 'pbt_hopper_10'],
        ['pb2rand_hopper_11', 'pb2rand_hopper_17', 'pb2rand_hopper_10'],
        ['skip', 'pb2mix_hopper_17', 'pb2mix_hopper_10'],
        ['bgpbt_hopper_11', 'bgpbt_hopper_17', 'bgpbt_hopper_10'],
        ['firepbt_hopper_11', 'firepbt_hopper_17', 'firepbt_hopper_10'],
        ],
        ['pbt', 'pb2rand', 'pb2mix', 'bgpbt', 'firepbt'],
        ['small', 'medium', 'large'],
        seeds=list(range(7)), split=split,
        xlabel='Search space size', ylabel='Score↑',
        aggregate='iqm', uncertainty='iqr',
        yticks_pad=0.1, yticks_labelsize=0.9 * plt.rcParams['ytick.labelsize'], force_n_yticks=3,
        legend_kwargs=dict(bbox_to_anchor=(1, 1), loc='lower right',
                           ncol=3, columnspacing=0.2, handletextpad=-0.3,
                           fontsize=0.9 * plt.rcParams['legend.fontsize'],
                           borderpad=0.2)
    )

    plot_best_solution_fitness_avg_over_settings([
        ['pbt_humanoid_22', 'pbt_humanoid_23', 'pbt_humanoid_13'],
        ['pb2rand_humanoid_22', 'pb2rand_humanoid_23', 'pb2rand_humanoid_13'],
        ['skip', 'pb2mix_humanoid_23', 'pb2mix_humanoid_13'],
        ['bgpbt_humanoid_22', 'bgpbt_humanoid_23', 'bgpbt_humanoid_13'],
        ['firepbt_humanoid_22', 'firepbt_humanoid_23', 'firepbt_humanoid_13'],
    ],
        ['pbt', 'pb2rand', 'pb2mix', 'bgpbt', 'firepbt'],
        ['small', 'medium', 'large'],
        seeds=list(range(7)), split=split,
        xlabel='Search space size', ylabel='Score↑',
        aggregate='iqm', uncertainty='iqr',
        yticks_pad=0.1, yticks_labelsize=0.75 * plt.rcParams['ytick.labelsize'], force_n_yticks=3,
        legend_kwargs=dict(bbox_to_anchor=(1, 1), loc='lower right',
                           ncol=3, columnspacing=0.2, handletextpad=-0.3,
                           fontsize=0.9 * plt.rcParams['legend.fontsize'],
                           borderpad=0.2)
    )

    crit_diff_diagram(['firepbt_hopper_10', 'firepbt_humanoid_13'], xlabel='Rank', ylabel='Search space')


def figure_7(split):
    #### vary population size (large space)

    plot_best_solution_fitness_avg_over_settings([
        ['pbt_fmnist_22', 'pbt_fmnist_20', 'pbt_fmnist_24'],
        ['pb2rand_fmnist_22', 'pb2rand_fmnist_20', 'pb2rand_fmnist_24'],
        ['pb2mix_fmnist_22', 'pb2mix_fmnist_20', 'pb2mix_fmnist_24'],
        ['bgpbt_fmnist_22', 'bgpbt_fmnist_20', 'bgpbt_fmnist_24'],
        ['firepbt_fmnist_22', 'firepbt_fmnist_20', 'firepbt_fmnist_24']],
        ['pbt', 'pb2rand', 'pb2mix', 'bgpbt', 'firepbt'],
        ['8', '22', '50'],
        seeds=list(range(5)), split=split, fitness_transform=lambda x: 100 * x,
        xlabel='Population size', ylabel='Accuracy↑',
        aggregate='iqm', uncertainty='iqr',
        yticks_pad=0.1, yticks_labelsize=0.9 * plt.rcParams['ytick.labelsize'], force_n_yticks=3,
        legend_kwargs=dict(bbox_to_anchor=(1, 1), loc='lower right',
                           ncol=3, columnspacing=0.2, handletextpad=-0.3,
                           fontsize=0.9 * plt.rcParams['legend.fontsize'],
                           borderpad=0.2)
    )

    plot_best_solution_fitness_avg_over_settings([
        ['pbt_c10_22', 'pbt_c10_20', 'pbt_c10_24'],
        ['pb2rand_c10_22', 'pb2rand_c10_20', 'pb2rand_c10_24'],
        ['pb2mix_c10_22', 'pb2mix_c10_20', 'pb2mix_c10_24'],
        ['bgpbt_c10_22', 'bgpbt_c10_20', 'bgpbt_c10_24'],
        ['firepbt_c10_22', 'firepbt_c10_20', 'firepbt_c10_24']],
        ['pbt', 'pb2rand', 'pb2mix', 'bgpbt', 'firepbt'],
        ['8', '22', '50'],
        seeds=list(range(5)), split=split, fitness_transform=lambda x: 100 * x,
        xlabel='Population size', ylabel='Accuracy↑',
        aggregate='iqm', uncertainty='iqr',
        yticks_pad=0.1, yticks_labelsize=0.9 * plt.rcParams['ytick.labelsize'], force_n_yticks=3,
        legend_kwargs=dict(bbox_to_anchor=(1, 1), loc='lower right',
                           ncol=3, columnspacing=0.2, handletextpad=-0.3,
                           fontsize=0.9 * plt.rcParams['legend.fontsize'],
                           borderpad=0.2)
    )

    crit_diff_diagram(['firepbt_fmnist_20', 'firepbt_c10_20'], xlabel='Rank', ylabel='Population size')

    plot_best_solution_fitness_avg_over_settings([
        ['pbt_hopper_15', 'pbt_hopper_10', 'pbt_hopper_16'],
        ['pb2rand_hopper_15', 'pb2rand_hopper_10', 'pb2rand_hopper_16'],
        ['pb2mix_hopper_15', 'pb2mix_hopper_10', 'pb2mix_hopper_16'],
        ['bgpbt_hopper_15', 'bgpbt_hopper_10', 'bgpbt_hopper_16'],
        ['firepbt_hopper_15', 'firepbt_hopper_10', 'firepbt_hopper_16']],
        ['pbt', 'pb2rand', 'pb2mix', 'bgpbt', 'firepbt'],
        ['8', '22', '50'],
        seeds=list(range(7)), split=split,
        xlabel='Population size', ylabel='Score↑',
        aggregate='iqm', uncertainty='iqr',
        yticks_pad=0.1, yticks_labelsize=0.8 * plt.rcParams['ytick.labelsize'], force_n_yticks=3,
        legend_kwargs=dict(bbox_to_anchor=(1, 1), loc='lower right',
                           ncol=3, columnspacing=0.2, handletextpad=-0.3,
                           fontsize=0.9 * plt.rcParams['legend.fontsize'],
                           borderpad=0.2)
    )

    plot_best_solution_fitness_avg_over_settings([
        ['pbt_humanoid_18', 'pbt_humanoid_13', 'pbt_humanoid_19'],
        ['pb2rand_humanoid_18', 'pb2rand_humanoid_13', 'pb2rand_humanoid_19'],
        ['pb2mix_humanoid_18', 'pb2mix_humanoid_13', 'pb2mix_humanoid_19'],
        ['bgpbt_humanoid_18', 'bgpbt_humanoid_13', 'bgpbt_humanoid_19'],
        ['firepbt_humanoid_18', 'firepbt_humanoid_13', 'firepbt_humanoid_19']],
        ['pbt', 'pb2rand', 'pb2mix', 'bgpbt', 'firepbt'],
        ['8', '22', '50'],
        seeds=list(range(7)), split=split,
        xlabel='Population size', ylabel='Score↑',
        aggregate='iqm', uncertainty='iqr',
        yticks_pad=0.1, yticks_labelsize=0.75 * plt.rcParams['ytick.labelsize'], force_n_yticks=3,
        legend_kwargs=dict(bbox_to_anchor=(1, 1), loc='lower right',
                           ncol=3, columnspacing=0.2, handletextpad=-0.3,
                           fontsize=0.9 * plt.rcParams['legend.fontsize'],
                           borderpad=0.2)
    )

    crit_diff_diagram(['firepbt_hopper_10', 'firepbt_humanoid_13'], xlabel='Rank', ylabel='Population size')


def figure_8():
    ### wall-clock time

    ##### (a)
    plot_time_avg_over_settings([
        ['pbt_hopper_14', 'pbt_hopper_12', 'pbt_hopper_10', 'pbt_hopper_13'],
        ['pb2rand_hopper_14', 'pb2rand_hopper_12', 'pb2rand_hopper_10', 'pb2rand_hopper_13'],
        ['pb2mix_hopper_14', 'pb2mix_hopper_12', 'pb2mix_hopper_10', 'pb2mix_hopper_13'],
        ['bgpbt_hopper_14', 'bgpbt_hopper_12', 'bgpbt_hopper_10', 'bgpbt_hopper_13'],
        ['firepbt_hopper_14', 'firepbt_hopper_12', 'firepbt_hopper_10', 'firepbt_hopper_13']],
        ['pbt', 'pb2rand', 'pb2mix', 'bgpbt', 'firepbt'],
        ['3', '10', '30', '100'],
        seeds=list(range(7)),
        xlabel='Number of steps', ylabel='Time (hours)↓',
        aggregate='mean', uncertainty='std',
        cumulative=True, time_type='train_and_eval_times',
        time_div_by=3600,
        figsize=(6, 3),
        legend_on_the_side=True,
    )

    #### (b)
    plot_time_avg_over_settings([
        ['pbt_hopper_11', 'pbt_hopper_17', 'pbt_hopper_10'],
        ['pb2rand_hopper_11', 'pb2rand_hopper_17', 'pb2rand_hopper_10'],
        ['skip', 'pb2mix_hopper_17', 'pb2mix_hopper_10'],
        ['bgpbt_hopper_11', 'bgpbt_hopper_17', 'bgpbt_hopper_10'],
        ['firepbt_hopper_11_timing', 'firepbt_hopper_17', 'firepbt_hopper_10_timing'],
        ],
        ['pbt', 'pb2rand', 'pb2mix', 'bgpbt', 'firepbt'],
        ['small', 'medium', 'large'],
        seeds=list(range(7)),
        xlabel='Search space', ylabel='Time (minutes)↓',
        aggregate='mean', uncertainty='std',
        cumulative=True, time_type='update_times',
        time_div_by=60,
        figsize=(6, 3),
        legend_on_the_side=True,
    )

    #### (c)
    plot_time_avg_over_settings([
        ['pbt_hopper_15', 'pbt_hopper_10', 'pbt_hopper_16'],
        ['pb2rand_hopper_15', 'pb2rand_hopper_10', 'pb2rand_hopper_16'],
        ['pb2mix_hopper_15', 'pb2mix_hopper_10', 'pb2mix_hopper_16'],
        ['bgpbt_hopper_15', 'bgpbt_hopper_10', 'bgpbt_hopper_16'],
        ['firepbt_hopper_15', 'firepbt_hopper_10_timing', 'firepbt_hopper_16'],
        ],
        ['pbt', 'pb2rand', 'pb2mix', 'bgpbt', 'firepbt'],
        ['8', '22', '50'],
        seeds=list(range(7)),
        xlabel='Population size', ylabel='Time (minutes)↓',
        aggregate='mean', uncertainty='std',
        cumulative=True, time_type='update_times',
        time_div_by=60,
        time_div_by_per_setting=[8, 22, 50],
        figsize=(6, 3),
        legend_on_the_side=True,
    )


def figure_9():
    ### Generalization

    #### Cifar100
    plot_fitness_at_ticks_avg_many_seeds_many_exps(['pbt_c100_01', 'pb2rand_c100_01', 'pb2mix_c100_01', 'bgpbt_c100_01', 'firepbt_c100_01'],
                                   fitness_transform=lambda x: 100*x, yscale='linear', seeds=list(range(5)),
                                                   ylim={'bottom': 60, 'top': 76.5}, xlabel='SGD steps', ylabel='Accuracy↑',
                                                   aggregate='iqm', uncertainty='iqr',
                                                   legend_kwargs=dict(columnspacing=0.7,
                                                                      fontsize=0.9 * plt.rcParams['legend.fontsize'],
                                                                      borderpad=0.2, handlelength=0.7,
                                                                      handletextpad=0.2, ncol=1),
                                                   figsize=(5, 3),
                                                   legend_on_the_side=True)

    #### TinyImageNet
    plot_fitness_at_ticks_avg_many_seeds_many_exps(['pbt_timg_12', 'pb2rand_timg_12', 'pb2mix_timg_12', 'bgpbt_timg_12', 'firepbt_timg_12'],
                                   fitness_transform=lambda x: 100*x, yscale='linear', seeds=list(range(5)),
                                                   ylim={'bottom': 20, 'top': 50}, xlabel='SGD steps', ylabel='Accuracy↑',
                                                   aggregate='iqm', uncertainty='iqr',
                                                   legend_kwargs=dict(columnspacing=0.7,
                                                                      fontsize=0.9 * plt.rcParams['legend.fontsize'],
                                                                      borderpad=0.2, handlelength=0.7,
                                                                      handletextpad=0.2, ncol=1),
                                                   legend_on_the_side=True,
                                                   xaxis_number_of_zeros=3,
                                                   figsize=(5,3),
                                                   )


    #### Pusher
    plot_fitness_at_ticks_avg_many_seeds_many_exps(['pbt_pusher_01', 'pb2rand_pusher_01', 'pb2mix_pusher_01', 'bgpbt_pusher_01', 'firepbt_pusher_01'],
                                   yscale='linear', seeds=list(range(7)),
                                   ylim={'bottom': -350, 'top': -180}, xlabel='Environment steps', ylabel='Score↑',
                                   aggregate='iqm', uncertainty='iqr',
                                   if_leftalign_xlabel=True,
                                   pad_inches=0.0,
                                   legend_on_the_side=True,
                                   figsize=(5,3),
                                   legend_kwargs=dict(columnspacing=0.7,
                                                      fontsize=0.9 * plt.rcParams['legend.fontsize'],
                                                      borderpad=0.2, handlelength=0.7,
                                                      handletextpad=0.2, ncol=1),
                                   yticks_pad=0, yticks_labelsize=0.9 * plt.rcParams['ytick.labelsize'],
                                   force_n_yticks=3,
                                   )

    #### Walker
    plot_fitness_at_ticks_avg_many_seeds_many_exps(
        ['pbt_walker_01', 'pb2rand_walker_01', 'pb2mix_walker_01', 'bgpbt_walker_01', 'firepbt_walker_01'],
        yscale='linear', seeds=list(range(7)),
        ylim={'bottom': 800, 'top': 1300}, xlabel='Environment steps', ylabel='Score↑',
        aggregate='iqm', uncertainty='iqr',
        if_leftalign_xlabel=True,
        pad_inches=0.0,
        legend_on_the_side=True,
        figsize=(5, 3),
        legend_kwargs=dict(columnspacing=0.7,
                           fontsize=0.9 * plt.rcParams['legend.fontsize'],
                           borderpad=0.2, handlelength=0.7,
                           handletextpad=0.2, ncol=1),
        yticks_pad=0, yticks_labelsize=0.9 * plt.rcParams['ytick.labelsize'],
        force_n_yticks=3,
    )

def figure_11():
    ### TinyImagenet with small search space
    plot_fitness_at_ticks_avg_many_seeds_many_exps(
        ['pbt_timg_13', 'pb2rand_timg_13', 'bgpbt_timg_13', 'firepbt_timg_13'],
        fitness_transform=lambda x: 100 * x, yscale='linear', seeds=list(range(5)),
        ylim={'bottom': 15, 'top': 40}, xlabel='SGD steps', ylabel='Accuracy↑',
        aggregate='iqm', uncertainty='iqr',
        legend_kwargs=dict(columnspacing=0.7,
                           fontsize=0.9 * plt.rcParams['legend.fontsize'],
                           borderpad=0.2, handlelength=0.7,
                           handletextpad=0.2, ncol=1),
        legend_on_the_side=True,
        xaxis_number_of_zeros=3,
        figsize=(5, 3),
    )

def figure_15(split):
    # HEATMAP

    plot_iqm_diff_heatmap(
        [
            ['pbt_c10_28', 'pbt_c10_27', 'pbt_c10_26', 'pbt_c10_25'],
            ['pb2rand_c10_28', 'pb2rand_c10_27', 'pb2rand_c10_26', 'pb2rand_c10_25'],
            ['bgpbt_c10_28', 'bgpbt_c10_27', 'bgpbt_c10_26', 'bgpbt_c10_25'],
            ['firepbt_c10_28', 'firepbt_c10_27', 'firepbt_c10_26', 'firepbt_c10_25']
        ],
        [
            ['pbt_c10_18', 'pbt_c10_17', 'pbt_c10_16', 'pbt_c10_15'],
            ['pb2rand_c10_18', 'pb2rand_c10_17', 'pb2rand_c10_16', 'pb2rand_c10_15'],
            ['bgpbt_c10_18', 'bgpbt_c10_17', 'bgpbt_c10_16', 'bgpbt_c10_15'],
            ['firepbt_c10_18', 'firepbt_c10_17', 'firepbt_c10_16', 'firepbt_c10_15']
        ],
        ['pbt', 'pb2rand', 'bgpbt', 'firepbt'],
        ['3', '10', '30', '100'],
        seeds=list(range(5)), split=split, fitness_transform=lambda x: 100 * x,
        xlabel='Number of steps', ylabel='Algorithm',
        aggregate='iqm', title='', vmin=-3, vmax=3
    )

    plot_iqm_diff_heatmap([
            ['pbt_humanoid_38', 'pbt_humanoid_37', 'pbt_humanoid_36', 'pbt_humanoid_35'],
            ['pb2rand_humanoid_38', 'pb2rand_humanoid_37', 'pb2rand_humanoid_36', 'pb2rand_humanoid_35'],
            ['pb2mix_humanoid_38','pb2mix_humanoid_37', 'pb2mix_humanoid_36', 'pb2mix_humanoid_35'],
            ['bgpbt_humanoid_38','bgpbt_humanoid_37', 'bgpbt_humanoid_36', 'bgpbt_humanoid_35'],
            ['firepbt_humanoid_38','firepbt_humanoid_37', 'firepbt_humanoid_36', 'firepbt_humanoid_35']
        ],
        [
            ['pbt_humanoid_17', 'pbt_humanoid_14', 'pbt_humanoid_13', 'pbt_humanoid_12'],
            ['pb2rand_humanoid_17', 'pb2rand_humanoid_14', 'pb2rand_humanoid_13', 'pb2rand_humanoid_12'],
            ['pb2mix_humanoid_17', 'pb2mix_humanoid_14', 'pb2mix_humanoid_13', 'pb2mix_humanoid_12'],
            ['bgpbt_humanoid_17', 'bgpbt_humanoid_14', 'bgpbt_humanoid_13', 'bgpbt_humanoid_12'],
            ['firepbt_humanoid_17', 'firepbt_humanoid_14', 'firepbt_humanoid_13', 'firepbt_humanoid_12']
        ],
        ['pbt', 'pb2rand', 'pb2mix', 'bgpbt', 'firepbt'],
        ['3', '10', '30', '100'],
        seeds=list(range(7)),
        split=split,
        xlabel='Number of steps', ylabel='Algorithm', title='',
        aggregate='iqm', fmt=".0f", vmin=-2500, vmax=2500,
    )

def figure_16(split):
    # vary perturbation factor in fire-pbt on rl tasks

    plot_best_solution_fitness_avg_over_settings([
        ['firepbt_hopper_10', 'firepbt_hopper_27'],
    ],
        ['firepbt'],
        ['2.0', '1.25'],
        seeds=list(range(7)), split=split,
        xlabel='Perturbation factor', ylabel='Score↑',
        aggregate='iqm', uncertainty='iqr',
        yticks_pad=0.1, yticks_labelsize=0.9 * plt.rcParams['ytick.labelsize'], force_n_yticks=3,
        legend_kwargs=dict(bbox_to_anchor=(1, 1), loc='lower right',
                           ncol=3, columnspacing=0.2, handletextpad=-0.3,
                           fontsize=0.9 * plt.rcParams['legend.fontsize'],
                           borderpad=0.2)
    )

    plot_best_solution_fitness_avg_over_settings([
        ['firepbt_humanoid_13', 'firepbt_humanoid_39'],
    ],
        ['firepbt'],
        ['2.0', '1.25'],
        seeds=list(range(7)), split=split,
        xlabel='Perturbation factor', ylabel='Score↑',
        aggregate='iqm', uncertainty='iqr',
        yticks_pad=0.1, yticks_labelsize=0.75 * plt.rcParams['ytick.labelsize'], force_n_yticks=3,
        legend_kwargs=dict(bbox_to_anchor=(1, 1), loc='lower right',
                           ncol=3, columnspacing=0.2, handletextpad=-0.3,
                           fontsize=0.9 * plt.rcParams['legend.fontsize'],
                           borderpad=0.2)
    )

def compute_gpu_hours():
    ####### Runtimes
    clean_runtimes()

    ################## Fig. 5 (a,b)
    exp_names = [f'{algo}_{task}_{exp_indices}'
                 for algo in ['pbt', 'pb2rand', 'bgpbt', 'firepbt']
                 for task in ['fmnist', 'c10']
                 for exp_indices in [15, 16, 17, 18]]
    store_runtimes(exp_names, seeds_include=(0,1,2,3,4))

    ################## Fig. 5 (c)
    exp_names = [f'{algo}_{task}_{exp_indices}'
                 for algo in ['pbt', 'pb2rand', 'pb2mix', 'bgpbt', 'firepbt']
                 for task in ['hopper']
                 for exp_indices in [10, 12, 13, 14]]
    store_runtimes(exp_names, seeds_include=(0, 1, 2, 3, 4, 5, 6))

    ################## Fig. 5 (d)
    exp_names = [f'{algo}_{task}_{exp_indices}'
                 for algo in ['pbt', 'pb2rand', 'pb2mix', 'bgpbt', 'firepbt']
                 for task in ['humanoid']
                 for exp_indices in [12, 13, 14, 17]]
    store_runtimes(exp_names, seeds_include=(0, 1, 2, 3, 4, 5, 6))

    ################## Fig. 6 (a,b)
    exp_names = [f'{algo}_{task}_{exp_indices}'
                 for algo in ['pbt', 'pb2rand', 'pb2mix', 'bgpbt', 'firepbt']
                 for task in ['fmnist', 'c10']
                 for exp_indices in [19, 20]]
    store_runtimes(exp_names, seeds_include=(0,1,2,3,4))

    ################# Fig. 6 (c)
    exp_names = [f'{algo}_{task}_{exp_indices}'
                 for algo in ['pbt', 'pb2rand', 'bgpbt', 'firepbt']
                 for task in ['hopper']
                 for exp_indices in [11, 17]]
    store_runtimes(exp_names, seeds_include=(0, 1, 2, 3, 4, 5, 6))
    store_runtimes([f'pb2mix_hopper_17'], seeds_include=(0, 1, 2, 3, 4, 5, 6))

    ################# Fig. 6 (d)
    exp_names = [f'{algo}_{task}_{exp_indices}'
                 for algo in ['pbt', 'pb2rand', 'bgpbt', 'firepbt']
                 for task in ['humanoid']
                 for exp_indices in [22, 23]]
    store_runtimes(exp_names, seeds_include=(0, 1, 2, 3, 4, 5, 6))
    store_runtimes([f'pb2mix_humanoid_23'], seeds_include=(0, 1, 2, 3, 4, 5, 6))

    ################# Fig. 7 (a,b)
    exp_names = [f'{algo}_{task}_{exp_indices}'
                 for algo in ['pbt', 'pb2rand', 'pb2mix', 'bgpbt', 'firepbt']
                 for task in ['fmnist', 'c10']
                 for exp_indices in [22, 24]]
    store_runtimes(exp_names, seeds_include=(0,1,2,3,4))

    ################ Fig. 7 (c)
    exp_names = [f'{algo}_{task}_{exp_indices}'
                 for algo in ['pbt', 'pb2rand', 'pb2mix', 'bgpbt', 'firepbt']
                 for task in ['hopper']
                 for exp_indices in [15, 16]]
    store_runtimes(exp_names, seeds_include=(0, 1, 2, 3, 4, 5, 6))

    ################ Fig. 7 (d)
    exp_names = [f'{algo}_{task}_{exp_indices}'
                 for algo in ['pbt', 'pb2rand', 'pb2mix', 'bgpbt', 'firepbt']
                 for task in ['humanoid']
                 for exp_indices in [18, 19]]
    store_runtimes(exp_names, seeds_include=(0, 1, 2, 3, 4, 5, 6))

    sum_runtimes()

if __name__ == '__main__':
    set_plot_style_paper()

    figure_2()
    # figure_3()
    # figure_4a()
    # figure_4b()

    # figure_5(split='val') # split='test' for Fig. 12
    # figure_6(split='val') # split='test' for Fig. 13
    # figure_7(split='val') # split='test' for Fig. 14
    # figure_8()
    # figure_9()

    # critical difference diagrams of Fig. 10 are created in code for the figure
    # of the associated experiments (search for crit_diff_diagram in this file)

    # figure_11()

    # compute_gpu_hours()

    # figure_15(split='val')
    # figure_16(split='val')