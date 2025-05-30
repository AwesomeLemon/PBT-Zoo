import random
import scipy.stats

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor


def smooth(c_vals, exp_dir):
    gpr = GaussianProcessRegressor(random_state=0, normalize_y=True)

    inp_range = np.arange(0, 1, 1 / len(c_vals)).reshape(-1, 1)
    gpr.fit(inp_range, c_vals)
    smoothed = gpr.predict(inp_range)
    if_viz = False
    if if_viz:
        plt.plot(inp_range, c_vals, label='original')
        plt.plot(inp_range, smoothed, label='smoothed')
        plt.legend()
        # random hexadecimal string as a name
        p = Path(exp_dir) / 'smooth'
        p.mkdir(parents=True, exist_ok=True)
        random_hex = ''.join(random.choice('0123456789ABCDEF') for _ in range(10))
        plt.savefig(p / f'smooth_{random_hex}.png')
        plt.close()
    return smoothed


def find_overlap(c1_smooth, c2_smooth):
    # - this uses smoothed curves
    curve1_vals = c1_smooth
    curve2_vals = c2_smooth
    overlap_start = None
    if curve1_vals[0] < curve2_vals[0]:
        for i in range(1, len(curve1_vals)):
            if curve1_vals[i] >= curve2_vals[0]:
                break

        if curve1_vals[i] >= curve2_vals[0]:
            overlap_start = i, 0
    else:
        for i in range(len(curve2_vals)):
            if curve2_vals[i] >= curve1_vals[0]:
                break

        if curve2_vals[i] >= curve1_vals[0]:
            overlap_start = 0, i

    if overlap_start is None:
        return None, None

    l = min(len(curve1_vals) - overlap_start[0], len(curve2_vals) - overlap_start[1])
    return overlap_start, l


def best_score_diff(curve1, curve2, curve1_vals_smooth, curve2_vals_smooth, exp_dir):
    # curve1 and curve2 are lists of tuples (t, score). Note that this score is not symmetric
    # - find start of overlapping sections
    assert len(curve1) > 1, 'fewer than 2 points is not a curve'

    def _inner(curve1_vals, curve2_vals, c1_smooth, c2_smooth):
        overlap_start, l = find_overlap(c1_smooth, c2_smooth)

        if overlap_start is None:
            return None

        bsd = max(curve1_vals[overlap_start[0]:overlap_start[0] + l]) - \
                max(curve2_vals[overlap_start[1]:overlap_start[1] + l])
        return bsd

    curve1_vals = [x[1] for x in curve1]
    curve2_vals = [x[1] for x in curve2]
    bsd = _inner(curve1_vals, curve2_vals, curve1_vals_smooth, curve2_vals_smooth)
    if bsd is None:
        n_steps = 10
        c1_min, c1_max = min(curve1_vals), max(curve1_vals)
        c2_min, c2_max = min(curve2_vals), max(curve2_vals)

        bsd = 0
        if c1_min > c2_max:
            delta_min = c1_min - c2_max
            delta_max = c1_min - c2_min

            if np.allclose(delta_min, delta_max):
                return 0

            if delta_max < delta_min:
                delta_min, delta_max = delta_max, delta_min
            step = (delta_max - delta_min) / n_steps

            bsd_w_deltas = []
            for delta in np.arange(delta_min, delta_max, step):
                curve1_vals_shifted = [x - delta for x in curve1_vals]
                curve1_vals_shifted_smooth = smooth(curve1_vals_shifted, exp_dir)
                bsd_w_delta = _inner(curve1_vals_shifted, curve2_vals,
                                     curve1_vals_shifted_smooth, curve2_vals_smooth)
                if bsd_w_delta is None:
                    bsd_w_delta = 0
                bsd_w_deltas.append(bsd_w_delta)
            bsd = max(max(bsd_w_deltas), 0)

    return bsd


def binom_test(curve1, curve2, c1_smooth, c2_smooth):
    # curve1 and curve2 are lists of tuples (t, score). Note that this score is not symmetric
    # - same as best_score_diff, smooth & find overlap
    curve1_vals = [x[1] for x in curve1]
    curve2_vals = [x[1] for x in curve2]
    overlap_start, l = find_overlap(c1_smooth, c2_smooth)
    if overlap_start is None:
        if min(curve1_vals) > max(curve2_vals):
            return 0 # curve1 is always better
        if max(curve1_vals) < min(curve2_vals):
            return 2 # curve1 is always worse => return a large number (returning 1 is not enough because we compare against p_stat + smth_with_max_value_1)

        # if the curves are weird, try running the test on the entire curves
        overlap_start = 0, 0
        l = min(len(curve1_vals), len(curve2_vals))

    n_success = sum([v1 > v2 for v1, v2 in zip(curve1_vals[overlap_start[0]:overlap_start[0] + l],
                                               curve2_vals[overlap_start[1]:overlap_start[1] + l])])
    pvalue = scipy.stats.binomtest(n_success, l, alternative='greater').pvalue
    print(f'{n_success=} {overlap_start=} {l=}')

    print('<<<' * 10)
    print(f'{curve1_vals[overlap_start[0]:overlap_start[0] + l]}')
    print(f'{curve2_vals[overlap_start[1]:overlap_start[1] + l]}')
    print('>>>'*10)

    return pvalue
