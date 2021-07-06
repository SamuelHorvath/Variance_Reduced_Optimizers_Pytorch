import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.utils import get_best_runs, get_key
import numpy as np
import os

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['legend.fontsize'] = 'x-large'
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['axes.labelsize'] = 'x-large'

PLOT_PATH = './plots/'
MARKERS = ['o', 'v', 's', 'P', 'p', '*', 'H', 'X', 'D']


def plot(exps, log_scale=True, legend=None, file=None,
         x_label='epochs', y_label=None):
    fig, ax = plt.subplots()
    metric_name = get_key(exps[0].train_metric) + exps[0].metric

    for i, exp in enumerate(exps):
        runs = get_best_runs(exp)
        print(exp.method, '...', len(runs))
        plot_mean_std(ax, runs, metric_name, i)

    if log_scale:
        ax.set_yscale('log')
    if legend is not None:
        ax.legend(legend)

    ax.set_xlabel(x_label)
    if y_label is None:
        ax.set_ylabel(metric_name)
    else:
        ax.set_ylabel(y_label)

    fig.tight_layout()
    if file is not None:
        os.makedirs(PLOT_PATH, exist_ok=True)
        plt.savefig(PLOT_PATH + file + '.pdf')
    plt.show()


def plot_mean_std(ax, runs, metric_name, i):
    # filter out non-complete runs
    max_len = np.max([len(run[metric_name]) for run in runs])
    runs = [run for run in runs if len(run[metric_name]) == max_len]

    if metric_name.startswith('train'):
        quant = np.array([run[metric_name][1:] for run in runs])
    else:
        quant = np.array([run[metric_name] for run in runs])
    axis = 0

    mean = np.nanmean(quant, axis=axis)
    std = np.nanstd(quant, axis=axis)

    x = np.arange(0, len(mean))
    ax.plot(x, mean, marker=MARKERS[i], markersize=12, markevery=len(x) // 10)
    ax.fill_between(x, mean + std, mean - std, alpha=0.4)
