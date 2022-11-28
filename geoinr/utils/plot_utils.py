import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


def build_plot_from_horizon_metrics(scalar_means: np.ndarray, residual_means: np.ndarray,
                                    variance: np.ndarray, filename: str):
    scale_for_variance = 20 / 0.001
    variance_s = variance * scale_for_variance

    n = scalar_means.size
    x = np.arange(n)
    width = 0.3 * np.exp(-(n - 1)**2) + 0.3
    legend_txt = str(round(variance.min(), 4)) + r' $min$ $s_{var}$'

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.scatter(x, scalar_means, s=variance_s, label=legend_txt, alpha=0.4)
    ax.plot(x, scalar_means, lw=3)
    ax.set_xlabel('horizon i')
    ax.set_ylabel('scalar mean')
    ax.legend(loc="upper left", markerscale=0.3, frameon=False)
    ax2 = ax.twinx()
    ax2.bar(x, residual_means, width=width, color="red", alpha=0.4)
    ax2.set_ylabel("residual")
    ax.set_xticks(x)
    fig.savefig(filename)


def build_plot_from_unit_metrics(class_ids: np.ndarray, residual_means: np.ndarray, filename: str):

    n = residual_means.size
    width = 0.3 * np.exp(-(n - 1)**2) + 0.3

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.bar(class_ids, residual_means, width=width, color="red")
    ax.set_xlabel('Class Ids')
    ax.set_ylabel('Residual Means')
    ax.set_xticks(class_ids)
    fig.savefig(filename)


# yy = np.linspace(-1, 1, 50)
# rr = np.random.uniform(0.039, 2.938, 50)
# ss = np.random.uniform(0.001, 0.017, 50)
# build_plot_from_horizon_metrics(yy, rr, ss, "test1.png")
#
# y = np.array([0.805, 0.770])
# s = np.array([0.00017, 0.05977])
# r = np.array([0.018, 0.022])
# build_plot_from_horizon_metrics(y, r, s, "test2.png")
#
# t = 5
