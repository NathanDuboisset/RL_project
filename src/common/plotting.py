import matplotlib.pyplot as plt
import numpy as np

from common.evaluation import EpisodeStats, ecdf


def plot_distributions(results: dict[str, EpisodeStats], path=None, title: str = ""):
    """Return histogram, boxplot (with outliers) and ECDF, plus the length
    histogram: the layout of Figures 2-5 of the report. Returns are not clipped.
    Saves and closes the figure if `path` is given, otherwise returns it (notebooks)."""
    names = list(results)
    colors = [f"C{i}" for i in range(len(names))]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))

    all_returns = np.concatenate([results[n].returns for n in names])
    bins = np.linspace(all_returns.min(), all_returns.max(), 41)
    for name, c in zip(names, colors):
        axes[0].hist(results[name].returns, bins=bins, alpha=0.45, label=name, color=c)
    axes[0].set(title="Episode return", xlabel="Return", ylabel="Frequency")
    axes[0].legend()

    bp = axes[1].boxplot([results[n].returns for n in names], tick_labels=names,
                         showfliers=True, patch_artist=True)
    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c)
        box.set_alpha(0.5)
    axes[1].set(title="Return boxplot (with outliers)", ylabel="Return")

    for name, c in zip(names, colors):
        axes[2].plot(*ecdf(results[name].returns), label=name, color=c, linewidth=2)
    axes[2].set(title="Return ECDF", xlabel="Return", ylabel="Cumulative proportion")
    axes[2].grid(alpha=0.2)
    axes[2].legend()

    all_lengths = np.concatenate([results[n].lengths for n in names])
    lbins = np.linspace(0, all_lengths.max(), 31)
    for name, c in zip(names, colors):
        axes[3].hist(results[name].lengths, bins=lbins, alpha=0.45, label=name, color=c)
    axes[3].set(title="Episode length", xlabel="Steps", ylabel="Frequency")
    axes[3].legend()

    n_eps = {results[n].returns.size for n in names}
    fig.suptitle(f"{title} (n={'/'.join(map(str, sorted(n_eps)))} episodes)")
    fig.tight_layout()
    if path is None:
        return fig
    fig.savefig(path, dpi=150)
    plt.close(fig)
