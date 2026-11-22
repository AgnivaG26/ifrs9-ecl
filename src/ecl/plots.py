"""Shared chart styling (validated categorical palette, recessive chrome)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def new(figsize=(9, 4.2), nrows=1, ncols=1):
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, facecolor=SURFACE)
    for a in (ax.flat if hasattr(ax, "flat") else [ax]):
        style(a)
    return fig, ax


def title(ax, text):
    ax.set_title(text, color=INK, loc="left", fontsize=12)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
