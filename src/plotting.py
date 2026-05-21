from __future__ import annotations

from functools import wraps


def configure_matplotlib(plt) -> None:
    """Apply thesis-readable defaults to all generated matplotlib figures."""
    if getattr(plt, "_cody_plotting_configured", False):
        return

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.22,
            "font.size": 16,
            "axes.titlesize": 21,
            "axes.labelsize": 18,
            "axes.titlepad": 14,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
            "legend.title_fontsize": 15,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.borderaxespad": 0.8,
        }
    )

    original_subplots = plt.subplots

    @wraps(original_subplots)
    def readable_subplots(*args, **kwargs):
        figsize = kwargs.get("figsize")
        if figsize is None:
            kwargs["figsize"] = (11, 7)
        else:
            width, height = figsize
            kwargs["figsize"] = (max(width * 1.18, 10), max(height * 1.18, 6))
        return original_subplots(*args, **kwargs)

    figure_cls = plt.Figure
    original_savefig = figure_cls.savefig

    @wraps(original_savefig)
    def readable_savefig(self, *args, **kwargs):
        kwargs.setdefault("dpi", 220)
        kwargs.setdefault("bbox_inches", "tight")
        kwargs.setdefault("pad_inches", 0.22)
        return original_savefig(self, *args, **kwargs)

    axes_cls = plt.Axes
    original_legend = axes_cls.legend

    @wraps(original_legend)
    def readable_legend(self, *args, **kwargs):
        handles, labels = self.get_legend_handles_labels()
        label_count = len([label for label in labels if label and not label.startswith("_")])
        if "bbox_to_anchor" not in kwargs:
            kwargs.setdefault("loc", "upper center")
            kwargs["bbox_to_anchor"] = (0.5, 1.22)
            kwargs.setdefault("ncol", min(4, max(1, label_count)))
        kwargs.setdefault("frameon", True)
        kwargs.setdefault("fontsize", 14)
        return original_legend(self, *args, **kwargs)

    plt.subplots = readable_subplots
    figure_cls.savefig = readable_savefig
    axes_cls.legend = readable_legend
    plt._cody_plotting_configured = True
