import matplotlib.pyplot as plt


def configure_plots():

    plt.rcParams.update({

        "font.size": 9,

        "axes.labelsize": 9,

        "xtick.labelsize": 8,

        "ytick.labelsize": 8,

        "legend.fontsize": 8,

        "lines.linewidth": 1.4,

        "lines.markersize": 4.5,

        "figure.dpi": 150,

        "savefig.dpi": 300,

        "savefig.bbox": None,

        "axes.grid": True,

        "grid.alpha": 0.25,

        "legend.frameon": False,

        "pdf.fonttype": 42,

        "ps.fonttype": 42,
    })


def figure_size(width="single",ratio=0.68):
    widths={"single":3.35,"double":6.9}
    w=widths[width]
    return w,ratio*w