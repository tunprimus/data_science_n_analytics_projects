#!/usr/bin/env python3
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

# Define constants
GOLDEN_RATIO = 1.618033989
FIG_WIDTH = 20
FIG_HEIGHT = FIG_WIDTH / GOLDEN_RATIO
FIG_SIZE = (FIG_WIDTH, FIG_HEIGHT)
FIG_DPI = 72
RANDOM_SAMPLE_SIZE = 13
RANDOM_SEED = 42
FONT_SIZE = FIG_HEIGHT
TITLE_SIZE = 23

colours = cycler(color=plt.get_cmap("tab10").colors)  # ["b", "r", "g"]

mpl.style.use("ggplot")
mpl.rcParams["figure.figsize"] = FIG_SIZE
mpl.rcParams["axes.facecolor"] = "white"
mpl.rcParams["axes.grid"] = True
mpl.rcParams["axes.prop_cycle"] = colours
mpl.rcParams["axes.linewidth"] = 1
mpl.rcParams["axes.autolimit_mode"] = "round_numbers"
mpl.rcParams["grid.color"] = "lightgray"
mpl.rcParams["xtick.color"] = "black"
mpl.rcParams["ytick.color"] = "black"
mpl.rcParams["font.size"] = FONT_SIZE
mpl.rcParams["figure.titlesize"] = TITLE_SIZE
mpl.rcParams["figure.dpi"] = FIG_DPI
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.directory"] = "../../reports/figures"
mpl.rcParams["ps.papersize"] = "A4"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

