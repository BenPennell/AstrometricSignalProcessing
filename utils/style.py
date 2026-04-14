import matplotlib.pyplot as plt
import matplotlib
import numpy as np

'''
    Style choices for making plots how I like them
'''

# set tex font
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}",
})

# default ticks, font sizes
#matplotlib.rcParams['figure.figsize'] = (12, 10)
matplotlib.rcParams["axes.labelsize"] = 18
matplotlib.rcParams["axes.titlesize"] = 18
matplotlib.rcParams["legend.fontsize"] = 16
matplotlib.rcParams["figure.titlesize"] = 30
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14

# default to black
matplotlib.rcParams['axes.prop_cycle'] = plt.cycler(color=['black'])

# legend
matplotlib.rcParams['legend.frameon'] = False
matplotlib.rcParams["legend.handletextpad"] = 0

# presets
hist_defaults = {
    "histtype": "step",
    "linewidth": 3,
}

hist_cdf = {
    "cumulative": True,
    "density": True
}

dline = {
    "linestyle": "--",
    "color": "r",
}

# pastel colours for the solution types
colors = plt.get_cmap('Accent')(np.linspace(0, 1, 8))[:5]