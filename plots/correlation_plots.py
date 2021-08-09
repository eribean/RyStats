from itertools import product

import numpy as np

from bokeh.plotting import figure
from bokeh.layouts import row

from RyStats.plots.colormap import *


__all__ = ['correlation_image', 'loading_image']


def correlation_image(the_data, labels=None, cmap=0):
    """Outputs a bokeh figure containing the correlation_matrix.

    Use this to generate an image of the correlation matrix for easy viewing.

    Args:
        the_data: input correlation_matrix
        labels: list of strings corresponding to the correlation matrix
        cmap: [(0) | 1] 0 is Red / Blue, 1 is Green / Purple

    Returns:
        figure_handle: bokeh figure handle to be used with show(), legend is affixed
                       to the figure
    """
    colormap = [COLORMAP, COLORMAPGP][cmap]
    binned_data = np.digitize(the_data, np.linspace(-1, 1, len(colormap)))
    color = np.array(colormap)[np.rot90(binned_data, 2)-1].flatten().tolist()

    # Adjust the transparency value
    orig = np.rot90(np.abs(the_data), 2)
    alpha = orig + 0.3
    alpha[orig >= 0.25] = 0.8
    alpha[orig >= 0.6] = 1.0
    alpha = alpha.flatten().tolist()

    orientation = np.pi / 3
    if labels is None:
        labels = list(map(lambda x: str(x), range(the_data.shape[0])))
        orientation = 0.0

    xname, yname = list(zip(*product(labels, labels)))

    data = dict(xname=list(reversed(xname)), yname=list(reversed(yname)),
                colors=color, alphas=alpha,
                count=list(reversed(the_data.round(2).flatten())))

    # Create the figure
    p = figure(x_axis_location="above", tools="hover,save",
               y_range=list(reversed(labels)), x_range=labels,
               tooltips = [('Names', '@yname, @xname'), ('Correlation', '@count')])

    p.plot_width = 800
    p.plot_height = 800
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "5pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = orientation

    p.rect('xname', 'yname', 0.9, 0.9, source=data,
           color='colors', line_color=None, alpha='alphas',
           hover_line_color='black', hover_color='colors')

    return row(p, colormap_legend(cmap))


def loading_image(the_data, cutoff=0.2, q_labels=None, f_labels=None,
                   plot_width=800, plot_height=800):
    """Outputs a bokeh figure containing the loading matrix.

    Args:
        the_data: input matrix to visualize
        cutoff: suppress all values less than cutoff: |data| < cutoff = 0
        q_labels: list of strings corresponding to the questions (rows)
        f_labels: list of strings corresponding to the factors (columns)
        plot_width: (int) width of plot in pixels
        plot_height: (int) height of plot in pixels

    Returns:
        figure_handle: bokeh figure handle to be used with show(), legend is affixed
                       to the figure
    """
    the_new_data = the_data.T.copy()
    mask = np.abs(the_new_data) < cutoff
    the_new_data[mask] = 0

    # Make the labels
    orientation = np.pi / 3
    if q_labels is None:
        q_labels = list(map(lambda x: str(x), range(the_data.shape[0])))

    if f_labels is None:
        f_labels = list(map(lambda x: str(x), range(the_data.shape[1])))
        orientation = 0.0

    colormap = COLORMAP
    binned_data = np.digitize(the_new_data, np.linspace(-1, 1, len(colormap)))
    color = np.array(colormap)[np.rot90(binned_data, 2)-1].flatten().tolist()

    # Adjust the transparency value
    orig = np.rot90(np.abs(the_new_data), 2)
    alpha = orig + 0.3
    alpha[orig >= 0.25] = 0.8
    alpha[orig >= 0.6] = 1.0
    alpha = alpha.flatten().tolist()
    xname, yname = list(zip(*product(f_labels, q_labels)))

    data = dict(xname=list(reversed(xname)), yname=list(reversed(yname)),
                colors=color, alphas=alpha,
                count=list(reversed(the_new_data.round(2).flatten())))

    # Create the figure
    p = figure(x_axis_location="above", tools="hover,save",
               y_range=list(reversed(q_labels)), x_range=f_labels,
               tooltips = [('Names', '@yname, @xname'), ('Value', '@count')])

    p.plot_width = plot_width
    p.plot_height = plot_height
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "5pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = orientation

    p.rect('xname', 'yname', 0.9, 0.9, source=data,
           color='colors', line_color=None, alpha='alphas',
           hover_line_color='black', hover_color='colors')

    return row(p, colormap_legend())