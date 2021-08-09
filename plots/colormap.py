import numpy as np

from bokeh.plotting import figure


__all__ = ["colormap_legend", "COLORMAP", "COLORMAPGP"]


# Fixed colormap of 21 levels
COLORMAP = ["#0000ff", "#0d0df3", "#1b1be7", "#2929db", "#3636cf", "#4444c4",
            "#5252b8", "#5f5fac", "#6d6da0", "#7a7a94", "#888888", "#947b7b",
            "#a06d6d", "#ac5f5f", "#b85252", "#c34444", "#cf3636", "#db2929",
            "#e71b1b", "#f30e0e", "#ff0000"]

COLORMAPGP = ["#00e900", "#0bdc0c", "#15cf18", "#20c224", "#2ab530", "#35a83b",
              "#3f9b44", "#4a8e4d", "#558256", "#5f755f", "#6a6868", "#675e86",
              "#6555a4", "#624cc3", "#6042e1", "#5e39ff", "#612eff", "#6522ff",
              "#6917ff", "#6c0bff", "#7000ff"]


def colormap_legend(cmap=0):
    """Returns bokeh figure containing the colormap"""
    colormap = [COLORMAP, COLORMAPGP][cmap]
    the_data = np.linspace(-1, 1, 21)
    labels = list(map(lambda x: str(x.round(2)), the_data[::2]))

    # Match the transparency of the
    orig = np.abs(the_data)
    alpha = orig + 0.3
    alpha[orig >= 0.25] = 0.8
    alpha[orig >= 0.6] = 1.0
    alpha = alpha.tolist()

    # Create the figure
    p = figure(y_axis_location="right", tools="", y_range=[-0.5, 20.5])

    # Formatting Options
    p.plot_width = 50
    p.plot_height = 400
    p.toolbar.logo = None
    p.toolbar_location = None
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "8pt"
    p.axis.major_label_text_font_style = 'bold'
    p.axis.major_label_standoff = 2
    p.xaxis.major_label_text_font_size = "0pt"
    p.xaxis.minor_tick_line_color = None
    p.yaxis.ticker = list(range(0, 21, 2))
    p.yaxis.major_label_overrides = {ndx: label for ndx, label in
                                     zip(range(0, 21, 2), labels)}

    # Create the legend
    p.rect(np.zeros(21), np.arange(21), 0.9, 0.9, color=colormap,
           alpha=alpha, line_color=None)

    return p