import numpy as np

from bokeh.plotting import figure


__all__ = ["scree_plot"]


def scree_plot(eigenvalues, parallel_eigs=None, plot_difference=True):
    """Creates a scree plot from eigenvalues.

    Plots extracted eigenvalues and possibly results from parallel
    analysis on same axes.

    Args:
        eigenvalues: input array of eigenvalues
        parallel_eigs:  (optional) input array of eigenvalues from parallel analysis
        plot_difference: (Boolean) use difference of eigenvalues and parallel_eigs

    Returns:
        bokeh_handle: bokeh figure handle to be used with show(), if parallel_eigs is
                      supplied, then the crossing point is marked
    """
    input_x = np.arange(eigenvalues.size) + 1

    if parallel_eigs is None:
        parallel_eigs = np.zeros_like(eigenvalues)

    difference_eigs = eigenvalues - parallel_eigs

    # first ndx where eigenvalue is zero
    factor_ndx = np.where(difference_eigs < 0)[0]

    # bokeh figure
    p = figure(plot_width=600, plot_height=400)

    if factor_ndx.size > 0:
        factor_ndx = factor_ndx[0]
        valid_eigs = eigenvalues[:factor_ndx]
        valid_pe = parallel_eigs[:factor_ndx]

        invalid_eigs = eigenvalues[factor_ndx:]
        invalid_pe = parallel_eigs[factor_ndx:]

        if plot_difference:
            p.line(input_x[:factor_ndx], valid_eigs - valid_pe, line_width=2, 
                   legend_label=f"Valid Eigenvalues = {valid_eigs.size}")
            p.line(input_x[factor_ndx-1:factor_ndx+1], 
                   [valid_eigs[-1] - valid_pe[-1], invalid_eigs[0]-invalid_pe[0]], 
                   line_width=2, line_color="purple", line_dash="dotted")
            p.line(input_x[factor_ndx:], invalid_eigs - invalid_pe, 
                   line_width=2, line_color="red", legend_label="Invalid Factors")

            p.circle(input_x[:factor_ndx], valid_eigs - valid_pe, 
                     size=4, fill_color="white")
            p.circle(input_x[factor_ndx:], invalid_eigs - invalid_pe, 
                     size=4, color='red', fill_color="white")

        else:
            p.line(input_x[:factor_ndx], valid_eigs , line_width=2, 
                   legend_label=f"Valid Eigenvalues = {valid_eigs.size}")
            p.line(input_x[factor_ndx-1:factor_ndx+1], [valid_eigs[-1], invalid_eigs[0]], 
                   line_width=2, line_color="purple", line_dash="dotted")
            p.line(input_x[factor_ndx:], invalid_eigs, line_width=2, 
                   line_color="red", legend_label="Invalid Factors")

            p.circle(input_x[:factor_ndx], valid_eigs, size=4, fill_color="white")
            p.circle(input_x[factor_ndx:], invalid_eigs, size=4, color='red', fill_color="white")

            p.line(input_x, parallel_eigs, line_width=2, color='black', 
                   line_dash="dashed", legend_label="PA Eigenvalues")
            p.circle(input_x, parallel_eigs, size=4, fill_color='white', color='black')
    else:
        p.line(input_x, eigenvalues, line_width=2)
        p.circle(input_x, eigenvalues, fill_color='white', size=4)

        if parallel_eigs.sum() > 0:
            p.line(input_x, parallel_eigs, line_width=2, color='black', line_dash="dashed")
            p.circle(input_x, parallel_eigs, fill_color='white', size=4, color="black")

    return p