from itertools import product
from IPython.core.display import display, HTML

import numpy as np


__all__ = ['loading_table']


def loading_table(the_factors, cutoff=0.25, q_labels=None, f_labels=None,
                  html_only=False):
    """Creates a factor correlation table.

    Create an html table with maximum loading bolded and small loadings suppressed

    Args:
        the_factors: input array with individual factors as columns
        cutoff: value at which to suppress output (optional)
        q_labels: Labels for each row (optional)
        f_labels: Labels for each column (optional)
        html_only: (Boolean) Return html as a string

    Returns:
        output: rendered table | css/html string of table
    """
    rows, cols = the_factors.shape

    if q_labels is None:
        q_labels = tuple(map(lambda x: f"{x+1}", range(rows)))

    if f_labels is None:
        f_labels = tuple(map(lambda x: "Factor {}".format(x+1),
                         range(cols)))

    mask = np.abs(the_factors) < cutoff
    max_values = np.argmax(np.abs(the_factors), axis=1)

    css_string = """<style>
    table{
        background-color: #e4e4e4;
        width: TBApx;
        table-layout: auto;
        border-top: 5px double;
        border-bottom: 5px double;
        border-collapse: collapse;}

    tr {border-bottom: 1px solid black;}
    tr:nth-child(even) {background-color: #fdfdfd
    ;}
    td, th {text-align: right;}

    td, th {padding: 0.125em 0.5em 0.25em 0.5em;
            line-height: 1;}

    .left_a {text-align: left;}
    .max_v {font-weight: bold; color: blue;}
    </style>"""
    css_string = css_string.replace("TBA", str((cols + 1) * 100))
    the_string = ('\n<table>' +
                  '<tr style="border-bottom: 2px solid black;"><th></th>')

    for label in f_labels:
        the_string += '<th height="35px">{}</th>'.format(label)
    the_string += "</tr>"

    for row in range(rows):
        the_string += f'<tr><td class="left_a" height="30px">{q_labels[row]}</td>'
        for col in range(cols):
            if mask[row, col]:
                the_string += '<td>--&nbsp&nbsp</td>'
            elif max_values[row] == col:
                the_string += f'<td class="max_v">{the_factors[row, col]:.2f}</td>'
            else:
                the_string += f'<td>{the_factors[row, col]:.2f}</td>'
        the_string += "</tr>"
    the_string += "</table>"

    total_string = css_string + the_string

    # Not typically a fan of two returns, but needed for rendering
    if html_only:
        return total_string
    else:
        return display(HTML(total_string), metadata={'isolated': True})
