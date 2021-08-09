[![RyStats Actions Status](https://github.com/eribean/RyStats/workflows/RyStats/badge.svg)](https://github.com/eribean/RyStats/actions)
[![codecov.io](https://codecov.io/gh/eribean/RyStats/coverage.svg?branch=main)](https://codecov.io/gh/eribean/RyStats)
[![CodeFactor](https://www.codefactor.io/repository/github/eribean/RyStats/badge)](https://www.codefactor.io/repository/github/eribean/RyStats)
[![PyPI version](https://badge.fury.io/py/RyStats.svg)](https://badge.fury.io/py/RyStats)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)


# RyStats
A collection of statistical packages writtten in python.

## Factor Analysis
Several factor analysis routines are implemented including:

1. Principal Components Analysis
2. Principal Axis Factoring
3. Maximum Likelihood
4. Minimum Rank

### Rotation
1. Orthogonal / Oblique
2. Procrustes

### Dimensionality Assessment
 1. Parallel Analysis
 2. MAP (Minimum Average Partial)

### Misc.
1. Polychoric Correlations

# Dependencies
1. Numpy
2. Scipy
3. Bokeh (for plotting)

# Plotting Tools
Plotting Tools require Bokeh and are made to be used within a Jupyter Notebook

### Plot Functions
1. correlation_image: creates a heatmap of a correlation matrix
2. loading_image: creates a heatmap of a loadings matrix
3. scree_plot: plots the eigenvalues for dimensionality assessment
4. loading_table: creates a table that displays the loadings matrix

# Installation
```
pip install . -t $PYTHONPATH --upgrade
```


## Unittests

**Pytest** with coverage
```
pytest --cov=RyStats --cov-report term
```

**Nosetest** with coverage
```
nosetests --with-coverage --cover-package=RyStats
```

## Contact

Ryan Sanchez  
ryan.sanchez@gofactr.com

## License

MIT License

Copyright (c) 2021 Ryan C. Sanchez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
