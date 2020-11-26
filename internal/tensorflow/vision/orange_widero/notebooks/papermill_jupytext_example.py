# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
# ---

# ### Papermill + Jupytext example
#
#
# The following notebook, although in jupytext's .py format, can be executed by papermill by running -
# ```
# jupytext papermill_jupytext_example.py -o /tmp/papermill_jupytext_example.ipynb --set-kernel -
# ```
#
# to create an ipynb, and then -
# ```
# papermill /tmp/papermill_jupytext_example.ipynb /tmp/papermill_jupytext_example_executed.ipynb -p gamma 3
# ```
#
# to convert the executed notebook to html format run -
# ```
# jupyter nbconvert --to html /tmp/papermill_jupytext_example_executed.ipynb
# ```
#
# this generates the file -
# ```
# /tmp/papermill_jupytext_example_executed.html
# ```

import numpy as np
import matplotlib.pyplot as plt

# Note that the following cell has a tag named ``parameters``. 
# You can see a cell's tags by choosing "View => Cell Toolbar => Tags" in the jupyter menu.

# + tags=["parameters"]
alpha = 0
beta = 1
gamma = 2
# -

print('alpha is %d' % alpha)
print('beta is %d' % beta)

x = np.linspace(-2, 2, 50)
plt.plot(x, x**gamma, '.-')
plt.title(r'$\gamma$ = %d' % gamma)
plt.ylabel(r'$x^\gamma$')
plt.xlabel(r'$x$')
plt.show()
