# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: stereo_dlo
#     language: python
#     name: stereo_dlo
# ---

import numpy as np
import matplotlib.pyplot as plt

min_z = 2.
max_z = 80.

# +
fig, ax = plt.subplots(2, 1, figsize=(30,20), facecolor=(1,1,1))


z0s = [0, 10, 20 ,30]
# z0s = [10, 20 ,30]
# z0s = [20]


for z0 in z0s:
    z_tilde_inv = np.linspace(1/(max_z+z0), 1/(min_z+z0), 256)
    z = (1/z_tilde_inv)-z0
    ax[0].plot(z[::-1], np.arange(256)[::-1], '-o', label=f"$z_0$={z0}")
    ax[1].plot(z[::-1][:-1], np.diff(z[::-1]), '-o', label=f"$z_0$={z0}")
ax[0].legend()
ax[0].grid()
ax[1].legend()
ax[1].grid()
# -


