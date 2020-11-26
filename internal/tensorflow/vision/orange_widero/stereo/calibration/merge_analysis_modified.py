# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
# ---

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# +
# imports
import os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('dark_background')

# + code_folding=[]
# prepare
# sector = 'front'
# sector = 'rear'
sector = 'RCR'
if sector is 'front':
    cams = ['3cams','FCR','park_front','FCL']
elif sector is 'rear':
    cams = ['3cams','RCR','RCL','park_rear']
elif sector is 'FCL':
    cams = ['2cams', 'park_left','park_front']
elif sector is 'FCR':
    cams = ['2cams', 'park_front','park_right']
elif sector is 'RCL':
    cams = ['2cams', 'park_rear', 'park_left']
elif sector is 'RCR':
    cams = ['2cams', 'park_right', 'park_rear']
load_dir = '/mobileye/algo_STEREO3/ohrl/tolerance/results/' + sector + '/'
z_bins = np.array([ 2.05882353,  6.17647059, 10.29411765, 14.41176471, 18.52941176,
       22.64705882, 26.76470588, 30.88235294, 35.        , 39.11764706,
       43.23529412, 47.35294118, 51.47058824, 55.58823529, 59.70588235,
       63.82352941, 67.94117647, 85.        ])  # forgot to save it, copied
ind_binz40 = np.argmin( np.abs(z_bins-40))

# +
# %matplotlib notebook
fig1 = plt.figure()
plt.gca().set_prop_cycle(plt.cycler('color',
                plt.cm.gist_rainbow(np.linspace(0, 1, 4))))
for i, cam in enumerate(cams):
    file_temp = np.load(load_dir + cam + '.npz')
    angs_pix = file_temp['angs_pix']
    ea_medianz = file_temp['ea_medianz']
    norm_added_med = ( ea_medianz[:,ind_binz40] - ea_medianz[0, ind_binz40])  / ea_medianz[0, ind_binz40]
    plt.plot(angs_pix, norm_added_med,'o' ,label=cam)
    plt.grid(which='both', alpha=0.2)
    plt.legend()
plt.xlabel('angle [pix @ main level -1]', fontsize=18)
plt.ylabel('median absolute error (norm.)'  ,fontsize=18)
plt.title('z = %d [m]' % z_bins[ind_binz40] , fontsize=18)
plt.show()

# plt.savefig(load_dir+'combined_cams_40m.png')
# plt.savefig(load_dir+'combined_cams_40m.pdf')
    
