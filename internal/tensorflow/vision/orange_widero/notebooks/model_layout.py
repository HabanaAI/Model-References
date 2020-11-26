# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
# ---

# +
import numpy as np
import matplotlib.pyplot as plt
# #%matplotlib notebook
from stereo.data.vis_utils import view_pptk, depth_visualization, blend_depth
from stereo.data.lidar_utils import interp_lidar
from stereo.models.pred import StereoPredictorCLI
import imp
import os
import sys
from glob import glob

import tensorflow as tf


# -

def calc_model(arch_func_, verbose=False):
    flags = tf.compat.v1.app.flags
    FLAGS = flags.FLAGS

    num_srnd = 2
    im_sz = [310, 720]
    origin = np.array([[155, 360]])
    focal = np.array([[700, 700]])
    left, right = -origin[0, 0], im_sz[1] - origin[0, 0]
    bottom, top = -origin[0, 1], im_sz[0] - origin[0, 1]
    x, y = np.meshgrid(np.arange(left, right), np.arange(bottom, top))
    x = (x[np.newaxis, :, :, np.newaxis]).astype('float32')
    y = (y[np.newaxis, :, :, np.newaxis]).astype('float32')

    tf.compat.v1.reset_default_graph()
    inp1 = tf.compat.v1.placeholder(tf.float32, [1, 1 + num_srnd, im_sz[0], im_sz[1]], name="inp1")
    lidar1 = tf.compat.v1.placeholder(tf.float32, [1, im_sz[0], im_sz[1]], name="lidar1")
    mask1 = tf.compat.v1.placeholder(tf.uint8, [1, im_sz[0], im_sz[1]], name="mask1")
    T_cntr_srnd = tf.compat.v1.placeholder(tf.float32, [1, num_srnd, 3], name="T_cntr_srnd")
    RT12 = tf.compat.v1.placeholder(tf.float32, [1, 4, 4], name="RT12")

    out1 = arch_func_(inp1, T_cntr_srnd, focal, origin, x, y)

    graph = tf.compat.v1.get_default_graph()
    
    s = []
    f = []
    n = []
    for op in graph.get_operations():
        if len(op.values()) != 1:
            continue
        tens = op.values()[0]
        if tens.name[-14:] == '/kernel/read:0':
        
            out = tens.graph.get_tensor_by_name(tens.consumers()[0].name + ":0").shape
            w = tens.shape
            size = w[0] * w[1] * w[2] * w[3] + w[3]
            flop = 2 * size.value * out[1].value * out[2].value
            mac = (w[0] * w[1] * w[2] * w[3]).value * out[1].value * out[2].value
            name = tens.consumers()[0].name[:-7]
            
            s.append(size.value)
            f.append(flop)
            n.append(name)
            
            if (verbose):
                print (name + ":")
                print ("Weights: " + str(w))
                print ("Output: " + str(out))
                print ("size: " + str(size) + ", FLOP: " + str(flop))
                print ("MAC: " + str(mac))

        if tens.name[-25:] == '/deconv2d/Variable/read:0':
            out = None
            i = -1
            j = 0
            for tout in tens.consumers():
                i += 1
                if tout.type == "Conv2DBackpropInput":
                    j=i
                    out = tens.graph.get_tensor_by_name(tout.name + ":0").shape
                    
            w = tens.shape
            size = w[0] * w[1] * w[2] * w[3]
            flop = 2 * size.value * out[1].value * out[2].value
            mac = size.value * out[1].value * out[2].value
            name = tens.consumers()[j].name[:-17]
            
            s.append(size.value)
            f.append(flop)
            n.append(name)
            
            if (verbose):
                print (name + ":")
                print ("Weights: " + str(w))
                print ("Output: " + str(out))
                print ("size: " + str(size) + ", FLOP: " + str(flop))
                print ("MAC: " + str(mac))
                
        if tens.name[-42:] == '/corr_transform/ImageProjectiveTransform:0':
            out = tens.consumers()[0].values()[0].consumers()[0].values()[0].consumers()[0].values()[0].shape
            inp = tens.shape
            flop = 3 * inp[3].value * out[0].value * out[2].value * out[3].value
            name = tens.name[:-42]
            
            f.append(flop)
            s.append(0)
            n.append(name)
            
            if (verbose):
                print (name + ":")
                print("Input: " + str(inp))
                print ("Output: " + str(out))
                print ("size: 0" + ", FLOP: " + str(flop))
                
    return s, f, n



# +
#[op.values() for op in graph.get_operations()][:]

# +
arch_func_ = imp.load_source('arch','../stereo/models/arch/sdsv3_v0.4.0_stereo_single_phase_corr_fc_diet4.py').arch

s, f, n = calc_model(arch_func_, True)
# -

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.semilogy(np.array(s, dtype = "float32"))
ax.semilogy(np.array(f, dtype = "float32"))
ax.set_xticks(np.arange(len(n)))
ax.set_xticklabels(n, rotation="vertical")
ax.grid(which = "both", b = True, axis="both")
plt.show()

max(f)

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
leg = []
for path in glob("../stereo/models/arch/*.py"):
    arch = imp.load_source('arch',path).arch
    s, f, n = calc_model(arch)
    ax.scatter(x=[sum(s)], y=[sum(f)], label=path[22:])
    ax.set_yscale('log')
    ax.set_xscale('log')
    #break
    leg.append(path[22:])
ax.legend(leg, framealpha= 0.3)
plt.show()
