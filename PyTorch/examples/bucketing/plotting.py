# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def plotter(shapes_lst, out_name='hist.svg', tags=None):
    if tags is None:
        tags = map(str, range(len(shapes_lst)))
    for shapes, tag in tqdm(zip(shapes_lst, tags)):
        print('plotting', tag)
        hist, bins = np.histogram(shapes, bins=30)
        plt.plot((bins[1:] + bins[:-1])/2, hist/sum(hist), '.-', label=tag)
        plt.xlabel('lengths')
        plt.ylabel('probability')
        plt.legend()
        plt.savefig(out_name)


# TODO Remove
def plot_bucket_analysis_results1(results):
    jitter = [(0.1,1), (0,0), (0.1,1), (0.1,1), (-0.1,-1), (-0.1,-1)]
    for j, algoidx in zip(jitter, results.keys()):
        res = results[algoidx]
        print(f'plotting algo {algoidx}')
        buckets = []
        avgwastes = []
        times = []
        for num_bucket in res:
            buckets += [num_bucket]
            avgwastes += [res[num_bucket]['avgwaste']]
            times += [res[num_bucket]['time']]
        plt.plot(buckets, avgwastes, '.-', label=f'{algoidx}')
        for x,y,lbl in zip(buckets,avgwastes,times):
            label = "{:.1f}".format(lbl)
            plt.annotate(label, # this is the text
                        (x+j[0],y+j[1]), # these are the coordinates to position the label
                        textcoords="offset points", # how to position the text
                        xytext=(0,0), # distance from text to points (x,y)
                        ha='center')
    plt.legend()
    plt.xlabel('num buckets')
    plt.ylabel('avg waste')
    plt.savefig('bucket_analysis.svg')


def plot_bucket_analysis_results(results, out_name):
    x = []
    y = []
    alltimes = []
    names = []
    expected_keys = ['const_bucket', 'uniform_bucket', 'percentile_bucket', 'lloyd_max_bucketing', 'lp_bucket', 'brute_force_min_pad_waste']
    if sorted(list(results.keys())) == sorted(expected_keys):
        algo_keys = expected_keys
    else:
        algo_keys = sorted(list(results.keys()))
    for algoidx in algo_keys:
        res = results[algoidx]
        print(f'plotting algo {algoidx}')
        buckets = []
        avgwastes = []
        times = []
        for num_bucket in res:
            buckets += [num_bucket]
            avgwastes += [res[num_bucket]['avgwaste']]
            times += [res[num_bucket]['time']]
        y += [avgwastes]
        alltimes += [times]
        x += [buckets]
        names += [algoidx]
    miny = np.min(y) * 0.95
    y = np.array(y) - miny
    # all arrays are numalgo x numbuckets
    # converting time to log to calculate bar thickness based on time duration
    log_times = np.log(alltimes)
    bar_thickness = log_times - np.min(log_times) + 1
    bar_thickness_flat = bar_thickness.T.flatten()
    bar_thickness_per_bucket = np.sum(bar_thickness,0)
    buckets = [i for i in x[0]]
    allxloc = []
    for algid in range(len(names)):
        xloc = []
        for bktid in range(len(buckets)):
            num = sum(bar_thickness_flat[:bktid*len(names) + algid]) + (bktid*np.mean(bar_thickness_per_bucket)*0.15)
            xloc += [num]
        allxloc += [xloc]

    colors = ['#7f6d5f', '#557f2d', 'y', 'r', 'b', 'cyan', 'black', 'pink']
    if len(names) > len(colors):
        print('Less colors specified, please add more colors to the list above')
    bars = []
    for col, xloc, waste, barthick, nm in zip(colors, allxloc, y, bar_thickness, names):
        print(nm)
        bar = plt.bar(xloc, waste, color=col, width=barthick, edgecolor='white', label=nm, align='edge', bottom=miny)
        bars += [bar]

    for idx, rect in enumerate(sum(bars[1:], bars[0])):
        algid = idx // len(buckets)
        bktid = idx % len(buckets)
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height + miny, f'{alltimes[algid][bktid]:.0f}', ha='center', va='bottom', fontsize=7)


    plt.xlabel('Number of buckets')
    plt.xticks((np.cumsum(bar_thickness_per_bucket) - (bar_thickness_per_bucket/2)) + ((np.mean(bar_thickness_per_bucket)*0.15) * np.arange(len(buckets))), buckets)
    plt.legend()
    plt.ylabel('Average waste')
    plt.savefig(out_name)
