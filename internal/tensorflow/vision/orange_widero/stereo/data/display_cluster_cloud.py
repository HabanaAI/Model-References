import pptk
import numpy as np
import os,argparse

def display_clip_cloud(clip_name, predump_dir='/mobileye/algo_STEREO3/stereo/data/predump.v2/', specific_cluster=None):
    from stereo.data.predump_v2 import PreDumpIndexV2
    from tqdm import tqdm

    predump_index = PreDumpIndexV2(predump_dir)
    cluster_dir = os.path.join(predump_index.lidar_path(clip_name), 'clusters')

    if not os.path.exists(cluster_dir):
        raise ValueError("No cluster data for this clip")

    first = True
    clusters = None
    for fn in tqdm(os.listdir(cluster_dir)):
        try:
            gi_dict = np.load(os.path.join(cluster_dir, fn), allow_pickle=True)
            clusters_temp = gi_dict['clusters']
            wpoints_gi_temp = gi_dict['wpoints_gi']
            wpoints_main_temp = gi_dict['wpoints_main']
            temp_mask = np.logical_and(clusters_temp[:,1] > 0, wpoints_main_temp[:,1] < -0.9)
            clusters_temp[temp_mask,1] = -1
            if first:
                clusters = clusters_temp
                wpoints_gi = wpoints_gi_temp
                wpoints_main = wpoints_main_temp
                first = False
            else:
                clusters = np.r_[clusters, clusters_temp]
                wpoints_gi = np.r_[wpoints_gi, wpoints_gi_temp]
                wpoints_main = np.r_[wpoints_main, wpoints_main_temp]
        except:
            continue

    if clusters is None or len(clusters) == 0:
        raise ValueError("Cluster dir for this clip is empty")


    clust_mask = np.logical_and.reduce((clusters[:, 1] >= 0, wpoints_main[:,1] < 1.5,
                                        np.sqrt(wpoints_main[:,2]**2 + wpoints_main[:,1]**2 + wpoints_main[:,0]**2) < 15.))
    # clust_mask = clusters[:,1] >= 0
    C = np.mod(clusters[clust_mask][:, 1] + 1, 20) / 20.
    alt_pnts = wpoints_gi[clust_mask]
    if specific_cluster is not None:
        alt_clusters = clusters[clust_mask]
        C[alt_clusters[:,1] == specific_cluster] = 30.
    # alt_pnts[:, 0] *= -1
    alt_pnts = np.c_[alt_pnts[:, 0], alt_pnts[:, 2], alt_pnts[:, 1]]
    v = pptk.viewer(alt_pnts)
    v.set(point_size=.02)
    v.attributes(C)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Display cluster cloud.')
    parser.add_argument('clip_name', help='clip name')
    parser.add_argument('--cluster', help='specific cluster to highlight', default=None)

    args = vars(parser.parse_args())

    clip_name = args['clip_name']
    display_clip_cloud(clip_name, specific_cluster=args['cluster'])