import numpy as np
from PIL import Image
import os
from matplotlib import gridspec
from scipy.ndimage import binary_erosion
from stereo.common.gt_packages_wrapper import FoFutils

# plt.style.use('dark_background')

MODEL_NAMES = ['deeplab'] # list of available models

def seg_mod_init(model_name, with_classes=False):
    """
    Initializing Deeplab segmentation model
    :return: CNN function for segmentation. Runs on PIL images, outputs label for each pixel,
            where label=10 is sky.
    """
    if model_name == 'deeplab':
        from stereo.data.deeplab import DeepLabModel
        MODEL_FNAME = 'deeplab_cityscapes_xception71_trainfine_2018_09_08.tar.gz'
        MODEL_DIR = '/mobileye/algo_STEREO3/stereo/extern/deeplab'
        MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FNAME)
        model = DeepLabModel(MODEL_PATH)
        if with_classes:
            classes = np.asarray([
                'road', 'sidewalk', 'building', 'wall', 'fence',
                'pole', 'traffic light', 'traffic sign',
                'vegetation', 'terrain', 'sky',
                "person", "rider",
                'car', 'truck', 'bus', 'on rails', 'motorcycle', 'bicycle'])
    if with_classes:
        return model, classes
    else:
        return model


def prep_to_seg(image_np):
    """
    Flipping numpy images and converting to PIL for segmentation net
    :param im_np: Image in numpy array format
    :return: Image in PIL format
    """
    im_flip = np.flipud(image_np)
    pil = Image.fromarray(im_flip)
    return pil


def run_seg(model, pil, im_shape):
    """
    Inferring segmentation with net
    :param model: Deeplab net model
    :param pil: Image in PIL format
    :param im_shape: Shape of the required output image in numpy array format. Same as shape of original numpy image,
    :return: Segmentation map in numpy array format
    """
    _, seg_map_flip = model.run(pil)
    seg_map = np.flipud(seg_map_flip.astype('uint8'))
    seg_map_im = np.array(
        Image.fromarray(seg_map).resize((im_shape[1], im_shape[0]), Image.NEAREST))  # Image.LANCZOS caused problems
    return seg_map_im


def im_from_clip(clip, gi, cam_name, pyr_level):
    """
    Rectified image from clip object
    :param clip: AVMClip / FOFutils
    :param gi:
    :param cam_name:
    :param pyr_level:
    :return: Rectified image (numpy array)
    """
    if isinstance(clip, FoFutils):
        clip2 = clip.get_clip(cam=cam_name,grab_index=gi)
    else:
        clip2 =clip
    Metemp = clip2.get_frame(grab_index=gi, camera_name=cam_name)[0]['pyr'][pyr_level]
    im = clip2.rectify_image(Metemp, cam=cam_name, level=pyr_level).im
    return im


def segment_image_from_clip(model, clip, gi, cam_name, pyr_level):
    """
    Segmenting an image.
    :param model: Deeplab CNN model.
    :param clip:  AVMCLip / FoFutils
    :param gi:
    :param cam_name:
    :param pyr_level:
    :return:
    """
    im = im_from_clip(clip, gi, cam_name, pyr_level)
    pil = prep_to_seg(im)
    seg_map_im = run_seg(model, pil, im.shape)
    seg_sky0 = (seg_map_im == 10)
    # erosion_iters = np.ceil(5 *(2 ** (-1 - pyr_level ) )).astype(int) # 5 iterations for level -1 is arbitrary, seems to complete poles
    # seg_sky = binary_erosion(seg_sky0, iterations=erosion_iters)
    return im, seg_sky0, seg_map_im


def rand_points_cond(cond, n_samps):
    """
    Choose random points of image that satisfy the condition `cond`
    :param cond: (n_x, n_y) ndarrray; boolean or {0/1}
    :param n_samps: number of image points to be sampled. < cond.size
    :return x_samp: (n_samps,)
    :return y_samp: (n_samps,)
    :return cam_points: (2, n_samps) 2D camera points (x,y) satisfying `cond`
    """
    x, y = np.indices(cond.shape)
    x_cond = x[cond]
    y_cond = y[cond]
    n_cond = x_cond.size
    ind_rand = np.random.choice(n_cond, n_samps, replace=False)
    x_samp = x_cond[ind_rand]
    y_samp = y_cond[ind_rand]
    cam_points = np.stack((x_samp, y_samp))  # (2, n_samps)
    return x_samp, y_samp, cam_points

def rand_seg_sky_points(model, clip, gi, cam_name, pyr_level, n_sky_samp=100, z_sky=1000, erosion=False):
    """
    Choose random points from segmentation of sky.

    :param model:
    :param clip:
    :param gi:
    :param cam_name:
    :param pyr_level:
    :param n_sky_samp:
    :param z_sky:
    :param erosion:
    :return sky_wpoints_lidar: (3, n_sky_samp)
    :return sky_cam_points:  (2, n_sky_samp)
    :return im:
    :return seg_sky:
    """
    im, seg_sky0, _ = segment_image_from_clip(model, clip, gi, cam_name, pyr_level)
    if erosion:
        seg_sky = binary_erosion(seg_sky0, iterations=5)
    else:
        seg_sky =  seg_sky0
    vis_seg(im, seg_sky.astype(int), seg_sky0.astype(int))
    cond_sky = (seg_sky.astype(int) ==1 )
    _, _, sky_cam_points = rand_points_cond(cond_sky.T, n_sky_samp)
    RT_lid_to_cam, focal, origin_pix, nv_rect, nh_rect = \
        lidar_transformation_params(clip, gi, cam_name, pyr_level)
    sky_wpoints_lidar = im2world(sky_cam_points, z_sky, focal, origin_pix) # (3, n_points)
    show_im_lidar(im, sky_cam_points, z_sky)
    return sky_wpoints_lidar, sky_cam_points, im, seg_sky

def seg_colormap():
    """ Colormap for segmentation of 19 labels """
    from matplotlib import colors
    color_list = [(1, 0.1, 0.1), (1, 1, 0), (0.1, 1, 1), (0., 0.45, 0.), (0.4, 0.6, 1),
                  (0.9, 0.4, 0.4), (0, 0, 0), (0.3, 0.15, 0.15), (0.1, 1, 0.1), (0.8, 0.45, 0),
                  (0.25, 0, 0.2), (1, 0.75, 0.8), (0.3, 0.3, 0.3), (0.5, 0.5, 0), (1, 0, 1),
                  (0.5, 0, 1), (0.35, 0., 0.), (0, 0.55, 0.35), (0.1, 0.1, 0.5)] # 19 labels
    colormap = colors.ListedColormap(color_list, name='discr')
    return colormap

# Function for visualizing segmentation
def vis_seg(im, seg_map_im, seg_map_im0=None, label_names=None):
    """
    Show image, segmentation, and image on top of segmentation.
    Optionally, a previous version of the segmentation `seg_map_im0` will be also shown (second to image).
    """
    import matplotlib.pyplot as plt
    N_LABELS = 19
    if label_names is None:
        label_names = np.asarray([
            'road', 'sidewalk', 'building', 'wall', 'fence',
            'pole', 'traffic light', 'traffic sign',
            'vegetation', 'terrain', 'sky',
            "person", "rider",
            'car', 'truck', 'bus', 'on rails', 'motorcycle', 'bicycle'])
    if seg_map_im0 is None:
        nplots = 4
        seg_map_im0 = seg_map_im
        grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])
    else:
        nplots = 5
        grid_spec = gridspec.GridSpec(1, 5, width_ratios=[6, 6, 6, 6, 1])
    cmap1 = seg_colormap()
    plt.figure(figsize=(16, 4))
    ax1 = plt.subplot(grid_spec[0])
    plt.imshow(im, origin='lower', cmap='gray')

    ax2 = plt.subplot(grid_spec[1], sharex=ax1, sharey=ax1)
    h_im = plt.imshow(seg_map_im0, origin='lower', cmap=cmap1, vmin=0, vmax=N_LABELS - 1)
    uniq_labels = np.unique(seg_map_im0)
    colors = h_im.cmap(h_im.norm(uniq_labels))
    colors1 = colors.reshape((-1, 1, 4))[:, :, :4]

    if nplots == 5:
        ax5 = plt.subplot(grid_spec[2], sharex=ax1, sharey=ax1)
        plt.imshow(seg_map_im, origin='lower', cmap=cmap1, vmin=0, vmax=N_LABELS - 1)

    ax3 = plt.subplot(grid_spec[nplots - 2], sharex=ax1, sharey=ax1)
    plt.imshow(im, origin='lower', cmap='gray')
    plt.imshow(seg_map_im, origin='lower', cmap=cmap1, alpha=0.5, vmin=0, vmax=N_LABELS - 1)

    ax4 = plt.subplot(grid_spec[nplots - 1])
    plt.imshow(uniq_labels.reshape((-1, 1)), cmap=cmap1, vmin=0, vmax=N_LABELS - 1);  # interpolation='nearest')
    ax4.yaxis.tick_right()
    plt.yticks(range(len(uniq_labels)), label_names[uniq_labels])
    plt.xticks([], [])
    ax4.tick_params(width=0.0)
    plt.grid('off')
    plt.show()

class Labeler(object):
    """
    Class for handling segmentations/labelings.
    """
    def __init__(self, model_list):
        """
        Initialize models.
        """
        self.models = {}
        self.classes = {}
        self.MODEL_NAMES = MODEL_NAMES
        if type(model_list) is str:
            model_list = [model_list]

        for model_name in model_list:
            assert model_name in self.MODEL_NAMES, "Model %s not available" % model_name
            self.models[model_name], self.classes[model_name]= \
                seg_mod_init(model_name, with_classes=True)


    def generate_segmentation(self, model_name, image):
        """
        Run inference.
        :param model_name: str; model to run inference on.
        :param image: ndarray (ny, nx);
        :return segmentation_mask: np array, same shape as `image`. Values {0,1,...,n} correspond `segmentation_classes`
        :return segmentation_classes: list of strings, each class correspond to a label according to its position in list.
        """
        im_shape = image.shape
        pil_im = prep_to_seg(image) # convert to pil image
        segmentation_mask = run_seg(self.models[model_name], pil_im, im_shape)
        segmentation_classes_list = self.classes[model_name]
        segmentation_classes = dict(zip(range(len(segmentation_classes_list)),
                                        segmentation_classes_list))
        return segmentation_mask, segmentation_classes


    def visualize_segmentation(self, model_name, im, seg_map_im, seg_map_im0=None):
        """
        Wrapper for `vis_seg` in `Labeler`. Useful for monitoring.
        :param model_name:
        :param im:
        :param seg_map_im:
        :param seg_map_im0:
        :return:
        :rtype:
        """
        vis_seg(im, seg_map_im, seg_map_im0, label_names=self.classes[model_name])