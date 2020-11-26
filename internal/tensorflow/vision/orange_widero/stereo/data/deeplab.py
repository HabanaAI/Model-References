import os
import tarfile
import numpy as np
import tensorflow as tf

from matplotlib import gridspec
from matplotlib import pyplot as plt
from PIL import Image


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.compat.v1.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
      A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
      label: A 2D array with integer type, storing the segmentation label.

    Returns:
      result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
      ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


MODEL_FNAME = 'deeplab_cityscapes_xception71_trainfine_2018_09_08.tar.gz'
MODEL_DIR = '/mobileye/algo_STEREO3/stereo/extern/deeplab'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FNAME)

LABEL_NAMES = np.asarray([
    'road', 'sidewalk', 'building', 'wall', 'fence',
    'pole', 'traffic light', 'traffic sign',
    'vegetation', 'terrain', 'sky',
    "person", "rider",
    'car', 'truck', 'bus', 'on rails', 'motorcycle', 'bicycle'])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

# other models
""" Some other models
_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
_MODEL_URLS = {
    'mobilenetv2_coco_voctrainaug':
        'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
    'mobilenetv2_coco_voctrainval':
        'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
    'xception_coco_voctrainaug':
        'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
    'xception_coco_voctrainval':
        'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
    'mobilenetv2_coco_cityscapes_trainfine':
        "deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz",
    "xception65_cityscapes_trainfine":
        "deeplabv3_cityscapes_train_2018_02_06.tar.gz",
    'xception71_dpc_cityscapes_trainfine':
        "deeplab_cityscapes_xception71_trainfine_2018_09_08.tar.gz",
    "xception71_dpc_cityscapes_trainval":
        "deeplab_cityscapes_xception71_trainvalfine_2018_09_08.tar.gz",
}
"""


def vis_segmentation(image, seg_map):
    """Visualizes input image, segmentation map and overlay view."""
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    ax1 = plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1], sharex=ax1,sharey=ax1)
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2], sharex=ax1,sharey=ax1)
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.show()


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

if __name__ == '__main__':

    import cv2
    from PIL import Image
    plt.style.use('dark_background')
#     %matplotlib notebook
    image_dir = '/homes/ofers/sandbox/deeplab/examples'
    image_names = ['18-08-15_10-37-17_Alfred_Front_0094_249507.ppm', 
                   '18-08-14_09-11-28_Alfred_Front_0022_130784.ppm',
                   '18-12-27_14-01-18_Alfred_JER5_Front_0116_208003.ppm',
                   '19-05-14_13-36-57_Alfred_Front_0031_60419.ppm',
                   '18-08-15_10-37-17_Alfred_Front_0094_249507.ppm',
                   '18-12-27_14-01-18_Alfred_JER5_Front_0116_209831.ppm',
                   '19-05-14_13-36-57_Alfred_Front_0031_61407.ppm']

    model = DeepLabModel(MODEL_PATH)

    for image_name in image_names:

        print(image_name)
        path = os.path.join(image_dir, image_name)
        im = cv2.imread(path)
        pil = Image.fromarray(im)
        _, seg_map = model.run(pil)
        seg_map = seg_map.astype('uint8')
        seg_map = np.array(Image.fromarray(seg_map).resize((im.shape[1], im.shape[0]), Image.NEAREST))
        vis_segmentation(im, seg_map)
