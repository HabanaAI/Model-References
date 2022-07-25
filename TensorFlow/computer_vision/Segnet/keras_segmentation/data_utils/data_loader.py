import itertools
import os
import random
import six
import numpy as np
import cv2
import tensorflow as tf

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found, disabling progress bars")

    def tqdm(iter):
        return iter


from keras_segmentation.models.config import IMAGE_ORDERING
from .augmentation import augment_seg

DATA_LOADER_SEED = 0
np.random.seed(DATA_LOADER_SEED)
random.seed(DATA_LOADER_SEED)
class_colors = [(random.randint(0, 255), random.randint(
    0, 255), random.randint(0, 255)) for _ in range(5000)]


nprandomseed = 2019

class DataLoaderError(Exception):
    pass


def get_pairs_from_paths(images_path, segs_path, ignore_non_matching=False):
    """ Find all the images from the images_path directory and
        the segmentation images from the segs_path directory
        while checking integrity of data """

    ACCEPTABLE_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp"]
    ACCEPTABLE_SEGMENTATION_FORMATS = [".png", ".bmp"]

    image_files = []
    segmentation_files = {}

    for dir_entry in sorted(os.listdir(images_path)):
        if os.path.isfile(os.path.join(images_path, dir_entry)) and \
                os.path.splitext(dir_entry)[1] in ACCEPTABLE_IMAGE_FORMATS:
            file_name, file_extension = os.path.splitext(dir_entry)
            image_files.append((file_name, file_extension,
                                os.path.join(images_path, dir_entry)))

    for dir_entry in sorted(os.listdir(segs_path)):
        if os.path.isfile(os.path.join(segs_path, dir_entry)) and \
           os.path.splitext(dir_entry)[1] in ACCEPTABLE_SEGMENTATION_FORMATS:
            file_name, file_extension = os.path.splitext(dir_entry)
            full_dir_entry = os.path.join(segs_path, dir_entry)
            if file_name in segmentation_files:
                raise DataLoaderError("Segmentation file with filename {0}"
                                      " already exists and is ambiguous to"
                                      " resolve with path {1}."
                                      " Please remove or rename the latter."
                                      .format(file_name, full_dir_entry))

            segmentation_files[file_name] = (file_extension, full_dir_entry)

    return_value = []
    # Match the images and segmentations
    for image_file, _, image_full_path in image_files:
        if image_file in segmentation_files:
            return_value.append((image_full_path,
                                segmentation_files[image_file][1]))
        elif ignore_non_matching:
            continue
        else:
            # Error out
            raise DataLoaderError("No corresponding segmentation "
                                  "found for image {0}."
                                  .format(image_full_path))

    return return_value


def get_image_array(image_input,
                    width, height,
                    imgNorm="sub_mean", ordering='channels_first'):
    """ Load image array from input """

    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_image_array: path {0} doesn't exist"
                                  .format(image_input))
        img = cv2.imread(image_input, 1)
    else:
        raise DataLoaderError("get_image_array: Can't process input type {0}"
                              .format(str(type(image_input))))

    if imgNorm == "sub_and_divide":
        img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
    elif imgNorm == "sub_mean":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        img = img[:, :, ::-1]
    elif imgNorm == "divide":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = img/255.0

    if ordering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
    return img


def get_segmentation_array(image_input, nClasses,
                           width, height, no_reshape=False, loss_type=0):
    """ Load segmentation array from input """

    seg_labels = np.zeros((height, width, nClasses))

    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_segmentation_array: "
                                  "path {0} doesn't exist".format(image_input))
        img = cv2.imread(image_input, 1)
    else:
        raise DataLoaderError("get_segmentation_array: "
                              "Can't process input type {0}"
                              .format(str(type(image_input))))

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    img = img[:, :, 0]

    for c in range(nClasses):
        seg_labels[:, :, c] = (img == c).astype(int)

    if not no_reshape:
        if (loss_type==1 or loss_type==2):
            seg_labels = np.reshape(seg_labels, (width*height, nClasses))

    return seg_labels


def verify_segmentation_dataset(images_path, segs_path,
                                n_classes, deterministic=False, show_all_errors=False):
    try:
        img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
        if not len(img_seg_pairs):
            print("Couldn't load any data from images_path: "
                  "{0} and segmentations path: {1}"
                  .format(images_path, segs_path))
            return False

        return_value = True

        for im_fn, seg_fn in tqdm(img_seg_pairs):
            img = cv2.imread(im_fn)
            seg = cv2.imread(seg_fn)
            # Check dimensions match
            if not img.shape == seg.shape:
                return_value = False
                print("The size of image {0} and its segmentation {1} "
                      "doesn't match (possibly the files are corrupt)."
                      .format(im_fn, seg_fn))
                if not show_all_errors:
                    break
            else:
                max_pixel_value = np.max(seg[:, :, 0])
                if max_pixel_value >= n_classes:
                    return_value = False
                    print("The pixel values of the segmentation image {0} "
                          "violating range [0, {1}]. "
                          "Found maximum pixel value {2}"
                          .format(seg_fn, str(n_classes - 1), max_pixel_value))
                    if not show_all_errors:
                        break
        if return_value:
            print("Dataset verified! ")
        else:
            print("Dataset not verified!")
        return return_value
    except DataLoaderError as e:
        print("Found error during data loading\n{0}".format(str(e)))
        return False


def image_segmentation_generator(images_path, segs_path, batch_size,
                                 n_classes, input_height, input_width,
                                 output_height, output_width, deterministic=False,
                                 do_augment=False,
                                 augmentation_name="aug_all",
                                 num_shards=1,
                                 shard_id=0,
                                 loss_type=0):

    img_seg_pairs = get_pairs_from_paths(images_path, segs_path)

    if num_shards > 1:
        np.random.seed(nprandomseed)
    if not deterministic:
        random.shuffle(img_seg_pairs)

    if num_shards > 1:
        process_subset_size = len(img_seg_pairs) // num_shards
        ids_from = process_subset_size*shard_id
        ids_to = ids_from + process_subset_size
        img_seg_pairs_ = img_seg_pairs[ids_from : ids_to]
        # make sure all samples are used
        img_seg_pairs_ += img_seg_pairs[process_subset_size*num_shards:]
        img_seg_pairs = img_seg_pairs_
        #print(f'Image Generator : [ {shard_id} ] , {len(img_seg_pairs)} - {img_seg_pairs[:10]}')
        print(f'Ids from to : [{shard_id}], {ids_from} to {ids_to}')

    zipped = itertools.cycle(img_seg_pairs)

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            im, seg = next(zipped)
            im = cv2.imread(im, 1)
            seg = cv2.imread(seg, 1)

            if do_augment:
                im, seg[:, :, 0] = augment_seg(im, seg[:, :, 0],
                                               augmentation_name)

            X.append(get_image_array(im, input_width,
                                     input_height, ordering=IMAGE_ORDERING))
            Y.append(get_segmentation_array(
                seg, n_classes, output_width, output_height, loss_type=loss_type))

        if not (loss_type==1 or loss_type==2):
            Y = np.reshape(Y, [batch_size, output_height, output_width, n_classes])
        yield np.array(X), np.array(Y)


def create_segmentation_list(generator, iterations):

    X=[]
    Y=[]
    print("Genering list: ", iterations)
    for itr in tqdm(range(iterations)):
        img,seg = next(generator)
        X.append(img)
        Y.append(seg)

    X_concat = np.concatenate(X, axis=0)
    Y_concat = np.concatenate(Y, axis=0)

    return X_concat, Y_concat

def cached_image_generator(generator, num_shards, shard_id, batch_size, total_img_num, deterministic=False):

    shard_img = total_img_num//num_shards + total_img_num%num_shards if shard_id == num_shards-1 else total_img_num//num_shards
    padding = batch_size - shard_img%batch_size if shard_img%batch_size != 0 else 0
    X, y = create_segmentation_list(generator, shard_img + padding)

    # create a pipeline from a generator given full dataset X,y. This should prevent 2GB protobuf error
    def create_pipeline(X, y):
        def create_gen():
            train_len = X.shape[0]
            def generator():
                k = 0
                while True:
                    yield X[k%train_len, ...], y[k%train_len, ...]
                    k += 1
            return generator
        get_shp = lambda item : item.shape[1:]
        output_signature=(tf.TensorSpec(shape=get_shp(X), dtype=tf.float32),
                          tf.TensorSpec(shape=get_shp(y), dtype=tf.float32))

        return tf.data.Dataset.from_generator(create_gen(), output_signature=output_signature)

    dataset = create_pipeline(X, y)
    buffer_size = X.shape[0]
    print("Train ds ", total_img_num, "buffer_size", buffer_size)

    seed = tf.random.set_seed(nprandomseed) if deterministic else None
    dataset = dataset.shuffle(buffer_size, seed).repeat().batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
