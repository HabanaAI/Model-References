"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""
import tensorflow.compat.v1 as tf
tf.enable_resource_variables()
import tensorflow.compat.v1.keras.backend as K
from tensorflow.compat.v1.keras.callbacks import Callback
import os
import sys
import time
import numpy as np
import random
import argparse

import imgaug.augmenters as iaa # https://github.com/aleju/imgaug (pip3 install imgaug)

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2017"

############################################################
#  Dataset
############################################################

class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, class_ids=None,
                  class_map=None, return_coco=False, auto_download=False, limit=None, num_shards=1,shard_id=0):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        if auto_download is True:
            self.auto_download(dataset_dir, subset, year)

        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            if args.deterministic:
                image_ids = sorted(list(set(image_ids)))
            else:
                image_ids = list(set(image_ids))
        else:
            # All images
            if args.deterministic:
                image_ids = sorted(list(coco.imgs.keys()))
            else:
                image_ids = list(coco.imgs.keys())
        if limit:
            # if also using horovod all workers will see the same sample due to the random seed reset
            random.seed(0)
            image_ids = random.sample(image_ids,min(limit,len(image_ids)))

        if num_shards > 1:
            process_subet_size = len(image_ids) // num_shards
            ids_from = process_subet_size*shard_id
            ids_to = ids_from + process_subet_size
            image_ids_ = image_ids[ids_from : ids_to]
            # make sure all samples are used
            image_ids_ += image_ids[process_subet_size*num_shards:]
            image_ids = image_ids_
            # print(len(image_ids),image_ids[:10])

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

    def auto_download(self, dataDir, dataType, dataYear):
        """Download the COCO dataset/annotations if requested.
        dataDir: The root directory of the COCO dataset.
        dataType: What to load (train, val, minival, valminusminival)
        dataYear: What dataset year to load (2014, 2017) as a string, not an integer
        Note:
            For 2014, use "train", "val", "minival", or "valminusminival"
            For 2017, only "train" and "val" annotations are available
        """

        # Setup paths and file names
        if dataType == "minival" or dataType == "valminusminival":
            imgDir = "{}/{}{}".format(dataDir, "val", dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, "val", dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format("val", dataYear)
        else:
            imgDir = "{}/{}{}".format(dataDir, dataType, dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, dataType, dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format(dataType, dataYear)

        # Create main folder if it doesn't exist yet
        if not os.path.exists(dataDir):
            os.makedirs(dataDir)

        # Download images if not available locally
        if not os.path.exists(imgDir):
            os.makedirs(imgDir)
            print("Downloading images to " + imgZipFile + " ...")
            with urllib.request.urlopen(imgURL) as resp, open(imgZipFile, 'wb') as out:
                shutil.copyfileobj(resp, out)
            print("... done downloading.")
            print("Unzipping " + imgZipFile)
            with zipfile.ZipFile(imgZipFile, "r") as zip_ref:
                zip_ref.extractall(dataDir)
            print("... done unzipping")
        print("Will use images in " + imgDir)

        # Setup annotations data paths
        annDir = "{}/annotations".format(dataDir)
        if dataType == "minival":
            annZipFile = "{}/instances_minival2014.json.zip".format(dataDir)
            annFile = "{}/instances_minival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0"
            unZipDir = annDir
        elif dataType == "valminusminival":
            annZipFile = "{}/instances_valminusminival2014.json.zip".format(dataDir)
            annFile = "{}/instances_valminusminival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0"
            unZipDir = annDir
        else:
            annZipFile = "{}/annotations_trainval{}.zip".format(dataDir, dataYear)
            annFile = "{}/instances_{}{}.json".format(annDir, dataType, dataYear)
            annURL = "http://images.cocodataset.org/annotations/annotations_trainval{}.zip".format(dataYear)
            unZipDir = dataDir

        # Download annotations if not available locally
        if not os.path.exists(annDir):
            os.makedirs(annDir)
        if not os.path.exists(annFile):
            if not os.path.exists(annZipFile):
                print("Downloading zipped annotations to " + annZipFile + " ...")
                with urllib.request.urlopen(annURL) as resp, open(annZipFile, 'wb') as out:
                    shutil.copyfileobj(resp, out)
                print("... done downloading.")
            print("Unzipping " + annZipFile)
            with zipfile.ZipFile(annZipFile, "r") as zip_ref:
                zip_ref.extractall(unZipDir)
            print("... done unzipping")
        print("Will use annotations in " + annFile)

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results

class COCOSchedualer(Callback):
    def __init__(self,initial_lr=0.02,gamma=0.1,drops=[60,80],warmup_steps=1000,warmup_factor=0.001,steps_per_epoch=None,lr_attr_name=None,verbose=True):
        super().__init__()
        self.steps = 0
        self.initial_lr=initial_lr
        self.gamma=gamma
        self.drops=drops
        self.warmup_steps=warmup_steps
        self.warmup_rate=(initial_lr - initial_lr*warmup_factor)/warmup_steps
        self.lr_attr_name=lr_attr_name
        self.steps_per_epoch=steps_per_epoch
        self.phase_index=0
        self.verbose=verbose

    def on_epoch_begin(self, epoch, logs=None):
        self.epochs=epoch
        if self.lr_attr_name is None:
            if not hasattr(self.model.optimizer, 'lr'):
                if not hasattr(self.model.optimizer, 'learning_rate'):
                    raise ValueError('Optimizer must have an "lr" or "learning_rate" attribute.')
                self.lr_attr_name='learning_rate'
            else:
                self.lr_attr_name='lr'

        lr = self.epoch_schedual(epoch)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(getattr(self.model.optimizer,self.lr_attr_name), lr)

    def on_epoch_end(self, epoch, logs=None):
        #find out how many steps per epoch
        if epoch==0 and self.steps_per_epoch is None:
            self.steps_per_epoch=self.steps

    def on_batch_begin(self, batch, logs=None):
        if self.epochs>0 and self.steps_per_epoch is None:
            return
        elif self.steps_per_epoch is not None:
            self.steps = self.epochs*self.steps_per_epoch+batch

        if self.steps < self.warmup_steps:
            lr = self.warmup_schedual(self.steps)
            K.set_value(getattr(self.model.optimizer,self.lr_attr_name), lr)

    def on_batch_end(self, batch, logs=None):
        #find out how many steps per epoch
        if self.steps_per_epoch is None and self.epochs==0:
            self.steps+=1

    def warmup_schedual(self,global_steps):
        lr = self.initial_lr - (self.warmup_steps-global_steps-1)*self.warmup_rate
        if self.verbose:
            print(f'\tupdating learning rate at step {global_steps} to {lr:0.5f}\t')
        return lr

    def epoch_schedual(self,epoch):
        while len(self.drops)>0 and epoch >= self.drops[0]:
            self.drops.pop(0)
            self.phase_index+=1
            if self.verbose:
                print(f'\n\nupdating phase index at epoch {epoch} to {self.phase_index}. New lr is {self.initial_lr*self.gamma**self.phase_index:0.5f}')

        lr = self.initial_lr*self.gamma**self.phase_index
        return lr

def evaluate_coco(model, dataset, coco,eval_type="bbox,segm", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)
    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)

    # Load results. This modifies results with additional attributes.
    with open(os.path.join(model.model_dir, f'coco_results_{len(image_ids)}.txt'), "w") as file:
        file.write(str(results))
    coco_results = coco.loadRes(results)
    # Evaluate
    for e_type in eval_type.split(','):
        cocoEval = COCOeval(coco, coco_results, e_type)
        cocoEval.params.imgIds = coco_image_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

############################################################
#  Training
############################################################


def run_coco(args):
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Year: ", args.year)
    print("Logs: ", args.logs)
    print("Auto Download: ", args.download)

############################################################
#  Configurations
############################################################
    if args.deterministic:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.reset_default_graph()
        SEED=0
        os.environ['PYTHONHASHSEED']=str(SEED)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        random.seed(SEED)
        np.random.seed(SEED)
        tf.set_random_seed(SEED)

    if args.device in ['HPU']:
        from test.test_helpers.test_utils import load_habana_module
        load_habana_module()

    dev_str = f'/device:{args.device}:0'
    print(f'Selected device: {dev_str}')

    is_master = True
    hvd = None

    if args.gpus < 0:
        config = tf.ConfigProto(device_count={'GPU': 0})
        K.set_session(tf.Session(config=config))
        print('running on cpu')

    if args.using_horovod and args.command == "train":
        import horovod.tensorflow.keras as hvd
        hvd.init()
        confighorovod = tf.ConfigProto()
        confighorovod.gpu_options.visible_device_list = str(hvd.local_rank())
        K.set_session(tf.Session(config=confighorovod))
        is_master = hvd.local_rank() == 0
        if not is_master:
            tf.get_logger().setLevel(tf.logging.FATAL)

    elif args.command == "evaluate":
        if args.gpus > -1:
            confighorovod = tf.ConfigProto()
            confighorovod.gpu_options.visible_device_list = str(args.gpus)
            K.set_session(tf.Session(config=confighorovod))

    class CocoConfig(Config):
        """Configuration for training on MS COCO.
        Derives from the base Config class and overrides values specific
        to the COCO dataset.
        """
        # Give the configuration a recognizable name
        NAME = "coco"
        if hvd:
            _GPU_COUNT = hvd.size()
            GPU_COUNT = 1 #fix batch size as IMAGES_PER_GPU
        else:
            _GPU_COUNT = abs(args.gpus)
            GPU_COUNT = _GPU_COUNT

        if args.fchollet_fix:
            BGR = True
            ## mean pixel is in RGB format to match original settings
            MEAN_PIXEL = [123.68, 116.78, 103.94]
        elif args.BGR or 'kapp_' in args.backbone:
            ## BGR/caffe format
            BGR = True
            MEAN_PIXEL = [103.94, 116.78, 123.68]
        else:
            ## default RGB mode
            BGR = False
            MEAN_PIXEL = [123.68, 116.78, 103.94]

        GT_NOISE_STD = 0

        QUICK_TEST = args.quick_test
        ## these can be used to run with dynamic shapes
        BIN_PADDING = None # 8
        IMAGE_RESIZE_MODE = "square" # "pad64"
        DYNAMIC_ANCHORS = False # True
        PRESET_LAYERS_TRAIN = args.train_layers
        if args.dynamic:
            IMAGE_RESIZE_MODE = "pad64"
            DYNAMIC_ANCHORS = True

        if BIN_PADDING or IMAGE_RESIZE_MODE in ['no_pad', 'pad64'] or QUICK_TEST:
            IMAGES_PER_GPU = 1
        else:
            IMAGES_PER_GPU = 2
        # Override if specified.
        if args.images_per_gpu is not None:
            IMAGES_PER_GPU = args.images_per_gpu
        # always evaluate using same number of samples regardless of number of gpus
        VAL_SAMPLES = 1600
        if QUICK_TEST:
            VAL_SAMPLES = 1
        _BATCH_SIZE = _GPU_COUNT*IMAGES_PER_GPU
        VALIDATION_STEPS = None # VAL_SAMPLES//_BATCH_SIZE
        if args.validation_steps is not None:
            VALIDATION_STEPS = args.validation_steps
        # lr is scaled with respect to the actual number of gpus
        LEARNING_RATE = 0.02 * (_BATCH_SIZE / 16)**0.5
        DETERMINISTIC = args.deterministic
        if args.deterministic:
            LEARNING_RATE = 0
        STEPS_PER_EPOCH = None # 5000
        PYRAMID_ROI_CUSTOM_OP = True if args.custom_roi == '1' else False
        LEARNING_MOMENTUM_CONST = True if args.momentum_const == '1' else False
        COMBINED_NMS_OP = True if args.combined_nms == '1' else False
        if args.xl_inputs:
            TRAIN_ROIS_PER_IMAGE = 512
            ROI_POSITIVE_RATIO = 0.25
            IMAGE_MIN_DIM_TRAIN = [640, 672, 704, 736, 768, 800, 832]
            IMAGE_MIN_DIM_VAL = 832
            IMAGE_MAX_DIM = 1344
        else:
            TRAIN_ROIS_PER_IMAGE = 256
            ROI_POSITIVE_RATIO = 0.33
            IMAGE_MIN_DIM_TRAIN = [640, 672, 704, 736, 768, 800]
            IMAGE_MIN_DIM_VAL = 800
            IMAGE_MAX_DIM = 1024
        if QUICK_TEST:
            TRAIN_ROIS_PER_IMAGE = 20
            IMAGE_MAX_DIM = 512
        if args.clip_norm > 0:
            GRADIENT_CLIP_NORM = args.clip_norm
        else:
            GRADIENT_CLIP_NORM = None
        # Number of classes (including background)
        NUM_CLASSES = 1 + 80  # COCO has 80 classes
        BACKBONE=args.backbone
        RPN_ONLY=args.rpn_only
        ### schedual settings
        WARMUP = 1000
        if args.warmup_steps is not None:
            WARMUP = args.warmup_steps
        if QUICK_TEST:
            WARMUP = 1
        if RPN_ONLY:
            DROPS = [40, 60]
            TOT_EPOCHS = 70
        else:
            if args.short: ## short regime
                DROPS = [77, 154]
                TOT_EPOCHS = 175
            else :## long regime
                DROPS = [210, 280]
                TOT_EPOCHS = 300

        if args.epochs is not None:
            TOT_EPOCHS = args.epochs

        if args.steps_per_epoch is not None:
            STEPS_PER_EPOCH = args.steps_per_epoch

        if STEPS_PER_EPOCH is not None:
            _SCHEDUAL_RATIO = max(STEPS_PER_EPOCH // 1000, 1)
        else:
            _SCHEDUAL_RATIO = max((117280 // _BATCH_SIZE) // 1000,1)
        for i,v in enumerate(DROPS):
            DROPS[i] = int(v/_SCHEDUAL_RATIO + 0.5)
        del i
        del v
        if args.epochs is None:
            TOT_EPOCHS = int(TOT_EPOCHS/_SCHEDUAL_RATIO + 0.5)
    class InferenceConfig(CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.001


    if args.command == "train":
        config = CocoConfig()
        mode = "training"
    else:
        config = InferenceConfig()
        mode = "inference"

    with tf.device("/device:CPU:0"):
        model = modellib.MaskRCNN(dev_str, mode=mode, config=config, model_dir=args.logs, hvd=hvd)

    exclude = None
    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        with tf.device(dev_str):
            model_path = model.get_imagenet_weights()
    else:
        model_path = args.model
        if 'r101_imagenet_init.h5' in args.model:
            exclude = r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)|(anchors.*)|(mask\_.*)|"

    # Load weights
    if is_master:
        config.display()
        model.keras_model.summary()
        print("Loading weights", model_path)
    if 'keras' not in args.model:
        # keras backbone weights are automatically loaded during build
        with tf.device(dev_str):
            model.load_weights(model_path, by_name=True, exclude=exclude, resume=args.resume, verbose=is_master)
    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        num_shards = 1
        shard_id = 0
        if hvd:
            num_shards = hvd.local_size()
            shard_id = hvd.local_rank()
        dataset_train = CocoDataset()
        dataset_train.load_coco(args.dataset, "train", year=args.year, auto_download=args.download,
                                num_shards=num_shards,shard_id=shard_id)

        if args.year in '2014':
            dataset_train.load_coco(args.dataset, "valminusminival", year=args.year, auto_download=args.download,
                                    num_shards=num_shards,shard_id=shard_id)

        dataset_train.prepare()
        # Validation dataset
        dataset_val = CocoDataset()
        val_type = "val" if args.year in '2017' else "minival"
        dataset_val.load_coco(args.dataset, val_type, year=args.year, auto_download=args.download,
                              num_shards=num_shards,shard_id=shard_id,limit=config.VAL_SAMPLES)
        dataset_val.prepare()

        augmentation = iaa.Fliplr(0.5)
        callbacks = []

        ## add callbacks here
        schedule = COCOSchedualer(config.LEARNING_RATE,warmup_steps=config.WARMUP,gamma=0.1,drops=config.DROPS,verbose=is_master)
        callbacks += [schedule]

        external_callbacks = getattr(args, 'external_callbacks', None)
        if external_callbacks is not None:
            callbacks.extend(external_callbacks)

        if is_master:
            print("Training Resnet stage 3+nobn")
        with tf.device("/device:CPU:0"):
            model.train(dev_str, dataset_train, dataset_val,
                    learning_rate= config.LEARNING_RATE,
                    epochs=config.TOT_EPOCHS,
                    layers=config.PRESET_LAYERS_TRAIN,
                    augmentation=augmentation,
                    custom_callbacks=callbacks,
                    dump_tf_timeline=args.dump_tf_timeline,
                    profile=args.profile)

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = CocoDataset()
        val_type = "val" if args.year in '2017' else "minival"
        coco = dataset_val.load_coco(args.dataset, val_type, year=args.year, return_coco=True, auto_download=args.download, limit=args.limit if args.limit > 0 else None)
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(len(dataset_val.image_info)))
        evaluate_coco(model, dataset_val, coco)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--year', required=False,
                        default=DEFAULT_DATASET_YEAR,
                        metavar="<year>",
                        help='Year of the MS-COCO dataset (2014 or 2017) (default={})'.format(DEFAULT_DATASET_YEAR))
    parser.add_argument('--model', required=True, default='keras',
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--backbone', required=False, default='kapp_Resnet101',
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,type=int,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--gpus', required=False,
                        default=1,type=int,
                        metavar="<gpu count>",
                        help='num gpus to use')
    parser.add_argument('--download', action='store_true',
                        help='Automatically download and unzip MS-COCO files (default=False)')
    parser.add_argument('--rpn-only',action='store_true',
                        help='build only rpn segment (default=False)')
    parser.add_argument('--resume',action='store_true',
                        help='training should resume from provided checkpoint (default=True)')
    parser.add_argument('--using_horovod',action='store_true',
                        help='Use horovod')
    parser.add_argument('--short',action='store_true',
                        help='Use short regime')
    parser.add_argument('--xl_input',action='store_true',
                        help='Use extra large inputs')
    parser.add_argument('--clip_norm', required=False,
                        default=5.0,type=float,
                        help='gradient clipping norm, use a negative values to not use')
    parser.add_argument('--train-layers', required=False,default='3+nobn',
                        metavar="e.g., '3+', '3+nobn'",
                        help="layers to train")
    parser.add_argument('--dynamic',action='store_true',
                        help='Use dynamic shapes')
    parser.add_argument('--xl_inputs',action='store_true',
                        help='Use xl image shapes to match detectron2')
    parser.add_argument('--BGR',action='store_true',
                        help='flip input channels and normalization to BGR, matching legcy opencv mode. This should '
                             'match the original settings of the backbone (e.g. fchollet and pytorch MSRA imported '
                             'weights use BGR)')
    parser.add_argument('--fchollet_fix', action='store_true',
                        help='fix preprocessing to use bgr format and match input normaliztion to the original setting from:'
                             ' https://github.com/fchollet/deep-learning-models/blob/3fdb8ced1b51ebb8b74b42731add80a2f1947630/imagenet_utils.py#L11'
                             ' this should only be done for models using the fchollet r50 weights')
    parser.add_argument('--epochs', required=False,
                        default=None, type=int,
                        metavar='<epochs count>',
                        help='Number of epochs. Overrides \'--short\' and \'--rpn-only\'')
    parser.add_argument('--steps_per_epoch', required=False,
                        default=None, type=int,
                        metavar='<steps per epoch count>',
                        help='Number of steps per single epoch.')
    parser.add_argument('--warmup_steps', required=False,
                        default=None, type=int,
                        metavar='<warmup step count>',
                        help='Number of warmup steps.')
    parser.add_argument('--validation_steps', required=False,
                        default=None, type=int,
                        metavar='<validation step count>',
                        help='Number of validation steps.')
    parser.add_argument('--quick_test', action='store_true', default=False,
                        help='Quick sanity check. Test only, it will affect convergence!')
    parser.add_argument('--device', choices=['CPU', 'HPU', 'GPU'], default='HPU')
    parser.add_argument('--profile', action='store_true', default=False,
                        help='Enables validation and TensorBoard log callbacks (if \'--dump_tf_timeline\' is not used).')
    parser.add_argument('--dump_tf_timeline', action='store_true', default=False,
                        help='Gathers addtional metadata.')
    parser.add_argument('--custom_roi', choices=['0', '1'], default='0',
                        help='Enables use of custom op tf_falsh.ops.pyramid_roi_align if set to 1')
    parser.add_argument('--momentum_const', choices=['0', '1'], default='1',
                        help='Enables use of momentum as Const (not as Variable) in ResourceApplyKerasMomentum if set to 1')
    parser.add_argument('--combined_nms', choices=['0', '1'], default='1',
                        help='Enables use of tf.image.combined_non_max_supression if set to 1')
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='Enables deterministic behavior to test features. Pass in a checkpoint .h5 file as --model')
    parser.add_argument('--images_per_gpu', required=False,
                        default=None, type=int,
                        metavar='<images per GPU>',
                        help='Number of images simultanously processed during step by one processing unit.')
    args = parser.parse_args()
    run_coco(args)