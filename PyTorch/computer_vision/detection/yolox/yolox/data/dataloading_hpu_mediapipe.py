###########################################################################
# Copyright (C) 2025 Habana Labs, Ltd. an Intel Company
###########################################################################

# Customized mediapipe based on HPUMediaPipe in:
# - habana_frameworks/medialoaders/torch/media_dataloader_mediapipe.py
#
# Customized iterator based on HPUSsdPytorchIterator in:
# - habana_frameworks/mediapipe/plugins/iterator_pytorch.py

from habana_frameworks.mediapipe.mediapipe import MediaPipe
from habana_frameworks.mediapipe import fn
from habana_frameworks.mediapipe.media_types import imgtype as it
from habana_frameworks.mediapipe.media_types import ftype as ft
from habana_frameworks.mediapipe.media_types import decoderStage as ds

from habana_frameworks.mediapipe.plugins.iterator_pytorch import HPUGenericPytorchIterator
from torch import hsplit
from torch import narrow

import time     # for random seed

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

class YoloxMediaPipe(MediaPipe):
    instance_count = 0

    def __init__(self, a_torch_transforms=None, a_root=None, a_annotation_file=None, a_batch_size=1, a_shuffle=False, a_drop_last=True, a_prefetch_count=1, a_num_instances=1, a_instance_id=0, a_device=None):
        self.super_init         = False

        self.root               = a_root
        self.annotation_file    = a_annotation_file
        self.a_batch_size       = a_batch_size
        self.shuffle            = a_shuffle
        self.drop_last          = a_drop_last
        self.a_prefetch_count   = a_prefetch_count
        self.num_instances      = a_num_instances
        self.instance_id        = a_instance_id
        self.device             = a_device

        transform = a_torch_transforms
        if not isinstance(transform.val, bool):
            raise ValueError("Unsupported value of transform.val ", transform.val)

        if not isinstance(transform.trans_val, transforms.Compose):
            raise ValueError("transform.trans_val should be of type torchvision.transforms")

        num_resize    = 0
        resize_width  = None
        resize_height = None
        res_pp_filter = None
        for t in transform.trans_val.transforms:
            # see docs for transform.Resize for how the values for size, max_size should be interpreted
            # https://pytorch.org/vision/main/generated/torchvision.transforms.Resize.html#torchvision.transforms.Resize
            if isinstance(t, transforms.Resize):
                self.resize_transform = t

                # validate resize H/W
                check_maxsize = False
                if num_resize > 0:
                    raise ValueError("Only one Resize transform is allowed")

                num_resize += 1
                if isinstance(t.size, int):
                    resize_height = t.size
                    resize_width = t.size
                    check_maxsize = True
                elif (isinstance(t.size, tuple) or (isinstance(t.size, list))) and len(t.size) == 1:
                    resize_height = t.size[0]
                    resize_width = t.size[0]
                    check_maxsize = True
                elif (isinstance(t.size, tuple) or (isinstance(t.size, list))) and len(t.size) == 2:
                    resize_height = t.size[0]
                    resize_width = t.size[1]
                else:
                    raise ValueError("Unsupported size: transforms.Resize ")

                if (resize_width % 2 != 0) or (resize_height % 2 != 0):
                    raise ValueError("Unsupported w:h for resize (must be multiple of 2)")

                # resize_height will be same as resize_width in this case (max_size only applies if size is length 1)
                if (check_maxsize == True) and (t.max_size != None) and (resize_width > t.max_size):
                    raise ValueError("max_size must be greater than resize width/height for transform: " + str(type(t)))

                # map interpolation mode from transform to mediapipe ftype
                if t.interpolation == InterpolationMode.BILINEAR:
                    res_pp_filter = ft.BI_LINEAR  # 3
                elif t.interpolation == InterpolationMode.NEAREST:
                    res_pp_filter = ft.NEAREST  # 2
                elif t.interpolation == InterpolationMode.BICUBIC:
                    res_pp_filter = ft.BICUBIC  # 4
                elif t.interpolation == InterpolationMode.BOX:
                    res_pp_filter = ft.BOX  # 6
                elif t.interpolation == InterpolationMode.LANCZOS:
                    res_pp_filter = ft.LANCZOS  # 1
                elif t.interpolation == InterpolationMode.HAMMING:
                    print("Warning: InterpolationMode.HAMMING not supported, using InterpolationMode.BILINEAR")
                    res_pp_filter = ft.BI_LINEAR
                else:
                    raise ValueError("Unsupported InterpolationMode:" + t.interpolation)

            else:
                raise ValueError("Unsupported transform type:" + str(type(t)))

        if num_resize != 1:
            raise ValueError("Unsupported resize count")

        # for clarity save the resized dimensions only (in this stage we don't know/care the original decoded image size)
        self.resize_width  = resize_width
        self.resize_height = resize_height
        self.res_pp_filter = res_pp_filter

        print("MediaDataloader num instances {} instance id {}".format(
            self.num_instances, self.instance_id))

        YoloxMediaPipe.instance_count += 1
        pipename = "{}:{}".format(self.__class__.__name__, YoloxMediaPipe.instance_count)
        pipename = str(pipename)

        # TODO - print log info in one place
        #print("transform Resize: w:h ", resize_width, resize_height, " interpolation: ", t.interpolation, " max_size: ", t.max_size)
        #print("Decode w:h ", self.decode_width, self.decode_height, " , Crop disabled")
        #print("MediaDataloader shuffle is ", self.shuffle)
        #print("MediaDataloader output type is ", self.media_output_dtype)

        # init MediaPipe class
        self.super_init = True
        super().__init__(device=a_device, batch_size=a_batch_size, prefetch_depth=a_prefetch_count, pipe_name=pipename)

    def __del__(self):
        if self.super_init == True:
            super().__del__()

    def definegraph(self):
        # random seed
        seed_mediapipe = int(time.time_ns() % (2**31 - 1))

        # read images and metadata from CocoReader
        self.input = fn.CocoReader(root=self.root, annfile=self.annotation_file, seed=seed_mediapipe, shuffle=self.shuffle,
            drop_remainder=self.drop_last, num_slices=self.num_instances, slice_index=self.instance_id, partial_batch=True, pad_remainder=True)
        
        # decode with resizing
        output_image_size = [self.resize_width, self.resize_height]
        self.decode = fn.ImageDecoder(output_format=it.RGB_P, resize=output_image_size, resampling_mode=self.res_pp_filter, decoder_stage=ds.ENABLE_ALL_STAGES)

        # convert RGB_P to BGR_P
        self.csc = fn.ColorSpaceConversion(colorSpaceMode=1)

        # run pipeline: read image, decode+resize, CSC       
        jpegs, ids, sizes, boxes, labels, lengths, batch = self.input()
        images = self.decode(jpegs)
        images = self.csc(images)
        
        return images, ids, sizes, boxes, labels, lengths, batch

class YoloxPytorchIterator(HPUGenericPytorchIterator):
    def __init__(self, mediapipe, pad_last_batch=True):
        super().__init__(mediapipe=mediapipe)
        self.pad_last_batch = pad_last_batch

    def __next__(self):
        # lengths is not returned from iterator
        images, ids, sizes, boxes, labels, lengths, batch = self.pipe.run()

        images_tensor   = self.proxy_device.get_tensor(images.dev_addr)
        ids_tensor      = self.proxy_device.get_tensor(ids.dev_addr)
        sizes_tensor    = self.proxy_device.get_tensor(sizes.dev_addr)
        img_size        = list(hsplit(sizes_tensor, 2))  # Split the tensor
        b_size          = self.pipe.getBatchSize()
        img_size[0]     = img_size[0].reshape(b_size)
        img_size[1]     = img_size[1].reshape(b_size)

        boxes_tensor  = self.proxy_device.get_tensor(boxes.dev_addr)
        labels_tensor = self.proxy_device.get_tensor(labels.dev_addr)
        batch_tensor  = self.proxy_device.get_tensor(batch.dev_addr)

        # get number of valid images, the rest are padding (reader repeats last image)
        batch_tensor_cpu = batch_tensor.to('cpu')
        batch_np = batch_tensor_cpu.numpy()
        batch = batch_np[0]

        # by default we return a full batch of images, with padding for the last partial batch,
        #   to avoid graph recompilation of the model - this is why images_tensor is not narrow'ed here
        # caller can check ids_tensor.size(0) to get the number of *valid* images and just discard the rest
        if batch < b_size:
            if not self.pad_last_batch:
                images_tensor   = narrow(images_tensor, 0, 0, batch)

            ids_tensor      = narrow(ids_tensor, 0, 0, batch)
            img_size[0]     = narrow(img_size[0], 0, 0, batch)
            img_size[1]     = narrow(img_size[1], 0, 0, batch)
            boxes_tensor    = narrow(boxes_tensor, 0, 0, batch)
            labels_tensor   = narrow(labels_tensor, 0, 0, batch)

        # return tuple in the same format as default PT iterator so we don't have to change calling code
        return images_tensor, boxes_tensor, img_size, ids_tensor
