# Mask-RCNN model chagnes
The MaskRCNN script is based on https://github.com/matterport/Mask_RCNN

## Model Changes
* Support for Habana device was added.
* Horovod support.
* Runnable from TensorFlow 2.2, with keras replaced by tf.keras
* Int64 and assert workarounds.
* Script modified to offload dynamic shapes to CPU or use padding.
* Script improvements:
	* Improved root directory path detection in coco.py
	* Coco script is runnable as a module
	* Improved pool of parameters (selectable number of epochs and steps, TensorFlow timeline dump, post-training validation can be disabled, more deterministic run)
* Added support for tf.image.combined_non_max_suppression.
* Added custom Pyramid Roi Align functional blocks – fuses multiple tf.image.crop_and_resize and tf.nn.avg_pool2d.
* Backbone changed to “kapp_ResNet50”.
* SGD_with_colocate_grad – fix BW to use the same device as FW.
* CocoScheduler – learning rate regime.
