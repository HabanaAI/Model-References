from . import segnet

model_from_name = {}

model_from_name["segnet"] = segnet.segnet
model_from_name["vgg_segnet"] = segnet.vgg_segnet