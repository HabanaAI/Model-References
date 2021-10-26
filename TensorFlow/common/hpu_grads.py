###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################
from tensorflow.python.framework import ops
from TensorFlow.common.library_loader import habana_ops

@ops.RegisterGradient("PyramidRoiAlign")
def _PyramidRoiAlignGrad(op, grad):
  # get op tensor indices
  num_pyramid_levels = len(op.inputs) - 3
  num_boxes_tensor_index = num_pyramid_levels
  box_coords_tensor_index = num_pyramid_levels + 1
  box_to_level_map_tensor_index = num_pyramid_levels + 2
  # get image_pyramid shapes
  image_pyramid_shapes = []
  image_pyramid = []
  for pyramid_level in range(num_pyramid_levels):
    image = op.inputs[pyramid_level]
    if image.get_shape()[1:3].is_fully_defined():
      image_shape = image.get_shape()[1:3]
    else:
      raise ValueError('_pyramid_roi_align_grad: supports only static shapes with image_pyramid')
    image_pyramid_shapes.append(image_shape)
    image_pyramid.append(image)
  # gradient computation
  image_pyramid_grads = habana_ops.pyramid_roi_align_grad_images(image_pyramid, grad, op.inputs[num_boxes_tensor_index],
            op.inputs[box_coords_tensor_index], op.inputs[box_to_level_map_tensor_index],
            image_pyramid_shapes = image_pyramid_shapes,
            sampling_ratio = op.get_attr('sampling_ratio'),
            use_abs_coords = op.get_attr('use_abs_coords'),
            scale_to_level = op.get_attr('scale_to_level'))
  return image_pyramid_grads + [None, None, None]

@ops.RegisterGradient("HabanaCorrelation")
def _HabanaCorrelationGrad(op, *grads):
    """Returns grad * (y*x^(y-1), z*log(x))."""
    grads_list = list(grads)
    cntr = op.inputs[0]
    srnd = op.inputs[1]
    scales = op.inputs[2]
    trans_x = op.inputs[3]
    trans_y = op.inputs[4]
    interp = op.outputs[1]
    grid = op.outputs[2]
    grads = habana_ops.ops.habana_correlation_grad(grad_in=grads_list[0], cntr=cntr, interp=interp, grid=grid, name=op.name+"_grad")
    return grads.cntr_grad, grads.srnd_grad, scales, trans_x, trans_y
