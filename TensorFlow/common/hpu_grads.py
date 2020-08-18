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
  for pyramid_level in range(num_pyramid_levels):
    image = op.inputs[pyramid_level]
    if image.get_shape()[1:3].is_fully_defined():
      image_shape = image.get_shape()[1:3]
    else:
      raise ValueError('_pyramid_roi_align_grad: supports only static shapes with image_pyramid')
    image_pyramid_shapes.append(image_shape)
  # gradient computation
  image_pyramid_grads = habana_ops.pyramid_roi_align_grad_images(grad, op.inputs[num_boxes_tensor_index],
            op.inputs[box_coords_tensor_index], op.inputs[box_to_level_map_tensor_index],
            num_pyramid_levels = num_pyramid_levels,
            image_pyramid_shapes = image_pyramid_shapes,
            sampling_ratio = op.get_attr('sampling_ratio'))
  return image_pyramid_grads + [None, None, None]