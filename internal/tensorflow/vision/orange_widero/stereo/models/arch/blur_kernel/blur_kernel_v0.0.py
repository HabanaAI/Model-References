import tensorflow as tf

from stereo.interfaces.implements import implements_format
from stereo.models.arch_utils import blur_kernel_conv2d


arch_format = ("arch", "blur_kernel")
@implements_format(*arch_format)
def arch(main_to_main,
         frontCornerLeft_to_frontCornerLeft,
         frontCornerRight_to_frontCornerRight,
         rearCornerLeft_to_rearCornerLeft,
         rearCornerRight_to_rearCornerRight,
         rear_to_rear,
         kernel_rad = 2):

    cntrs = {'main_to_main': main_to_main,
             'frontCornerLeft_to_frontCornerLeft': frontCornerLeft_to_frontCornerLeft,
             'frontCornerRight_to_frontCornerRight': frontCornerRight_to_frontCornerRight,
             'rearCornerLeft_to_rearCornerLeft': rearCornerLeft_to_rearCornerLeft,
             'rearCornerRight_to_rearCornerRight': rearCornerRight_to_rearCornerRight,
             'rear_to_rear' : rear_to_rear}
    cam_srnd = {'main_to_main': ['main_to_frontCornerLeft', 'main_to_frontCornerRight', 'main_to_parking_front'],
                'frontCornerLeft_to_frontCornerLeft': ['frontCornerLeft_to_parking_front', 'frontCornerLeft_to_parking_left'],
                'frontCornerRight_to_frontCornerRight': ['frontCornerRight_to_parking_front', 'frontCornerRight_to_parking_right'],
                'rearCornerLeft_to_rearCornerLeft': ['rearCornerLeft_to_parking_rear', 'rearCornerLeft_to_parking_left'],
                'rearCornerRight_to_rearCornerRight': ['rearCornerRight_to_parking_rear', 'rearCornerRight_to_parking_right'],
                'rear_to_rear' : ['rear_to_rearCornerLeft', 'rear_to_rearCornerRight', 'rear_to_parking_rear']}
    all_srnd = ['main_to_frontCornerLeft', 'main_to_frontCornerRight', 'main_to_parking_front',
                'frontCornerLeft_to_parking_front', 'frontCornerLeft_to_parking_left',
                'frontCornerRight_to_parking_front', 'frontCornerRight_to_parking_right',
                'rearCornerLeft_to_parking_rear', 'rearCornerLeft_to_parking_left',
                'rearCornerRight_to_parking_rear', 'rearCornerRight_to_parking_right',
                'rear_to_rearCornerLeft', 'rear_to_rearCornerRight', 'rear_to_parking_rear']

    blurred = {}
    for cntr in cntrs.keys():
        for srnd in cam_srnd[cntr]:
            blurred[srnd] = blur_kernel_conv2d(cntrs[cntr], kernel_rad=kernel_rad, name=srnd)
    outs = []
    for srnd in all_srnd:
        outs.append(blurred[srnd])
    return tf.concat(outs, axis=1)

