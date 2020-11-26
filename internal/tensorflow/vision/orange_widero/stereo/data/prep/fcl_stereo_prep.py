import numpy as np
from stereo.data.prep.prep_utils import prep_view, prep_input_structure_view

def prep_frame(dataSetIndex, frame, view_names, fail_missing_lidar=True, inferenceOnly=False, views=None):
    if views is None:
        views = dataSetIndex.read_views(frame=frame, view_names=view_names)
    fcl_frame = prep_view(views, 'frontCornerLeft_to_frontCornerLeft', 'parking_front_to_frontCornerLeft',
                          'parking_left_to_frontCornerLeft',
                          fail_missing_lidar=fail_missing_lidar, inferenceOnly=inferenceOnly)
    if fcl_frame:
        return [fcl_frame]
    return []



