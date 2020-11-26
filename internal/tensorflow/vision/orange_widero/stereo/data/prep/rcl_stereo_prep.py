from stereo.data.prep.prep_utils import prep_view

def prep_frame(dataSetIndex, frame, view_names, fail_missing_lidar=True, inferenceOnly=False, views=None):
    if views is None:
        views = dataSetIndex.read_views(frame=frame, view_names=view_names)
    rcl_frame = prep_view(views, 'rearCornerLeft_to_rearCornerLeft', 'parking_left_to_rearCornerLeft',
                          'parking_rear_to_rearCornerLeft',
                          fail_missing_lidar=fail_missing_lidar, inferenceOnly=inferenceOnly)
    if rcl_frame:
        return [rcl_frame]
    return []



