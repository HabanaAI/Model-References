from stereo.data.prep.prep_utils import prep_view

def prep_frame(dataSetIndex, frame, view_names, fail_missing_lidar=True, inferenceOnly=False, views=None):
    if views is None:
        views = dataSetIndex.read_views(frame=frame, view_names=view_names)
    rcr_frame = prep_view(views, 'rearCornerRight_to_rearCornerRight', 'parking_rear_to_rearCornerRight',
                          'parking_right_to_rearCornerRight',
                          fail_missing_lidar=fail_missing_lidar, inferenceOnly=inferenceOnly)
    if rcr_frame:
        return [rcr_frame]
    return []




