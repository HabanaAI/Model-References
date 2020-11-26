from stereo.data.prep.prep_utils import prep_view_v23

def prep_frame(dataSetIndex, frame, view_names, inferenceOnly=False, views=None, fail_missing_lidar=None):
    if views is None:
        views = dataSetIndex.read_views(frame=frame, view_names=view_names)
    rcl_frame = prep_view_v23(views, 'rearCornerLeft_to_rearCornerLeft',
                               ['parking_rear_to_rearCornerLeft', 'parking_left_to_rearCornerLeft'],
                               inferenceOnly=inferenceOnly)
    if rcl_frame:
        return [rcl_frame]
    return []