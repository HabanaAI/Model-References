from stereo.data.prep.prep_utils import prep_view_v23

def prep_frame(dataSetIndex, frame, view_names, inferenceOnly=False, views=None, fail_missing_lidar=None):
    if views is None:
        views = dataSetIndex.read_views(frame=frame, view_names=view_names)
    rear_frame = prep_view_v23(views, 'rear_to_rear',
                               ['rearCornerRight_to_rear', 'rearCornerLeft_to_rear', 'parking_rear_to_rear'],
                               inferenceOnly=inferenceOnly)
    if rear_frame:
        return [rear_frame]
    return []