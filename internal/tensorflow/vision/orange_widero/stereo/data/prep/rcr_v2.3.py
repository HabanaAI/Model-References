from stereo.data.prep.prep_utils import prep_view_v23

def prep_frame(dataSetIndex, frame, view_names, inferenceOnly=False, views=None, fail_missing_lidar=None):
    if views is None:
        views = dataSetIndex.read_views(frame=frame, view_names=view_names)
    rcr_frame = prep_view_v23(views, 'rearCornerRight_to_rearCornerRight',
                               ['parking_right_to_rearCornerRight', 'parking_rear_to_rearCornerRight'],
                               inferenceOnly=inferenceOnly)
    if rcr_frame:
        return [rcr_frame]
    return []