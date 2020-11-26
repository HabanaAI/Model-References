from stereo.data.prep.prep_utils import prep_view_v23

def prep_frame(dataSetIndex, frame, view_names, inferenceOnly=False, views=None, fail_missing_lidar=None):
    if views is None:
        views = dataSetIndex.read_views(frame=frame, view_names=view_names)
    fcl_frame = prep_view_v23(views, 'frontCornerLeft_to_frontCornerLeft',
                               ['parking_left_to_frontCornerLeft', 'parking_front_to_frontCornerLeft'],
                               inferenceOnly=inferenceOnly)
    if fcl_frame:
        return [fcl_frame]
    return []