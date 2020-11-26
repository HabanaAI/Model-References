from stereo.data.prep.prep_utils import prep_view_v23

def prep_frame(dataSetIndex, frame, view_names, inferenceOnly=False, views=None, fail_missing_lidar=None):
    if views is None:
        views = dataSetIndex.read_views(frame=frame, view_names=view_names)
    main_frame = prep_view_v23(views, 'main_to_main',
                               ['frontCornerLeft_to_main', 'frontCornerRight_to_main', 'parking_front_to_main'],
                               inferenceOnly=inferenceOnly)
    if main_frame:
        return [main_frame]
    return []