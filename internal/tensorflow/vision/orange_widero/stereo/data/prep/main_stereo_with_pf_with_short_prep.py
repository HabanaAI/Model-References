from stereo.data.prep.prep_utils import prep_view_n

def prep_frame(dataSetIndex, frame, view_names, fail_missing_lidar=True, inferenceOnly=False, views=None):
    if views is None:
        views = dataSetIndex.read_views(frame=frame, view_names=view_names)
    main_frame = prep_view_n(views, 'main_to_main', ['frontCornerLeft_to_main', 'frontCornerRight_to_main', 'parking_front_to_main'],
                           fail_missing_lidar=fail_missing_lidar, inferenceOnly=inferenceOnly, useShort=True)
    if main_frame:
        return [main_frame]
    return []

    
