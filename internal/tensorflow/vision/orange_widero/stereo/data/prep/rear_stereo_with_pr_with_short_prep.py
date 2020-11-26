from stereo.data.prep.prep_utils import prep_view_n

def prep_frame(dataSetIndex, frame, view_names, fail_missing_lidar=True, inferenceOnly=False, views=None):
    if views is None:
        views = dataSetIndex.read_views(frame=frame, view_names=view_names)
    main_frame = prep_view_n(views, 'rear_to_rear', ['rearCornerRight_to_rear', 'rearCornerLeft_to_rear', 'parking_rear_to_rear'],
                           fail_missing_lidar=fail_missing_lidar, inferenceOnly=inferenceOnly, useShort=True, useVD_notMoving=True)
    if main_frame:
        return [main_frame]
    return []
