from stereo.data.prep.prep_utils import prep_view_v23
from stereo.data.lidar_utils import visibility_estimation_imgs


def prep_frame(dataSetIndex, frame, view_names, inferenceOnly=False, views=None, fail_missing_lidar=None):
    """
    This prep_frame function is like main_v2.3.py's but adds short visibility estimation lidar filtering
    """

    if views is None:
        views = dataSetIndex.read_views(frame=frame, view_names=view_names)

    occlusion_params = {"vis_est_kwargs": {"k": 50, "upper_bound": 80, "thresh": 0.9437}}
    views['main_to_main']['lidars']['short'] = \
        visibility_estimation_imgs(views['main_to_main']['lidars']['short'], occlusion_params)

    main_frame = prep_view_v23(views, 'main_to_main',
                               ['frontCornerLeft_to_main', 'frontCornerRight_to_main', 'parking_front_to_main'],
                               inferenceOnly=inferenceOnly)
    if main_frame:
        return [main_frame]

    return []
