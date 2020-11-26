import pytest

################################
#  test_clip_name_to_sess_name #
################################
clips_to_sessions = \
    [
        ("19-05-26_11-51-03_Alfred_Left_0002", "19-05-26_11_51_03_Alfred_3Loops"),
        ("Town07_1587473531", "1587473531")
    ]


@pytest.mark.parametrize("clip_to_session", clips_to_sessions)
def test_clip_name_to_sess_name(clip_to_session):
    from stereo.data.clip_utils import clip_name_to_sess_name
    clip_name, session_name = clip_to_session
    assert session_name == clip_name_to_sess_name(clip_name)


################################
#  test_clip_name_to_sess_path #
################################
clips_to_sessions_paths = \
    [
        ("19-05-26_11-51-03_Alfred_Left_0002", "/mobileye/QA_Algo_06/DC/REM/JER5/DATACO182/19-05-26_11_51_03_Alfred_3Loops"),
        ("19-12-08_14-57-29_Bella_Front_0066", "/mobileye/Auto_Veh_09/DC/Israel/19-12-08_13-57-55_Bella_DATACO-142_1loop")
    ]


@pytest.mark.parametrize("clip_to_session_path", clips_to_sessions_paths)
def test_clip_name_to_sess_path(clip_to_session_path):
    from stereo.data.clip_utils import clip_name_to_sess_path
    clip_name, session_path = clip_to_session_path
    assert session_path == clip_name_to_sess_path(clip_name)


##############################
#  test_sess_name_to_vehicle #
##############################
sessions_names_to_vehicles = \
    [
        ("19-05-26_11_51_03_Alfred_3Loops", "Alfred"),
        ("19-12-08_13-57-55_Bella_DATACO-142_1loop", "Bella")
    ]


@pytest.mark.parametrize("session_name_to_vehicle", sessions_names_to_vehicles)
def test_sess_name_to_vehicle(session_name_to_vehicle):
    from stereo.data.clip_utils import sess_name_to_vehicle
    session_name, vehicle = session_name_to_vehicle
    assert vehicle == sess_name_to_vehicle(session_name)


#################################
#  test_clip_name_to_clips_dict #
#################################
clips_names_to_clips_dicts = \
    [
        ('19-12-08_14-57-29_Bella_Front_0066',
         {
             'Front': '19-12-08_14-57-29_Bella_Front_0066',
             'Left': '19-12-08_14-57-29_Bella_Left_0066',
             'Rear': '19-12-08_14-56-51_Bella_Rear_0066',
             'Right': '19-12-08_14-56-50_Bella_Right_0066'
         }
         )
    ]


@pytest.mark.parametrize("clip_to_clips_dict", clips_names_to_clips_dicts)
def test_clip_name_to_clips_dict(clip_to_clips_dict):
    from stereo.data.clip_utils import clip_name_to_clips_dict
    clip_name, clip_dicts = clip_to_clips_dict
    assert clip_dicts == clip_name_to_clips_dict(clip_name)


###################################
#  test_get_transformation_matrix #
###################################
from devkit.clip import MeClip
params_list = \
    [
        (MeClip("19-12-08_14-56-51_Bella_Rear_0066"), "frontCornerLeft", "main",
         [
            [0.71506558,  0.01379774, -0.6989212, -0.58545212],
            [0.01306899,  0.9993666,  0.03309985,  0.19064065],
            [0.69893521, -0.03280275,  0.71443233, -0.68079081],
            [0.0,  0.0,  0.0,  1.0]
        ])
    ]


@pytest.mark.parametrize("params", params_list)
def test_get_transformation_matrix(params):
    from stereo.data.clip_utils import get_transformation_matrix
    from numpy.linalg import norm
    clip, source_camera, target_camera, expected_RT = params
    RT = get_transformation_matrix(clip, source_camera, target_camera)
    assert norm(expected_RT - RT) < 1e-7
