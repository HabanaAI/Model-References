import pytest


#####################
#  test_load_format #
#####################
formats = \
    [
        ("eval", None, ['clip_name', 'gi', 'center_im', 'inp_ims', 'ground_truth', 'x']),
        ("arch", "four_views_polys", ["I_cntr", "I_srnd_0", "I_srnd_1", "I_srnd_2", "T_cntr_srnd", "focal", "origin",
                                      "steps", "steps_polys"]),
        ("loss", "lidar_only", ["im_lidar_inv", "im_lidar_short_inv", "im_mask"])
    ]


@pytest.mark.parametrize("params", formats)
def test_load_format(params):
    from stereo.interfaces.implements import load_format
    fmt_type, fmt, expected_list = params
    assert load_format(fmt_type, fmt) == expected_list


#####################
#  test_load_format #
#####################
dataset_names = \
    [
        ("main.v2.3", """
            {   
                "format_name": "four_views",
                "compressed": true,
                "s3": "s3://mobileye-team-stereo/data/tf_datasets/v2.3.1/main",
                "prep": "main_v2.3","views_names": ["main_to_main", "frontCornerLeft_to_main", 
                "frontCornerRight_to_main", "parking_front_to_main"]
            }""")
    ]


@pytest.mark.parametrize("params", dataset_names)
def test_load_dataset_attributes(params):
    from stereo.interfaces.implements import load_dataset_attributes
    import json
    dataset_name, expected_json = params
    assert load_dataset_attributes(dataset_name)[0] == json.loads(expected_json)
