import pytest
import sys
from tests.tests_utils import get_all_versions, clean_tmp

versions_list = get_all_versions()


@pytest.mark.skip  # prediction causes seg fault right now for some reason
@pytest.mark.parametrize("version_json", versions_list)
def test_pred_version(version_json):
    from stereo.prediction.vidar.pred_version import main as pred_version_main
    params = "-j %s -c 18-08-26_15-46-17_Alfred_Front_0034 --out_dir /tmp/ --part_num 250 --part_ind 0" % version_json
    sys.argv[1:] = params.split(" ")
    try:
        pred_version_main()
    finally:
        clean_tmp(version_json)
