from tests.tests_utils import get_all_models, clean_tmp
import pytest
import sys


jsons_to_test = get_all_models()


@pytest.mark.parametrize("conf_json", jsons_to_test)
def test_train_sm(conf_json, local):
    from stereo.models.train_sm import main as train_sm_main
    params = [conf_json]
    if local:
        params.append('--debug')
    print("params: %s" % params)
    sys.argv[1:] = params
    try:
        train_sm_main()
    finally:
        clean_tmp(conf_json)
