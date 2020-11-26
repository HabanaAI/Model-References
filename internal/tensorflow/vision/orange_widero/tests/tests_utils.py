import os
import shutil


def get_all_models():
    stereo_base_conf_dir = 'stereo/models/conf/stereo'
    models = os.listdir(stereo_base_conf_dir)
    return [os.path.join(stereo_base_conf_dir, conf_json) for conf_json in models]


def get_all_versions():
    base_conf_dir = 'stereo/prediction/vidar/vidar_versions'
    versions = [j for j in os.listdir(base_conf_dir) if j.endswith("json")]
    versions_paths = [os.path.join(base_conf_dir, conf_json) for conf_json in versions]
    return versions_paths


def clean_tmp(full_path_version_json):
    version_json = os.path.basename(full_path_version_json)[:-5]  # without .json
    try:
        shutil.rmtree(os.path.join('/tmp/', version_json))
    except OSError:
        pass
