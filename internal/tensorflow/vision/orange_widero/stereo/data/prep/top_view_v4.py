import numpy as np
import os
from devkit.clip import MeClip


def prep_frame(frame, is_pred=False):
    base_path = "/mobileye/algo_STEREO3/stereo/data/sest"
    input_path = os.path.join(base_path, "vidar_to_top_view/v4")
    label_path = os.path.join(base_path, "labels/v2.2")
    clip_name, gi = frame.split('@')
    gi = int(gi)
    input_image_path = os.path.join(input_path, clip_name, "%07d.npy" % gi)
    input_image = np.expand_dims(np.load(input_image_path).astype(np.float32), -1)
    label_image_path = os.path.join(label_path, clip_name, "%07d.npy" % gi)
    label_image = np.expand_dims(np.load(label_image_path), -1)
    clip = MeClip(clip_name)
    clip_meta = clip.get_frame(grab_index=gi, only_meta=True)[0]['meta']
    return \
        [{
            "inp": input_image,
            "label": label_image,
            "clip_name": np.array(clip_name),
            "gi": np.int32(gi),
            "yaw_rate": clip_meta['vehicleYawRateDegreesPerSecond'],
            "speed": clip_meta['speed']
        }]
