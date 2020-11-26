import numpy as np
import os


def prep_frame(frame, is_pred=False):
    base_path = "/mobileye/algo_STEREO3/stereo/data/sest"
    input_path = os.path.join(base_path, "vidar_to_top_view/v3")
    label_path = os.path.join(base_path, "labels/v2.1")
    clip_name, gi = frame.split('@')
    gi = int(gi)
    input_image_path = os.path.join(input_path, clip_name, "%07d.npy" % gi)
    input_image = np.load(input_image_path).astype(np.float32).transpose((1, 2, 0))
    label_image_path = os.path.join(label_path, clip_name, "%07d.npy" % gi)
    label_image = np.expand_dims(np.load(label_image_path), -1)
    return \
        [{
            "inp": input_image,
            "label": label_image,
            "clip_name": np.array(clip_name),
            "gi": np.int32(gi)
        }]
