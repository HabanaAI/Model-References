import numpy as np
import os


def prep_frame(frame, num_of_frames=5, is_pred=False):
    base_path = "/mobileye/algo_STEREO3/stereo/data/sest"
    input_path = os.path.join(base_path, "vidar_to_top_view/v2")
    label_path = os.path.join(base_path, "labels/v2.1")
    clip_name, gi = frame.split('@')
    gi = int(gi)
    input_images = []
    for i in range(num_of_frames):
        try:
            input_image_path = os.path.join(input_path, clip_name, "%07d.npy" % (gi - 4*i))
            input_image = np.load(input_image_path).astype(np.float32)[:, :, 0]
            input_images.append(input_image)
        except IOError as e:
            print("couldn't load %s frames back" % num_of_frames)
            if not is_pred:
                raise e
            input_images.append(input_images[i - 1])

    input_images = np.transpose(input_images, (1, 2, 0))
    label_image_path = os.path.join(label_path, clip_name, "%07d.npy" % gi)
    label_image = np.expand_dims(np.load(label_image_path), -1)
    return \
        [{
            "inp": input_images,
            "label": label_image,
            "clip_name": np.array(clip_name),
            "gi": np.int32(gi)
        }]
