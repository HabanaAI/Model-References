# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: local-venv
#     language: python
#     name: local-venv
# ---

import devkit
from me_pjs import PythonJobServer, SchedulerType
from devkit.clip import MeClip


def get_frame(args):
  clip_name, frame = args
  clip = MeClip(clip_name)
  frame = clip.get_frame(frame=frame, camera_name='main', tone_map='ltm')
  return frame[0]['pyr'][-2].im


clips = ['18-12-26_11-08-30_Alfred_Front_0015', '19-01-10_16-01-06_Front_0036', '19-05-14_16-03-39_Alfred_Front_0044']
frames = [7, 17, 27]
arg_tuples = zip(clips, frames)

pjs = PythonJobServer(SchedulerType.MEJS, folder='/mobileye/algo_STEREO3/ofers/test', full_copy_modules=[devkit])

out = pjs.run(get_frame, arg_tuples)

# +
import matplotlib.pyplot as plt

for im in out:
    plt.imshow(im, origin='lower', cmap='gray')
    plt.show()
# -


