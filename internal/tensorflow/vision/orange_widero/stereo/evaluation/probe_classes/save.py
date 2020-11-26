import os

from stereo.evaluation.probe import Probe
from stereo.common.s3_utils import my_save

class Save(Probe):
    def summarize(self):
        return

    def concatenate(self):
        return

    def my_init(self):
        return

    def my_update(self, eval_out, clip_name_batch, gi_batch):
        for b in range(self.batch_size):
            self.save_output({o: eval_out[o][b] for o in eval_out}, clip_name_batch[b], gi_batch[b])

    def my_summarize(self):
        return

    def my_concatenate(self, dict_lst):
        return

    def save_output(self, data, clip_name, gi):
        output_base_path = "s3://mobileye-habana/mobileye-team-stereo/inference"
        if type(clip_name) != str:
            clip_name = clip_name.decode('utf-8')
        output_full_path = os.path.join(output_base_path, self.model_name, self.restore_iter, self.cam, clip_name,
                                        "%07d.pkl" % int(gi))
        my_save(output_full_path, data)
