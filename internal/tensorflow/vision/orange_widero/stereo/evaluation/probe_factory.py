'''
To add a new child class to the factory:
    1. Import the class
    2. Add it to the self.probes dictionary in __init__
'''

from stereo.evaluation.probe_classes.analyzer import Analyzer
from stereo.evaluation.probe_classes.close_lidar_probe import closeLidar
from stereo.evaluation.probe_classes.save import Save

class probeFactory(object):

    def __init__(self, probe_name="testAnalyzer"):

        self.probes = {
            "Analyzer": Analyzer,
            "closeLidar": closeLidar,
            "save": Save
        }

        if probe_name not in self.probes.keys():
            raise ValueError("You must add your probe child class to probe_factory.py")
        self.probe_name = probe_name

    def get_probe_obj(self, json_path, out_dir, restore_iter, idx, num_instances, cam, loss_name, dataset_name, blacklist,
                 suffix, debug, batch_size=12, local=False):

        return self.probes[self.probe_name](json_path, out_dir, restore_iter, idx, num_instances, cam, loss_name, dataset_name, blacklist,
                 suffix, debug, batch_size, local)